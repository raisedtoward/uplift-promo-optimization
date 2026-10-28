import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from atlas.uplift.learners import SLearner, UpliftTreeLearner, UpliftRandomForestLearner, UpliftBaseModel

from src import config as cfg


def fit_learners(train_data: pd.DataFrame) -> Dict[str, object]:
    """
    multi-treatment uplift learners compared in the research notebook, URF is the one used for decisions
    """
    logger = logging.getLogger(cfg.LOGGER_NAME)
    fitted_learners = {
        'S': SLearner(
            base_model = UpliftBaseModel(model_class = LGBMClassifier, model_parameters = cfg.boost_params),
            treatment_name = cfg.treatment_col,
            add_treatment_features_interaction = False,
            logger_name = cfg.LOGGER_NAME,
        ),
        'UT': UpliftTreeLearner(max_depth = 10, min_samples_leaf = 30, seed = cfg.seed, logger_name = cfg.LOGGER_NAME),
        'URF': UpliftRandomForestLearner(n_estimators = 50, max_depth = 10, min_samples_leaf = 30, seed = cfg.seed, logger_name = cfg.LOGGER_NAME),
    }
    X, y, t = train_data[cfg.uplift_features], train_data[cfg.target_col], train_data[cfg.treatment_col]
    for learner_type, learner in fitted_learners.items():
        logger.info(msg = f'fitting {learner_type} learner')
        learner.fit(X = X, y = y, t = t)
    return fitted_learners


def predict_uplift(learner, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    returns uplift_predictions (N, 4) - uplift_1..uplift_4 vs control and control_prediction (N,) - prediction_0
    """
    uplift_predictions, control_prediction = learner.predict(X = data[learner.features], return_baseline = True)
    uplift_predictions.index = data.index
    control_prediction.index = data.index
    return uplift_predictions, control_prediction


def convert_uplift_to_target_predictions(uplift_predictions: pd.DataFrame, control_prediction: pd.Series) -> pd.DataFrame:
    target_predictions = pd.concat(objs = [control_prediction, uplift_predictions.add(other = control_prediction, axis = 'index')], axis = 1)
    target_predictions.columns = [c.replace('uplift', 'prediction') for c in target_predictions.columns]
    return target_predictions


def fit_aov_model(train_data: pd.DataFrame) -> LGBMRegressor:
    """
    imputes AOV for users without orders in the last 90 days, trained on ordering users only
    """
    ordering_users = train_data.loc[train_data['aov_90d'] > 0, :]
    aov_model = LGBMRegressor(random_state = cfg.seed, verbosity = -1)
    aov_model.fit(X = ordering_users[cfg.aov_predictive_features], y = ordering_users['aov_90d'])
    return aov_model


def get_aov_estimate(aov_model: LGBMRegressor, data: pd.DataFrame) -> pd.Series:
    """
    actual aov_90d where available, aov_90d_forecast for the rest
    """
    data = data.copy()
    data['aov_90d_forecast'] = aov_model.predict(X = data[cfg.aov_predictive_features])
    data['aov_estimate'] = data['aov_90d'].fillna(value = 0)
    data.loc[data['aov_estimate'] == 0, 'aov_estimate'] = data.loc[data['aov_estimate'] == 0, 'aov_90d_forecast']
    return data['aov_estimate']


def fit_utilization_model(train_data: pd.DataFrame) -> LGBMClassifier:
    """
    P(voucher_used = 1 | D = d, FD = fd, X = x), fitted on voucher arms only (no voucher can't be used)
    """
    utilization_model_mask = (train_data[cfg.treatment_col] != cfg.CONTROL_LABEL)
    utilization_model = LGBMClassifier(**cfg.boost_params)
    utilization_model.fit(
        X = train_data.loc[utilization_model_mask, cfg.utilization_model_features],
        y = train_data.loc[utilization_model_mask, cfg.utilization_col],
    )
    return utilization_model


def get_utilization_estimate(utilization_model: LGBMClassifier, data: pd.DataFrame) -> pd.DataFrame:
    """
    (N, 4) U_estimate_1,...,U_estimate_4 - expected utilization if voucher level t was offered
    """
    U_X = data.loc[:, cfg.utilization_model_features].copy()
    U_estimate = []
    for t_label in cfg.treatment_labels:
        if t_label == cfg.CONTROL_LABEL:
            continue
        U_X[cfg.treatment_col] = t_label
        U_estimate_t = pd.Series(data = utilization_model.predict_proba(X = U_X)[:, 1], index = U_X.index, name = f'U_estimate_{t_label}')
        U_estimate.append(U_estimate_t)
    return pd.concat(objs = U_estimate, axis = 1)


def get_arm_level_mappings(train_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    group averages by (voucher, FD) used by the 1st and 2nd spend formulations, (FD x voucher levels 1..4) tables
    """
    def _pivot(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
        gb = data.groupby(by = [cfg.aux_treatment_col, cfg.treatment_col])[value_col].mean().unstack(level = -1)
        return gb.loc[:, [t for t in cfg.treatment_labels if t != cfg.CONTROL_LABEL]]

    return {
        ### P(voucher_used = 1 | ordered = 1, D = d, FD = fd) - 1st formulation
        'utilization_rate_given_order_mapping': _pivot(data = train_data.loc[train_data[cfg.target_col] == 1, :], value_col = cfg.utilization_col),
        ### E[voucher_cost | voucher_used = 1, D = d, FD = fd] - 1st formulation
        'avg_cost_mapping': _pivot(data = train_data.loc[train_data[cfg.utilization_col] == 1, :], value_col = cfg.cost_col),
        ### P(voucher_used = 1 | D = d, FD = fd) - 2nd formulation
        'utilization_rate_mapping': _pivot(data = train_data, value_col = cfg.utilization_col),
    }


def map_by_free_delivery(mapping: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
    return mapping.loc[data[cfg.aux_treatment_col].to_numpy(), :].to_numpy(dtype = float)


def get_expected_voucher_cost(aov: pd.Series) -> np.ndarray:
    """
    (N, 4) cost of a used voucher, min(d * AOV, 9 CU)
    """
    discounts = np.array(object = [cfg.labels_mapping[t] for t in cfg.treatment_labels if t != cfg.CONTROL_LABEL]) / 100
    discounts_grid = np.tile(A = discounts, reps = (len(aov), 1))
    return np.clip(a = discounts_grid * aov.to_numpy()[:, None], a_min = 0, a_max = cfg.voucher_cap)
