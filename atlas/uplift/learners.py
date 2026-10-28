import warnings
from typing import Any, Callable, Dict, List, Tuple, Union
import logging
import multiprocessing as mp
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from IPython.display import Image

from causalml.inference.tree import uplift_tree_plot
from causalml.inference.tree import DecisionTree, UpliftTreeClassifier, UpliftRandomForestClassifier

from atlas.utils import ArrayLike1d, BaseModel

from pulsar.common.validation import assert_treatment_labels
from pulsar.common.constants import CONTROL_LABEL, TREATMENT_LABEL, TREATMENT_COL

DEFAULT_SEED = 8

LGBM_DEFAULT_PARAMS = {
    'random_state': DEFAULT_SEED,
    'seed': DEFAULT_SEED,
    'verbosity': -1,
}


### wrappers for sklearn-api-like models to use inside meta-learners
class UpliftBaseModel:
    def __init__(
        self,
        model_class,
        model_parameters: Dict = None
    ):
        """
        this class should implement fit and predict methods
        """

        model_parameters = model_parameters or {}
        if model_class in (LGBMClassifier, LGBMRegressor):
            model_parameters = {**LGBM_DEFAULT_PARAMS, **model_parameters}
        self.model = model_class(**model_parameters)
        self.proba_model_types = (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LGBMClassifier)

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        **kwargs,
    ):
        if isinstance(self.model, (LGBMClassifier, LGBMRegressor)):
            self.model.set_params(verbosity = -1)
        self.model.fit(X, y, **kwargs)

    def predict(self, X: pd.DataFrame, **kwargs) -> Union[pd.Series, np.ndarray]:
        if isinstance(self.model, self.proba_model_types):
            return self.model.predict_proba(X, **kwargs)[:, 1]
        else:
            return self.model.predict(X, **kwargs)


class UpliftRegressionModel(UpliftBaseModel):
    def __init__(self, *args, **kwargs):
        """
        this class should check if underlying model is for regression
        """
        super().__init__(*args, **kwargs)
        self.allowed_model_types = (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, LGBMRegressor)
        if not isinstance(self.model, self.allowed_model_types):
            raise TypeError(
                f"provided model should be one of the allowed regression models type: "
                f"{self.allowed_model_types}"
            )


class BaseLearner(BaseModel):
    def __init__(self):
        ### 4 (or similarly 1/4) would correspond to 80%/20% to be allowed, but 81%/19% would exceed the threshold
        self.max_treatment_control_imbalance = 9
        self.treatment_features = None
        self.control_features = None
        self.features = None
        self.treatment_name = None
        self.control_label = CONTROL_LABEL

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        t: ArrayLike1d,
        **kwargs,
    ):
        raise NotImplementedError("not implemented in base class")

    def predict(self, X: pd.DataFrame, **kwargs):
        raise NotImplementedError("not implemented in base class")

    def _check_treatment(self, t: ArrayLike1d, force_binary: bool) -> pd.Series:

        t = pd.Series(t, name = self.treatment_name)
        assert_treatment_labels(labels=t, force_binary=force_binary, force_consecutive=True)
        if force_binary:
            group_shares = t.value_counts(normalize = True)
            group_shares /= group_shares.loc[0]
            group_shares = np.maximum(group_shares, 1 / group_shares)
            if any(group_shares > self.max_treatment_control_imbalance):
                raise ValueError(
                    f"treatment share and control share of population should not differ more than "
                    f"x{self.max_treatment_control_imbalance:.4f}, got distribution: {group_shares:.4f}"
                )
        return t

    def _check_inputs(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        t: ArrayLike1d,
        force_binary_treatment: bool,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:

        t = self._check_treatment(t, force_binary = force_binary_treatment)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        if not all(X.index == y.index):
            raise ValueError("indices of X and y do not match")

        return X, y, t

    def _prepare_target_2groups(
        self,
        y: ArrayLike1d,
        t: ArrayLike1d,
    ) -> Tuple[pd.Series, pd.Series]:

        if not all(y.index == t.index):
            raise ValueError("indices of y and t do not match")
        treatment_mask = (t != self.control_label)
        y_treatment = y.loc[treatment_mask]
        y_control = y.loc[~treatment_mask]

        return y_treatment, y_control

    def _prepare_features_2groups(
        self,
        X: pd.DataFrame,
        t: ArrayLike1d,
    ) -> Tuple[pd.DataFrame, pd.Series]:

        if not all(X.index == t.index):
            raise ValueError("indices of X and t do not match")
        treatment_mask = (t != self.control_label)
        X_treatment = X.loc[treatment_mask, self.treatment_features]
        X_control = X.loc[~treatment_mask, self.control_features]

        return X_treatment, X_control

    def _check_X(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Index, None]]:

        index = None
        if isinstance(X, pd.DataFrame):
            if not list(X.columns) == self.features:
                raise ValueError(f"provided features do not match the ones on which learner was fitted: {self.features}")
            index = X.index
            X = X.values

        return X, index

    def subset_features(self, X: pd.DataFrame) -> pd.DataFrame:
        all_features = set(self.features)
        missing_features = list(all_features.difference(X.columns))
        if len(missing_features):
            raise AttributeError(f"{missing_features} features are missing from X data")
        X = X.loc[:, self.features]
        return X


class SLearner(BaseLearner):
    """
    Supports multiple treatments vs single control design
    """
    def __init__(
        self,
        base_model: UpliftBaseModel,
        add_treatment_features_interaction: bool = False,
        max_treatment_control_imbalance: float = 9.0,
        treatment_name: str = TREATMENT_COL,
        logger_name: str = None,
    ):
        """
        base_model should implement fit and predict methods
        """
        super().__init__()
        if not isinstance(base_model, UpliftBaseModel):
            raise TypeError("base model should be of `UpliftBaseModel` type")
        self.base_model = base_model
        self.interaction = add_treatment_features_interaction
        self.features = None
        self.max_treatment_control_imbalance = max(max_treatment_control_imbalance, 1 / max_treatment_control_imbalance)
        self.treatment_name = treatment_name
        self.treatment_groups = None
        self.logger_name = logger_name

    def _prepare_features(
        self,
        X: pd.DataFrame,
        t: ArrayLike1d,
    ) -> pd.DataFrame:
        """
        this is different from inherited method due to feature interactions being potentially included
        """
        if not all(X.index == t.index):
            raise ValueError("indices of X and t do not match")
        if self.interaction:
            init_columns = X.columns
            X = pd.concat([X, X * t.values[:, None], t], axis = 1)
            X.columns = list(init_columns) + [f'{c}__{self.treatment_name}' for c in init_columns] + [self.treatment_name]
        else:
            X = pd.concat([X, t], axis = 1)
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        t: ArrayLike1d,
        **kwargs,
    ):
        logger = logging.getLogger(self.logger_name)

        X, y, t = self._check_inputs(X, y, t, force_binary_treatment=False)
        self.features = list(X.columns)
        self.treatment_groups = sorted(t.unique())
        X = self._prepare_features(X, t)
        logger.info("Checks passed, fitting model..")
        self.base_model.fit(X, y, **kwargs)
        logger.info("Model fitted, exiting..")

    def predict(
        self,
        X: pd.DataFrame,
        return_baseline: bool = False,
        **kwargs,
    ) -> Union[Union[pd.DataFrame, np.ndarray], Tuple[Union[pd.DataFrame, np.ndarray], pd.Series]]:
        logger = logging.getLogger(self.logger_name)

        X = self.subset_features(X)
        X_control = self._prepare_features(
            X,
            t = pd.Series(np.ones(len(X)) * self.control_label, index = X.index, name = self.treatment_name)
        )
        logger.info(f"Calculating control (treatment {self.control_label}) group prediction..")
        control_prediction = pd.Series(
            self.base_model.predict(X_control, **kwargs),
            name = f'prediction_{self.control_label}',
            index = X.index,
        )

        prediction_output = pd.DataFrame(
            np.zeros((len(X), len(self.treatment_groups) - 1)),
            columns = [f'uplift_{t}' for t in self.treatment_groups if t != self.control_label],
            index = X.index,
        )
        for t in self.treatment_groups:
            if t == self.control_label:
                continue
            X_treatment = self._prepare_features(
                X,
                t = pd.Series(np.ones(len(X)) * t, index=X.index, name=self.treatment_name)
            )
            logger.info(f"Calculating treatment {t} group prediction..")
            treatment_prediction = self.base_model.predict(X_treatment, **kwargs)
            prediction_output[f'uplift_{t}'] = treatment_prediction - control_prediction

        if return_baseline:
            return prediction_output, control_prediction
        else:
            return prediction_output


class TLearner(BaseLearner):
    def __init__(
        self,
        treatment_model: UpliftBaseModel,
        control_model: UpliftBaseModel,
        max_treatment_control_imbalance: float = 9.0,
        treatment_name: str = TREATMENT_COL,
        logger_name: str = None,
    ):
        """
        treatment_model and control_model, should implement fit and predict methods
        """
        super().__init__()
        if not isinstance(treatment_model, UpliftBaseModel):
            raise TypeError("treatment_model should be of `UpliftBaseModel` type")
        if not isinstance(control_model, UpliftBaseModel):
            raise TypeError("control_model should be of `UpliftBaseModel` type")
        self.treatment_model = treatment_model
        self.control_model = control_model
        self.treatment_calibration_model = None
        self.control_calibration_model = None
        self.treatment_features = None
        self.control_features = None
        self.features = None
        self.max_treatment_control_imbalance = max(max_treatment_control_imbalance, 1/max_treatment_control_imbalance)
        self.treatment_name = treatment_name
        self.logger_name = logger_name

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        t: ArrayLike1d,
        treatment_features: List[str] = None,
        control_features: List[str] = None,
        treatment_kwargs: Dict = None,
        control_kwargs: Dict = None,
    ):
        logger = logging.getLogger(self.logger_name)

        X, y, t = self._check_inputs(X, y, t, force_binary_treatment=True)

        if treatment_features:
            self.treatment_features = treatment_features
        else:
            self.treatment_features = list(X.columns)
        if control_features:
            self.control_features = control_features
        else:
            self.control_features = list(X.columns)
        self.features = list(set(self.treatment_features + self.control_features))

        X_treatment, X_control = self._prepare_features_2groups(X, t)
        y_treatment, y_control = self._prepare_target_2groups(y, t)

        treatment_kwargs = treatment_kwargs or {}
        control_kwargs = control_kwargs or {}

        logger.info("Fitting treatment model..")
        self.treatment_model.fit(X_treatment, y_treatment, **treatment_kwargs)
        logger.info("Fitting control model..")
        self.control_model.fit(X_control, y_control, **control_kwargs)
        logger.info("Both models fitted, exiting..")

    def calibrate(
        self,
        y_true: ArrayLike1d,
        y_score: ArrayLike1d,
        segment: str,
    ):
        logger = logging.getLogger(self.logger_name)

        if segment not in ('treatment', 'control'):
            raise ValueError("segment should be one of the ('treatment', 'control')")
        model = UpliftBaseModel(LogisticRegression, {'penalty': None})
        logger.info(f"Fitting calibration model for {segment} segment..")
        model.fit(y_score[:, None], y_true)
        if segment == 'treatment':
            self.treatment_calibration_model = model
        else:
            self.control_calibration_model = model
        logger.info("Model fitted, exiting..")

    def predict(
        self,
        X: pd.DataFrame,
        prediction_type: str = 'uplift',
    ) -> pd.Series:
        logger = logging.getLogger(self.logger_name)
        if prediction_type not in ('uplift', 'treatment', 'control'):
            raise ValueError("prediction_type should be one of the ('uplift', 'treatment', 'control')")
        X = pd.DataFrame(X)

        all_features = set(self.treatment_features + self.control_features)
        missing_features = list(all_features.difference(X.columns))
        if len(missing_features):
            raise AttributeError(f"{missing_features} features are missing from X data")

        X_treatment = X.loc[:, self.treatment_features]
        X_control = X.loc[:, self.control_features]
        if prediction_type in ('treatment', 'uplift'):
            logger.info("Calculating treatment prediction..")
            treatment_prediction = self.treatment_model.predict(X_treatment)
            if self.treatment_calibration_model:
                treatment_prediction_calibrated = self.treatment_calibration_model.predict(treatment_prediction[:, None])
            else:
                treatment_prediction_calibrated = treatment_prediction
        if prediction_type == 'treatment':
            return treatment_prediction_calibrated
        if prediction_type in ('control', 'uplift'):
            logger.info("Calculating control prediction..")
            control_prediction = self.control_model.predict(X_control)
            if self.control_calibration_model:
                control_prediction_calibrated = self.control_calibration_model.predict(control_prediction[:, None])
            else:
                control_prediction_calibrated = control_prediction
        if prediction_type == 'control':
            return control_prediction_calibrated
        return treatment_prediction_calibrated - control_prediction_calibrated


class XLearner(BaseLearner):
    """
    Supports single treatment vs single control design
    """
    def __init__(
        self,
        treatment_base_model,
        control_base_model,
        treatment_imputed_model,
        control_imputed_model,
        max_treatment_control_imbalance: float = 9.0,
        treatment_name: str = TREATMENT_COL,
        logger_name: str = None,
    ):
        """
        all 4 models should implement fit and predict methods
        """
        super().__init__()
        if not isinstance(treatment_base_model, UpliftBaseModel):
            raise TypeError("treatment_base_model should be of `UpliftBaseModel` type")
        if not isinstance(control_base_model, UpliftBaseModel):
            raise TypeError("control_base_model should be of `UpliftBaseModel` type")
        if not isinstance(treatment_imputed_model, UpliftRegressionModel):
            raise TypeError("treatment_imputed_model should be of `UpliftRegressionModel` type")
        if not isinstance(control_imputed_model, UpliftRegressionModel):
            raise TypeError("control_imputed_model should be of `UpliftRegressionModel` type")
        self.treatment_base_model = treatment_base_model
        self.control_base_model = control_base_model
        self.treatment_imputed_model = treatment_imputed_model
        self.control_imputed_model = control_imputed_model
        self.weighting_g_function = None
        self.treatment_features = None
        self.control_features = None
        self.g_features = None
        self.features = None
        self.max_treatment_control_imbalance = max(max_treatment_control_imbalance, 1/max_treatment_control_imbalance)
        self.treatment_name = treatment_name
        self.logger_name = logger_name

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        t: ArrayLike1d,
        treatment_features: List[str] = None,
        control_features: List[str] = None,
        g_features: List[str] = None,
        weighting_g_function: Callable = None,
        treatment_kwargs_stage_1: Dict = None,
        control_kwargs_stage_1: Dict = None,
        treatment_kwargs_stage_2: Dict = None,
        control_kwargs_stage_2: Dict = None,
    ):
        logger = logging.getLogger(self.logger_name)

        X, y, t = self._check_inputs(X, y, t, force_binary_treatment=True)

        self.treatment_features = treatment_features or list(X.columns)
        self.control_features = control_features or list(X.columns)

        if weighting_g_function is not None:
            self.weighting_g_function = weighting_g_function
            if not isinstance(weighting_g_function, float):
                self.g_features = g_features or list(X.columns)

        self.features = list(set(self.treatment_features + self.control_features + (self.g_features or [])))

        X_treatment, X_control = self._prepare_features_2groups(X, t)
        y_treatment, y_control = self._prepare_target_2groups(y, t)

        treatment_kwargs_stage_1 = treatment_kwargs_stage_1 or {}
        control_kwargs_stage_1 = control_kwargs_stage_1 or {}
        treatment_kwargs_stage_2 = treatment_kwargs_stage_2 or {}
        control_kwargs_stage_2 = control_kwargs_stage_2 or {}

        logger.info("Fitting treatment base model..")
        self.treatment_base_model.fit(X_treatment, y_treatment, **treatment_kwargs_stage_1)
        logger.info("Fitting control base model..")
        self.control_base_model.fit(X_control, y_control, **control_kwargs_stage_1)
        logger.info("Calculating imputed effects..")
        D_control = self.treatment_base_model.predict(X_control) - y_control
        D_treatment = y_treatment - self.control_base_model.predict(X_treatment)
        logger.info("Fitting second stage treatment model..")
        self.treatment_imputed_model.fit(X_treatment, D_treatment, **treatment_kwargs_stage_2)
        logger.info("Fitting second stage control model..")
        self.control_imputed_model.fit(X_control, D_control, **control_kwargs_stage_2)

    def predict(
        self,
        X: pd.DataFrame,
        weighting_g_function: Callable = None,
        g_features: List[str] = None,
    ) -> pd.Series:
        logger = logging.getLogger(self.logger_name)

        if weighting_g_function is None:
            if self.weighting_g_function is None:
                raise ValueError(
                    "you should provide weighting function for meta-models either in this call or in class constructor"
                )
            else:
                g = self.weighting_g_function
                g_features = self.g_features
        else:
            g = weighting_g_function

        logger.info("Calculating control model prediction..")
        tau_0 = self.control_imputed_model.predict(X.loc[:, self.control_features])  # E[D_0|X]

        logger.info("Calculating treatment model prediction..")
        tau_1 = self.treatment_imputed_model.predict(X.loc[:, self.treatment_features])  # E[D_1|X]

        logger.info("Calculating weights..")
        if isinstance(g, float):
            gx = g
        elif callable(g):
            gx = g(X.loc[:, g_features])
        elif isinstance(g, UpliftBaseModel):
            gx = g.predict(X.loc[:, g_features])
        else:
            raise TypeError("weighting_g_function should be float, callable or 'UpliftBaseModel'")

        return gx * tau_0 + (1 - gx) * tau_1


class CTLearner(BaseLearner):
    """
    Class Transformation Method (Transformed Outcome) as described in Gutierrez, Gerardy and based on Athey, Imbens
    Supports single treatment vs single control design
    """
    def __init__(
        self,
        base_model: UpliftBaseModel,
        # propensity_model,
        max_treatment_control_imbalance: float = 9.0,
        treatment_name: float = TREATMENT_COL,
        logger_name: str = None,
    ):
        """
        base_model should implement fit and predict methods
        """
        super().__init__()
        if not isinstance(base_model, UpliftRegressionModel):
            raise TypeError("base model should be of `UpliftRegressionModel` type")
        self.base_model = base_model
        self.features = None
        self.max_treatment_control_imbalance = max(max_treatment_control_imbalance, 1/max_treatment_control_imbalance)
        self.treatment_name = treatment_name
        self.logger_name = logger_name

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        t: ArrayLike1d,
        p: Union[ArrayLike1d, float] = 0.5,
        **kwargs
    ):
        logger = logging.getLogger(self.logger_name)

        X, y, t = self._check_inputs(X, y, t, force_binary_treatment=False)

        self.features = list(X.columns)

        # it's a bit of hack to work-around cases where p being passed for full dataset in kwargs,
        # but X, y, t are coming from a smaller cv-fold
        if isinstance(p, pd.Series):
            if len(p) != len(y):
                try:
                    p = p[y.index]
                    warnings.warn(
                        f"parameter p has larger length than rest of arguments, "
                        f"subset was taken successfully by index"
                    )
                except Exception as e:
                    warnings.warn(f"parameter p has larger length than rest of arguments")
                    raise e

        # transformed_outcome = y * (t / p - (1 - t) / (1 - p))
        transformed_outcome = y * (t - p) / (p * (1 - p))

        logger.info("Fitting transformed outcome model..")
        self.base_model.fit(X, transformed_outcome, **kwargs)

    def predict(self, X: pd.DataFrame, clip: bool = True) -> Union[pd.Series, np.ndarray]:
        logger = logging.getLogger(self.logger_name)

        X = self.subset_features(X)

        logger.info("Calculating prediction from regression model..")
        tau = self.base_model.predict(X) # E[Y*|X]
        if clip:
            logger.info("Clipping values outside of [-1, 1]..")
            clipped_tau = np.clip(tau, -1, 1)
            return clipped_tau
        return tau


class UpliftTreeLearner(BaseLearner):
    """
    Wrapper for causalml.inference.tree.UpliftTreeClassifier
    Supports multiple treatments vs single control design
    """
    def __init__(
        self,
        split_criteria: str = 'KL',
        max_depth: int = 5,
        max_features: int = 10,
        min_samples_leaf: int = 50,
        min_samples_treatment: int = 10,
        n_reg: int = 0,
        normalization: bool = True,
        max_treatment_control_imbalance: float = 9.0,
        treatment_name: str = TREATMENT_COL,
        seed: int = 8,
        logger_name: str = None,
    ):
        """
        `split_criteria` should be one of the ('CTS', 'KL', 'ED', 'Chi')
        """
        super().__init__()
        self._fitted = False
        self.features = None
        self.max_treatment_control_imbalance = max(max_treatment_control_imbalance, 1/max_treatment_control_imbalance)
        self.treatment_name = treatment_name
        self.logger_name = logger_name

        split_criterias = ('CTS', 'KL', 'ED', 'Chi') ### 'DDP'?
        if split_criteria not in split_criterias:
            raise ValueError(f"unknown 'split_criteria', should be one of the {split_criterias}")

        self.model_params = {
            'evaluationFunction': split_criteria,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_treatment': min_samples_treatment,
            'n_reg': n_reg,
            'normalization': normalization,
            'random_state': seed,
        }

        self.control_group_label = '0'
        # self.model = UpliftTreeClassifierWithTypeCheckingFixed(
        self.model = UpliftTreeClassifier(
            control_name = self.control_group_label, ### yes, hardcoded to be consistent with other learners for binary T/C flags
            **self.model_params
        )

    def plot_tree(self):
        if self._fitted:
            dot_graph = uplift_tree_plot(self.model.fitted_uplift_tree, self.features)
            tree_image = Image(dot_graph.create_png())
            # print(type(tree_image))
            return tree_image
        else:
            raise AttributeError('learner is not fitted yet')

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        t: ArrayLike1d,
        **kwargs
    ):
        logger = logging.getLogger(self.logger_name)

        X, y, t = self._check_inputs(X, y, t, force_binary_treatment=False)
        self.features = list(X.columns)

        logger.info("Fitting uplift tree model..")
        t_fit = t.astype('str').values  # yes, it's that weird
        X_fit, y_fit = X.values, y.values
        self.model.fit(X = X_fit, treatment = t_fit, y = y_fit, **kwargs)
        self._fitted = True

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_baseline: bool = False,
        **kwargs,
    ) -> Union[Union[pd.DataFrame, np.ndarray], Tuple[Union[pd.DataFrame, np.ndarray], pd.Series]]:

        logger = logging.getLogger(self.logger_name)
        logger.info("Calculating prediction..")

        X = self.subset_features(X)
        X_pred, index = self._check_X(X)

        # prediction_output = self.model.predict(X_pred, full_output = True)
        prediction_output = self.model.predict(X_pred, **kwargs,)
        ### returns an array of n-1 uplifts for treatments 1,2,...,n-1
        prediction_output = pd.DataFrame(prediction_output, columns = self.model.classes_)
        output_columns = []
        for c, group in zip(prediction_output.columns, self.model.classes_):
            if c == self.control_group_label:
                control_prediction = prediction_output[self.control_group_label].rename(f'prediction_{self.control_group_label}')
            else:
                prediction_output[f'uplift_{c}'] = prediction_output[c] - prediction_output[self.control_group_label]
                output_columns.append(f'uplift_{c}')

        if isinstance(X, pd.DataFrame):
            prediction_output.index = index
            prediction_output = prediction_output.loc[:, output_columns]
            control_prediction.index = index
        else:
            prediction_output = prediction_output.loc[:, output_columns].values
            control_prediction = control_prediction.values

        if return_baseline:
            return prediction_output, control_prediction
        else:
            return prediction_output


class UpliftRandomForestLearner(BaseLearner):
    """
    Wrapper for causalml.inference.tree.UpliftRandomForestClassifier
    Supports multiple treatments vs single control design
    """
    def __init__(
        self,
        split_criteria: str = 'KL',
        n_estimators: int = 10,
        max_depth: int = 5,
        max_features: int = 10,
        min_samples_leaf: int = 50,
        min_samples_treatment: int = 10,
        n_reg: int = 0,
        normalization: bool = True,
        n_jobs: int = 1,
        joblib_prefer: str = None,
        seed: int = 8,
        max_treatment_control_imbalance: float = 9.0,
        treatment_name: str = TREATMENT_COL,
        logger_name: str = None,
        # progress_bar: bool = False,
    ):
        """
        `split_criteria` should be one of the ('CTS', 'KL', 'ED', 'Chi')
        """

        self._fitted = False
        self.features = None
        self.max_treatment_control_imbalance = max(max_treatment_control_imbalance, 1/max_treatment_control_imbalance)
        self.treatment_name = treatment_name
        self.logger_name = logger_name

        split_criterias = ('CTS', 'KL', 'ED', 'Chi') ### 'DDP'?
        if split_criteria not in split_criterias:
            raise ValueError(f"unknown 'split_criteria', should be one of the {split_criterias}")

        if n_jobs == -1:
            n_jobs = max(1, mp.cpu_count() - 2)
        elif n_jobs >= 1:
            n_jobs = min(n_jobs, mp.cpu_count() - 2)
        else:
            raise ValueError("n_jobs parameters should be either -1 or >= 1")

        self.model_params = {
            'evaluationFunction': split_criteria,
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_treatment': min_samples_treatment,
            'n_reg': n_reg,
            'normalization': normalization,
            'random_state': seed,
            'n_jobs': n_jobs,
            'joblib_prefer': joblib_prefer,
        }

        # self.model = UpliftRandomForestClassifierWithRandomStateBugFixed(
        #     progress_bar = progress_bar,
        self.control_group_label = '0'
        self.model = UpliftRandomForestClassifier(
            control_name = self.control_group_label, ### yes, hardcoded to be consistent with other learners for binary T/C flags
            **self.model_params
        )

    def plot_tree(self, tree_index: int):
        if self._fitted:
            n_estimators = len((self.model.uplift_forest))
            if not 0 <= tree_index <= n_estimators - 1:
                raise ValueError(f"index out of bounds for current model's n_estimators: {n_estimators}")
            dot_graph = uplift_tree_plot(self.model.uplift_forest[tree_index].fitted_uplift_tree, self.features)
            tree_image = Image(dot_graph.create_png())
            # print(type(tree_image))
            return tree_image
        else:
            raise AttributeError("learner is not fitted yet")

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike1d,
        t: ArrayLike1d,
        **kwargs
    ):
        logger = logging.getLogger(self.logger_name)

        X, y, t = self._check_inputs(X, y, t, force_binary_treatment=False)
        self.features = list(X.columns)

        logger.info("Fitting uplift random forest model..")
        t_fit = t.astype('str').values  # yes, it's that weird
        X_fit, y_fit = X.values, y.values
        self.model.fit(X = X_fit, treatment = t_fit, y = y_fit, **kwargs)
        self._fitted = True

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_baseline: bool = False,
        **kwargs,
    ) -> Union[Union[pd.DataFrame, np.ndarray], Tuple[Union[pd.DataFrame, np.ndarray], pd.Series]]:

        logger = logging.getLogger(self.logger_name)
        logger.info("Calculating predictions..")

        X = self.subset_features(X)
        X_pred, index = self._check_X(X)

        ### for whatever reason, RF class requires full_output argument to emit control group predictions as well
        ### Tree class doesn't even have that argument
        prediction_output = self.model.predict(X_pred, full_output = True, **kwargs)
        # prediction_output = pd.DataFrame(prediction_output, columns = self.model.classes_)
        ### returns pd.DataFrame ['0','1',...,'n-1', 'recommended_treatment','delta_1',...,'delta_n-1','max_delta'] in full_output
        ### otherwise it is an np.array of 'delta_1',...,'delta_n-1'

        output_columns = []
        for c, group in zip(prediction_output.columns, self.model.classes_):
            if c == self.control_group_label:
                control_prediction = prediction_output[self.control_group_label].rename(f'prediction_{self.control_group_label}')
            else:
                prediction_output[f'uplift_{c}'] = prediction_output[c] - prediction_output[self.control_group_label]
                output_columns.append(f'uplift_{c}')

        if isinstance(X, pd.DataFrame):
            prediction_output.index = index
            prediction_output = prediction_output.loc[:, output_columns]
            control_prediction.index = index
        else:
            prediction_output = prediction_output.loc[:, output_columns].values
            control_prediction = control_prediction.values

        if return_baseline:
            return prediction_output, control_prediction
        else:
            return prediction_output
