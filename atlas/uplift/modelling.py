from typing import Any, Callable, Dict, List, Union, Tuple
import logging
import io
from collections import defaultdict
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from causalml.feature_selection.filters import FilterSelect

from atlas.interfaces import SampleKeeper, ModelKeeper
from atlas.modelling import calc_na_filler_adaptive, make_predefined_split, perform_cross_validation_scoring
from atlas.uplift.learners import (
    BaseLearner,
    SLearner, TLearner, XLearner, CTLearner,
    UpliftTreeLearner, UpliftRandomForestLearner,
    UpliftRegressionModel, UpliftBaseModel
)
from atlas.utils import display
from atlas.uplift.evaluation import (
    get_percentile_uplift_chart, get_uplift_calibration,
    get_cumulative_uplift_chart, get_uplift_grid,
    get_lift_plot, get_uplift_gain_plot, get_qini_plot,
    calc_qini, calc_auqc, calc_auuc, calc_uplift_at_k,
    _assert_y_normalize_version
)

from atlas.interfaces import model_n_jobs_wrapper
from atlas.evaluation import (
    PSI,
    roc_auc_score_safe,
    calc_permutation_importances,
    plot_importances,
    calc_model_metric_lift_by_features,
    plot_model_metric_lift_curves,
    get_model_metric_lift_results_df,
    find_optimal_lift_step,
)

from pulsar.visualization.constants import VARIANTS_COLORS
from pulsar.common.constants import CONTROL_LABEL


def get_top_uplift_features(
    data: pd.DataFrame,
    target_name: str,
    treatment_name: str,
    features: List[str] = None,
    na_filler: float = None,
    selection_methods: List[str] = None,
    n_top: int = 10,
    control_label: int = CONTROL_LABEL,
) -> Dict[str, Any]:

    # selection_methods = ['F', 'LR', 'KL', 'ED', 'Chi']
    ### LR not always converges and ~0.7 ranking correlation with F; KL and Chi ~ .99 ranking correlation
    selection_methods = selection_methods or ['F', 'KL', 'ED', ]

    features = features or [f for f in data.columns if f not in [target_name, treatment_name]]

    treatment_labels = [v for v in sorted(data[treatment_name].unique()) if v != control_label]

    results = {}
    for treatment_label in tqdm(treatment_labels):

        filter_selector = FilterSelect()
        filter_results = []
        for selection_method in tqdm(selection_methods):
            top_importance_uplift_features = filter_selector.get_importance(
                data = data.fillna(na_filler),
                features = features,
                y_name = target_name,
                experiment_group_column = treatment_name,
                ### this would work with integers, even though strings are formally required, so we don't remap
                treatment_group = treatment_label,
                control_group = control_label,
                method = selection_method
            )
            # display(top_importance_uplift_features)
            filter_results.append(top_importance_uplift_features)

        filter_results_df = filter_results[0][['feature', 'rank']]

        for df, sm in zip(filter_results[1:], selection_methods[1:]):
            filter_results_df = pd.merge(
                filter_results_df, df[['feature', 'rank']], on = 'feature', how = 'outer',
                suffixes = ('', f'_{sm}')
            )

        filter_results_df.columns = ['feature', f'rank_{selection_methods[0]}'] + list(filter_results_df.columns[2:])
        filter_results_df.set_index(['feature'], inplace = True)
        filter_results_df['avg_rank'] = filter_results_df.mean(axis = 1)
        filter_results_df.sort_values(by = 'avg_rank', ascending = True, inplace = True)

        # top_singles = filter_results_df.index[(filter_results_df <= n_top).sum(axis=1) > 0]  ### at least 1 ranking in top

        results[treatment_label] = {
            'ranking': filter_results_df,
            # 'best_features': best_features,
        }

    for t_label in results:
        d = results[t_label]['ranking']
        d.columns = pd.MultiIndex.from_product([[t_label], d.columns])
        results[t_label]['ranking'] = d

    rankings_global = pd.concat(
        [results[t_label]['ranking'] for t_label in results], axis=1,
    )
    rankings_global['global_avg_rank'] = (
        rankings_global.loc[:, pd.IndexSlice[:, 'avg_rank']].mean(axis=1)
    )
    rankings_global = rankings_global.sort_values(by='global_avg_rank', ascending=True)
    top_average = list(rankings_global.index[:n_top])  ### top by averaged ranking
    best_features = (
        top_average
        # + [f for f in top_singles if f not in top_average]
    )

    results = {
        'ranking': rankings_global,
        'best_features': best_features,
    }

    return results


def run_uplift_evaluation_tests(
    sample_keeper: SampleKeeper,
    score_name: str,
    base_sample_key: str,
    sample_keys: List = None,
    k_target_threshold: Union[float, int] = 20,
    percentile_chart_n_buckets: int = 10,
    calibration_n_buckets: int = 10,
    calc_psi: bool = True,
    y_normalize_version: str = 'last',
    verbose: bool = True,
) -> Dict[str, Any]:

    _assert_y_normalize_version(y_normalize_version)

    sample_keys = sample_keys or sample_keeper.active
    if base_sample_key not in sample_keys:
        raise KeyError("'base_sample_key' should be present in 'sample_keys'")

    colormap = dict(zip(sample_keys, VARIANTS_COLORS))

    metrics_results = {}

    ### QINI
    metrics_results['qini'] = {}
    metrics_results['qini']['values'] = {}
    # qini_results_unnormalized = get_qini_plot(
    #     *sample_keeper.get_TST(k, score_name=scoring_name),
    #     group_by_forecast = True, x_normalize = False, y_normalize = False,
    #     show_ideal_Q = True, show_ideal_q0 = True, show_random = True, figsize = (8,5)
    # )
    # display(qini_results_unnormalized['figure'])
    qini_results = get_qini_plot(
        *sample_keeper.get_TST(base_sample_key, score_name=score_name),
        group_by_forecast = True, x_normalize = True, y_normalize = True,
        y_normalize_version = y_normalize_version,
        show_ideal_Q = True, show_ideal_q0 = True, show_random = True,
        figsize = (12, 7),
        label = base_sample_key,
        color = colormap[base_sample_key],
    )
    fig, ax = qini_results['figure'], qini_results['ax']
    # if verbose:
        # display(fig)

    for k in sample_keys:
        qini = calc_qini(*sample_keeper.get_TST(k, score_name=score_name))
        metrics_results['qini']['values'][k] = qini
        if k == base_sample_key:
            continue
        qini_results = get_qini_plot(
            *sample_keeper.get_TST(k, score_name=score_name),
            group_by_forecast = True, x_normalize = True, y_normalize = True,
            y_normalize_version = y_normalize_version,
            show_ideal_Q = False, show_ideal_q0 = False, show_random = False,
            ax = ax,
            label = k, color = colormap[k],
        )
        # AUQC_model = auc(*qini_results['curves'][k])
        # AUQC_random = auc(*qini_results['curves']['random'])
        # AUQC_Q = auc(*qini_results['curves']['Q'])
        # qini = (AUQC_model - AUQC_random) / (AUQC_Q - AUQC_random)
    if verbose:
        display(fig)
    metrics_results['qini']['figure'] = fig
    metrics_results['qini']['results'] = qini_results

    ### UPLIFT GAIN / AUUC
    metrics_results['auuc'] = {}
    metrics_results['auuc']['values'] = {}
    uplift_gain_results = get_uplift_gain_plot(
        *sample_keeper.get_TST(base_sample_key, score_name=score_name),
        group_by_forecast = True, x_normalize = True, y_normalize = True,
        y_normalize_version = y_normalize_version,
        show_ideal_Q = True,
        show_random = True,
        figsize = (12, 7),
        label = base_sample_key,
        color = colormap[base_sample_key],
    )
    fig, ax = uplift_gain_results['figure'], uplift_gain_results['ax']
    # display(fig)
    for k in sample_keys:
        auuc = calc_auuc(*sample_keeper.get_TST(k, score_name=score_name), y_normalize_version = y_normalize_version,)
        metrics_results['auuc']['values'][k] = auuc
        if k == base_sample_key:
            continue
        uplift_gain_results = get_uplift_gain_plot(
            *sample_keeper.get_TST(k, score_name=score_name),
            group_by_forecast = True, x_normalize = True, y_normalize = True,
            y_normalize_version = y_normalize_version,
            show_ideal_Q = False,
            show_random = False,
            ax = ax,
            label = k, color = colormap[k],
        )
        # auuc = auc(*uplift_gain_results['curves'][k])

    if verbose:
        display(fig)
    metrics_results['auuc']['figure'] = fig
    metrics_results['auuc']['results'] = uplift_gain_results

    ### UPLIFT PERCENTILE CHART
    metrics_results['percentile_uplift_chart'] = {}
    metrics_results['percentile_uplift_chart']['results'] = {}
    start_bucket_marker = 1
    for k in sample_keys:
        uplift_chart_results = get_percentile_uplift_chart(
            *sample_keeper.get_TST(k, score_name=score_name),
            n_buckets = percentile_chart_n_buckets,
            start_bucket_marker = start_bucket_marker,
        )
        if verbose:
            print(k)
            display(uplift_chart_results['figure'])
            display(uplift_chart_results['chart'].head(percentile_chart_n_buckets))
        metrics_results['percentile_uplift_chart']['results'][k] = uplift_chart_results

    ### LIFT CHART
    metrics_results['lift_chart'] = {}
    metrics_results['lift_chart']['results'] = {}
    metrics_results['lift_chart']['grid'] = {}
    k_grid = [5, 10, 20, 30, 50, 80, 100]
    if k_target_threshold not in k_grid:
        k_grid = sorted([k_target_threshold] + k_grid)

    for k in sample_keys:
        uplift_grid = get_uplift_grid(
            *sample_keeper.get_TST(k, score_name=score_name),
            k_grid = k_grid
        )
        metrics_results['lift_chart']['grid'][k] = uplift_grid

        # uplift_chart_results = get_cumulative_uplift_chart(
        #     *sample_keeper.get_TST(k, score_name=score_name),
        #     group_by_forecast = True
        # )
        # display(uplift_chart_results['chart'].head())
        lift_plot_results = get_lift_plot(
            *sample_keeper.get_TST(k, score_name=score_name),
            x_normalize=True,
            show_total=True,
            k = k_target_threshold,
            x_start_percentage = 0.01,
        )
        overall_actionable_lift = k_target_threshold / 100 * lift_plot_results['uplift_at_k'] + (1 - k_target_threshold / 100) * 0
        if verbose:
            print(k)
            display(uplift_grid)
            display(lift_plot_results['figure'])
            print(f"overall_actionable_lift@{k_target_threshold}: {overall_actionable_lift:.4f}")
        metrics_results['lift_chart']['results'][k] = lift_plot_results

    ### CALIBRATION
    metrics_results['calibration'] = {}
    metrics_results['calibration']['results'] = {}
    # qcut_bucketizer = partial(pd.qcut, labels = False, retbins = False, duplicates = 'drop')
    start_bucket_marker = 1
    for k in sample_keys:
        calibration_results = get_uplift_calibration(
            *sample_keeper.get_TST(k, score_name=score_name),
            n_buckets = calibration_n_buckets,
            # bucketizer = qcut_bucketizer,
            start_bucket_marker = start_bucket_marker,
            n_iterations = 100,
        )
        if verbose:
            print(k)
            display(calibration_results['figure'])
        metrics_results['calibration']['results'][k] = calibration_results

    ### PSI
    if calc_psi:
        metrics_results['psi'] = {}
        metrics_results['psi']['treatment_groups'] = {}
        metrics_results['psi']['target_groups'] = {}

        treatment_name = sample_keeper.treatment_name
        target_name = sample_keeper.target_name
        psi = PSI()
        max_buckets = 10
        for k in sample_keys:
            psi_df = sample_keeper[k][[score_name, target_name, treatment_name]]
            psi_sample_size_total = 100000
            if len(psi_df) >= psi_sample_size_total:
                psi_df = psi_df.sample(n = psi_sample_size_total, random_state = 8)

            psi_results = psi.get_psi_results(
                psi_df.query(f'{treatment_name} == 1'),
                psi_df.query(f'{treatment_name} == 0'),
                features = [score_name],
                target_name = target_name,
                discrete = False,
                rounding = None,
                max_buckets = max_buckets,
                labels = ('treatment', 'control')
            )
            if verbose:
                print(f"'{k}' sample, T vs C")
                display(psi_results['psi_df'])
                for f in psi_results['psi_dict']:
                    for kk in psi_results['psi_dict'][f]:
                        display(psi_results['psi_dict'][f][kk]['figure'])

            metrics_results['psi']['treatment_groups'][k] = psi_results

            ### reversing split logic for treatment and target flags
            psi_results = psi.get_psi_results(
                psi_df.query(f'{target_name} == 1'),
                psi_df.query(f'{target_name} == 0'),
                features = [score_name],
                target_name = treatment_name,
                discrete = False,
                rounding = None,
                labels = (f'{target_name} == 1', f'{target_name} == 0')
            )
            if verbose:
                print(f"'{k}' sample, responses vs non-responses")
                display(psi_results['psi_df'])
                for f in psi_results['psi_dict']:
                    for kk in psi_results['psi_dict'][f]:
                        display(psi_results['psi_dict'][f][kk]['figure'])

            metrics_results['psi']['target_groups'][k] = psi_results

    return metrics_results


def display_uplift_evaluation_results(
    metric_view: str,
    evaluation_results: Dict,
    learner_types: List[str],
    sample_keys: List[str],
) -> None:

    metric_views = (
        'lift_chart',
        'percentile_uplift_chart',
        'qini',
        'auuc',
        'psi',
        'calibration',
    )

    if metric_view not in metric_views:
        raise ValueError(f"unknown metric_view, should be one of the {metric_view}")

    l = len(sample_keys)

    for learner_type in learner_types:
        print('-'*90 + learner_type + '-'*90)

        if metric_view in ['qini', 'auuc']:
            figure = evaluation_results[learner_type][metric_view]['figure']
            display(figure)
            print(f"{metric_view}: " + "|".join(f"{key} = {v:.4f}" for key, v in evaluation_results[learner_type][metric_view]['values'].items()))

        elif metric_view in ['psi']:
            trial_key = list(evaluation_results[learner_type][metric_view]['treatment_groups'].keys())[0]
            scoring_name = list(evaluation_results[learner_type][metric_view]['treatment_groups'][trial_key]['psi_dict'].keys())[0]
            res = evaluation_results[learner_type][metric_view]['treatment_groups'][trial_key]['psi_dict'][scoring_name]
            psi_split_views = list(res.keys())
            ll = len(res)
            fig, axes = plt.subplots(ll, l, figsize = (l * 10, 15), squeeze=False)
            for a in axes:
                for ax in a:
                    ax.set_axis_off()
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
            plt.close()
            for kk, a in zip(psi_split_views, axes):
                for k, ax in zip(sample_keys, a):
                    res = evaluation_results[learner_type][metric_view]['treatment_groups'][k]['psi_dict'][scoring_name]
                    figure = res[kk]['figure']
                    file_object = io.BytesIO()
                    figure.savefig(file_object, bbox_inches='tight')
                    file_object.seek(0)
                    img = matplotlib.image.imread(file_object, 'png')
                    ax.imshow(img)
                    ax.set_title(k)
            display(fig)

        elif metric_view in ['lift_chart', 'percentile_uplift_chart', 'calibration']:
            fig, axes = plt.subplots(1, l, figsize = (l * 10, 20), squeeze = False)
            axes = axes[0]
            for ax in axes:
                ax.set_axis_off()
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
            plt.close()

            for k, ax in zip(sample_keys, axes):
                figure = evaluation_results[learner_type][metric_view]['results'][k]['figure']
                file_object = io.BytesIO()
                figure.savefig(file_object, bbox_inches='tight')
                file_object.seek(0)
                img = matplotlib.image.imread(file_object, 'png')
                ax.imshow(img)
                ax.set_title(k)
            # fig.suptitle(learner_type)
            plt.tight_layout()
            plt.close()
            display(fig)


def fit_learner_simple(
    sample_keeper: SampleKeeper,
    learner_type: str,
    train_sample_key: str,
    propensity_model: UpliftBaseModel = None,
    propensity_features: List[str] = None,
    max_treatment_control_imbalance: float = 0.6,
    lgbm_params: Dict = None,
    learner_params: Dict = None,
    learner: BaseLearner = None,
    logger_name: str = None,
) -> BaseLearner:

    default_boost_params = {
        'verbosity': -1,
        'learning_rate': 0.005,
        'n_estimators': 300,
        'boosting_type': 'gbdt',
        'max_depth': 5,
        'num_leaves': 32,
        'min_child_samples': 100,
        'reg_lambda': 0.1,
        'reg_alpha': 0.1,
        'subsample': 0.7,
    }

    boost_params = lgbm_params or default_boost_params

    X_train = sample_keeper.get_X(train_sample_key)
    if learner_type in ["UT", "URF"]:
        if X_train.isna().sum(axis = 0).any() > 0:
            raise ValueError(
                f"{learner_type} learner is not supporting missing values in X data, "
                f"check your '{train_sample_key}' sample"
            )

    if learner is not None:
        if not isinstance(learner, BaseLearner):
            raise TypeError(f"unknown learner type: {type(learner)}")
        if not (
            0 == 1
            or (learner_type == 'S' and isinstance(learner, SLearner))
            or (learner_type == 'T' and isinstance(learner, TLearner))
            or (learner_type == 'X' and isinstance(learner, XLearner))
            or (learner_type == 'CT' and isinstance(learner, CTLearner))
            or (learner_type == 'UT' and isinstance(learner, UpliftTreeLearner))
            or (learner_type == 'URF' and isinstance(learner, UpliftRandomForestLearner))
        ):
            raise ValueError("'learner_type' argument and actual provided learner's type do not match")


    if learner_type == 'S':
        if learner is None:
            learner = SLearner(
                # UpliftBaseModel(LogisticRegression, {'penalty': 'l2', 'C': 10}),
                base_model = UpliftBaseModel(LGBMClassifier, boost_params),
                treatment_name = sample_keeper.treatment_name,
                max_treatment_control_imbalance = max_treatment_control_imbalance,
                add_treatment_features_interaction = False,
                logger_name = logger_name,
            )
        learner.fit(
            *sample_keeper.get_XYT(train_sample_key),
            **{'categorical_feature': sample_keeper.cat_features,},
        )

    elif learner_type == 'T':
        if learner is None:
            learner = TLearner(
                # UpliftBaseModel(LogisticRegression, {'penalty': 'l2', 'C': 10}),
                treatment_model = UpliftBaseModel(LGBMClassifier, boost_params),
                control_model = UpliftBaseModel(LGBMClassifier, boost_params),
                treatment_name = sample_keeper.treatment_name,
                max_treatment_control_imbalance = max_treatment_control_imbalance,
                logger_name = logger_name,
            )
        learner.fit(
            *sample_keeper.get_XYT(train_sample_key),
            control_kwargs = {'categorical_feature': sample_keeper.cat_features},
            treatment_kwargs = {'categorical_feature': sample_keeper.cat_features},
        )

    elif learner_type == 'X':
        if learner is None:
            treatment_model = UpliftBaseModel(LGBMClassifier, boost_params)
            control_model = UpliftBaseModel(LGBMClassifier, boost_params)
            treatment_imp_model = UpliftRegressionModel(LGBMRegressor, boost_params)
            control_imp_model = UpliftRegressionModel(LGBMRegressor, boost_params)
            learner = XLearner(
                treatment_base_model = treatment_model,
                control_base_model = control_model,
                treatment_imputed_model = treatment_imp_model,
                control_imputed_model = control_imp_model,
                treatment_name = sample_keeper.treatment_name,
                max_treatment_control_imbalance = max_treatment_control_imbalance,
                logger_name = logger_name,
            )
        learner.fit(
            *sample_keeper.get_XYT(train_sample_key),
            weighting_g_function = propensity_model,
            g_features = propensity_features,
            control_kwargs_stage_1 = {'categorical_feature': sample_keeper.cat_features},
            control_kwargs_stage_2 = {'categorical_feature': sample_keeper.cat_features},
            treatment_kwargs_stage_1 = {'categorical_feature': sample_keeper.cat_features},
            treatment_kwargs_stage_2 = {'categorical_feature': sample_keeper.cat_features},
        )
    elif learner_type == 'CT':
        eps = 1e-5
        p = 0.5
        if propensity_model is not None and isinstance(propensity_model, UpliftBaseModel) and propensity_features is not None:
            p = np.clip(
                propensity_model.predict(sample_keeper.get_X(train_sample_key, features=propensity_features)),
                eps,
                1-eps,
            )

        if learner is None:
            learner = CTLearner(
                UpliftRegressionModel(LGBMRegressor, boost_params),
                treatment_name = sample_keeper.treatment_name,
                max_treatment_control_imbalance = max_treatment_control_imbalance,
                logger_name = logger_name,
            )
        learner.fit(
            *sample_keeper.get_XYT(train_sample_key),
            p = p,
        )
    elif learner_type in ['UT', 'URF']:
        # na_filler = calc_na_filler_adaptive(sample_keeper.get_X(train_sample_key))
        # def X_fillna(X: pd.DataFrame):
        #     return X.fillna(na_filler)
        if learner is None:
            if learner_type == 'UT':
                default_ut_params = {'split_criteria': 'KL', 'max_depth': 7, 'seed': 8}
                ut_params = learner_params or default_ut_params
                learner = UpliftTreeLearner(
                    **ut_params,
                    treatment_name = sample_keeper.treatment_name,
                    max_treatment_control_imbalance = max_treatment_control_imbalance,
                    logger_name = logger_name,
                )
            else:
                default_urf_params = {'split_criteria': 'KL', 'max_depth': 7, 'n_estimators': 10, 'n_jobs': 1, 'seed': 8}
                urf_params = learner_params or default_urf_params
                learner = UpliftRandomForestLearner(
                    **urf_params,
                    treatment_name = sample_keeper.treatment_name,
                    max_treatment_control_imbalance = max_treatment_control_imbalance,
                    logger_name = logger_name,
                )
        learner.fit(
            *sample_keeper.get_XYT(train_sample_key),
            # X_fillna(sample_keeper.get_X(train_sample_key)),
            # sample_keeper.get_Y(train_sample_key),
            # sample_keeper.get_T(train_sample_key),
        )

    return learner


def run_uplift_optuna_hyperparameters_search(
    sample_keeper: SampleKeeper,
    train_sample_key: str,
    test_sample_key: str,
    learner_type: str,
    scorer: Callable,
    greater_is_better: bool,
    propensity_model: UpliftBaseModel = None,
    propensity_features: List[str] = None,
    max_treatment_control_imbalance: float = 0.6,
    n_trials_total: int = 10,
    score_weights: Dict[str, float] = None,
    cv_strategy: Union[str, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold] = None,
    n_splits: int = 3,
    logger_name: str = None,
    n_jobs: int = 1,
) -> Tuple[BaseLearner, float]:

    seed = 8
    # n_cross_val_trials = int(n_trials_total * cross_val_steps_share)
    score_weights = score_weights or {'cv': 1.0, 'ps': 0.0}
    cv_weight = score_weights['cv'] / (score_weights['cv'] + score_weights['ps'])
    ps_weight = score_weights['ps'] / (score_weights['cv'] + score_weights['ps'])
    score_weights['cv'], score_weights['ps'] = cv_weight, ps_weight

    learners_prefixes = ["S", "T", "X", "CT", "UT", "URF"]
    if learner_type not in learners_prefixes:
        raise ValueError(f"unknown learner_type, should be one of the {learners_prefixes}")

    X_train, X_test = sample_keeper.get_X(train_sample_key), sample_keeper.get_X(test_sample_key)
    n_features = len(X_train.columns)

    if learner_type in ["UT", "URF"]:
        if (X_train.isna().sum(axis = 0).any() > 0) or (X_test.isna().sum(axis = 0).any() > 0):
            raise ValueError(
                f"{learner_type} learner is not supporting missing values in X data, "
                f"check your '{train_sample_key}' or '{test_sample_key}' samples"
            )
        # na_filler = calc_na_filler_adaptive(sample_keeper.get_X(train_sample_key))
        # def X_fillna(X: pd.DataFrame):
        #     return X.fillna(na_filler)
        # X_train, X_test = X_fillna(X_train), X_fillna(X_test)
        # y = sample_keeper.get_Y(train_sample_key)
        # t = sample_keeper.get_T(train_sample_key)

    fit_kwargs = {}
    learner_logger_name = None
    all_features = sample_keeper.all_features
    cat_features = sample_keeper.cat_features
    propensity_features = propensity_features or []

    if learner_type == 'X':
        features = list(set(all_features + propensity_features))
    else:
        features = all_features
    non_tuned_params = {
        'treatment_name': sample_keeper.treatment_name,
        'logger_name': learner_logger_name,
        'max_treatment_control_imbalance': max_treatment_control_imbalance,
    }
    uplift_tree_split_criterias = ('CTS', 'KL', 'ED', 'Chi')
    eps = 1e-5
    p = 0.5
    weighting_g_function = 0.5
    if propensity_model is not None and isinstance(propensity_model, UpliftBaseModel) and propensity_features is not None:
        p = pd.Series(np.clip(
            propensity_model.predict(sample_keeper.get_X(train_sample_key, features=propensity_features)),
            eps,
            1 - eps,
        ))
        weighting_g_function = propensity_model
    if learner_type == 'S':
        learner_class = SLearner
        non_tuned_params = {
            **non_tuned_params,
            **{'add_treatment_features_interaction': False},
        }
        fit_kwargs = {'categorical_feature': cat_features}
    elif learner_type == 'T':
        learner_class = TLearner
        fit_kwargs = {
            'control_kwargs': {'categorical_feature': cat_features},
            'treatment_kwargs': {'categorical_feature': cat_features}
        }
    elif learner_type == 'X':
        learner_class = XLearner
        fit_kwargs = {
            'weighting_g_function': weighting_g_function,
            'g_features': propensity_features,
            'treatment_features': all_features,
            'control_features': all_features,
            'control_kwargs_stage_1': {'categorical_feature': cat_features},
            'control_kwargs_stage_2': {'categorical_feature': cat_features},
            'treatment_kwargs_stage_1': {'categorical_feature': cat_features},
            'treatment_kwargs_stage_2': {'categorical_feature': cat_features},
        }
    elif learner_type == 'CT':
        learner_class = CTLearner
    elif learner_type == 'UT':
        learner_class = UpliftTreeLearner
    elif learner_type == 'URF':
        learner_class = UpliftRandomForestLearner

    def get_tuned_params(learner_type: str, param_groups: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        if learner_type == 'S':
            tuned_params = {
                'base_model': UpliftBaseModel(LGBMClassifier, param_groups[1])
            }
        elif learner_type == 'CT':
            tuned_params = {
                'base_model': UpliftRegressionModel(LGBMRegressor, param_groups[1]),
            }
        elif learner_type == 'T':
            tuned_params = {
                'treatment_model': UpliftBaseModel(LGBMClassifier, param_groups[1]),
                'control_model': UpliftBaseModel(LGBMClassifier, param_groups[2]),
            }
        elif learner_type == 'X':
            tuned_params = {
                'treatment_base_model': UpliftBaseModel(LGBMClassifier, param_groups[1]),
                'control_base_model': UpliftBaseModel(LGBMClassifier, param_groups[2]),
                'treatment_imputed_model': UpliftRegressionModel(LGBMRegressor, param_groups[3]),
                'control_imputed_model': UpliftRegressionModel(LGBMRegressor, param_groups[4]),
            }
        elif learner_type in ['UT', 'URF']:
            tuned_params = param_groups[1]
        else:
            raise ValueError(f"unknown learner_type '{learner_type}'")

        return tuned_params

    def optuna_objective(
        trial: optuna.Trial,
        # use_ps: bool = True,
    ):
        # Suggest hyperparameters
        if learner_type in ['S', 'CT']:
            param_groups = {
                i: {
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'n_jobs': n_jobs,
                    'learning_rate': trial.suggest_float(f'learning_rate__{i}', 1e-4, 0.1, log = True),
                    'num_leaves': trial.suggest_int(f'num_leaves__{i}', 31, 128),
                    'max_depth': trial.suggest_int(f'max_depth__{i}', 3, 10),
                    'min_child_samples': trial.suggest_int(f'min_child_samples__{i}', 50, 300, step = 50),
                    'n_estimators': trial.suggest_int(f'n_estimators__{i}', 50, 500, step = 50),
                    'subsample': trial.suggest_float(f'subsample__{i}', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float(f'colsample_bytree__{i}', 0.5, 1.0),
                }
                for i in range(1, 2)
            }
        elif learner_type == 'T':
            param_groups = {
                i: {
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'n_jobs': n_jobs,
                    'learning_rate': trial.suggest_float(f'learning_rate__{i}', 1e-4, 0.1, log = True),
                    'num_leaves': trial.suggest_int(f'num_leaves__{i}', 31, 128),
                    'max_depth': trial.suggest_int(f'max_depth__{i}', 3, 10),
                    'min_child_samples': trial.suggest_int(f'min_child_samples__{i}', 50, 300, step = 50),
                    'n_estimators': trial.suggest_int(f'n_estimators__{i}', 50, 500, step = 50),
                    'subsample': trial.suggest_float(f'subsample__{i}', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float(f'colsample_bytree__{i}', 0.5, 1.0),
                }
                for i in range(1, 3)
            }
        elif learner_type == 'X':
            param_groups = {
                i: {
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'n_jobs': n_jobs,
                    'learning_rate': trial.suggest_float(f'learning_rate__{i}', 1e-4, 0.1, log = True),
                    'num_leaves': trial.suggest_int(f'num_leaves__{i}', 31, 128),
                    'max_depth': trial.suggest_int(f'max_depth__{i}', 3, 10),
                    'min_child_samples': trial.suggest_int(f'min_child_samples__{i}', 50, 300, step = 50),
                    'n_estimators': trial.suggest_int(f'n_estimators__{i}', 50, 500, step = 50),
                    'subsample': trial.suggest_float(f'subsample__{i}', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float(f'colsample_bytree__{i}', 0.5, 1.0),
                }
                for i in range(1, 5)
            }
        elif learner_type == 'UT':
            param_groups = {
                i: {
                    'seed': seed,
                    'split_criteria': trial.suggest_categorical(f'split_criteria__{i}', choices = uplift_tree_split_criterias),
                    'max_depth': trial.suggest_int(f'max_depth__{i}', 3, 10),
                    'min_samples_leaf': trial.suggest_int(f'min_samples_leaf__{i}', 50, 300, step = 50),
                    'min_samples_treatment': trial.suggest_int(f'min_samples_treatment__{i}', 10, 300, step = 50),
                    'normalization': trial.suggest_categorical(f'normalization__{i}', choices = [True]),
                    'n_reg': trial.suggest_int(f'n_reg__{i}', 0, 100, step = 20),
                    'max_features': trial.suggest_int(f'max_features__{i}', min(5, n_features), n_features, step = min(5, n_features)),
                }
                for i in range(1, 2)
            }
        elif learner_type == 'URF':
            param_groups = {
                i: {
                    'seed': seed,
                    'n_jobs': n_jobs,
                    'joblib_prefer': 'processes',
                    'progress_bar': True,
                    'n_estimators': trial.suggest_int(f'n_estimators__{i}', 10, 200, step = 50),
                    'split_criteria': trial.suggest_categorical(f'split_criteria__{i}', choices = uplift_tree_split_criterias),
                    'max_depth': trial.suggest_int(f'max_depth__{i}', 3, 10),
                    'min_samples_leaf': trial.suggest_int(f'min_samples_leaf__{i}', 50, 300, step = 50),
                    'min_samples_treatment': trial.suggest_int(f'min_samples_treatment__{i}', 10, 300, step = 50),
                    'normalization': trial.suggest_categorical(f'normalization__{i}', choices = [True]),
                    'n_reg': trial.suggest_int(f'n_reg__{i}', 0, 100, step = 20),
                    'max_features': trial.suggest_int(f'max_features__{i}', min(5, n_features), n_features, step = min(5, n_features)),
                }
                for i in range(1, 2)
            }
        else:
            raise NotImplementedError

        tuned_params = get_tuned_params(learner_type = learner_type, param_groups = param_groups)
        learner = learner_class(**non_tuned_params, **tuned_params)

        trial_number = trial.number

        scores = {}
        for split_strategy in ['cv', 'ps']:
            if score_weights[split_strategy] == 0:
                scores[split_strategy] = 0
                continue
            if split_strategy == 'cv':
                X, y, t = sample_keeper.get_XYT(train_sample_key, features = features)
                if isinstance(cv_strategy, (GroupKFold, StratifiedGroupKFold)):
                    groups = sample_keeper.get_id(train_sample_key)
                    cv = cv_strategy
                elif isinstance(cv_strategy, str):
                    if cv_strategy.lower().startswith('group'):
                        groups = sample_keeper.get_id(train_sample_key)
                        cv = StratifiedGroupKFold(n_splits = n_splits)
                    else:
                        groups = None
                        cv = StratifiedKFold(n_splits = n_splits)
                else:
                    cv = cv_strategy
                    groups = None
                cv_kwargs = {
                    'cv': cv,
                    'n_splits': n_splits,
                    'seed': 8,
                    'shuffle': True,
                    'stratify': True,
                }
            else:
                (X_train_PS, y_train_PS, t_train_PS), PS = make_predefined_split(
                    *sample_keeper.get_XY(train_sample_key, features = features),
                    *sample_keeper.get_XY(test_sample_key, features = features),
                    sample_keeper.get_T(train_sample_key),
                    sample_keeper.get_T(test_sample_key),
                )
                X, y, t = (X_train_PS, y_train_PS, t_train_PS)
                groups = None
                # grid_refit_flag = False
                cv_kwargs = {
                    'cv': PS
                }

            if learner_type == 'CT' and isinstance(p, pd.Series):
                fit_kwargs['p'] = pd.Series(p, index = X.iloc[:len(p)].index)

            model_keeper = ModelKeeper(model = learner)
            cross_val_results = perform_cross_validation_scoring(
                model_keeper = model_keeper,
                scoring_method = 'uplift',
                scorer = scorer,
                X = X,
                y = y,
                t = t,
                groups = groups,
                fit_kwargs = fit_kwargs,
                **cv_kwargs,
            )
            test_scores = cross_val_results['test_scores']
            if len(test_scores) > 1:
                # eps = 1e-5
                # score = test_scores.mean() / max(test_scores.std(), eps)
                score = test_scores.mean()
            else:
                score = test_scores.mean()

            scores[split_strategy] = score

        # score = np.mean(scores)
        ### larger weight for pr-split as we assume test/val sample is from oot-period
        score = score_weights['cv'] * scores['cv'] + score_weights['ps'] * scores['ps']
        logger = logging.getLogger(logger_name)
        logger.info(f"trial #{trial_number}: score = {score:.4f}")
        return score

    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed = seed)  # setting seed to make the sampler behave in a deterministic way

    study = optuna.create_study(
        direction = 'maximize' if greater_is_better else 'minimize',
        sampler = sampler,
    )
    study.optimize(
        func = optuna_objective,
        n_trials = n_trials_total,
        show_progress_bar = True,
        gc_after_trial = True,
    )
    # study.optimize(
    #     func = partial(optuna_objective, use_ps = True),
    #     n_trials = n_trials_total - n_cross_val_trials,
    #     show_progress_bar = True,
    # )
    optuna.logging.set_verbosity(verbosity)

    best_trial, best_params = study.best_trial, study.best_params

    parsed_param_groups = defaultdict(lambda: {})
    for k, v in best_params.items():
        p_name, group = k.split('__')
        parsed_param_groups[int(group)][p_name] = v

    tuned_params = get_tuned_params(learner_type = learner_type, param_groups = parsed_param_groups)
    # print(tuned_params)

    learner = learner_class(**non_tuned_params, **tuned_params)
    learner.fit(*sample_keeper.get_XYT(train_sample_key, features = features), **fit_kwargs)
    y_predicted = learner.predict(sample_keeper.get_X(test_sample_key, features = features))
    y_true, t = sample_keeper.get_Y(test_sample_key), sample_keeper.get_T(test_sample_key)
    score = scorer(y_true, y_predicted, t)

    return learner, score


def fit_learner_and_score(
    sample_keeper: SampleKeeper,
    learner_type: str,
    base_score_name: str,
    train_sample_key: str,
    test_sample_key: str,
    k_target_threshold: Union[float, int] = 20,
    propensity_model: UpliftBaseModel = None,
    propensity_features: List[str] = None,
    max_treatment_control_imbalance: float = 0.6,
    tune_hyperparameters: bool = False,
    lgbm_params: Dict = None,
    learner_params: Dict = None,
    learner: BaseLearner = None,
    score_weights: Dict[str, float] = None,
    cv_strategy: Union[str, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold] = None,
    cv_n_splits: int = 3,
    n_trials: int = 10,
    n_jobs: int = 1,
    logger_name: str = None,
    verbose: bool = False,
) -> Dict[str, Union[ModelKeeper, str, Dict[str, Any]]]:

    learners_prefixes = ["S", "T", "X", "CT", "UT", "URF"]
    if learner_type not in learners_prefixes:
        raise ValueError(f"unknown learner_type, should be one of the {learners_prefixes}")

    scoring_name = f"{learner_type}_{base_score_name}"

    treatment_name = sample_keeper.treatment_name

    propensity_features = propensity_features or []
    propensity_model = propensity_model or 0.5

    key_metric_name, key_metric = 'qini', calc_qini
    # auqc_scorer = partial(calc_auqc, group_by_forecast=True, x_normalize=True, y_normalize=True, y_normalize_version='last')
    qini_scorer = partial(calc_qini, group_by_forecast=True, x_normalize=True, y_normalize=True, y_normalize_version = 'total',)

    if tune_hyperparameters:
        learner, score = run_uplift_optuna_hyperparameters_search(
            sample_keeper = sample_keeper,
            learner_type = learner_type,
            train_sample_key = train_sample_key,
            test_sample_key = test_sample_key,
            scorer = qini_scorer,
            propensity_model = propensity_model,
            propensity_features = propensity_features,
            max_treatment_control_imbalance = max_treatment_control_imbalance,
            greater_is_better = True,
            n_trials_total = n_trials,
            cv_strategy = cv_strategy,
            n_splits = cv_n_splits,
            score_weights = score_weights,
            logger_name = logger_name,
            n_jobs = n_jobs,
        )
    else:
        learner = fit_learner_simple(
            sample_keeper = sample_keeper,
            learner_type = learner_type,
            train_sample_key = train_sample_key,
            propensity_model = propensity_model,
            propensity_features = propensity_features,
            max_treatment_control_imbalance = max_treatment_control_imbalance,
            lgbm_params = lgbm_params,
            learner_params = learner_params,
            learner = learner,
            logger_name = logger_name,
        )

    artifacts = {}
    artifacts['extra'] = {}

    if learner_type == 'T':
        del sample_keeper.markup['treatment_name']
        for k in sample_keeper.active:
            sample_keeper[k][f'treatment_model_{scoring_name}'] = learner.predict(sample_keeper.get_X(k), prediction_type='treatment')
            sample_keeper[k][f'control_model_{scoring_name}'] = learner.predict(sample_keeper.get_X(k), prediction_type='control')
            sample_keeper[k][f'group_model_{scoring_name}'] = -1
            for t_label, t_group in zip([0, 1], ('control', 'treatment')):
                mask = (sample_keeper[k][treatment_name] == t_label)
                sample_keeper[k].loc[mask, f'group_model_{scoring_name}'] = sample_keeper[k].loc[mask, f'{t_group}_model_{scoring_name}']

        artifacts['extra']['group_scoring_names'] = [
            f'control_model_{scoring_name}', f'treatment_model_{scoring_name}', f'group_model_{scoring_name}',
        ]
        artifacts['extra']['group_dynamic_results'] = {}
        for t_label, t_group in zip([0, 1], ('control', 'treatment')):
            with sample_keeper.apply_filters([f"{treatment_name} == {t_label}"]):
                sample_keeper.set_markup(score_name = f'{t_group}_model_{scoring_name}')
                dynamic_results = sample_keeper.get_dynamic_statistics(
                    metrics = {'rocauc': roc_auc_score_safe},
                    filtered = True,
                )
                fig = dynamic_results['figure']
                if verbose:
                    display(fig)
                artifacts['extra']['group_dynamic_results'][t_group] = dynamic_results
        sample_keeper.set_markup(treatment_name = treatment_name,)

    sample_keeper.set_markup(score_name = scoring_name,)
    if learner_type == 'X':
        ### cause features for propensity model might be different, so providing them explicitly, both keepers need them
        x_features = list(set(sample_keeper.all_features + propensity_features))
        model_keeper = ModelKeeper(model = learner, features = x_features)
        sample_keeper.score_samples(
            model_keeper = model_keeper, score_name = scoring_name, method = 'predict',
            features = x_features,
        )
    elif learner_type in ['UT', 'URF']:
        model_keeper = ModelKeeper(
            model = learner, features = sample_keeper.all_features,
            # preprocessing = X_fillna
        )
        sample_keeper.score_samples(model_keeper = model_keeper, score_name = scoring_name, method = 'predict')
    else:
        model_keeper = ModelKeeper(model = learner, features = sample_keeper.all_features)
        sample_keeper.score_samples(model_keeper = model_keeper, score_name = scoring_name, method = 'predict')

    dynamic_results = sample_keeper.get_dynamic_statistics(
        metrics = {
            key_metric_name: key_metric,
            f'uplift_at_{k_target_threshold}': partial(calc_uplift_at_k, k = k_target_threshold)
        },
        filtered = True,
    )
    fig = dynamic_results['figure']
    if verbose:
        display(fig)
    artifacts['dynamic_results'] = dynamic_results

    return {
        'model_keeper': model_keeper,
        'scoring_name': scoring_name,
        'artifacts': artifacts,
    }


def run_learner_stepwise_feature_selection(
    sample_keeper: SampleKeeper,
    learner: BaseLearner,
    features: List[str],
    train_sample_key: str,
    test_sample_key: str,
    permutation_sample_key: str,
    key_metric: Callable,
    key_metric_name: str,
    k_target_threshold: Union[float, int] = 20,
    n_features_step: int = None,
    n_jobs: int = 1,
    max_display_features: int = 50,
    **fit_kwargs,
) -> List[str]:

    learner = model_n_jobs_wrapper(learner, n_jobs = 1)

    k = permutation_sample_key
    permutation_importances = calc_permutation_importances(
        model = learner,
        n_shuffles = 2,
        n_jobs = n_jobs,
        seed = 8,
        X = sample_keeper.get_X(k, features),
        y_true = sample_keeper.get_Y(k),
        treatment = sample_keeper.get_T(k),
        metric = key_metric,
        greater_is_better = True,
        prediction_method = 'value',
        verbose = False,
    )

    fig = plot_importances(
        importances = permutation_importances,
        n_feat_to_plot = max_display_features,
        plot_type = 'bar',
        title = 'permutation',
        color = 'darkorchid',
        fontsize = 10,
        figsize = (20, 10),
        show = False
    )
    display(fig)

    # permutation_ordered_features = list(permutation_importances.index)
    permutation_ordered_features = list(permutation_importances.query("importance > 0").index)

    if n_features_step is None:
        n_features_step = min(5, max(1, len(permutation_ordered_features) // 10))
    current_features = permutation_ordered_features[:]

    # with sample_keeper.apply_filters([f"{treatment_name} == {t_label}"]):
    # key_metric_name = 'qini'
    # key_metric = calc_qini

    metric_lift_results = calc_model_metric_lift_by_features(
        model = learner,
        features_sorted = current_features,
        X_train = sample_keeper.get_X(train_sample_key, features=current_features),
        y_train = sample_keeper.get_Y(train_sample_key),
        treatment = sample_keeper.get_T(train_sample_key),
        is_uplift_model = True,
        ignore_model_type = False,
        group_features_by = n_features_step,
        metrics = {
            key_metric_name: key_metric,
            # 'auqc': calc_auqc,
            # 'auuc': calc_auuc,
            f'uplift@{k_target_threshold}': partial(calc_uplift_at_k, k = k_target_threshold)
        },
        zero_model_metric_value = 0.0,
        default_prediction_method = 'value',
        # prediction_methods = {key_metric_name: 'value'},
        eval_samples = {
            k: sample_keeper.get_XYT(k, features=current_features)
            for k in sample_keeper.active if k != train_sample_key
        },
        train_name = train_sample_key,
        eval_on_train = True,
        cat_features = [f for f in sample_keeper.cat_features if f in current_features],
        verbose = False,
        progress_bar = True,
        n_jobs = n_jobs,
        **fit_kwargs,
    )

    plotted_lift_curves = plot_model_metric_lift_curves(
        metric_lift_results,
        use_steps_num_on_axis = True,
        max_factors_to_plot = None,
        xlabelrotation = 90,
        drop_zero = True,
        figsize = (20, 15)
    )

    lift_res_df = get_model_metric_lift_results_df(metric_lift_results)
    display(plotted_lift_curves)
    display(lift_res_df)

    optimal_lift_step_results = find_optimal_lift_step(
        steps_df = lift_res_df,
        metric_name = key_metric_name,
        train_sample_key = 'train',
        test_sample_key = test_sample_key,
    )
    # print(optimal_lift_step_results['optimal_step'], optimal_lift_step_results['optimal_features'])
    display(optimal_lift_step_results['evaluation_data'])
    return optimal_lift_step_results['optimal_features']
