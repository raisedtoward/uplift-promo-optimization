from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import logging
import warnings
import itertools
from copy import deepcopy
from collections import defaultdict
from io import StringIO
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy
import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pathos.multiprocessing
import optuna
import hiplot
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, PredefinedSplit, StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.feature_selection import mutual_info_classif, f_classif

from atlas.evaluation import (
    PSI,
    calc_psi_discrete,
    calc_hhi_discrete,
    calc_gini,
    calc_permutation_importances,
    calc_shap_importances,
    plot_importances,
    calc_model_metric_lift_by_features,
    plot_model_metric_lift_curves,
    get_model_metric_lift_results_df,
    find_optimal_lift_step,
)
from atlas.interfaces import SampleKeeper, ModelKeeper, model_n_jobs_wrapper
from atlas.encoders import HotEncoder
from atlas.utils import display, ArrayLike1d, BaseModel

from pulsar.common.validation import assert_treatment_labels


def calc_na_filler_adaptive(data: pd.DataFrame) -> Dict[str, float]:
    df_agg = data.agg(['min', 'max', 'std'])
    lb = df_agg.loc['min', :]
    near_zero_one_mask = np.isclose(lb, 0) | np.isclose(lb, 1)
    fillers = {c: -1.0 for c in lb[near_zero_one_mask].index}
    fillers.update(dict(np.rint(
        df_agg.loc['min', lb[~near_zero_one_mask].index]
        -
        np.log(df_agg.loc['std', lb[~near_zero_one_mask].index])
    )))
    return fillers


def create_id_groups(
    ids: pd.Series,
    n_splits: int = 3,
    seed: int = None
) -> pd.Series:

    unique_ids = sorted(set(ids))
    n_ids = len(unique_ids)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(unique_ids)
    split_size = n_ids // n_splits
    split_ids = np.zeros(n_ids)
    for i in range(n_splits):
        split_ids[i*split_size:(i+1)*split_size] = i
    id_to_split = dict(zip(unique_ids, split_ids))
    split_ids = ids.map(id_to_split).astype(int)
    return split_ids


def cramers_v(cross_tab: pd.DataFrame, correction: bool = True) -> float:
    """
    Args:
        cross_tab: Crosstab between two variables.
        correction: Whether correction should be applied or not
    Returns:
        The Cramer's V corrected statistic for the two variables.
    """

    chi2 = scipy.stats.chi2_contingency(cross_tab, correction=correction)[0]
    n = cross_tab.sum().sum()
    phi2 = chi2 / n
    r = cross_tab.shape[0]
    k = cross_tab.shape[1] if len(cross_tab.shape) > 1 else 1

    # Deal with NaNs later on
    with np.errstate(divide="ignore", invalid="ignore"):
        phi2corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / (n - 1.0))
        rcorr = r - ((r - 1.0) ** 2.0) / (n - 1.0)
        kcorr = k - ((k - 1.0) ** 2.0) / (n - 1.0)
        rkcorr = min((kcorr - 1.0), (rcorr - 1.0))
        if rkcorr == 0.0:
            corr = 1.0
        else:
            corr = np.sqrt(phi2corr / rkcorr)
    return corr


def get_top_features_cramer(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = None,
    use_na: bool = True,
    correction: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:

    corr_values = []
    cat_features = X.columns
    for f in tqdm(cat_features):
        x = X[f]
        if use_na:
            x = x.fillna('__NONE__')
        cross_tab = pd.crosstab(y, x) #index, columns
        corr_value = cramers_v(cross_tab.T, correction=correction)
        corr_values.append(corr_value)
        if verbose:
            print(f"{f}: {corr_value:.3f}")
            display(cross_tab.div(cross_tab.sum(axis = 1), axis = 0))
    return (
        pd.DataFrame({'Cramer_V': corr_values}, index=cat_features)
        .sort_values(by='Cramer_V', ascending=False).iloc[:top_n, ]
    )


def get_categorical_correlation_matrix(
    data: pd.DataFrame,
    cat_features: List[str] = None,
    use_na: bool = True,
) -> pd.DataFrame:

    if not cat_features:
        cat_features = data.columns
    corr_mat_categorical = np.zeros((len(cat_features), len(cat_features)))
    np.fill_diagonal(corr_mat_categorical, 1.0)
    corr_mat_categorical = pd.DataFrame(
        corr_mat_categorical, index=cat_features, columns=cat_features,
    )

    for f1, f2 in itertools.combinations(cat_features, 2):
        x, y = data[f1], data[f2]
        if use_na:
            x, y = x.fillna('__NONE__'), y.fillna('__NONE__')
        cross_tab = pd.crosstab(x, y)
        corr_value = cramers_v(cross_tab, correction=True)
        corr_mat_categorical.loc[f1, f2] = corr_mat_categorical.loc[f2, f1] = corr_value
    return corr_mat_categorical


def compare_to_uniform(x: pd.Series) -> float:
    vc = x.value_counts(normalize=True)[sorted(x.value_counts().index)]
    uf = np.ones(len(vc)) / len(vc)
    return calc_psi_discrete(vc, uf)


def compare_to_prior_discrete_distribution(
    x: pd.Series,
    prior: pd.Series
) -> float:
    vc = x.value_counts(normalize=True)
    prior = prior / np.sum(prior)
    return calc_psi_discrete(vc, prior)


def get_top_features_by_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = None,
    method: str = 'spearman',
) -> pd.DataFrame:

    y = pd.Series(y)
    y.index = X.index
    corr_methods = ['pearson', 'spearman', 'kendall']
    corr_method = method if method in corr_methods else "spearman"
    corr_with_target = pd.DataFrame(X.corrwith(other=y, axis=0, method=corr_method), columns=['Correlation'])
    corr_with_target['Abs Correlation'] = corr_with_target.applymap(abs)
    return corr_with_target.sort_values(by='Abs Correlation', ascending=False).iloc[:top_n, ]


def get_top_features_by_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = None,
    cat_features: List[str] = None,
    n_estimators: int = 50,
) -> pd.DataFrame:

    cat_features = cat_features or []
    model = lgb.LGBMClassifier(
        n_estimators = n_estimators,
        verbosity = -1
    )
    scores = []
    for f in tqdm(X.columns):
        X_fit = X[[f]]
        model.fit(
            X_fit,
            y,
            categorical_feature=[f] if f in cat_features else None
        )
        scores.append(calc_gini(y, model.predict_proba(X_fit)[:, 1]))
    res = pd.DataFrame({'gini': scores}, index = X.columns)
    return res.sort_values(by = 'gini', ascending = False).iloc[:top_n, :]


def get_top_features_by_gini(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = None,
    cat_features: List[str] = None,
    method: str = "raw",
    **kwargs,
) -> pd.DataFrame:

    if method == 'woe':
        raise NotImplementedError("")
        # woe_train, _, _, _ = get_woe_factors(
        #     bin_minimal_size=0.1, bin_size_increase=0.1, is_monotone=True,
        #     X_train=dataframe, y_train=y
        # )
        # res = pd.DataFrame(np.zeros(len(dataframe.columns)), index=dataframe.columns, columns=['gini'])
        # for f in tqdm(dataframe.columns):
        #     res.loc[f, 'gini'] = calc_gini(y, woe_train['WOE_' + f], ordering='asc')
        # return res.sort_values(by='gini', ascending=False).iloc[:top, ]
    elif method == 'lbgm':
        return get_top_features_by_lgbm(
            X = X,
            y = y,
            top_n = top_n,
            cat_features = cat_features,
            **kwargs,
        )
    else:
        res = pd.DataFrame(np.zeros(len(X.columns)), index=X.columns, columns=['gini'])
        for f in tqdm(X.columns):
            res.loc[f, 'gini'] = np.abs(calc_gini(y, X[f], ordering='asc'))
        return res.sort_values(by='gini', ascending=False).iloc[:top_n, ]


def get_top_features_by_gini_two_samples(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = None,
    method: str = "lgbm",
    cat_features: List[str] = None,
    rel_change_threshold: float = 0.05,
    min_threshold: float = 0.1,
    model_params: Dict[str, Any] = None,
    **kwargs,
) -> pd.DataFrame:

    cat_features = cat_features or []

    gini_train_dict = {}
    gini_test_dict = {}

    model_params = model_params or {}

    if method == 'woe':
        raise NotImplementedError("")
        # woe_train, woe_list_samples, _, _ = get_woe_factors(
        #     bin_minimal_size=0.1, bin_size_increase=0.1,
        #     is_monotone=True, X_train=X_train, y_train=y_train,
        #     other_samples_list=[X_test]
        # )
        # woe_test = woe_list_samples[0]
        # for f in tqdm(X_train.columns):
        #     gini_train_dict[f] = calc_gini(y_train, woe_train['WOE_' + f], ordering='asc')
        #     gini_test_dict[f] = calc_gini(y_test, woe_test['WOE_' + f], ordering='asc')
    else:
        for f in tqdm(X_train.columns):
            if method == 'raw':
                score_train, score_test = X_train[f], X_test[f]
            elif method == 'lgbm':
                model = lgb.LGBMClassifier(
                    verbosity = kwargs.get("verbosity", -1),
                    **model_params,
                )
                model.fit(X_train[[f]], y_train, categorical_feature=[f] if f in cat_features else None)
                score_train = model.predict_proba(X_train[[f]])[:, 1]
                score_test = model.predict_proba(X_test[[f]])[:, 1]
            gini_train_dict[f] = np.abs(calc_gini(y_train, score_train, ordering='asc'))
            gini_test_dict[f] = np.abs(calc_gini(y_test, score_test, ordering='asc'))

    selected_features = []
    min_gini_values = []

    for k in gini_train_dict.keys():
        if (gini_train_dict[k] - gini_test_dict[k] < rel_change_threshold) and (gini_test_dict[k] >= min_threshold):
            selected_features.append(k)
            min_gini_values.append(min(gini_train_dict[k], gini_test_dict[k]))

    return (
        pd.DataFrame(min_gini_values, index=selected_features, columns=['gini'])
        .sort_values(by='gini', ascending=False).iloc[:top_n, ]
    )


def get_top_features_by_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = None,
    discrete_features: Union[str, bool, np.ndarray,] = "auto",
) -> pd.DataFrame:

    res = mutual_info_classif(X, y, discrete_features)
    return (
        pd.DataFrame(
            res,
            index=X.columns,
            columns=['Mutual Information'],
        ).sort_values(by='Mutual Information', ascending=False).iloc[:top_n, ]
    )


def get_top_features_by_fvalue(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = None,
) -> pd.DataFrame:

    res = f_classif(X=X, y=y)
    return (
        pd.DataFrame(
            {'F-value': res[0], 'P-value': res[1]}, index=X.columns
        ).sort_values(by='F-value', ascending=False).iloc[:top_n, ]
    )


def get_top_features_by_psi(
    data: pd.DataFrame,
    target_name: str,
    features: List[str] = None,
    cat_features: List[str] = None,
    top_n: int = None,
    max_buckets: int = 20,
    na_filler: float = None,
    base: str = 'all',
    plot: bool = False,
    figsize: Tuple[int, int] = (15, 8),
    verbose: bool = False,
    n_jobs: int = 1,
    logger_name: str = None
) -> pd.DataFrame:

    logger = logging.getLogger(logger_name)

    cat_features = cat_features or {}
    if not features:
        features = [f for f in data.columns if f != target_name]
    cat_features = [f for f in features if f in cat_features]
    num_features = [f for f in features if f not in cat_features]
    if verbose:
        logger.info(f"categorical features: {cat_features}")
        logger.info(f"numeric features: {num_features}")

    if not na_filler and num_features:
        na_filler = calc_na_filler_adaptive(data[num_features])
        na_filler = {f: v for f, v in na_filler.items() if data[f].isna().sum() > 0}
        if verbose:
            logger.info(f"missing values would be replaced as following: {na_filler}")

    target_mask = (data[target_name] > 0).astype(bool)
    X_target, X_nontarget = data.loc[target_mask, :], data.loc[~target_mask, :]
    target_label, nontarget_label = 'target', 'non-target'
    target_color, nontarget_color = 'salmon', 'lightskyblue'

    psi = PSI()
    psi_results = pd.DataFrame(index = [], columns = (target_label, nontarget_label))

    psi_results_num = []
    psi_results_cat = []

    def show_figures(res: Dict):
        result_dict = res['psi_dict']
        for f in result_dict:
            for k in result_dict[f]:
                display(result_dict[f][k]['figure'])

    if base == 'all':
        ### checking target and non-target subsets against all population
        for X, l, c in zip(
            (X_target, X_nontarget), (target_label, nontarget_label), (target_color, nontarget_color)
        ):
            if num_features:
                res = psi.get_psi_results(
                    data.fillna(na_filler),
                    X.fillna(na_filler),
                    labels = ('all population', l),
                    colors = ('black', c),
                    target_name = None,
                    features = num_features,
                    discrete = False,
                    max_buckets = max_buckets,
                    merge_empty_bins = True,
                    figsize = figsize,
                    n_jobs = n_jobs,
                    do_plotting = plot,
                )
                psi_results_num.append(res['psi_df'].rename({'all': l}, axis = 1))
                if plot:
                    show_figures(res)
            if cat_features:
                res = psi.get_psi_results(
                    data[cat_features].astype(str),
                    X[cat_features].astype(str),
                    labels = ('all population', l),
                    colors = ('black', c),
                    features = cat_features,
                    discrete = True,
                    figsize = figsize,
                    n_jobs = n_jobs,
                    do_plotting = plot,
                )
                psi_results_cat.append(res['psi_df'].rename({'all': l}, axis = 1))
                if plot:
                    show_figures(res)
    elif base in ('target', 'non-target'):
        ### checking target and non-target subsets against each other
        if base == 'target':
            labels = target_label, nontarget_label
            colors = target_color, nontarget_color
        else:
            labels = nontarget_label, target_label
            colors = nontarget_color, target_color
        if num_features:
            if base == 'target':
                expected, actual = X_target.fillna(na_filler), X_nontarget.fillna(na_filler)
            else:
                expected, actual = X_nontarget.fillna(na_filler), X_target.fillna(na_filler)
            res = psi.get_psi_results(
                expected, actual,
                labels = labels,
                colors = colors,
                target_name = None,
                features = num_features,
                discrete = False,
                max_buckets = max_buckets,
                merge_empty_bins = True,
                figsize = figsize,
                n_jobs = n_jobs,
                do_plotting = plot,
            )
            psi_results_num.append(res['psi_df'])
            if plot:
                show_figures(res)

        if cat_features:
            if base == 'target':
                expected, actual = X_target[cat_features].astype(str), X_nontarget[cat_features].astype(str)
            else:
                expected, actual = X_nontarget[cat_features].astype(str), X_target[cat_features].astype(str)
            res = psi.get_psi_results(
                expected, actual,
                labels = labels,
                colors = colors,
                features = cat_features,
                discrete = True,
                figsize = figsize,
                n_jobs = n_jobs,
                do_plotting = plot,
            )
            psi_results_cat.append(res['psi_df'])
            if plot:
                show_figures(res)
    else:
        raise ValueError("unknown value for 'base' parameter - should be either 'all', 'target' or 'non-target'")

    if num_features:
        psi_results = pd.concat([psi_results, pd.concat(psi_results_num, axis = 1)], axis = 0)
    if cat_features:
        psi_results = pd.concat([psi_results, pd.concat(psi_results_cat, axis = 1)], axis = 0)

    psi_results['psi'] = psi_results.mean(axis = 1)

    return psi_results[['psi']].sort_values(by = 'psi', ascending = False).iloc[:top_n, :]


def drop_highly_correlated_features(
    corr_mat: pd.DataFrame,
    threshold: float = 0.99,
    rankings: Dict[str, int] = None,
    verbose: bool = True,
    logger_name: str = None,
):
    """
    `rankings` should be accessible by feature name as a key
    higher value - better feature
    otherwise dropping at random
    """

    logger = logging.getLogger(logger_name)

    na_filler = 2  ### to prevent incorrect masking over main diagonal when some correlations are nan
    seed = 8
    np.random.seed(seed)
    random_rankings = [1, 2]

    high_corr_drop_features = set()

    keep = np.triu(np.ones(corr_mat.shape), k = 1).astype('bool').reshape(corr_mat.size) ## excluding main diagonal
    correlations = pd.DataFrame(corr_mat.fillna(na_filler).stack()[keep])

    if rankings:
        ## starting with worse rankings
        features = list(filter(
            lambda f: f in corr_mat.columns,
            map(lambda x: x[0], sorted(rankings.items(), key = lambda x: -x[1]))
        ))
        iterator = itertools.combinations(features, 2)
        n = len(features)
        total = int(n * (n - 1) / 2)
    else:
        iterator = correlations.index
        total = len(iterator)
    for (f1, f2) in tqdm(iterator, total = total):
        if rankings:
            r1, r2 = rankings[f1], rankings[f2]
            v = corr_mat.loc[f1, f2]
        else:
            v = correlations.loc[(f1, f2), 0]
            r1, r2 = np.random.choice(random_rankings, size = 2, replace=False)
        if np.abs(v) >= threshold:
            if r1 <= r2 and f2 not in high_corr_drop_features:
                f_drop = f2
            elif r1 <= r2 and f2 in high_corr_drop_features:
                f_drop = None
            elif r1 > r2 and f1 not in high_corr_drop_features:
                f_drop = f1
            elif r1 > r2 and f1 in high_corr_drop_features:
                f_drop = None
            else:
                f_drop = None
            if f_drop:
                high_corr_drop_features.add(f_drop)
                if verbose:
                    logger.info(f"corr('{f1}'[rank: {r1}], '{f2}'[rank: {r2}]): {v:.5f}")
                    logger.info(f"dropping '{f_drop}'")
    return high_corr_drop_features


def rebalance_target_rate(
    data: pd.DataFrame,
    target_col: str,
    target_rate: float = 0.5,
    strategy: str = 'oversampling',
    seed: int = 8,
    reset_index: bool = False,
) -> pd.DataFrame:

    if sorted(data[target_col].unique()) != [0, 1]:
        raise ValueError("target is not binary")

    if np.isclose(target_rate, 0.0) or np.isclose(target_rate, 1.0) or target_rate < 0 or target_rate > 1:
        raise ValueError("target_rate should be in (0, 1) interval")

    target_mask = (data[target_col] > 0).astype(bool)
    X_target = data.loc[target_mask, :]
    X_nontarget = data.loc[~target_mask, :]

    l = len(data)
    t, nt = len(X_target), len(X_nontarget)

    observed_tr = t / l
    eps = 0.005
    if abs(observed_tr - target_rate) <= eps:
        return data

    strategy = strategy.lower()
    if strategy in ('oversampling', 'upsampling', 'over', 'up'):
        strategy = 'up'
    elif strategy in ('undersampling', 'downsampling', 'under', 'down'):
        strategy = 'down'
    else:
        raise ValueError(f"unknown strategy '{strategy}'")

    x = target_rate
    if observed_tr < target_rate:
        # either need to drop from non-target, or add from target
        if strategy == 'up':
            # oversampling from target class
            # (t + dt) / (t + dt + nt) = x
            # dt = (nt - t * (1-x)) / (1-x)
            dt = int((nt * x - t * (1-x)) / (1-x))
            res = pd.concat([
                X_target,
                X_target.sample(n = dt, replace = True, random_state = seed),
                X_nontarget
            ], axis = 0)
        else:
            # undersampling from non-target class
            # (t) / (t + n_nt) = x
            # n_nt = t * (1-x) / x
            n_nt = int(t * (1-x) / x)
            res = pd.concat([
                X_target,
                X_nontarget.sample(n = n_nt, replace = False, random_state = seed),
            ], axis = 0)
    else:
        # either need to drop from target, or add from non-target
        if strategy == 'up':
            # oversampling from non-target class
            # (t) / (t + dt + nt) = x
            # dt = (t * (1-x) - nt * x) / x
            dt = int((t * (1-x) - nt * x) / x)
            res = pd.concat([
                X_target,
                X_nontarget,
                X_nontarget.sample(n = dt, replace = True, random_state = seed),
            ], axis = 0)
        else:
            # undersampling from target class
            # (n_t) / (n_t + nt) = x
            # n_t = nt * x / (1 - x)
            n_t = int(nt * x / (1 - x))
            res = pd.concat([
                X_target.sample(n = n_t, replace = False, random_state = seed),
                X_nontarget
            ], axis = 0)

    if reset_index:
        return res.reset_index(drop = True)
    return res


def _get_lgmb_booster(model: Union[lgb.Booster, lgb.LGBMModel]) -> lgb.Booster:
    if isinstance(model, lgb.Booster):
        return model
    elif isinstance(model, (lgb.LGBMModel, lgb.LGBMClassifier, lgb.LGBMRegressor,)):
        return model.booster_
    else:
        raise TypeError(f"provided model instance is of unknown type {type(model)}")


def get_lgbm_boosting_importances(
    model: Union[lgb.Booster, lgb.LGBMModel],
    imp_type: str = 'gain',
) -> pd.DataFrame:
    booster = _get_lgmb_booster(model)
    result_df = [
        [i, j] for i, j in zip(booster.feature_name(), booster.feature_importance(importance_type=imp_type))
    ]  # if j != 0
    result_df = (
        pd.DataFrame(result_df, columns=['feature', 'importance'])
        .sort_values(by='importance', ascending=False)
        .set_index('feature')
    )
    return result_df


def plot_boosting_importances(
    model: Union[lgb.Booster, lgb.LGBMModel],
    imp_type: str = 'gain',
    plot_type: str = 'h',
    n_feat_to_plot: int = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:

    result_df = get_lgbm_boosting_importances(model, imp_type).reset_index(drop = False)
    booster = _get_lgmb_booster(model)
    if plot_type == 'h':
        lgb.plot_importance(booster, figsize = figsize, importance_type = imp_type, max_num_features = n_feat_to_plot)
    elif plot_type == 'v':
        ax = result_df.iloc[:n_feat_to_plot].plot('feature', 'importance', kind='bar', color='c', figsize = figsize)
        ax.set_title(f"Feature Importance: {imp_type}", fontsize=10)
        ax.set_xlabel('')
        plt.show()
    else:
        warnings.warn(f"unknown plot_type '{plot_type}'")
    # return result_df


def make_boosting_feature_selection(
    sample_keeper: SampleKeeper,
    features: List[str] = None,
    n_jobs_model: int = 1,
    n_jobs_permutation: int = 1,
    n_shuffles: int = 1,
    objective: str = 'binary_classification',
    selection_strategy_agg: str = 'min',
    train_sample_key: str = None,
    eval_sample_key: str = None,
    n_early_stopping_rounds: int = None,
    eval_metric: Callable = None,
    cv = None,
    model_params: Dict = None,
    seed: int = 8,
    verbose: bool = False,
) -> Dict[str, Any]:

    """
    `eval_set` is used for permutation importances calculation while early_stopping is done via cv test folds
    `n_shuffles` is used for permutation importances calculation
    """

    if objective not in ('binary_classification', 'regression'):
        raise ValueError("objective is not one of the ('binary_classification', 'regression')")
    if selection_strategy_agg not in ('min', 'mean'):
        raise ValueError("selection_strategy_agg is one of the in ('min', 'mean')")

    if model_params is None:
        model_params = {
            'n_estimators': 1000,
            'subsample': 0.8,
            'max_depth': 5,
            'learning_rate': 0.01,
            'num_leaves': 32
        }

    if objective == 'binary_classification':
        lgbm_model = lgb.LGBMClassifier(
            random_state = seed,
            n_jobs = n_jobs_model,
            metric = 'None',
            verbosity = -1
        )
        if cv is None:
            cv = StratifiedKFold(n_splits = 2, shuffle = True, random_state = seed)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits = cv, shuffle = True, random_state = seed)
        if eval_metric is None:
            eval_metric = 'auc'
        prediction_method = 'proba'
        permutation_metric = roc_auc_score
        greater_is_better = True
    else:
        lgbm_model = lgb.LGBMRegressor(
            random_state = seed,
            n_jobs = n_jobs_model,
            metric = 'None',
            verbosity = -1
        )
        if cv is None:
            cv = KFold(n_splits = 2, shuffle = True, random_state = seed)
        elif isinstance(cv, int):
            cv = KFold(n_splits = cv, shuffle = True, random_state = seed)
        if eval_metric is None:
            eval_metric = 'mse'
        prediction_method = 'value'
        permutation_metric = mean_absolute_error
        greater_is_better = False
    lgbm_model.set_params(**model_params)

    lgb_importances = []
    permutation_importances = []
    shap_importances = []

    if features is None or (isinstance(features, str) and features == 'all'):
        features = sample_keeper.all_features

    if train_sample_key is None:
        train_sample_key = 'train'

    X_t, y_t = sample_keeper.get_XY(train_sample_key, features = features)
    X_t, y_t = X_t.reset_index(drop = True), y_t.reset_index(drop = True)
    for train_ix, test_ix in tqdm(cv.split(X_t, y_t), total = cv.n_splits):
        callbacks = [lgb.early_stopping(stopping_rounds=n_early_stopping_rounds)] if n_early_stopping_rounds else []
        lgbm_model.fit(
            X_t.iloc[train_ix, :], y_t.iloc[train_ix],
            categorical_feature = [f for f in sample_keeper.markup['cat_features'] if f in features],
            # early_stopping_rounds=n_early_stopping_rounds,
            callbacks = callbacks,
            eval_metric = eval_metric,
            eval_set = (X_t.iloc[test_ix, :], y_t.iloc[test_ix]),
            # verbose = False,
        )
        b_i = get_lgbm_boosting_importances(lgbm_model, 'gain').rename({'importance': 'gain'}, axis = 1)
        lgb_importances.append(b_i)
        if verbose:
            display(b_i)

        if eval_sample_key is None:
            eval_set = (X_t.iloc[test_ix, :], y_t.iloc[test_ix])
        elif isinstance(eval_sample_key, (str, list, tuple)):
            eval_set = sample_keeper.get_XY(eval_sample_key, features = features)
        else:
            raise ValueError("eval_sample_key should be a key for sample_keeper")

        i_cols = list(b_i.loc[b_i['gain'] > 0, :].index) # to save time on actually unused features when permuting

        if n_jobs_model > 1:
            # to not get conflicts from paralleling on permutation inside
            # booster = _get_lgmb_booster(lgbm_model).reset_parameter({'n_jobs': 1})
            booster = lgbm_model
            reset_n_jobs = booster.n_jobs
            booster.n_jobs = 1
        else:
            # booster = _get_lgmb_booster(lgbm_model)
            booster = lgbm_model
            reset_n_jobs = 0

        permutation_importances_i = calc_permutation_importances(
            model=booster,
            n_shuffles=n_shuffles,
            X=eval_set[0], y_true=eval_set[1],
            metric=permutation_metric,
            greater_is_better=greater_is_better,
            prediction_method='value',
            # prediction_method=prediction_method,
            seed=seed,
            columns = i_cols,
            n_jobs = n_jobs_permutation,
        ).rename({'importance': 'permutation'}, axis = 1)
        permutation_importances.append(permutation_importances_i)
        if reset_n_jobs:
            booster.n_jobs = reset_n_jobs
        if verbose:
            display(permutation_importances_i)

        sample_size = min(50000, len(eval_set[0]))
        shap_importances_i = calc_shap_importances(
            lgbm_model,
            data = eval_set[0].loc[:, features],
            features = features,
            method = 'tree',
            objective = objective,
            sample_size = sample_size,
        )['importances'].rename({'importance': 'shap'}, axis = 1)
        shap_importances.append(shap_importances_i)
        if verbose:
            display(shap_importances_i)

    def _cols_to_flat_index(df):
        df.columns = list(map(
            lambda x: f'_'.join(x), df.columns.to_flat_index()
        ))
        return df

    aggs = ['min', 'mean', 'std']
    lgb_imp = _cols_to_flat_index(pd.concat(lgb_importances).fillna(0.0).groupby(level=0).agg(aggs))
    perm_imp = _cols_to_flat_index(pd.concat(permutation_importances).fillna(0.0).groupby(level=0).agg(aggs))
    shap_imp = _cols_to_flat_index(pd.concat(shap_importances).fillna(0.0).groupby(level=0).agg(aggs))
    imps = pd.merge(lgb_imp, perm_imp, left_index=True, right_index=True, how = 'left')
    imps = pd.merge(imps, shap_imp, left_index=True, right_index=True, how = 'left').fillna(0.0)
    imps.sort_values(
        by=['shap_mean', 'permutation_mean', 'gain_mean', ],
        ascending=[False, False, False], inplace=True
    )

    relevant_features = {}
    for s in ['min', 'mean']:
        selection_cols = [f for f in imps.columns if f.endswith(s)]
        mask_query = " and ".join([f"({col} > 0)" for col in selection_cols])
        relevant_features[s] = list(imps.query(mask_query).index)

    return {
        'importances': imps,
        'important_always': relevant_features['min'],
        'important_average': relevant_features['mean'],
        'selected_features': relevant_features[selection_strategy_agg],
    }


def get_boost_ensemble(
    sample_keeper: SampleKeeper,
    cv: KFold,
    eval_metric: str = 'auc',
    early_stopping_rounds: int = 100,
    eval_set: Union[str, Tuple[pd.DataFrame, pd.Series]] = None,
    seed: int = 8,
    n_jobs: int = 1,
    model_params: Dict = None,
) -> List[lgb.LGBMModel]:

    if eval_set is None:
        eval_set = sample_keeper.get_XY('train')
    elif isinstance(eval_set, str):
        eval_set = sample_keeper.get_XY(eval_set)
    elif (
        isinstance(eval_set, tuple) and len(eval_set) == 2
        and isinstance(eval_set[0], (pd.DataFrame, np.ndarray))
        and len(eval_set[0]) == len(eval_set[1])
    ):
        pass
    else:
        raise ValueError("wrong eval set - should be either string value key for sample_keeper or tuple (X,y)")

    estimators = []
    for train_ix, test_ix in cv.split(sample_keeper.get_X('train'), sample_keeper.get_Y('train')):
        lgbm_model = lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs, metric='None')  ### 'None' is important!
        lgbm_model.set_params(**model_params)
        X, y = sample_keeper.get_XY('train')
        X, y = X.iloc[train_ix, :], y.iloc[train_ix]
        lgbm_model.fit(
            X, y,
            categorical_feature=sample_keeper.markup['cat_features'],
            early_stopping_rounds=early_stopping_rounds,  ### re-do to use callbacks
            eval_metric=eval_metric,
            eval_set=eval_set
        )
        estimators.append(lgbm_model)
    return estimators


def get_boosting_ensemble_prediction(
    X: pd.DataFrame,
    estimators: List[lgb.LGBMModel],
) -> np.ndarray:

    predictions = []
    for i, model in enumerate(estimators):
        if isinstance(model, lgb.LGBMClassifier):
            predictions.append(model.predict_proba(X)[:, 1].reshape(-1, 1))
        elif isinstance(model, lgb.LGBMRegressor):
            predictions.append(model.predict(X).reshape(-1, 1))
        else:
            raise TypeError(f"unknown model type ({type(model)}) for model {i}")
    return np.hstack(predictions).mean(axis=1)


def make_predefined_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    t_train: pd.Series = None,
    t_test: pd.Series = None,
) -> Tuple[Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]], PredefinedSplit]:
    """ returns stacked X, stacked y and PS object which will fit only on 'train' part """

    X_train_PS = pd.concat([X_train, X_test], ignore_index=True)
    y_train_PS = pd.concat([y_train, y_test], ignore_index=True)
    X_train_PS = X_train_PS.reset_index(drop=True)
    y_train_PS = y_train_PS.reset_index(drop=True)

    if t_train is not None and t_test is not None:
        t_train_PS = pd.concat([t_train, t_test], axis = 0, ignore_index=True)
        t_train_PS = t_train_PS.reset_index(drop=True)

    ### -1 means that we're training on it, 0 is for first test-fold
    test_fold = np.hstack([-np.ones(len(y_train)), np.zeros(len(y_test))])
    ### now scoring will be made on test-part only
    PS = PredefinedSplit(test_fold)

    if t_train is not None and t_test is not None:
        return (X_train_PS, y_train_PS, t_train_PS), PS

    return (X_train_PS, y_train_PS), PS


def remove_large_vif(
    X: pd.DataFrame,
    threshold: float = 5.0,
    drop_all_infinite: bool = False,
    has_constant: str = 'raise',
    verbose: bool = False,
    logger_name: str = None,
) -> List[str]:
    """
    Returns list of columns left after iterative removal of all the columns having VIF values above the threshold.
    """

    logger = logging.getLogger(logger_name)

    w = X.shape[1]
    logger.info(f"{w} columns in initial dataframe")

    X = X.copy()
    # modified - now adding constant as first column to fit linreg with intercept
    dropped = True
    X = add_constant(X, prepend=True, has_constant=has_constant)
    if X.shape[1] == w:
        msg = (
            "intercept column was not added to data, seems like you have constant features, "
            "you should not rely on this function's results"
        )
        logger.warning(msg)

    while dropped:
        dropped = False
        vif = [variance_inflation_factor(exog=X.values, exog_idx=i) for i in range(X.shape[1])][1:]  ### dropping constant
        if verbose:
            logger.info(f"current columns: {list(X.columns[1:])}, current vif values: {vif}")
        max_vif = np.max(vif)
        max_idx = np.argmax(vif) + 1  # +1 'cos we dropped vif for constant  - but that column still in X
        if max_vif > threshold or np.isnan(max_vif):

            if max_vif == np.inf:
                if drop_all_infinite:
                    iter_drop = [X.columns[idx + 1] for idx in range(len(vif)) if vif[idx] == max_vif]
                    logger.info(f"Dropping {iter_drop} with vif={max_vif}")
                    X.drop(iter_drop, axis=1, inplace=True)
                    dropped = True
                    logger.info(f"{X.shape[1] - 1} columns in dataframe currently")
                else:
                    logger.info(f"Dropping {X.columns[max_idx]} with vif={max_vif}")
                    X.drop([X.columns[max_idx]], axis=1, inplace=True)
                    dropped = True
                    logger.info(f"{X.shape[1] - 1} columns in dataframe currently")
            elif np.isnan(max_vif):
                raise Exception("got nan as vif value - check your data for constant features")
            else:
                logger.info(f"Dropping {iter_drop} with vif={max_vif}")
                X.drop([X.columns[max_idx]], axis=1, inplace=True)
                dropped = True
                logger.info(f"{X.shape[1] - 1} columns in dataframe currently")

    X.drop([X.columns[0]], axis=1, inplace=True)  ### dropping constant
    logger.info(f"{X.shape[1]} columns in final dataframe")

    return list(X.columns)


def _gridsearch_to_hiplot(grid):
    list_for_hiplot = []
    for score, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
        changed_params = {'<<score>>': score}
        for k, v in params.items():
            if isinstance(v, (np.integer, int)):
                changed_params[k] = int(v)
            else:
                changed_params[k] = v
        list_for_hiplot.append(changed_params)
    return list_for_hiplot


def _optuna_study_to_hiplot(study):
    list_for_hiplot = []
    for trial in study.trials:
        changed_params = {'<<score>>': trial.values[0]}
        for k, v in trial.params.items():
            if isinstance(v, (np.integer, int)):
                changed_params[k] = int(v)
            else:
                changed_params[k] = v
        list_for_hiplot.append(changed_params)
    return list_for_hiplot


def display_hiplot(iterable):
    hip_exp = hiplot.Experiment.from_iterable(iterable)
    hip_exp.display_data(hiplot.Displays.PARALLEL_PLOT).update({'hide': ['uid', 'from_uid']})
    hip_exp.display_data(hiplot.Displays.TABLE).update({'hide': ['from_uid']})
    hip_exp.display()


def perform_cross_validation_scoring(
    model_keeper: ModelKeeper,
    scorer: Callable,
    X: pd.DataFrame,
    y: ArrayLike1d,
    t: ArrayLike1d = None,
    groups: ArrayLike1d = None,
    cv: KFold = None,
    n_splits: int = 3,
    seed: int = 8,
    shuffle: bool = True,
    stratify: bool = True,
    fit_kwargs: Dict = None,
    scoring_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    :param model_keeper: ModelKeeper object wrapping and handling fit/score/predict methods
    :param scorer: evaluation function, callable with signature f(y_true, y_pred) or f(y_true, y_pred, t)
    :param X: features dataframe
    :param y: target variable
    :param t: treatment variable, optional for uplift models
    :param groups: group identifiers for enabling GroupKFold split
    :param cv: sklearn CV object, pre-defined split is also supported
    :param n_splits: how many splits to use if cv is not supplied
    :param seed: seed/random state to use if cv is not supplied
    :param shuffle: whether to shuffle before cross-validation
    :param stratify: whether cross-validation should be stratified
    :param fit_kwargs: kwargs passed to fit method for each of the fitting folds
    :param scoring_method: which scoring method of ModelKeeper should be used for making predictions
    :return:
    """

    if cv is None:
        warnings.filterwarnings("ignore", category = FutureWarning)
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits = n_splits)
        elif stratify:
            cv = StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = seed)
        else:
            cv = KFold(n_splits = n_splits, shuffle = shuffle, random_state = seed)
        warnings.filterwarnings("default", category = FutureWarning)
    elif isinstance(cv, (GroupKFold, StratifiedGroupKFold)):
        if groups is None:
            raise ValueError(f"'groups' attribute should be provided for GroupKFold split")

    train_fold_scores = []
    test_fold_scores = []

    def check_binary_treatment(t):
        try:
            assert_treatment_labels(labels = t, force_binary = True)
            return True
        except ValueError as e:
            return False

    if t is not None:
        if check_binary_treatment(t):
            ### for uplift models with binary treatment and binary target
            f_stratification = y - 2 * t
        else:
            f_stratification = y
        is_uplift_model = True
    else:
        f_stratification = y
        is_uplift_model = False

    if isinstance(cv, (GroupKFold, StratifiedGroupKFold)):
        splitted_indices = cv.split(X = X, y = f_stratification, groups = groups)
    else:
        splitted_indices = cv.split(X = X, y = f_stratification)
    for train_index, test_index in splitted_indices:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if is_uplift_model:
            t_train, t_test = t.iloc[train_index], t.iloc[test_index]
            model_keeper.fit(X_train, y_train, t_train, **fit_kwargs)
        else:
            model_keeper.fit(X_train, y_train, **fit_kwargs)

        y_pred_train = model_keeper.score(X_train, method = scoring_method)
        y_pred_test = model_keeper.score(X_test, method = scoring_method)

        if is_uplift_model:
            train_score = scorer(y_train, y_pred_train, t_train)
            test_score = scorer(y_test, y_pred_test, t_test)
        else:
            train_score = scorer(y_train, y_pred_train)
            test_score = scorer(y_test, y_pred_test)

        train_fold_scores.append(train_score)
        test_fold_scores.append(test_score)

    train_fold_scores = np.asarray(train_fold_scores)
    test_fold_scores = np.asarray(test_fold_scores)

    return {
        'train_scores': train_fold_scores,
        'test_scores': test_fold_scores,
        'train_mean': train_fold_scores.mean(),
        'test_mean': test_fold_scores.mean(),
        'train_std': train_fold_scores.std(),
        'test_std': test_fold_scores.std(),
    }


def calc_linear_regression_stats(
    X_data: pd.DataFrame,
    target: pd.Series,
    model: BaseModel,
    no_intercept: bool = False,
):
    ### some statistics are not fully consistent with sm.api
    ### if all ohe-hot features (w/o dropping one extra) are present and fit doesn't include intercept
    prediction = model.predict(X_data)

    SSR = np.sum((prediction - target) ** 2)
    SST = np.sum(target ** 2)
    SST_corrected = np.sum((target - target.mean()) ** 2)
    N = len(X_data)

    X = X_data.values
    X_intercept = sm.add_constant(X_data).values

    if no_intercept:
        r2 = 1 - SSR / SST
        X_rank = np.linalg.matrix_rank(X)
        df_model = X_rank
        df_total = N
    else:
        r2 = 1 - SSR / SST_corrected
        X_rank = np.linalg.matrix_rank(X_intercept)
        df_model = X_rank - 1
        df_total = N - 1

    df_residual = N - X_rank

    # print(SSR, SST, SST_corrected, df_model, df_residual, df_total)

    r2_adj = 1 - (1 - r2) * df_total / df_residual
    f_value = r2 * df_residual / ((1 - r2) * df_model)
    f_distr = scipy.stats.f(dfn=df_model, dfd=df_residual)
    f_pvalue = 1 - f_distr.cdf(f_value)
    # f_pvalue = pf(f_value, df1 = df_model, df2 = df_residual, lower.tail = FALSE)

    return {
        'r2': r2,
        'r2_adj': r2_adj,
        'f_value': f_value,
        'f_pvalue': f_pvalue
    }


def fit_linear_regression(
    data: pd.DataFrame,
    target_name: str,
    features: List[str] = None,
    cat_features: List[str] = None,
    normalize: bool = False,
    apply_one_hot: bool = False,
    one_hot_kwargs: Dict = None,
    no_intercept: bool = False,
) -> Dict[str, Any]:

    cat_features = cat_features or []

    num_features = [f for f in features if f not in cat_features]

    if normalize:
        st_scaler = StandardScaler()
        numeric_data = pd.DataFrame(st_scaler.fit_transform(data[num_features]), columns=num_features)
    else:
        st_scaler = None
        numeric_data = data[num_features].copy()

    if apply_one_hot:
        one_hot_kwargs = one_hot_kwargs or {}

        ohe = HotEncoder(**one_hot_kwargs)
        categorical_data = ohe.fit_transform(data, cat_features)
    else:
        ohe = None
        categorical_data = data[cat_features].copy()

    x_data = pd.concat([numeric_data, categorical_data], axis=1)
    features = list(data.columns)

    if apply_one_hot:
        frm = ' + '.join(features)
    else:
        frm = ' + '.join(f'C({f})' if f in cat_features else f'{f}' for f in features)

    if no_intercept:
        frm = f"{target_name} ~ {frm} - 1"
    else:
        frm = f"{target_name} ~ {frm}"

    print(f'fitting model: {frm}')

    model = smf.ols(frm, data=pd.concat([data[target_name], x_data], axis=1)).fit()
    summary = model.summary()
    ols_results, coefficients, tests_statistics = summary.tables
    coefficients = pd.read_csv(StringIO(coefficients.as_csv()), header=0)
    coefficients.columns = ['feature', 'coef', 'stderr', 't', 'pvalue', 'ci_lower', 'ci_upper']
    # coefficients.sort_values(by = 'pvalue', inplace = True)
    display(coefficients)
    # if apply_one_hot:
    #     lr_stats = calc_linear_regression_stats(data, dataframe[target_name], model, no_intercept)
    #     r2, r2_adj, fvalue, f_pvalue = lr_stats.values()
    #     print(r2, r2_adj, fvalue, f_pvalue)

    r2, r2_adj, fvalue, f_pvalue = model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue
    msg = f"R-squared: {r2:.5f}, R-squared Adjusted: {r2_adj:.5f}, F-stat: {fvalue:.5f}, F-stat p-value: {f_pvalue:.5f}"
    print(msg)

    return {
        'scaler': st_scaler,
        'one_hot_encoder': ohe,
        'X_fit': x_data,
        'model': model,
        'coefficients': coefficients,
        'fit_stats': ols_results,
    }


def fit_model_simple(
    sample_keeper: SampleKeeper,
    train_sample_key: str,
    objective: str = 'binary_classification',
    lgbm_params: Dict = None,
    n_jobs: int = None,
    seed: int = 8,
) -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:

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

    if objective == 'binary_classification':
        model_class = lgb.LGBMClassifier
    elif objective == 'regression':
        model_class = lgb.LGBMRegressor
    else:
        raise ValueError(f"unknown objective '{objective}'")

    boost_params = lgbm_params or default_boost_params

    non_tuned_params = {
        'random_state': seed,
        'seed': seed,
        'verbosity': -1,
        'n_jobs': n_jobs,
        # 'metric': 'None',
    }

    model = model_class(
        **non_tuned_params,
    ).set_params(**boost_params)
    model.fit(
        *sample_keeper.get_XY(train_sample_key),
        **{'categorical_feature': sample_keeper.cat_features,},
    )
    return model


def run_optuna_hyperparameters_search(
    sample_keeper: SampleKeeper,
    train_sample_key: str,
    test_sample_key: str,
    scorer: Callable,
    greater_is_better: bool,
    objective: str = 'binary_classification',
    n_trials_total: int = 10,
    score_weights: Dict[str, float] = None,
    cv: Union[StratifiedKFold, KFold] = None,
    n_splits: int = 3,
    n_jobs: int = 1,
    seed: int = 8,
    logger_name: str = None,
) -> Tuple[Union[lgb.LGBMClassifier, lgb.LGBMRegressor], float]:

    # n_cross_val_trials = int(n_trials_total * cross_val_steps_share)
    score_weights = score_weights or {'cv': 1.0, 'ps': 0.0}
    cv_weight = score_weights['cv'] / (score_weights['cv'] + score_weights['ps'])
    ps_weight = score_weights['ps'] / (score_weights['cv'] + score_weights['ps'])
    score_weights['cv'], score_weights['ps'] = cv_weight, ps_weight

    all_features = sample_keeper.all_features
    cat_features = sample_keeper.cat_features
    fit_kwargs = {'categorical_feature': cat_features}

    features = all_features
    non_tuned_params = {
        'random_state': seed,
        'seed': seed,
        'verbosity': -1,
        'n_jobs': n_jobs,
    }

    if objective == 'binary_classification':
        model_class = lgb.LGBMClassifier
    elif objective == 'regression':
        model_class = lgb.LGBMRegressor
    else:
        raise ValueError(f"unknown objective '{objective}'")

    def optuna_objective(
        trial: optuna.Trial,
        # use_ps: bool = True,
    ):
        # Suggest hyperparameters
        i = 1
        tuned_params = {
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float(f'learning_rate__{i}', 1e-4, 0.1, log = True),
            'num_leaves': trial.suggest_int(f'num_leaves__{i}', 8, 128),
            'max_depth': trial.suggest_int(f'max_depth__{i}', 3, 10),
            'min_child_samples': trial.suggest_int(f'min_child_samples__{i}', 50, 300, step = 50),
            'n_estimators': trial.suggest_int(f'n_estimators__{i}', 50, 500, step = 50),
            'subsample': trial.suggest_float(f'subsample__{i}', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float(f'colsample_bytree__{i}', 0.5, 1.0),
        }

        model = model_class(**non_tuned_params, **tuned_params)

        trial_number = trial.number
        # if trial_number < n_cross_val_trials:
        #     use_ps = False
        # else:
        #     use_ps = True
        scores = {}
        for split_strategy in ['cv', 'ps']:
            if score_weights[split_strategy] == 0:
                scores[split_strategy] = 0
                continue
            if split_strategy == 'cv':
                # X, y, t = sample_keeper.get_XYT(train_sample_key, features = features)
                X, y = sample_keeper.get_XY(train_sample_key, features = features)
                cv_kwargs = {
                    'cv': cv,
                    'n_splits': n_splits,
                    'seed': seed,
                    'shuffle': True,
                    'stratify': True,
                }
            else:
                # (X_train_PS, y_train_PS, t_train_PS), PS = make_predefined_split(
                (X_train_PS, y_train_PS), PS = make_predefined_split(
                    *sample_keeper.get_XY(train_sample_key, features = features),
                    *sample_keeper.get_XY(test_sample_key, features = features),
                    # sample_keeper.get_T(train_sample_key),
                    # sample_keeper.get_T(test_sample_key),
                )
                # X, y, t = (X_train_PS, y_train_PS, t_train_PS)
                X, y = (X_train_PS, y_train_PS)
                # grid_refit_flag = False
                cv_kwargs = {
                    'cv': PS
                }

            model_keeper = ModelKeeper(model = model)
            cross_val_results = perform_cross_validation_scoring(
                model_keeper = model_keeper,
                scoring_method = None,  # should be inferred from the model_keeper through model and objective
                scorer = scorer,
                X = X,
                y = y,
                # t = t,
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
    optuna.logging.set_verbosity(verbosity)

    best_trial, best_params = study.best_trial, study.best_params

    parsed_param_groups = defaultdict(lambda: {})
    for k, v in best_params.items():
        p_name, group = k.split('__')
        parsed_param_groups[int(group)][p_name] = v

    tuned_params = parsed_param_groups[1]
    # print(tuned_params)

    model = model_class(**non_tuned_params, **tuned_params)
    model.fit(*sample_keeper.get_XY(train_sample_key, features = features), **fit_kwargs)
    y_predicted = model.predict(sample_keeper.get_X(test_sample_key, features = features))
    y_true = sample_keeper.get_Y(test_sample_key)
    score = scorer(y_true, y_predicted)

    return model, score


def fit_model_and_score(
    sample_keeper: SampleKeeper,
    score_name: str,
    train_sample_key: str,
    test_sample_key: str,
    tune_hyperparameters: bool = False,
    objective: str = 'binary_classification',
    lgbm_params: Dict = None,
    score_weights: Dict[str, float] = None,
    cv_n_splits: int = 3,
    n_trials: int = 10,
    n_jobs: int = 1,
    logger_name: str = None,
    verbose: bool = False,
) -> Dict[str, Union[ModelKeeper, str, Dict[str, Any]]]:

    if objective == 'binary_classification':
        key_metric_name, key_metric = 'rocauc', roc_auc_score
        greater_is_better = True
    elif objective == 'regression':
        key_metric_name, key_metric = 'mae', mean_absolute_error
        greater_is_better = False
    else:
        raise ValueError(f"unknown objective '{objective}'")
    # auqc_scorer = partial(calc_auqc, group_by_forecast=True, x_normalize=True, y_normalize=True, y_normalize_version='last')
    scorer = key_metric

    if tune_hyperparameters:
        model, score = run_optuna_hyperparameters_search(
            sample_keeper = sample_keeper,
            objective = objective,
            train_sample_key = train_sample_key,
            test_sample_key = test_sample_key,
            scorer = scorer,
            greater_is_better = greater_is_better,
            n_trials_total = n_trials,
            cv = None,
            score_weights = score_weights,
            n_splits = cv_n_splits,
            logger_name = logger_name,
            n_jobs = n_jobs,
        )
    else:
        model = fit_model_simple(
            sample_keeper = sample_keeper,
            objective = objective,
            train_sample_key = train_sample_key,
            lgbm_params = lgbm_params,
            n_jobs = n_jobs,
        )

    artifacts = {}
    # artifacts['extra'] = {}

    sample_keeper.set_markup(score_name = score_name,)
    model_keeper = ModelKeeper(model = model, features = sample_keeper.all_features)
    scoring_method = 'proba' if objective == 'binary_classification' else 'predict'
    sample_keeper.score_samples(model_keeper = model_keeper, score_name = score_name, method = scoring_method)

    dynamic_results = sample_keeper.get_dynamic_statistics(
        metrics = {
            key_metric_name: key_metric,
        },
        filtered = True,
    )
    fig = dynamic_results['figure']
    if verbose:
        display(fig)
    artifacts['dynamic_results'] = dynamic_results

    return {
        'model_keeper': model_keeper,
        'scoring_name': score_name,
        'artifacts': artifacts,
    }


def run_model_stepwise_feature_selection(
    sample_keeper: SampleKeeper,
    model: BaseModel,
    features: List[str],
    train_sample_key: str,
    test_sample_key: str,
    permutation_sample_key: str,
    key_metric: Callable,
    key_metric_name: str,
    greater_is_better: bool,
    objective: str = 'binary_classification',
    n_features_step: int = None,
    n_jobs: int = 1,
    seed: int = 8,
    max_display_features: int = 50,
    **fit_kwargs,
) -> List[str]:

    model = model_n_jobs_wrapper(model, n_jobs = 1)

    scoring_method = 'proba' if objective == 'binary_classification' else 'predict'

    permutation_importances = calc_permutation_importances(
        model = model,
        n_shuffles = 2,
        n_jobs = n_jobs,
        seed = seed,
        X = sample_keeper.get_X(permutation_sample_key, features),
        y_true = sample_keeper.get_Y(permutation_sample_key),
        # treatment = sample_keeper.get_T(permutation_sample_key),
        metric = key_metric,
        greater_is_better = greater_is_better,
        prediction_method = scoring_method,
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

    metric_lift_results = calc_model_metric_lift_by_features(
        model = model,
        features_sorted = current_features,
        X_train = sample_keeper.get_X(train_sample_key, features=current_features),
        y_train = sample_keeper.get_Y(train_sample_key),
        # treatment = sample_keeper.get_T(train_sample_key),
        is_uplift_model = False,
        ignore_model_type = False,
        group_features_by = n_features_step,
        metrics = {
            key_metric_name: key_metric,
        },
        zero_model_metric_value = 0.0,
        default_prediction_method = scoring_method,
        # prediction_methods = {key_metric_name: 'value'},
        eval_samples = {
            k: sample_keeper.get_XY(k, features=current_features)
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
