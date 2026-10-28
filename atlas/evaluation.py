from typing import Any, Callable, Iterable, Optional, Dict, List, Tuple, Union
import pathos
import itertools
import warnings
from copy import deepcopy
from collections import defaultdict
from functools import partial
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.collections import QuadMesh

#from IPython.display import display, Markdown, HTML, clear_output, Image
try:
    from ipywidgets import interact, interact_manual, fixed, widgets
    from ipywidgets import interactive, HBox, VBox

    def get_importances_widget(importances_df):
        L = len(importances_df)
        n_feat_to_plot = widgets.IntSlider(min=1, max=L, value=L, step = 1, description = '# features:')
        plot_type = widgets.RadioButtons(options=['bar', 'line'], description = 'Plot type:')
        plot_color = widgets.RadioButtons(options=['crimson', 'darkorchid', 'royalblue', 'springgreen'], description = 'Plot type:')
        plot_title = widgets.Text(value='', description='Title:')
        plot_ticksfontsize = widgets.IntSlider(min = 5, max = 20, value = 10, step = 1, description = 'fontsize:')
        widget_object = interactive(
            plot_importances, importances = fixed(importances_df),
            n_feat_to_plot = n_feat_to_plot, plot_type = plot_type, title = plot_title, color = plot_color,
            fontsize = plot_ticksfontsize, figsize = fixed((20, 10)), show = fixed(True)
        )
        widget_panel = VBox([HBox(widget_object.children[:-1]), widget_object.children[-1]])
        return widget_panel, widget_object

except ModuleNotFoundError:
    def get_importances_widget(*args, **kwargs):
        raise NotImplementedError("ipywidgets module is required")


import shap
import statsmodels
import statsmodels.nonparametric
import statsmodels.nonparametric.bandwidths
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import spearmanr
from scipy.signal import argrelextrema

from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.extmath import stable_cumsum

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import fbeta_score, make_scorer, log_loss

from pulsar.common.calc import calc_weighted_mean, calc_weighted_sum
from pulsar.discovery.distributions import bucketize, EmpiricalDistribution, fit_distribution
from pulsar.stattests.binomial import calc_binomial_test
from pulsar.stattests.ttest import calc_t_test_two_samples
from pulsar.visualization.constants import VARIANTS_COLORS, LINESTYLES, DEFAULT_PLOT_STYLE

from atlas.interfaces import ModelKeeper, model_n_jobs_wrapper, model_fit_wrapper, model_prediction_wrapper
from atlas.utils import filter_kwargs, defaultdict_to_dict, display, ArrayLike1d
# from atlas.pdpbox import pdp, info_plots
from pdpbox import pdp, info_plots


# SELECTED_COLORS = [
#     'darkorchid', 'lightskyblue', 'crimson', 'gold', 'forestgreen',
#     'darkmagenta', 'dodgerblue', 'lightcoral', 'navy', 'springgreen',
# ]
SELECTED_COLORS = VARIANTS_COLORS

LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot']


def calc_pdp(
    model,
    data_pdp: pd.DataFrame,
    target_name: str = None,
    feature_names: List[str] = None,
    cat_features: List[str] = None,
    X_pdp: pd.DataFrame = None,
    explain_features: List[str] = None,
    output_plots: bool = False,
    plot_ice: bool = True,
    ice_frac_to_plot: int = 1000,
    n_cluster_centers: int = None,
    figsize: Tuple[int, int] = None,
) -> Dict[str, Dict]:

    if not feature_names:
        if not X_pdp:
            raise ValueError(
                "provide either features to extract from data or provide ready X_pdp dataframe for model to predict on"
            )
        if not explain_features:
            raise ValueError(
                "provide either all features for model or X_pdp and a list of those that you want explained"
            )
    else:
        if not X_pdp:
            X_pdp = data_pdp.loc[:, feature_names].reset_index(drop = True)
        if not explain_features:
            explain_features = feature_names

    if not cat_features:
        cat_features = []

    if not output_plots:
        plt.ioff()

    if n_cluster_centers:
        cluster = True
    else:
        cluster = False

    ice_frac_to_plot = min(len(X_pdp), ice_frac_to_plot)

    pdp_results = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    for pdp_feature in tqdm(explain_features):
        try:
            show_percentile = False #True if pdp_feature not in cat_features else False
            grid_type = 'percentile'
            cust_grid_points = None
            forced_feature_type = None
            if pdp_feature in cat_features:
                grid_type = 'equal'
                cust_grid_points = np.unique(X_pdp[pdp_feature])
                forced_feature_type = 'categorical'

            if target_name:
                fig, axes, pdp_summary_target = info_plots.target_plot(
                    df = data_pdp,
                    feature = pdp_feature,
                    feature_name = pdp_feature,
                    target = target_name,
                    cust_grid_points = cust_grid_points,
                    grid_type = grid_type,
                    show_percentile = show_percentile,
                    forced_feature_type = forced_feature_type,
                    figsize = figsize,
                )
                pdp_results[pdp_feature]['target']['figure'] = fig
                pdp_results[pdp_feature]['target']['axes'] = axes
                pdp_results[pdp_feature]['target']['summary'] = pdp_summary_target

                if not output_plots:
                    plt.close(fig)

            fig, axes, pdp_summary_model = info_plots.actual_plot(
                model = model,
                X = X_pdp,
                feature = pdp_feature,
                feature_name = pdp_feature,
                cust_grid_points = cust_grid_points,
                grid_type = grid_type,
                show_percentile = show_percentile,
                forced_feature_type = forced_feature_type,
                figsize = figsize,
            )
            pdp_results[pdp_feature]['model']['figure'] = fig
            pdp_results[pdp_feature]['model']['axes'] = axes
            pdp_results[pdp_feature]['model']['summary'] = pdp_summary_model

            if not output_plots:
                plt.close(fig)

            pdp_isolate_out = pdp.pdp_isolate(
                model = model,
                dataset = X_pdp,
                model_features = feature_names,
                feature = pdp_feature,
                cust_grid_points = cust_grid_points,
                grid_type = grid_type,
                forced_feature_type = forced_feature_type,
            )

            fig, axes = pdp.pdp_plot(
                pdp_isolate_out = pdp_isolate_out,
                feature_name = pdp_feature,
                plot_lines = plot_ice,
                center = True,
                plot_pts_dist = True,
                frac_to_plot = ice_frac_to_plot,
                show_percentile = show_percentile,
                cluster = cluster,
                n_cluster_centers = n_cluster_centers,
                figsize = figsize,
            )

            if not output_plots:
                plt.close(fig)

            pdp_results[pdp_feature]['ice']['figure'] = fig
            pdp_results[pdp_feature]['ice']['axes'] = axes
            pdp_results[pdp_feature]['ice']['ice_lines'] = pdp_isolate_out.ice_lines

            pdp_results[pdp_feature] = {k: dict(v) for k, v in pdp_results[pdp_feature].items()}
        except Exception as e:
            print(f"something went wrong with feature '{pdp_feature}'")
            print(e)

    return dict(pdp_results)


def roc_auc_score_safe(y_true, y_score, **kwargs):
    if len(set(y_true)) == 1:
        return -1.0
    return roc_auc_score(y_true, y_score, **kwargs)


def calc_model_metric_lift_by_features(
    model,
    features_sorted: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    treatment: pd.Series = None,
    is_uplift_model: bool = False,
    ignore_model_type: bool = False,
    group_features_by: int = 1,
    metrics: Dict[str, Callable] = None,
    zero_model_metric_value: float = 0,
    prediction_methods: Dict[str, str] = None,
    default_prediction_method: str = 'value',
    eval_samples: Dict[str, Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, pd.Series]]] = None,
    eval_on_train: bool = True,
    train_name: str = 'train',
    n_jobs: int = 1,
    verbose: bool = False,
    progress_bar: bool = True,
    cat_features: List[str] = None,
    **fit_kwargs,
) -> Dict[str, List]:
    """
    eval_samples is a dict of tuples (X,y) or  (X,y,t) if 'is_uplift_model' is True
    if eval_on_train is True, then it's added to eval_samples under the `train_name` label

    metrics is a dict of 'metric_name': metric_func(y_true, y_pred) or metric_func(y_true, y_score)
    while it's possible to provide multiple metrics it's better to run separately
    with those which take labels and those which take scores

    also possible to provide corresponding dict with prediction_methods for metrics that go by the same keys
    otherwise for every metric that will be set as by default_prediction_method

    first level in results - metric, second - sample
    """

    if n_jobs > 1:
        model = model_n_jobs_wrapper(model, n_jobs = 1)
        n_processes = n_jobs
    else:
        n_processes = 1

    ### setting categorical features
    if cat_features is None:
        ### guessing categoricals
        cat_features = X_train.select_dtypes('object').columns.tolist()
    ### setting prediction methods and metric defaults
    metrics = metrics or {'accuracy': accuracy_score}
    prediction_methods = prediction_methods or {}
    eval_samples = eval_samples or {}
    for metric_key in metrics:
        if metric_key not in prediction_methods:
            prediction_methods[metric_key] = default_prediction_method

    ### avoiding type checkers for classic models
    if is_uplift_model:
        ignore_model_type = True
        if not len(y_train) == len(treatment):
            raise ValueError(f"lengths of target and treatment arrays should match")

    if eval_on_train:
        if is_uplift_model:
            eval_samples[train_name] = (X_train, y_train, treatment)
        else:
            eval_samples[train_name] = (X_train, y_train)

    ### setting lists of features with respect to categorical ones
    features_lists = []
    cat_features_lists = []
    L = len(features_sorted)
    current_group_n_features = 0
    all_features_buffer = []
    cat_features_buffer = []

    for i in range(L):
        all_features_buffer = features_sorted[0:i+1]
        current_group_n_features += 1
        if features_sorted[i] in cat_features:
            cat_features_buffer.append(features_sorted[i])
        if (current_group_n_features == group_features_by) or (i == (L - 1)):
            #### drop aggregated list into lists for fitting
            features_lists.append(deepcopy(all_features_buffer))
            cat_features_lists.append(deepcopy(cat_features_buffer))
            current_group_n_features = 0

    ### a little bit hackish, but..
    if is_uplift_model:
        fit_kwargs['t'] = treatment

    def _fit_evaluate_single_step(
        features_list, cat_features_list,
        model, X_train, y_train,
        ignore_model_type,
        fit_kwargs,
        verbose: bool = False,
    ):
        if verbose:
            print(f"fitting model on features: {features_list}")
            print(f"categorical features are: {cat_features_list}")

        model = model_fit_wrapper(
            model, X_train[features_list], y_train,
            cat_features = cat_features_list,
            ignore_model_type = ignore_model_type,
            **fit_kwargs
        )
        eval_results = defaultdict(lambda: defaultdict(float))

        for metric_key in metrics:
            metric_function = metrics[metric_key]
            predictor = model_prediction_wrapper(model, prediction_method = prediction_methods[metric_key])
            for sample_key in eval_samples:
                if is_uplift_model:
                    X, y, t = eval_samples[sample_key]
                    X = X[features_list]
                    eval_results[metric_key][sample_key] = metric_function(y, predictor(X), t)
                else:
                    X, y = eval_samples[sample_key]
                    X = X[features_list]
                    eval_results[metric_key][sample_key] = metric_function(y, predictor(X))

        return eval_results

    if n_processes == 1:
        eval_results = []
        iterator = zip(features_lists, cat_features_lists)
        if progress_bar:
            iterator = tqdm(iterator, total = len(features_lists))
        for f_list, cf_list in iterator:
            eval_results.append(
                _fit_evaluate_single_step(
                    f_list, cf_list,
                    model = model, X_train = X_train, y_train = y_train,
                    ignore_model_type = ignore_model_type,
                    fit_kwargs = fit_kwargs,
                    verbose = verbose,
                )
            )
    else:
        _fit_evaluate_single_step_partial = partial(
            _fit_evaluate_single_step,
            model = model, X_train = X_train, y_train = y_train,
            ignore_model_type = ignore_model_type, fit_kwargs = fit_kwargs,
            verbose = False,
        )

        def _f_unpacking_features_lists_tuple(t):
            return _fit_evaluate_single_step_partial(*t)

        with pathos.multiprocessing.Pool(processes = n_processes) as pool:
            iterator = pool.imap(
                _f_unpacking_features_lists_tuple,
                zip(features_lists, cat_features_lists)
            )
            if progress_bar:
                iterator = tqdm(iterator, total = len(features_lists))
            eval_results = list(iterator)

    uplift_results = defaultdict(lambda: defaultdict(lambda: list))
    for metric_key in metrics:
        for sample_key in eval_samples:
            uplift_results[metric_key][sample_key] = [zero_model_metric_value]
            for r in eval_results:
                uplift_results[metric_key][sample_key].append(r[metric_key][sample_key])
    uplift_results = defaultdict_to_dict(uplift_results)

    if group_features_by == 1:
        return {'steps': ['<none>'] + list(features_sorted), 'metrics': uplift_results, }
    else:
        return {'steps': [['<none>']] + features_lists, 'metrics': uplift_results, }


def plot_model_metric_lift_curves(
    metric_lift_results: Dict[str, List],
    max_factors_to_plot: Optional[int] = 50,
    use_steps_num_on_axis: bool = False,
    drop_zero: bool = True,
    xlabelrotation: int = 45,
    figsize: Tuple[int, int] = None,
    samples_keys: List[str] = None,
    reorder_keys: bool = True,
    color_groups: List[List[str]] = None,
    linestyle_groups: List[List[str]] = None,
):
    """
    visualizes results of calc_model_metric_lift_by_features
    """

    ### to add explicitly list of metrics_names to hang a widget on it

    colors = VARIANTS_COLORS
    colors = [mcolors.CSS4_COLORS.get(col) for col in colors]
    linestyles = LINESTYLES
    n_col, n_ls = len(colors), len(linestyles)

    if drop_zero:
        start_idx = 1
    else:
        start_idx = 0
    steps = metric_lift_results['steps'][start_idx:max_factors_to_plot]

    if use_steps_num_on_axis:
        steps = list(range(len(metric_lift_results['steps'])))[start_idx:max_factors_to_plot]
    metrics = metric_lift_results['metrics']
    count_f = np.arange(len(steps))

    def _get_color(sample_key, j):
        if color_groups is None:
            return colors[j % n_col]
        else:
            for i, g in enumerate(color_groups):
                if sample_key in g:
                    return colors[i % n_col]
        return 'black'

    def _get_linestyle(sample_key, j):
        if linestyle_groups is None:
            return 'solid'
        else:
            for i, g in enumerate(linestyle_groups):
                if sample_key in g:
                    return linestyles[i % n_ls]
        return 'solid'

    def _reorder_keys(keys):
        # hack for trying to achieve consistent colouring throughout the tests by default
        all_keys = list(keys)
        t_keys, nt_keys = [k for k in all_keys if 'train' in k], [k for k in all_keys if 'train' not in k]
        ts_keys, nt_keys = [k for k in nt_keys if 'test' in k], [k for k in nt_keys if 'test' not in k]
        return (*t_keys, *ts_keys, *nt_keys)

    plt.style.use(DEFAULT_PLOT_STYLE)
    if figsize is None:
        figsize = (20, 10 * len(metrics))
    fig, ax = plt.subplots(nrows = len(metrics), ncols = 1, figsize = figsize)
    if len(metrics) > 1:
        pass
    else:
        ax = [ax]
    for i, metric_key in enumerate(metrics.keys()):
        if samples_keys is not None:
            s_iterator = samples_keys
        else:
            s_iterator = list(metrics[metric_key].keys())
        if reorder_keys:
            s_iterator = _reorder_keys(s_iterator)
        for j, sample_key in enumerate(s_iterator):
            ax[i].set_title("Metric dependence on factors included")
            ax[i].plot(
                count_f, metrics[metric_key][sample_key][start_idx:max_factors_to_plot],
                linewidth = 4, label = sample_key, marker = 'o',
                color = _get_color(sample_key, j),
                linestyle = _get_linestyle(sample_key, j),
            )
            ax[i].set(ylabel = metric_key, xlabel = 'Feature addition steps')
            ax[i].set_xticks(count_f)
            if use_steps_num_on_axis:
                ax[i].set_xticklabels([f'step# {f}' for f in steps])
            else:
                ax[i].set_xticklabels([f'+ {f}' for f in steps])
            ax[i].set_xlim([-0.05, count_f[-1] + 0.05])
            ax[i].tick_params(axis = 'y', labelsize = 'medium', labelrotation = 0)
            ax[i].tick_params(axis = 'x', labelsize = 'medium', labelrotation = xlabelrotation)

    for a in ax:
        a.grid(True)
        a.legend(loc = "best", handlelength = 8)

    plt.tight_layout()
    plt.ioff()
    plt.close()

    return fig


def get_model_metric_lift_results_df(metric_lift_results: Dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{'steps': metric_lift_results['steps']},
            **{
                f'{metric}_{k}': metric_lift_results['metrics'][metric][k]
                for metric in metric_lift_results['metrics']
                for k in metric_lift_results['metrics'][metric]
            },
        }
    )


def find_optimal_lift_step(
    steps_df: pd.DataFrame,
    metric_name: str,
    train_sample_key: str,
    test_sample_key: str,
    alpha: float = 0.2,
) -> Dict[str, Union[int, List[str], pd.DataFrame]]:

    steps_df = steps_df.copy()
    if isinstance(steps_df['steps'].iloc[0], list):
        steps_df['n_features'] = steps_df['steps'].apply(len)
    else:
        steps_df['n_features'] = range(len(steps_df['steps']))
    steps_df = steps_df.iloc[1:]

    train_scores = np.array(list(steps_df[f'{metric_name}_{train_sample_key}']))
    test_scores = np.array(list(steps_df[f'{metric_name}_{test_sample_key}']))
    n_features = np.array(list(steps_df['n_features']))

    def penalty(x):
        return np.log1p(x)

    penalties = penalty(n_features)

    # Generalization gap
    gap = train_scores - test_scores

    df = pd.DataFrame({
        'step': range(1, len(test_scores) + 1),
        'n_features': n_features,
        'train_score': train_scores,
        'test_score': test_scores,
        'gap': gap,
        'penalty': penalties,
    })

    gap_threshold = np.quantile(df['gap'], 0.5)
    gap_threshold = (df['gap'].max() - df['gap'].min()) / 2
    df['gap_threshold'] = gap_threshold

    df['selection_score'] = (df['test_score'] - alpha * df['gap']) / df['penalty']

    local_max_indices = argrelextrema(test_scores, np.greater)[0]
    # print(local_max_indices)

    candidates = []
    for i in local_max_indices:
        if gap[i] < gap_threshold:
            selection_score = df['selection_score'].iloc[i]
            candidates.append((i, selection_score))

    # Always consider global max if no suitable local max
    if not candidates:
        optimal_index = int(np.argmax(test_scores))
    else:
        optimal_index = max(candidates, key=lambda x: x[1])[0]

    df['is_local_max'] = df.index.isin(local_max_indices)
    df['is_selected'] = (df.index == optimal_index)

    # tools.display_dataframe_to_user(name="Feature Selection Evaluation", dataframe=df)

    optimal_step = optimal_index + 1

    if isinstance(steps_df['steps'].iloc[0], list):
        optimal_features = steps_df.loc[optimal_step, 'steps']
    else:
        optimal_features = list(steps_df['steps'][:optimal_step])

    return {
        'optimal_step': optimal_step,
        'optimal_features': optimal_features,
        'evaluation_data': df,
    }


def calc_calibration_test(
    data: pd.DataFrame,
    target_col: str = None,
    score_col: str = None,
    group_col: str = None,
    weight_col: str = None,
    use_init_n: bool = False,
    test_type: str = 'binomial',
    alternative: str = 'two-sided',
    confidence_levels: Tuple[float, ...] = (0.95, 0.99),
    paired: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    group_width: int = 4,
    markerwidth: int = 10,
    markersize: int = 10,
    x_fontsize: int = 10,
    y_fontsize: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    grouping is possible for discrete factors by providing group_col parameter
    alternative is either 'two-sided', 'less' or 'greater'
    test_type is either 'binomial' ('b'), 't-test'('t') or 'vasicek' ('v') and the latter is not yet implemented

    null hypothesis would be rejected in favor of alternative with a given confidence level (1 - alpha)
    if calculated statistic wouldn't be placed inside the corresponding confidence interval
    or
    what is actually the same - when p-value would be less than given significance level alpha.
    Confidence level is usually set to 0.95 or 0.99, (alpha to 0.05 or 0.01)

    that said - if, for example, alternative is 'greater',
    then lower bound of interval around H0-value should be infinite
    and P(stat>upper_bound) = alpha

    for binomial test, intervals are always build around forecast
    and p for H0 is calculated as weighted average of data[score_col]

    for t-test here it is made accordingly
    underlying test for calculation of intervals is paired t-test by default
    you can change it to independent setting 'paired' to False

    there is always a mess with one-sided alternative hypothesis and fact/forecast placement in the args
    so, for example, when we want to be conservative and we don't care when PD is way too high compared to target rate,
    but we care when it is much less than the actual value and want test to fail in that case
    e.g. when we are trying to ensure that fact would not be greater than forecast (H0 not rejected) ----
    we set alternative to 'greater' - that way confidence interval below average forecast would be open to -inf
    and if fact lies below forecast - then it would be always okay

    for binomial test, weights could be of any positive float values,
    weighted average target and forecast are normalized through sum of weights
    if that is unwanted for some reason - set use_init_n to True - number of rows in data will be used

    for t-test, weights should be frequency weights and its sum must be greater than 1
    """

    fetch_columns = [col for col in [score_col, target_col, weight_col, group_col] if col is not None]

    if weight_col is None:
        weight_col = '__weights__'
        data = data[fetch_columns].assign(__weights__ = 1)

    if (data[weight_col] < 0).any():
        raise ValueError('negative weights are not allowed')
    elif data[weight_col].sum() <= 0:
        raise ValueError('sum of weights should be strictly positive')
    if any(data[target_col].isnull()):
        raise ValueError('some observations have missing target - remove them before applying this test')
    if any(data[score_col].isnull()):
        raise ValueError('some observations have missing score - remove them before applying this test')
    elif (any(data[score_col] > 1) or any(data[score_col] < 0)) and test_type in ['binomial', 'vasicek']:
        raise ValueError('some observations have probability outside the range of [0,1]')

    if group_col is not None:
        groups = sorted(data[group_col].unique())
        masks = [data[group_col] == value for value in groups]
    else:
        groups = ['All Sample']
        masks = [pd.Series([True for _ in range(len(data))], index = data.index)]

    calibration_stats_results = {}

    center_name, value_name, ci_name, volume_name = 'S_avg_w', 'T_avg_w', 'CI', 'Volume'

    for group, mask in zip(groups, masks):
        group_stats = {}
        group_stats['N'] = N = data.loc[mask, :].shape[0]
        group_stats['T'] = T = data.loc[mask, target_col].sum()
        group_stats['S'] = T = data.loc[mask, score_col].sum()
        group_stats['T_avg'] = T_avg = data.loc[mask, target_col].mean()
        group_stats['S_avg'] = S_avg = data.loc[mask, score_col].mean()
        group_stats['S_min'] = S_min = data.loc[mask, score_col].min()
        group_stats['S_max'] = S_max = data.loc[mask, score_col].max()

        if verbose:
            print(
                f"Number of observations: {N}, Target sum total: {T}, Target average: {T_avg}, Score average: {S_avg}"
            )

        group_stats['N_w'] = N_w = calc_weighted_sum(np.ones(N), data.loc[mask, weight_col])
        group_stats['T_w'] = T_w = calc_weighted_sum(data.loc[mask, target_col], data.loc[mask, weight_col])
        group_stats['S_w'] = S_w = calc_weighted_sum(data.loc[mask, score_col], data.loc[mask, weight_col])
        if not use_init_n:
            ### classic weighting
            group_stats[value_name] = T_avg_w = calc_weighted_mean(
                data.loc[mask, target_col], data.loc[mask, weight_col]
            )
            group_stats[center_name] = S_avg_w = calc_weighted_mean(
                data.loc[mask, score_col], data.loc[mask, weight_col]
            )
            group_stats[volume_name] = N_w
        else:
            ### normalizing on init obs count
            group_stats[value_name] = T_avg_w = T_w / N
            group_stats[center_name] = S_avg_w = S_w / N
            group_stats[volume_name] = N

        if verbose:
            print(
                f"Weighted number of observations: {N_w}, Targets weighted sum total: {T_w}, "
                f"Weighted target average: {T_avg_w}, Weighted score average: {S_avg_w}"
            )

        ci_results = []

        if test_type.lower() in ('binomial', 'binom', 'b'):
            for conf_level in confidence_levels:
                t_result = calc_binomial_test(
                    n = round(group_stats[volume_name], 0),
                    p = S_avg_w,
                    statistic = T_avg_w,
                    ratio = True,
                    alternative = alternative,
                    confidence_level = conf_level,
                )
                lower_bound, upper_bound = t_result['ci_h0']
                p_value = t_result['p_value']
                ci_results.append((conf_level, (lower_bound, upper_bound)))
            group_stats['p_value'] = p_value
        elif test_type.lower() in ('t', 't-test', 'ttest', 't_test'):
            for conf_level in confidence_levels:
                t_result = calc_t_test_two_samples(
                    x = data.loc[mask, target_col],
                    y = data.loc[mask, score_col],
                    weights = data.loc[mask, weight_col],
                    mu = 0.0,
                    paired = paired,
                    alternative = alternative,
                    confidence_level = conf_level,
                )
                lower_bound, upper_bound = t_result['ci_h0']
                p_value = t_result['p_value']
                ### note that ordering of x and y matters in the call above for pvalue when alternative is one-sided
                ### ci returned above is symmetric around 0 (fact - forecast) <==> l < fact - forecast < u
                ### note, that also it can be read like l < forecast - fact < u,
                ### as it just quantiles of t-distribution with calculated df
                ### but we want them to be symmetric around forecast, so:
                ci_results.append((conf_level, (S_avg_w + lower_bound, S_avg_w + upper_bound)))
            group_stats['p_value'] = p_value
        elif test_type in ('v', 'vasicek'):
            raise ValueError(f"{test_type} not implemented")
        else:
            raise ValueError(f"unknown test type {test_type}")

        group_stats[ci_name] = ci_results
        calibration_stats_results[group] = group_stats

    fig = plot_confidence_intervals(
        calibration_stats_results.items(),
        center_name = center_name, ci_name = ci_name, volume_name = volume_name, value_name = value_name,
        group_width = group_width, color_when_widening = False, figsize = figsize,
        markerwidth = markerwidth, markersize = markersize, x_fontsize = x_fontsize, y_fontsize = y_fontsize,
    )
    fig.axes[0].set_xlabel(group_col)

    return {
        'results': calibration_stats_results,
        'figure': fig,
    }


def calc_calibration_test_buckets(
    target: pd.Series,
    score: pd.Series,
    n_buckets: int = 10,
    force_even_buckets: bool = False,
    **kwargs,
) -> Dict[str, Any]:

    score_buckets = bucketize(
        values=score,
        n_buckets=n_buckets,
        force_even_buckets=force_even_buckets,
        merge_to_right_side=True,
        ecdf=None,
        force_consecutive_markers=True,
        start=1,
    )

    calibration_results = calc_calibration_test(
        data = pd.DataFrame({'score_bucket': score_buckets, 'score': score, 'target': target}),
        target_col = 'target', score_col = 'score', group_col = 'score_bucket',
        **kwargs,
    )

    return calibration_results


def parse_calibration_results(
    calibration_results: Dict,
    group_name: str = 'All Sample',
    info: str = 'brief',
) -> pd.DataFrame:

    def parse_ci(l):
        s = sorted(l, key = lambda x: x[0])
        labels = []
        values = []
        for ci in s:
            labels.insert(0, f'CI_{ci[0]}_low')
            labels.append(f'CI_{ci[0]}_high')
            values.insert(0, ci[1][0])
            values.append(ci[1][1])
        return pd.DataFrame(data = values, index = labels)

    results = calibration_results['results']
    ci = parse_ci(results[group_name]['CI'])

    if info == 'brief':
        d = {
            'Volume': 'Volume', 'T_avg_w': 'Average Target', 'S_avg_w': 'Average Score',
            'S_min': 'Min Score', 'S_max': 'Max Score',
        }
        result = pd.concat([pd.DataFrame({d[k]: [results[group_name][k]] for k in d}).transpose(), ci])
        result.columns = ['Value']
    else:
        raise NotImplementedError

    return result


def plot_confidence_intervals(
    results: Iterable[Tuple],
    center_name: str,
    ci_name: str,
    value_name: str,
    volume_name: str = 'N',
    color_when_widening: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    group_width: int = 4,
    markerwidth: int = 5,
    markersize: int = 5,
    x_fontsize: int = 10,
    y_fontsize: int = 10,
) -> plt.Figure:
    """
    takes sequence of tuples (group_name, group_results_dict)
    every inner dict should have data in specified format, values from which taken by names,
    but also it may hold some more additional data
    {
        volume_name: N_observations | Total weight of the group ,
        center_name: center value of confidence interval,
        ci_name: sequence of tuples like (ci_level_i, (lower_bound_i, upper_bound_i)),
        value_name: scalar value for which confidence interval hit is checked
    }
    color_when_widening is used to color value according to CI color which it got into first time
    there it is assumed that confidence levels are ordered in increasing manner
    """

    ### add volume of group on the second y axis behind each interval
    ### add placement series of other values onto the same CI in np.array or list if several scores are passed
    ### currently it is a single float stored under value_name

    ci_colors = [
        'gold', 'crimson', 'red',
        'darkmagenta', 'lightskyblue', 'mediumorchid', 'forestgreen', 'dodgerblue', 'springgreen', 'navy'
    ]
    colors = [mcolors.CSS4_COLORS.get(col) for col in ci_colors]
    n_col = len(colors)

    plt.style.use(DEFAULT_PLOT_STYLE)
    if len(results) > 1:
        figsize = (max(figsize[0], group_width * len(results)), min(12, max(figsize[1], len(results))))
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    plt.title('Calibration test', fontdict = {'fontsize': 16})
    # for s in ['top', 'bottom', 'right', 'left',]:
    #     ax0.spines[s].set_color('w')
    ax.xaxis.grid()

    ax.set_xlim(left = 0, right = len(results))

    xtickgroups = []
    lbs, ubs = [], []
    for j, (group, group_results) in enumerate(results):
        xtickgroups.append(group)
        xloc = j + 0.5
        xleftb = j + 0.3
        xrightb = j + 0.7
        i_color = len(group_results[ci_name])

        for i, ci_res in enumerate(group_results[ci_name]):
            ci_level, (lb, ub) = ci_res
            color = colors[i % n_col]
            if lb <= group_results[value_name] <= ub:
                ### remember into which interval we got first
                i_color = min(i, i_color)

            label = f"{round(ci_level * 100, 2)}% CI" if j == 0 else ''

            ### here we suppose that both boundaries cannot be infinite at the same time
            if lb == -np.inf:
                ubs.append(ub)
                ax.vlines(x = xloc, ymin = group_results[center_name], ymax = ub, color = color, label = label)
                ax.hlines(y = ub, xmin = xleftb, xmax = xrightb, color = color)
            elif ub == np.inf:
                lbs.append(lb)
                ax.vlines(x = xloc, ymin = lb, ymax = group_results[center_name], color = color, label = label)
                ax.hlines(y = lb, xmin = xleftb, xmax = xrightb, color = color,)
            else:
                lbs.append(lb)
                ubs.append(ub)
                ax.vlines(x = xloc, ymin = lb, ymax = ub, color = color, label = label)
                ax.hlines(y = lb, xmin = xleftb, xmax = xrightb, color = color,)
                ax.hlines(y = ub, xmin = xleftb, xmax = xrightb, color = color)

        # forecast
        ax.plot(
            xloc, group_results[center_name],
            label = 'Average Forecast' if j == 0 else '',
            linestyle = '', markersize = markersize, linewidth = markerwidth, marker = 'v', color = 'black'
        )
        lbs.append(group_results[center_name])
        ubs.append(group_results[center_name])

        # fact
        if color_when_widening:
            if i_color < len(group_results[ci_name]):
                ### lies inside some interval
                color = colors[i_color % n_col]
            else:
                color = 'grey'
        else:
            color = 'black'
        ax.plot(
            xloc, group_results[value_name],
            label = 'Average Actual Value' if j == 0 else '',
            linestyle = '', markersize = markersize, linewidth = markerwidth, marker = '^', color = color,
        )
        lbs.append(group_results[value_name])
        ubs.append(group_results[value_name])

    ax.legend(ncol = 4, loc = 'upper center')

    ax.set_xticks(ticks = np.arange(len(results)) + 0.5, labels = xtickgroups, fontsize = x_fontsize)
    plt.yticks(fontsize = y_fontsize)

    bounds_range = max(ubs) - min(lbs)
    ax.set_ylim(bottom = min(lbs) - 0.15 * bounds_range, top = max(ubs) + 0.15 * bounds_range)
    # plt.rc('legend', fontsize = int((y_fontsize + x_fontsize) / 2))
    plt.ioff()
    plt.close()

    return fig


def calc_psi_discrete(sh1_i, sh2_i, eps = 0.0001):
    sh1 = deepcopy(sh1_i)
    sh2 = deepcopy(sh2_i)
    sh2[sh2 <= 0] = eps
    #print(sh2)
    return ((sh1-sh2) * np.log(sh1/sh2)).sum()


def calc_hhi_discrete(cnts):
    J = len(cnts)
    return (sum((cnts/sum(cnts))**2) - 1.0 / J) / (1 - 1.0 / J)


def calc_gini(y_true, y_score, ordering = 'desc'):
    if ordering == 'desc':
        gini = round((roc_auc_score(y_true, y_score) * 2 - 1), 8)
    else:
        gini = round((roc_auc_score(y_true, (-1) * y_score) * 2 - 1), 8)
    return gini


class PSI:
    def __init__(self, eps = 0.0001, continuity_cutoff = 10, show = False):
        self.eps = eps
        self.continuity_cutoff = continuity_cutoff
        self.show = show

    @staticmethod
    def utl_binning_scott_ranges(
        x, bin_n_buckets = None,
        bin_q_threshold = 0.01, bin_n_buckets_max = 20,
        continuity_cutoff = 10, eps = 1e-10
    ):
        x_unique = pd.Series(x).unique()
        n_unique = len(x_unique)
        if n_unique <= 1:
            raise ValueError("cannot bin feature with only one value")
        if n_unique <= continuity_cutoff:
            bin_n_buckets = n_unique
            thresholds = [-np.inf] + [x + eps for x in sorted(x_unique)]
            return thresholds

        q_lower = q_upper = 0
        while q_lower == q_upper:
            q_lower = np.nanquantile(x, bin_q_threshold)
            q_upper = np.nanquantile(x, 1 - bin_q_threshold)
            bin_q_threshold /= 10

        if bin_n_buckets is None:
            # if the parameter is not set, then calculate some "recommended" number of buckets based on analytical step
            f_step = len(x) ** (-1 / 3) * 3.5 * np.std(x)
            bin_n_buckets = min(
                bin_n_buckets_max,
                np.ceil(q_upper - q_lower) / f_step + 1
            )  # should not be greater than 'bin_n_buckets_max' (50 by default)

        significance = - int(np.ceil(np.log10((q_upper - q_lower) / bin_n_buckets) - 2))
        f_step = np.round((q_upper - q_lower) / bin_n_buckets, decimals=significance)

        q_lower = np.ceil(q_lower * (10 ** significance)) / (10 ** significance)
        # q_upper = q_lower + np.floor((q_upper - q_lower) / f_step) * f_step

        thresholds = []
        for i in range(1, int(bin_n_buckets) + 1):
            thresholds.append(q_lower + (i - 1) * f_step)

        thresholds.insert(0, np.nanmin(x))
        thresholds.insert(len(thresholds), np.nanmax(x))

        if len(thresholds) > 3:
            frst = len(x[x < thresholds[1]])
            lst = len(x[x > thresholds[-2]])
            if lst / len(x) < 0.025:
                thresholds.pop(-2)
            if frst / len(x) < 0.025:
                thresholds.pop(1)

        return thresholds

    def plot_two_binned_distributions(
        self, shares1, shares2, bins_labels = None,
        labels = ('1', '2'),
        colors = ('springgreen', 'lightskyblue'),
        psi_value = None,
        title = None, ax = None, figsize = (8, 5),
    ):

        plt.style.use(DEFAULT_PLOT_STYLE)
        width = 0.4
        locs = np.arange(len(shares1))
        # plt.rc('xtick', labelsize = 8)
        xticklabels = {"rotation": 'vertical', "fontsize": 10}

        if ax is not None:
            fig = None
        else:
            fig, ax = plt.subplots(1, 1, figsize = figsize)

        rects1 = ax.bar(locs, shares1, width, color = colors[0], alpha = 0.8)
        rects2 = ax.bar(locs + width, shares2, width, color = colors[1], alpha = 0.8)

        if psi_value is not None:
            stats_str = f'PSI = {psi_value:.5g}'
        else:
            stats_str = ''

        ax.grid(True)
        ax.set_ylabel('Population Share', fontsize = 10)
        ax.set_xlabel('Bins', fontsize = 10)
        ax.set_title('Histogram' if not title else title, fontsize = 10)
        ax.set_xticks(locs+width/2)
        ax.tick_params(axis='both', which='major', labelsize=10)

        if bins_labels is not None:
            ax.set_xticklabels(bins_labels, **xticklabels)

        # if labels != ('', ''):
        leg = ax.legend((rects1[0], rects2[0]), labels, loc="upper right", fontsize = 10)
        leg.set_title(stats_str, prop = {"size": 10})
        plt.xlim((-1, len(shares1) + 1))

        if not self.show and fig is not None:
            # plt.tight_layout()
            plt.close(fig)

        return fig

    def plot_two_distributions(
        self, f1, f2,
        labels = ('1', '2'),
        colors = ('springgreen', 'lightskyblue'),
        title = None,
        ax = None, figsize = (8, 5), plot_margin = 0.05
    ):

        plt.style.use(DEFAULT_PLOT_STYLE)
        colors = [mcolors.CSS4_COLORS.get(col) for col in colors]
        N_col = len(colors)

        if not ((len(labels) == 2) and (labels[0] != labels[1])):
            raise ValueError(f"there must be two labels different from each other, got {labels}")
        score_name = 'value'
        label_name = 'label'
        df1 = pd.DataFrame({score_name: f1})
        df1[label_name] = labels[0]
        df2 = pd.DataFrame({score_name: f2})
        df2[label_name] = labels[1]
        dataframe = pd.concat([df1, df2], axis = 0)

        mins, maxs = [], []
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            ax.grid(True)
        else:
            fig = None

        ax.set_title('Density estimates' if not title else title, fontsize = 10)
        ax.set_xlabel('Value', fontsize = 10)
        ax.set_ylabel('Density', fontsize = 10)

        for i, label in enumerate(labels):
            subsample = dataframe.loc[dataframe[label_name] == label, score_name]
            mins.append(np.nanmin(subsample))
            maxs.append(np.nanmax(subsample))
            bw = statsmodels.nonparametric.bandwidths.bw_scott(subsample)
            if bw == 0:
                bw = 0.9 * np.std(subsample) * len(subsample)**(-1/5)
                sns.distplot(subsample, kde_kws = {'bw': bw}, ax = ax, label = label, color = colors[i % N_col])
            else:
                sns.histplot(subsample, stat = "density", kde = True, ax = ax, label = label, color = colors[i % N_col])

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(loc="upper right")

        #xmin, xmax = ax.get_xlim()
        xmin = min((1+plot_margin)*min(mins), (1-plot_margin)*min(mins))
        xmax = max((1+plot_margin)*max(maxs), (1-plot_margin)*max(maxs))
        if np.isclose(xmin, 0, atol=1e-4):
            xmin = xmin - max(plot_margin, 0.5*np.nanstd(dataframe[score_name]))
        elif np.isclose(xmax, 0, atol=1e-4) == 0:
            xmax = xmax + max(plot_margin, 0.5*np.nanstd(dataframe[score_name]))
        else:
            pass

        ax.set_xlim(xmin, xmax)
        if not self.show and fig is not None:
            # plt.tight_layout()
            plt.close(fig)

        return fig

    @staticmethod
    def calc_psi_shares(shares_1, shares_2, eps = 0.0001):
        """
        calculates psi on two samples where observations are already binned
        """
        sh1 = deepcopy(shares_1)
        sh2 = deepcopy(shares_2)
        sh1[sh1 <= 0] = eps
        sh2[sh2 <= 0] = eps

        return ((sh1 - sh2) * np.log(sh1 / sh2)).sum()

    def calc_psi_in_bins(
        self,
        expected,
        actual,
        bins,
        labels: Tuple[str, str],
        feature_name: str,
        colors: Tuple[str, str] = ('springgreen', 'lightskyblue'),
        ax: plt.Axes = None,
        do_plotting: bool = True,
        figsize: Tuple[int, int] = (8, 5),
    ):
        expected_bins = np.histogram(expected, bins)[0]
        expected_bins = expected_bins / len(expected)
        actual_bins = np.histogram(actual, bins)[0]
        actual_bins = actual_bins / len(actual)

        psi_value = self.calc_psi_shares(expected_bins, actual_bins, self.eps)

        if do_plotting:
            bins_labels = [f'[{l:.5g}, {r:.5g}]' for l, r in zip(bins[:-1], bins[1:])]
            fig = self.plot_two_binned_distributions(
                expected_bins, actual_bins,
                bins_labels = bins_labels, labels = labels, psi_value = psi_value,
                colors = colors,
                title = feature_name, ax = ax, figsize = figsize
            )
        else:
            fig = None

        return psi_value, fig

    @staticmethod
    def calc_discrete_bins_share(x):
        dataframe = pd.DataFrame({'x': x})
        dataframe['cnt'] = 1
        gb = dataframe.groupby('x').agg({'cnt': ['count']})
        gb.columns = ['Obs. Share']
        gb['Obs. Share'] /= gb['Obs. Share'].sum()
        return gb

    @staticmethod
    def _merge_empty_bins(bins, expected):
        # bins for histogram are [, ) except for the last, where it is (, ]
        counts, merged_bins = np.histogram(expected, bins)
        cdf = pd.DataFrame({'l': merged_bins[:-1], 'r': merged_bins[1:], 'c': counts})
        cdf['ccs'] = cdf['c'].cumsum()
        cdf = cdf.groupby(['ccs']).agg({'l': min, 'r': max, 'c': sum}).sort_values(by = 'l')
        merged_bins = np.hstack([cdf['l'].values, cdf['r'].values[-1:]])
        return merged_bins

    def calc_psi(
        self,
        expected: Union[pd.Series, np.ndarray],
        actual: Union[pd.Series, np.ndarray],
        discrete: bool = False,
        rounding: int = None,
        max_buckets: int = 20,
        merge_empty_bins: bool = True,
        labels: Tuple[str, str] = ('1', '2'),
        colors: Tuple[str, str] = ('springgreen', 'lightskyblue'),
        feature_name: str = '',
        do_plotting: bool = True,
        ax: plt.Axes = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> Dict[str, Union[float, plt.Figure]]:

        if rounding:
            expected = np.round(expected, rounding)
            actual = np.round(actual, rounding)

        if discrete:
            expected_df = self.calc_discrete_bins_share(expected)
            actual_df = self.calc_discrete_bins_share(actual)
            df = pd.merge(expected_df, actual_df, left_index = True, right_index = True, how = 'outer').fillna(0.0)
            df.columns = ['expected', 'actual']
            shares1, shares2 = df['expected'], df['actual']
            psi_value = self.calc_psi_shares(shares1, shares2)

            if do_plotting:
                psi_plot = self.plot_two_binned_distributions(
                    shares1, shares2,
                    bins_labels = [f'{v:.5g}' if isinstance(v, (int, float)) else f'{v}' for v in df.index],
                    labels = labels, colors = colors,
                    psi_value = psi_value,
                    title = ' ',
                    # title = feature_name,
                    figsize = figsize
                )
        else:
            try:
                bins = self.utl_binning_scott_ranges(
                    expected, bin_n_buckets_max = max_buckets, continuity_cutoff = self.continuity_cutoff
                )
            except ValueError as e:
                print(e)
                return {
                    'psi': np.nan,
                    'figure': None,
                }

            if merge_empty_bins:
                bins = self._merge_empty_bins(bins, expected)

            if ax is None and do_plotting:
                fig, axes = plt.subplots(1, 2, figsize = figsize)
                ax = axes[0]
            psi_value, psi_plot = self.calc_psi_in_bins(
                expected, actual,
                bins = bins,
                labels = labels, colors = colors,
                feature_name = None,
                #feature_name = feature_name,
                do_plotting = do_plotting,
                ax = ax
            )

            if psi_plot is None and do_plotting:
                _ = self.plot_two_distributions(
                    expected, actual,
                    labels = labels,
                    colors = colors,
                    title = None,
                    # title = feature_name,
                    ax = axes[1]
                )
                psi_plot = fig
            if do_plotting:
                plt.close(fig)

        if do_plotting:
            figure = psi_plot
            figure.suptitle(feature_name)
        else:
            figure = None
        return {
            'psi': psi_value,
            'figure': figure,
        }

    def _get_psi_results_dataframe(self, results: Dict[str, Union[float, plt.Figure]]):
        # {f:{'all':{'psi':, 'figure':}, 'target':{'psi':, 'figure':}, 'non-target':{'psi':, 'figure':}},..}
        features = list(results.keys())
        cuts = sorted(set(itertools.chain(*(results[f].keys() for f in features))))
        df = pd.DataFrame(columns = cuts, index = features)
        for f in features:
            for c in cuts:
                df.loc[f, c] = results[f].get(c, {}).get('psi', np.nan)
        return df.astype(float)

    def get_psi_results(
        self,
        df_expected: pd.DataFrame,
        df_actual: pd.DataFrame,
        features: List[str],
        na_filler: float = None,
        target_name: str = None,
        discrete: bool = False,
        rounding: int = None,
        max_buckets: int = 20,
        merge_empty_bins: bool = True,
        do_plotting: bool = True,
        labels: Tuple[str, str] = ('1', '2'),
        colors: Tuple[str, str] = ('springgreen', 'lightskyblue'),
        figsize: Tuple[int, int] = (12, 5),
        n_jobs: int = 1,
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:

        if isinstance(features, str):
            features = [features]

        n_processes = min(min(pathos.multiprocessing.cpu_count(), n_jobs), len(features))
        if len(features) <= 20:
            n_processes = 1

        def assert_non_empty(expected, actual, stage_message):
            if not len(expected) > 0:
                raise ValueError(f"{stage_message} first data sequence seems to be empty")
            if not len(actual) > 0:
                raise ValueError(f"{stage_message} second data sequence seems to be empty")

        kwargs = {
            'df_expected': df_expected,
            'df_actual': df_actual,
            'na_filler': na_filler,
            'target_name': target_name,
            'discrete': discrete,
            'rounding': rounding,
            'max_buckets': max_buckets,
            'merge_empty_bins': merge_empty_bins,
            'labels': labels,
            'colors': colors,
            'figsize': figsize,
        }

        def calc_psi_single_feature(
            f: str,
            df_expected: pd.DataFrame,
            df_actual: pd.DataFrame,
            na_filler: float = None,
            target_name: str = None,
            discrete: bool = False,
            rounding: int = None,
            max_buckets: int = 20,
            merge_empty_bins: bool = True,
            labels: Tuple[str, str] = ('1', '2'),
            colors: Tuple[str, str] = ('springgreen', 'lightskyblue'),
            figsize: Tuple[int, int] = (12, 5),
        ) -> Dict[str, Dict[str, Union[float, plt.Figure]]]:

            expected, actual = df_expected[f], df_actual[f]
            assert_non_empty(expected, actual, stage_message = f"all data, feature: '{f}'")
            if na_filler is not None:
                expected, actual = expected.fillna(na_filler), actual.fillna(na_filler)

            psi_all = self.calc_psi(
                expected, actual,
                discrete = discrete, rounding = rounding, max_buckets = max_buckets,
                labels = labels, colors = colors,
                merge_empty_bins = merge_empty_bins,
                feature_name = f, do_plotting = do_plotting,
                figsize = figsize
            )
            if np.isnan(psi_all['psi']):
                print(f"unable to calculate for {f} on all population with current settings, retrying with 'discrete' = True")
                psi_all = self.calc_psi(
                    expected, actual,
                    discrete = True, rounding = rounding, max_buckets = max_buckets,
                    labels = labels, colors = colors,
                    merge_empty_bins = merge_empty_bins,
                    feature_name = f, do_plotting = do_plotting,
                    figsize = figsize
                )
            if np.isnan(psi_all['psi']):
                print(f'unable to calculate for {f} on all population')

            psi_results_f = {'all': psi_all}

            if target_name:
                expected, actual = (
                    df_expected.query(f"{target_name} == 1")[f],
                    df_actual.query(f"{target_name} == 1")[f]
                )
                assert_non_empty(expected, actual, stage_message = f"target == 1, feature: '{f}'")
                if na_filler is not None:
                    expected, actual = expected.fillna(na_filler), actual.fillna(na_filler)

                psi_target = self.calc_psi(
                    expected, actual,
                    discrete = discrete, rounding = rounding, max_buckets = max_buckets,
                    labels = labels, colors = colors,
                    merge_empty_bins = merge_empty_bins,
                    feature_name = f + ' <<target>>',
                    do_plotting = do_plotting, figsize = figsize,
                )

                if np.isnan(psi_target['psi']):
                    print(f"unable to calculate for {f} on target population with current settings, retrying with 'discrete' = True")
                    psi_target = self.calc_psi(
                        expected, actual,
                        discrete = True, rounding = rounding, max_buckets = max_buckets,
                        labels = labels, colors = colors,
                        merge_empty_bins = merge_empty_bins,
                        feature_name = f + ' <<target>>',
                        do_plotting = do_plotting, figsize = figsize,
                    )
                if np.isnan(psi_target['psi']):
                    print(f'unable to calculate for {f} on target population')

                expected, actual = (
                    df_expected.query(f"{target_name} == 0")[f],
                    df_actual.query(f"{target_name} == 0")[f]
                )
                assert_non_empty(expected, actual, stage_message = f"target == 0, feature: '{f}'")
                if na_filler is not None:
                    expected, actual = expected.fillna(na_filler), actual.fillna(na_filler)

                psi_nontarget = self.calc_psi(
                    expected, actual,
                    discrete = discrete, rounding = rounding, max_buckets = max_buckets,
                    labels = labels, colors = colors,
                    merge_empty_bins = merge_empty_bins,
                    feature_name = f + ' <<non-target>>',
                    do_plotting = do_plotting, figsize = figsize
                )

                if np.isnan(psi_nontarget['psi']):
                    print(f"unable to calculate for {f} on non-target population with current settings, retrying with 'discrete' = True")
                    psi_nontarget = self.calc_psi(
                        expected, actual,
                        discrete = True, rounding = rounding, max_buckets = max_buckets,
                        labels = labels, colors = colors,
                        merge_empty_bins = merge_empty_bins,
                        feature_name = f + ' <<target>>',
                        do_plotting = do_plotting, figsize = figsize,
                    )
                if np.isnan(psi_nontarget['psi']):
                    print(f'unable to calculate for {f} on non-target population')

                psi_results_f['target'] = psi_target
                psi_results_f['non-target'] = psi_nontarget

            return psi_results_f

        def _mp_func_on_iterator(iterator, f):
            # This line is the strange hack for tqdm/mp in jupyter
            print(' ', end='', flush=True)
            return [f(arg) for arg in tqdm(iterator)]

        f_p = partial(calc_psi_single_feature, **kwargs,)
        if n_processes > 1:
            f_p = partial(_mp_func_on_iterator, f = f_p)
            with pathos.multiprocessing.Pool(processes = n_processes) as pool:
                cols_parts = np.array_split(features, n_processes)
                psi_results = pool.map(f_p, cols_parts)
                psi_results = {f: r for f, r in zip(features, itertools.chain(*psi_results))}
        else:
            psi_results = {f: f_p(f) for f in tqdm(features)}

        psi_results_df = self._get_psi_results_dataframe(psi_results)
        return {
            'psi_df': psi_results_df,
            'psi_dict': psi_results
        }


def get_boosting_importances(model: lgb.LGBMModel, importance_type: str = 'gain') -> pd.DataFrame:
    imp_type = model.importance_type
    if importance_type in ('gain', 'split'):
        model.importance_type = importance_type
    else:
        raise ValueError(f"unknown importance_type value '{importance_type}'")
    df = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by = 'importance', ascending = False)
    df['rank'] = range(1, len(df) + 1)

    model.importance_type = imp_type
    return df

def calc_permutation_importances(
    model,
    X,
    y_true,
    metric: Callable,
    treatment = None,
    greater_is_better: bool = True,
    columns: List[str] = None,
    n_shuffles: int = 10,
    seed: int = 8,
    prediction_method: str = 'proba',
    n_jobs: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    to do: should work with model keeper too

    `metric` is a func(y_true, y_pred) or func(y_true, y_score)
    `prediction_method` should correspond to `metric`
    """
    n_processes = min(pathos.multiprocessing.cpu_count(), n_jobs)

    try:
        from pandas.core.common import SettingWithCopyWarning as SettingWithCopyWarningClass
    except Exception as e:
        from pandas.errors import SettingWithCopyWarning as SettingWithCopyWarningClass

    reset_n_jobs = None
    if n_processes > 1:
        ### not applicable here cause it makes model to become not fitted
        # model = model_n_jobs_wrapper(model, n_jobs = 1)
        if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            reset_n_jobs = model.n_jobs
            model.n_jobs = 1

    def _get_predict_function(
        model, prediction_method,
    ):
        def _pf_proba(X, **kwargs):
            return model.predict_proba(X, **filter_kwargs(model.predict_proba, kwargs))
        def _pf_proba_1(X, **kwargs):
            return model.predict_proba(X, **filter_kwargs(model.predict_proba, kwargs))[:, 1]
        def _pf_predict(X, **kwargs):
            return model.predict(X, **filter_kwargs(model.predict, kwargs))
        def _pf_predict_threshold(X, **kwargs):
            return model.predict_by_threshold(X, **filter_kwargs(model.predict_by_threshold, kwargs))

        if prediction_method == 'proba':
            if isinstance(model, ModelKeeper):
                predict_function = _pf_proba
            else:
                predict_function = _pf_proba_1
        elif prediction_method in ('label', 'class', 'predict', 'value', 'uplift'):
            predict_function = _pf_predict
        elif prediction_method == 'threshold':
            if isinstance(model, ModelKeeper):
                predict_function = _pf_predict_threshold
            else:
                raise NotImplementedError("threshold prediction method is only available through ModelKeeper class")
        else:
            raise ValueError("wrong prediction method")

        return predict_function

    def _single_feature_permutation_importance(
        col, X, treatment, n_shuffles, predict_function, metric, seed, kwargs,
    ):
        np.random.seed(seed)
        m = 0
        warnings.filterwarnings("ignore", category = SettingWithCopyWarningClass)
        for _ in range(n_shuffles):
            saved = X[col].copy()
            X[col] = np.random.permutation(X[col])
            m += (
                metric(y_true, predict_function(X, **kwargs)) if treatment is None
                else metric(y_true, predict_function(X, **kwargs), treatment)
            )
            X[col] = saved
        m /= n_shuffles
        warnings.filterwarnings("default", category = SettingWithCopyWarningClass)
        return m

    def _mp_func_on_iterator(iterator, f):
        # This line is the strange hack for tqdm/mp in jupyter
        print(' ', end='', flush=True)
        return [f(arg) for arg in tqdm(iterator)]

    if columns is not None and len(columns) > 0:
        if isinstance(columns, str):
            columns_iterator = [columns]
        else:
            columns_iterator = columns
    else:
        columns_iterator = list(X.columns)

    predict_function = _get_predict_function(model, prediction_method)
    baseline = (
        metric(y_true, predict_function(X, **kwargs)) if treatment is None
        else metric(y_true, predict_function(X, **kwargs), treatment)
    )

    f_col = partial(
        _single_feature_permutation_importance,
        X = X, treatment = treatment,
        n_shuffles = n_shuffles, predict_function = predict_function, metric = metric, seed = seed,
        kwargs = kwargs,
    )
    f = partial(_mp_func_on_iterator, f = f_col)

    if n_processes > 1:
        with pathos.multiprocessing.Pool(processes = n_processes) as pool:
            cols_parts = np.array_split(columns_iterator, n_processes)
            imp = pool.map(f, cols_parts)
            imp = list(itertools.chain(*imp))
    else:
        imp = f(columns_iterator)

    if reset_n_jobs is not None:
        model.n_jobs = reset_n_jobs

    imp = pd.DataFrame(data = imp, columns=['importance'], index = columns_iterator)
    imp['importance'] = baseline - imp['importance']
    imp.sort_values(axis=0, ascending = False, by='importance', inplace = True)
    if greater_is_better:
        return imp
    else:
        imp = -imp
        return imp[::-1]


def get_shap_absolute_importances(features, shap_values):
    return pd.DataFrame({
        'feature': features,
        'importance': np.abs(shap_values).mean(axis = 0)
    }).sort_values(by = 'importance', ascending = False).set_index('feature')


def plot_shap_summary(
    shap_values, data_for_explanation, features,
    plot_max_features = None, plot_absolute = False, figsize = (10, 8),
    title = '',
):
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    shap.summary_plot(
        shap_values,
        data_for_explanation,
        feature_names = features,
        # auto_size_plot = False,
        plot_size = None,
        max_display = plot_max_features,
        show = False,
    )
    if not plot_absolute:
        plt.gcf().axes[-1].set_aspect(100)
        plt.gcf().axes[-1].set_box_aspect(100)
    if title:
        plt.title(title)
    plt.close()
    return fig


def calc_shap_importances(
    model,
    data: pd.DataFrame,
    features: List[str],
    method: str = 'tree',
    objective: str = 'binary_classification',
    sample_size: int = None,
    seed: int = None,
    plot_max_features: int = None,
    plot_absolute: bool = False,
    figsize: Tuple[int, int] = (10, 8),
):

    if sample_size:
        sample_size = min(sample_size, len(data))
        seed = seed or 8
        np.random.seed(seed)
        data_for_explanation = data[features].sample(n = sample_size, replace = False, random_state = seed)
    else:
        data_for_explanation = data[features]
    # y_for_explanation = data.loc[data_for_explanation.index, target_name].reset_index(drop = True)

    if method == 'tree':
        tree_explainer = shap.TreeExplainer(model, data = None)
        shap_values_list = tree_explainer.shap_values(data_for_explanation)
        expected_value = tree_explainer.expected_value
        explainer = tree_explainer
    elif method == 'kernel':
        def shap_decorator(model, features):
            def kernel_model(X):
                X = pd.DataFrame(X, columns = features)
                return model.predict(X)
            return kernel_model

        kernel_model = shap_decorator(model, features)
        kernel_explainer = shap.KernelExplainer(kernel_model, data_for_explanation)
        shap_values = kernel_explainer.shap_values(data_for_explanation)
        shap_values_list = [shap_values]
        expected_value = kernel_explainer.expected_value
        explainer = kernel_explainer
    else:
        raise ValueError(f"unknown method '{method}'")

    if objective == 'binary_classification':
        if method == 'tree':
            shap_values = shap_values_list[1]
        else:
            shap_values = shap_values_list[0]
    elif objective == 'regression':
        shap_values = shap_values_list
    else:
        raise ValueError(f"unknown objective '{objective}'")

    abs_importances_df = get_shap_absolute_importances(features, shap_values)
    if objective == 'binary_classification' and method == 'tree':
        ### to match shap absolute plot
        abs_importances_df *= 2

    figure = plot_shap_summary(
        shap_values_list if plot_absolute else shap_values,
        data_for_explanation, features,
        plot_absolute = plot_absolute, plot_max_features = plot_max_features, figsize = figsize,
    )

    return {
        'shap_values': shap_values_list,
        'importances': abs_importances_df,
        'figure': figure,
        'explainer': explainer,
    }


def plot_permutation_importances(*args, **kwargs):
    return plot_importances(*args, **kwargs)


def plot_importances(
    importances: pd.DataFrame,
    n_feat_to_plot: int = 10,
    plot_type: str = 'bar',
    color: str = 'royalblue',
    title: str = '',
    fontsize: int = 15,
    figsize: Tuple[int, int] = (20, 10),
    show: bool = False,
):

    plt.style.use(DEFAULT_PLOT_STYLE)
    n_feat_to_plot = min(importances.shape[0], n_feat_to_plot)
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    if plot_type == 'bar':
        sns.barplot(
            x = 'importance', y = importances.index[:n_feat_to_plot],
            data = importances[:n_feat_to_plot], color=color
        )
        plt.xlabel('')
        plt.yticks(fontsize = fontsize)
        plt.xticks(fontsize = fontsize)
    else:
        plt.plot(list(range(n_feat_to_plot)), importances[:n_feat_to_plot], color=color, marker='o', linewidth=3)
        plt.xticks(list(range(n_feat_to_plot)), importances.index[:n_feat_to_plot])
        plt.grid(axis='y')
        plt.xlim(-0.5, n_feat_to_plot - 0.5)
        plt.xticks(rotation=90, fontsize = fontsize)
        plt.yticks(fontsize = fontsize)

    plt.title(f"{title} importances", fontsize=15)
    if not show:
        plt.close()

    return fig


def get_permutation_importances_widget(permutation_importances_df):
    return get_importances_widget(permutation_importances_df)


def plot_auc_roc(y_true, y_score, ordering = 'desc'):
    if ordering == 'desc':
        fpr_best, tpr_best, _ = roc_curve(y_true, y_score)
    else:
        fpr_best, tpr_best, _ = roc_curve(y_true, (-1)*y_score)
    fig = plt.figure(figsize=(16, 10))

    roc_auc = auc(fpr_best, tpr_best)
    plt.plot(fpr_best, tpr_best, label = 'Fitted Model')
    plt.title("Gini = %0.4f, ROC AUC = %0.4f" %(roc_auc*2 - 1, roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
    plt.xlim([-0.005, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.ioff()
    plt.close()
    return fig


def get_confusion_matrix_advanced(
    y_true, y_pred,
    classes_dict = None,
    figsize = (16,10),
    annot_kws = None, xticklabels = None, yticklabels = None,
):
    if not annot_kws:
        annot_kws = {'fontsize':10}
    if not xticklabels:
        xticklabels = {"rotation": 'vertical', "fontsize": 10}
    if not yticklabels:
        yticklabels = {"rotation": 'horizontal', "fontsize": 10}
    cmap = sns.color_palette(palette='Blues', n_colors = 5)

    conf_matrix = pd.DataFrame(confusion_matrix(y_true = y_true, y_pred = y_pred))
    plt.figure(figsize = (1,1))
    ax_init = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap)
    plt.close()
    quadmesh = ax_init.findobj(QuadMesh)[0]
    init_facecolors = quadmesh.get_facecolors()

    df_matrix = pd.concat([pd.DataFrame(conf_matrix.sum(axis = 0)).transpose(),conf_matrix], axis = 0)
    df_matrix.index = ['Recognised'] + [*df_matrix.index[1:]]
    df_matrix['Support'] = df_matrix.sum(axis = 1)
    #df_matrix.fillna(0, inplace = True)

    fig = plt.figure(figsize = figsize)
    ax = sns.heatmap(df_matrix, annot=True, fmt='d', cmap = cmap, annot_kws = annot_kws)
    ax.set_xlabel('Predicted', fontsize = 15)
    ax.set_ylabel('True', rotation = 'horizontal', fontsize = 15)

    present_classes = sorted(set(y_true).union(y_pred))
    classes_cnt = len(present_classes)
    present_dict = dict(enumerate(present_classes)) ### from 0..N-1 index to label in vectors

    if classes_dict is not None:
        if classes_dict == 'keep':
            ### keeping what comes
            mapping_dict = present_dict
        else:
            ### verify that we get from classes_dict only classes which are in true or predicted
            mapping_dict = {k: classes_dict[present_dict[k]] for k in present_dict}
    else:
        ### creating 0..N-1 names
        mapping_dict = {l1: l1 for l1 in np.arange(classes_cnt)}

    _ = ax.set_yticks(np.arange(classes_cnt + 1) + 0.5)
    _ = ax.set_xticks(np.arange(classes_cnt + 1) + 0.5)
    _ = ax.set_yticklabels(labels = ['Recognised as'] + list(mapping_dict.values()), **yticklabels)
    _ = ax.set_xticklabels(labels = list(mapping_dict.values()) + ['Support'], **xticklabels)

    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()  # cells by rows

    facecolors[:classes_cnt] = np.array([0., 128/255, 135/255, 1])
    facecolors[np.arange(classes_cnt, len(facecolors), classes_cnt+1)] = np.array([0., 200/255, 127/255, 1])
    facecolors[classes_cnt] = np.array([0., 0., 0., 1])

    for i in range(0, classes_cnt+1):
        facecolors[((i+1)*(classes_cnt+1)):(i+2)*(classes_cnt+1)-1] = init_facecolors[i*classes_cnt: (i+1)*classes_cnt]

    quadmesh.set_facecolors(facecolors)
    conf_matrix.index = mapping_dict.values()
    conf_matrix.columns = mapping_dict.values()
    plt.tight_layout()
    plt.ioff()
    plt.close(fig)

    return conf_matrix, fig


def calc_powerstat(y_true, y_score, y_baseline = None, grouping = True, ordering = 'desc', return_curves = False):
    """
    todo: add weights
    """

    check_consistent_length(y_true, y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    N = len(y_true)
    N_target = sum(y_true)

    if y_baseline is None:
        auc_baseline = None
        y_baseline = np.ones(N)
    else:
        auc_baseline = -1
        check_consistent_length(y_true, y_baseline)
        assert_all_finite(y_baseline)

    df = pd.DataFrame({'score': y_score, 'ideal': y_true, 'baseline': y_baseline, 'target': y_true, 'count': np.ones(N)})
    curves = {}
    for key in ['score', 'ideal', 'baseline']:
        if ordering == 'desc':
            ascending = False
        else:
            ascending = True
        if grouping:
            calc_df = df.groupby(key).agg('sum').reset_index()
        else:
            calc_df = df
        calc_df = calc_df.sort_values(by = key, ascending = ascending)

        y_axis = np.r_[0, np.cumsum(calc_df['target']) / N_target]
        x_axis = np.r_[0, np.cumsum(calc_df['count']) / N]

        curves[key] = (x_axis, y_axis)

    auc_model, auc_ideal = auc(*curves['score']), auc(*curves['ideal'])
    auc_random = 0.5

    if auc_baseline is None:
        auc_baseline = auc_random
    else:
        auc_baseline = auc(*curves['baseline'])

    powerstat = (auc_model - auc_baseline) / (auc_ideal - auc_baseline)
    if return_curves:
        return powerstat, {'model': curves['score'], 'ideal': curves['ideal'], 'baseline': curves['baseline']}

    return powerstat


def plot_cap_curve(y_true, y_score, y_baseline = None, grouping = True, ordering = 'desc', figsize = (10, 6)):
    """
    todo: add weights
    """
    powerstat, curves = calc_powerstat(
        y_true, y_score,
        y_baseline = y_baseline,
        grouping = grouping,
        ordering = ordering,
        return_curves = True
    )

    x_model, y_model = curves['model']
    x_ideal, y_ideal = curves['ideal']
    x_baseline, y_baseline = curves['baseline']
    # if y_baseline is not None:
    #     x_baseline, y_baseline = curves['baseline']
    # else:
    #     x_baseline, y_baseline = None, None
    x_axis = x_model
    y_random = x_axis

    #auc_model = sum(np.diff(x_axis)*(y_model[1:]+y_model[:-1])/2)
    # auc_model = auc(x_axis,y_model)
    # auc_ideal = 1 - (N_target/N) / 2
    # x_ideal, y_ideal = np.array([0,N_target/N,1]), np.array([0,1,1])

    #plotting
    plt.style.use(DEFAULT_PLOT_STYLE)
    fig, ax = plt.subplots(1, 1, figsize = figsize)

    ax.set_title(f'Cumulative Accuracy Profile Curve\nGini(Powerstat) = {powerstat:.4f}', fontdict={'fontsize' : 15})
    # ax.xaxis.grid()
    ax.grid(True)

    #model
    est_model = ax.scatter(x = x_axis, y = y_model, s = 0.5, color = 'white', alpha = 0.5)
    filled = ax.fill_between(x_axis, y_model, y_random, alpha = 0.5, color = 'green', label = 'Fitted Model')
    ax.set_xlabel('% of cumulative observations')
    ax.set_ylabel('% of cumulative targets')
    ax.set_xticks(np.arange(0, 1.1, 0.1), [f'i%' for i in np.arange(0, 110, 10)])
    ax.set_yticks(np.arange(0, 1.1, 0.1), [f'i%' for i in np.arange(0, 110, 10)])
    #ideal/random
    ideal = ax.plot(x_ideal, y_ideal, linewidth = 1, color = 'black', linestyle = '--', label = 'Ideal model')
    random = ax.plot(x_axis, y_random, linewidth = 1, color = 'grey', linestyle = '--', label = 'Random model')
    if y_baseline is not None:
        baseline = ax.plot(x_baseline, y_baseline, linewidth = 1, color = 'crimson', linestyle = '--', label = 'Baseline model')
    ax.set_xlim(-0.001, 1.01)
    ax.set_ylim(-0.025, 1.01)
    # ax.legend(loc = 4, ncol = 1, bbox_to_anchor = (0.85, 0.165), prop = {'size' : 15})
    ax.legend(loc = 'best')
    plt.ioff()
    plt.close(fig)

    return fig


def calc_precision_recall_curve_binary(
    y_true: ArrayLike1d,
    y_score: ArrayLike1d,
    sample_weight: ArrayLike1d = None,
    use_grid: bool = False,
    grid_step: float = 0.01,
    thresholds_grid: List[float] = None,
    use_sklearn_bounds: bool = False
):
    """changed sklearn version where it's possible to set your own thresholds - much faster than the dummy one"""
    check_consistent_length(y_true, y_score)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)

    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        assert_all_finite(sample_weight)

    # make y_true a boolean vector
    y_true = (y_true == 1.0)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        check_consistent_length(y_true, sample_weight)
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0

    if use_grid:
        if thresholds_grid is not None:
            thresholds_grid = np.asarray(np.sort(thresholds_grid)[::-1])
        else:
            thresholds_grid = np.arange(0.0, 1.0, grid_step)[::-1] ### making grid with provided step
        threshold_idxs = []
        for t in thresholds_grid:
            mask = y_score >= t
            if all(mask):
                threshold_idxs.append(y_true.size - 1)
            else:
                threshold_idxs.append(np.max([np.argmin(mask) - 1,0]))
        threshold_idxs = np.asarray(threshold_idxs)
    else:
        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]   ### contains ix if s[ix+1] - s[ix] != 0
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]   ### == cumsum on target only in points where it changed
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps ### amount on every step minus tps, as every step is PredPositive corresponding to threshold

    if use_grid:
        thresholds = thresholds_grid
    else:
        thresholds = y_score[threshold_idxs]

    # fps: A count of false positives, at index i being the number of negative samples assigned a score >= thresholds[i].
    # The total number of negative samples is equal to fps[-1] (thus true negatives are given by fps[-1] - fps).

    # tps: An increasing count of true positives, at index i being the number of positive samples assigned a score >= thresholds[i]
    # The total number of positive samples is equal to tps[-1] (thus false negatives are given by tps[-1] - tps).

    # thresholds: Decreasing score values.

    # The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    # true positives and ``fp`` the number of false positives. The precision is
    # intuitively the ability of the classifier not to label as positive a sample
    # that is negative.

    # The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    # true positives and ``fn`` the number of false negatives. The recall is
    # intuitively the ability of the classifier to find all the positive samples.

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0   ### don't really get when that's gonna happen
    recall = tps / tps[-1]

    if use_sklearn_bounds:
        last_ind = tps.searchsorted(tps[-1])  # stop when full recall attained
    else:
        last_ind = len(tps) - 1

    sl = slice(last_ind, None, -1)  # and reverse the outputs so recall is decreasing

    # The last precision and recall values are 1. and 0. respectively and do not
    # have a corresponding threshold.  This ensures that the graph starts on the y axis.
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def plot_precision_recall_curves(
    prc_results_dict,
    beta = 1.0,
    use_single_y_axis = True,
    figsize = (20, 20),
    linewidth = 2,
):
    """each value in dict entrance should contain tuple of (prs, recs, thrs)"""

    fig = plt.figure(figsize = figsize)

    if use_single_y_axis:
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
        ax_pr = fig.add_subplot(gs[0])
        ax_p = fig.add_subplot(gs[1])
        ax_r = ax_p
        ax_d = fig.add_subplot(gs[2], sharex=ax_p)
    else:
        gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
        ax_pr = fig.add_subplot(gs[0])
        ax_p = fig.add_subplot(gs[1])
        ax_r = fig.add_subplot(gs[2], sharex=ax_p)
        ax_d = fig.add_subplot(gs[3], sharex=ax_p)
    axes = [ax_pr, ax_p, ax_r, ax_d]

    colors = SELECTED_COLORS
    colors = [mcolors.CSS4_COLORS.get(col) for col in colors]
    N_col = len(colors)

    for i, key in enumerate(prc_results_dict):
        label = key
        prs, recs, thrs = prc_results_dict[key]
        color = colors[i % N_col]

        thresholds_distribution = fit_distribution(thrs, empirical = True)['distribution']
        thresholds_distribution.plot_cdf(ax = ax_d, label = label, color = color, linewidth = linewidth)
        ax_d.set_ylabel("CDF")
        #### trapezoid area calculation
        trapezoid_auc = np.sum((recs[::-1][1:] - recs[::-1][:-1]) * (prs[::-1][1:] + prs[::-1][:-1]) / 2)

        ax_pr.plot(recs, prs, label = f'{label} pr-auc: {trapezoid_auc:.4f}', color = color, linewidth = linewidth)
        ax_pr.set(ylabel='Precision', xlabel='Recall')

        fbeta_levels = np.arange(0.1, 0.91, 0.1)
        recalls = np.arange(10e-5, 1.0, 1e-5)
        for fbeta in fbeta_levels:
            denom = ((1 + beta**2) * recalls - fbeta * (beta**2))
            precisions = fbeta * recalls[denom > 0] / denom[denom >0]
            recalls = recalls[denom > 0]
            precisions, recalls = precisions[precisions <= 1], recalls[precisions <= 1]
            ax_pr.plot(recalls, precisions, color = 'black', linestyle = 'dashed', linewidth = 0.5)
            ax_pr.annotate(
                f"f{beta:.2f} = {fbeta:.2f}", (recalls[-1], precisions[-1]),
                textcoords='offset points', xytext=(-30, -10), ha='center', size = 8
            )

        if use_single_y_axis:
            fss = []
            for pr, rc in zip(prs[:-1], recs[:-1]):
                fss.append((1 + beta**2) * pr * rc / ((beta**2) * pr + rc))

            ax_p.plot(
                thrs, fss, label = label+' F-{} score'.format(beta),
                linestyle = '-', linewidth = 0.5, color = color
            )
            ax_p.plot(
                thrs, prs[:-1], label = label+' Precision',
                linestyle = ':', color = color, linewidth = linewidth
            )
            ax_r.plot(
                thrs, recs[:-1], label = label+' Recall',
                linestyle = '--', color = color, linewidth = linewidth
            )
            ax_p.set(ylabel='Metric', xlabel='Threshold')

        else:
            ax_p.plot(thrs, prs[:-1], label = label, color = color, linewidth = linewidth)
            ax_p.set(ylabel='Precision', xlabel='Threshold')

            ax_r.plot(thrs, recs[:-1], label = label, color = color, linewidth = linewidth)
            ax_r.set(ylabel='Recall', xlabel='Threshold')

    for a in axes[:-1]:
        a.grid(True)
        a.legend(fontsize = 10)
        a.set_ylim(-0.05, 1.05)

    plt.close()
    plt.ioff()

    return fig


def calc_cutoff_stats_dynamic(
    data: pd.DataFrame,
    target_name: str,
    score_name: str,
    ascending: bool = False,
    beta: float = 1
):

    gb = data.groupby(score_name).agg({target_name: ['count', 'sum']}).sort_index(ascending = ascending)
    L = len(data)
    P = data[target_name].sum()
    N = L - P
    gb['PP'] = (gb[(target_name, 'count')].cumsum()).astype(int)
    gb['TP'] = (gb[(target_name, 'sum')].cumsum()).astype(int)
    gb['FP'] = (gb['PP'] - gb['TP']).astype(int)
    gb['FN'] = (P - gb['TP']).astype(int)
    gb['TN'] = (L - gb['TP'] - gb['FP'] - gb['FN']).astype(int)
    gb['PN'] = (L - gb['PP']).astype(int)
    gb['quantile'] = (1 - gb['PP'] / L)
    gb['precision'] = gb['TP'] / gb['PP']
    gb['recall'] = gb['TP'] / P
    gb[f'f-{beta}'] = (1 + beta**2) * gb['precision'] * gb['recall'] / ((beta**2) * gb['precision'] + gb['recall'])
    return gb.iloc[:, 2:]


def calc_optimal_cutoff(prs, recs, thrs, beta=1, min_recall_level=None, second_preference='precision'):
    """
    takes results of `calc_precision_recall_curve_binary`
    second_preference is either 'precision' or 'recall'
    `beta` in arguments is provided not squared and is being squared in calculations
    """
    fss = []

    for pr, rc in zip(prs[:-1], recs[:-1]):
        fss.append((1 + beta ** 2) * pr * rc / ((beta ** 2) * pr + rc))

    res = pd.DataFrame({'threshold': thrs, 'precision': prs[:-1], 'recall': recs[:-1], f'f-{beta}': fss})

    if min_recall_level is not None:
        warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)
        res_ok = res[res['recall'] >= min_recall_level]
        res_ok.sort_values(by=[f'f-{beta}', second_preference, 'threshold'], ascending=[False, False, True], inplace=True)

        res_not_ok = res[res['recall'] < min_recall_level]
        res_not_ok[f'f-{beta}'] = -1
        res_not_ok.sort_values(by=[second_preference, 'threshold'], ascending=[False, True], inplace=True)

        res = pd.concat([res_ok, res_not_ok], ignore_index=True)
        warnings.filterwarnings("default", category=pd.core.common.SettingWithCopyWarning)
    else:
        res.sort_values(by=[f'f-{beta}', second_preference, 'threshold'], ascending=[False, False, True], inplace=True)
        res.reset_index(drop=True, inplace=True)

    return res.loc[0, 'threshold'], res


def calc_spearman(
    data: pd.DataFrame,
    target_name: str,
    score_name: str,
    title: str = '',
    figsize: Tuple[int, int] = (10, 8),
    point_size: int = 10,
    marker: str = '+',
    color: str = 'black',
    plot_margin: float = 0.05,
    xlim: Tuple[float, float] = None,
    plot_ideal: Union[str, bool] = 'identity',
) -> Dict[str, Union[float, plt.Figure]]:

    rho = spearmanr(data[target_name], data[score_name])[0]

    fig, ax = plt.subplots(figsize = figsize)
    ax.set_title(f"{title} target vs score correlation, Spearman rho = {rho:.3f}")
    ax.scatter(data[target_name], data[score_name], s = point_size, marker = marker, color = color)

    xmin, xmax = min(data[target_name]), max(data[target_name])
    xmin = min((1 + plot_margin) * xmin, (1-plot_margin) * xmin)
    xmax = max((1 + plot_margin) * xmax, (1-plot_margin) * xmax)

    if np.isclose(xmin, 0, atol = 1e-4):
        xmin = xmin - min(plot_margin, 0.5 * np.std(data[target_name]))
    elif np.isclose(xmax, 0, atol = 1e-4) == 0:
        xmax = xmax + min(plot_margin, 0.5 * np.std(data[target_name]))

    x = np.linspace(xmin, xmax, 100)
    if isinstance(plot_ideal, bool) and plot_ideal:
        plot_ideal = 'identity'
    elif not isinstance(plot_ideal, str):
        plot_ideal = False

    if isinstance(plot_ideal, str):
        if plot_ideal == 'identity':
            y = x
        elif plot_ideal == 'minmax':
            ymin, ymax = min(data[score_name]), max(data[score_name])
            m = (ymax - ymin) / (xmax - xmin)
            b = ymin - m * xmin
            y = m * x + b
        elif plot_ideal == 'linear':
            from sklearn.linear_model import LinearRegression
            ### here actually predicting score by target, cause it's on X axis
            model = LinearRegression().fit(data[[target_name]], data[score_name])
            y = model.predict(pd.DataFrame(x, columns = [target_name]))
        else:
            raise ValueError(f"unknown 'plot_ideal' value: '{plot_ideal}'")

    if plot_ideal:
        ax.plot(x, y, linewidth = 2, color = "crimson", label = plot_ideal)
        ax.legend(loc = 'best')

    xlim = xlim or (None, None)
    xmin = xlim[0] or xmin
    xmax = xlim[1] or xmax
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(f"Target: '{target_name}'")
    ax.set_ylabel(f"Score: '{score_name}'")
    ax.grid(True)
    plt.close(fig)

    return {
        'rho': rho,
        'figure': fig,
    }


def calc_vif(
    data: pd.DataFrame,
    features: List[str],
    sort: bool = False,
) -> pd.DataFrame:

    # adding constant as first column to fit linreg with intercept
    X = add_constant(data[features])

    ### dropping constant
    vif = [variance_inflation_factor(exog = X.values, exog_idx = i) for i in range(X.shape[1])][1:]
    vif = pd.DataFrame({'feature': X.columns[1:], 'vif': vif, })
    if sort:
        vif = vif.sort_values(by = 'vif', ascending = False).reset_index(drop = True)

    return vif


def calc_rec_curve(
    data: pd.DataFrame,
    target_name: str,
    score_name: str,
    loss_function: Callable,
    loss_name: str = '',
    clip_max_loss_over_baseline: bool = True,
    baseline_forecast: Union[Callable, str] = 'mean',
    stride: int = 100,
    x_clip: float = None,
    linewidth: int = 2,
    figsize: Tuple[int, int] = (20, 8),
    return_curves = False,
) -> Dict[str, Union[pd.DataFrame, plt.Figure, Dict]]:

    if baseline_forecast == 'mean':
        baseline_forecast_value = data[target_name].mean()
    elif baseline_forecast == 'median':
        baseline_forecast_value = data[target_name].median()
    else:
        if callable(baseline_forecast):
            baseline_forecast_value = baseline_forecast(data[target_name])
        elif np.issubdtype(type(baseline_forecast), np.number) or isinstance(baseline_forecast, (int, float)):
            baseline_forecast_value = baseline_forecast

    # print(f"setting {baseline_forecast} as baseline value")

    rec_results = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    baseline_losses = loss_function(data[target_name], baseline_forecast_value)
    baseline_mean_loss = np.mean(baseline_losses)
    epsilon_hat = np.max(baseline_losses)

    baseline_df = pd.DataFrame({'loss': baseline_losses, 'count': 1})
    baseline_losses_ordering = np.argsort(baseline_losses)

    baseline_df = baseline_df.groupby('loss').agg('sum').reset_index()
    baseline_df['share'] = stable_cumsum(baseline_df['count'].values) / baseline_df['count'].sum()
    #np.cumsum(baseline_df['count']) / baseline_df['count'].sum()

    x_baseline_losses = np.hstack([
        [0],
        #np.unique(np.array(baseline_losses)[baseline_losses_ordering]),#[::stride],
        np.array(baseline_df['loss'])[::stride],
        [epsilon_hat]
    ])

    baseline_acc = np.hstack([[(baseline_losses <= 0).sum()], np.array(baseline_df['share'])[::stride], [1]])
    # baseline_acc = np.array([(baseline_losses <= loss_value).sum() for loss_value in tqdm(x_baseline_losses)]) / len(baseline_losses)
    baseline_aoc = 1 * epsilon_hat - auc(x_baseline_losses, baseline_acc)

    rec_results['baseline']['losses'] = baseline_losses
    rec_results['baseline']['mean_loss'] = baseline_mean_loss
    rec_results['baseline']['AOC'] = baseline_aoc
    if return_curves:
        rec_results['baseline']['REC_curve'] = (x_baseline_losses, baseline_acc)

    fig, ax = plt.subplots(figsize=(figsize))
    plt.plot(x_baseline_losses, baseline_acc, color = 'black', linestyle='--', linewidth = linewidth,
             label = f'REC-curve baseline, AOC = {baseline_aoc:.5f}, Mean Loss = {baseline_mean_loss:.5f}')
    plt.fill_between(x_baseline_losses, baseline_acc, 1.0, alpha=0.2, color='black', hatch = '/')
    plt.axvline(x=epsilon_hat, linestyle='--', linewidth = 0.5 * linewidth, color='black')

    if isinstance(score_name, str):
        score_names = [score_name]
    elif isinstance(score_name, Iterable):
        score_names = score_name

    colors = ['crimson', 'orchid', 'lightskyblue', 'lightcoral', 'dodgerblue', 'darkmagenta', 'cyan']

    aoc_results = []
    mean_loss_results = []
    max_loss_results = []

    for score_name, color in zip(score_names, itertools.cycle(colors)):

        model_losses = loss_function(data[target_name], data[score_name])
        model_losses_ordering = np.argsort(model_losses)

        model_mean_loss = np.mean(model_losses)
        epsilon_m = np.max(model_losses)

        mean_loss_results.append(model_mean_loss)
        max_loss_results.append(epsilon_m)

        if clip_max_loss_over_baseline:
            epsilon_m = min(epsilon_hat, epsilon_m)

        model_df = pd.DataFrame({'loss':model_losses,'count':1})
        model_df = model_df.groupby('loss').agg('sum').reset_index()
        model_df['share'] = stable_cumsum(model_df['count'].values) / model_df['count'].sum()
        #np.cumsum(model_df['count']) / model_df['count'].sum()

        x_model_losses = np.hstack([
            [0],
            #np.unique(np.array(model_losses)[model_losses_ordering])[::stride],
            np.array(model_df['loss'])[::stride],
            [epsilon_m]
        ])
        model_acc = np.hstack([
            [(model_losses <= 0).sum()],
            np.array(model_df['share'])[::stride],
            [1]
        ])
        # model_acc = np.array([(model_losses <= loss_value).sum() for loss_value in tqdm(x_model_losses)]) / len(model_losses)

        model_aoc = 1 * epsilon_m - auc(x_model_losses, model_acc)
        aoc_results.append(model_aoc)

        plt.plot(
            x_model_losses, model_acc,
            color = color, linewidth = linewidth,
            label = f'REC-curve {score_name}, AOC = {model_aoc:.5f}, Mean Loss = {model_mean_loss:.5f}'
        )
        plt.fill_between(x_model_losses, model_acc, 1.0, alpha=0.2, color=color, hatch = '|')
        plt.axvline(x=epsilon_m, linestyle='--', linewidth = 0.5 * linewidth, color=color)

        rec_results[score_name]['losses'] = model_losses
        rec_results[score_name]['mean_loss'] = model_mean_loss
        rec_results[score_name]['AOC'] = model_aoc
        if return_curves:
            rec_results[score_name]['REC_curve'] = (x_model_losses, model_acc)

    rec_results = {k:dict(v) for k, v in rec_results.items()}

    plt.axhline(y = 1, linestyle = '--', linewidth = 0.5, color = 'black')
    plt.axvline(x = epsilon_m, linestyle = '--', linewidth = 0.5 * linewidth, color = color)
    ax.set_xlabel(f'{loss_name} loss value')
    ax.set_ylabel('Accuracy')
    if x_clip:
        ax.set_xlim(-1e-8, x_clip)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc = 'lower right')
    ax.grid(True)
    plt.close(fig)

    summary_df = pd.DataFrame({
        'model': score_names+['baseline'],
        'aoc': aoc_results+[baseline_aoc],
        'mean_loss': mean_loss_results+[baseline_mean_loss],
        'max_loss': max_loss_results+[epsilon_hat],
    })

    return {
        'summary': summary_df.sort_values(by = 'aoc').reset_index(drop = True),
        'results': rec_results,
        'figure': fig
    }
