from typing import Any, Callable, Dict, List, Tuple, Union
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functools import partial

from sklearn.metrics import auc
from sklearn.isotonic import IsotonicRegression

from pulsar.discovery.distributions import bucketize
from pulsar.visualization.constants import DEFAULT_PLOT_STYLE

from atlas.uplift.learners import ArrayLike1d


### EVERYWHERE should replace pd.qcut for handcrafted with consecutive bins numbering
base_bucketizer = partial(bucketize, force_even_buckets = False, merge_to_right_side = True, ecdf = None, force_consecutive_markers = True, start = 1)


def _preprocess_xy_uplift_curve(x, y) -> Tuple[np.ndarray, np.ndarray]:
    x, y = np.array(x), np.array(y)
    nan_mask = np.isnan(y)
    x = x[~nan_mask]
    y = y[~nan_mask]
    if x[0] != 0 or y[0] != 0:
        x = np.r_[0, x]
        y = np.r_[0, y]
    return x, y


def _get_random_uplift_curve(x, y) -> Tuple[np.ndarray, np.ndarray]:
    return np.array([0, x[-1]]), np.array([0, y[-1]])


def _assert_y_normalize_version(y_normalize_version: str):
    if y_normalize_version not in ('total', 'treatment', 'last', 'abs_last'):
        raise ValueError(f"unknown 'y_normalize_version' value: '{y_normalize_version}'")


def _get_ideal_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    x_normalize: bool = True,
    y_normalize: bool = True,
    y_normalize_version: str = 'last',
    group_by_forecast: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """ adapted from scikit-uplift """

    y_true, treatment = np.array(y_true), np.array(treatment)

    n_control_responses = np.sum((y_true == 1) & (treatment == 0))
    n_treatment_nonresponses = np.sum((y_true == 0) & (treatment == 1))
    ### ideal curve through y_true and treatment - !todo: explanation!
    ideal_uplift_score = 2 * (y_true == treatment) + (y_true if n_control_responses > n_treatment_nonresponses else treatment)
    ideal_uplift_chart = get_cumulative_uplift_chart(y_true, ideal_uplift_score, treatment, group_by_forecast = group_by_forecast)['chart']
    if x_normalize:
        x_ideal = ideal_uplift_chart['percentage_cumulative']
    else:
        x_ideal = ideal_uplift_chart['n_all_cumulative']

    y_ideal = (
        ideal_uplift_chart['uplift_cumulative']
        *
        (ideal_uplift_chart['treatment_observations_cumulative'] + ideal_uplift_chart['control_observations_cumulative'])
    )
    if y_normalize:
        _assert_y_normalize_version(y_normalize_version)
        if y_normalize_version == 'total':
            y_ideal = y_ideal / ideal_uplift_chart['n_all_cumulative'].tail(1).values
        elif y_normalize_version == 'treatment':
            y_ideal = y_ideal / ideal_uplift_chart['treatment_observations_cumulative'].tail(1).values
        elif y_normalize_version == 'last':
            y_ideal = y_ideal / y_ideal.tail(1).values
        elif y_normalize_version == 'abs_last':
            y_ideal = y_ideal / abs(y_ideal.tail(1).values)

    x_ideal, y_ideal = _preprocess_xy_uplift_curve(x_ideal, y_ideal)

    return x_ideal, y_ideal


def _get_ideal_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    x_normalize: bool = True,
    y_normalize: bool = True,
    y_normalize_version: str = 'last',
    group_by_forecast: bool = True,
    dnd_effect: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true, treatment = np.array(y_true), np.array(treatment)

    ### ideal curve through y_true and treatment
    if dnd_effect:
        ideal_uplift_score = y_true * treatment - y_true * (1 - treatment)
        ideal_uplift_chart = get_cumulative_uplift_chart(y_true, ideal_uplift_score, treatment, group_by_forecast = group_by_forecast)['chart']

        ### !?shouldn't we take here only treatment cumulative volume actually..!?
        ### in scikit-lift or gutierrez, gerardy they use total volume for x, but original paper (Radcliffe, 2007) implies we should use treatment volume ..!?
        if x_normalize:
            x_ideal = ideal_uplift_chart['percentage_cumulative']
        else:
            x_ideal = ideal_uplift_chart['n_all_cumulative']

        y_ideal = (
            ideal_uplift_chart['uplift_cumulative']
            *
            ideal_uplift_chart['treatment_observations_cumulative']
        )
        y_ideal[0] = y_ideal[1]  # this is needed to make a trapezoid and not triangle

        if y_normalize:
            _assert_y_normalize_version(y_normalize_version)
            if y_normalize_version == 'total':
                y_ideal = y_ideal / ideal_uplift_chart['n_all_cumulative'].tail(1).values
            elif y_normalize_version == 'treatment':
                y_ideal = y_ideal / ideal_uplift_chart['treatment_observations_cumulative'].tail(1).values
            elif y_normalize_version == 'last':
                y_ideal = y_ideal / y_ideal.tail(1).values
            elif y_normalize_version == 'abs_last':
                y_ideal = y_ideal / abs(y_ideal.tail(1).values)
    else:
        Y_T, Y_C = y_true[treatment == 1].sum(), y_true[treatment == 0].sum()
        N_T, N_C = sum(treatment == 1), sum(treatment == 0)
        y_final = Y_T - Y_C * N_T / N_C

        # x_ideal, y_ideal = np.array([0, Y_T - Y_C, N_T + N_C]), np.array([0, y_final, y_final])
        x_ideal, y_ideal = np.array([0, y_final, N_T + N_C]), np.array([0, y_final, y_final])

        if x_normalize:
            x_ideal = x_ideal / x_ideal[-1]

        if y_normalize:
            _assert_y_normalize_version(y_normalize_version)
            if y_normalize_version == 'total':
                y_ideal = y_ideal / (N_T + N_C)
            elif y_normalize_version == 'treatment':
                y_ideal = y_ideal / N_T
            elif y_normalize_version == 'last':
                y_ideal = y_ideal / y_ideal[-1]
            elif y_normalize_version == 'abs_last':
                y_ideal = y_ideal / abs(y_ideal[-1])

    x_ideal, y_ideal = _preprocess_xy_uplift_curve(x_ideal, y_ideal)

    return x_ideal, y_ideal


def get_percentile_uplift_chart(
    y_true: np.ndarray,
    y_score: np.ndarray,
    treatment: np.ndarray,
    n_buckets: int = 10,
    bucketizer: Callable = base_bucketizer,
    start_bucket_marker: int = 1,
    figsize: Tuple[int, int] = (12, 10),
) -> Dict[str, Union[plt.Figure, pd.DataFrame]]:

    score_bucket_markers = bucketizer(y_score, n_buckets)
    unique_markers = np.unique(score_bucket_markers)
    n_actual_buckets = len(unique_markers)
    score_bucket_markers = n_actual_buckets - score_bucket_markers + start_bucket_marker  # now inverted and later ascending
    unique_markers = n_actual_buckets - unique_markers + start_bucket_marker
    df = pd.DataFrame({'bucket': score_bucket_markers, 'uplift_forecast': y_score, 'treatment': treatment, 'target': y_true})

    agg_f = df.groupby('bucket').agg({'uplift_forecast': 'mean'})
    agg_mf = df.groupby('bucket')[['uplift_forecast']].agg(['max', 'min'])
    agg_mf.columns = ['uplift_forecast_max', 'uplift_forecast_min']

    agg_a = df.groupby(['bucket', 'treatment']).agg({'target': ['count', 'sum']})
    agg_a = agg_a.unstack(-1)

    _replace_names = {'target_count': 'observations', 'target_sum': 'responses'}
    _replace_treatment = {0: 'control', 1: 'treatment'}
    agg_a.columns = [
        _replace_treatment[c[2]] + '_' + _replace_names[f'{c[0]}_{c[1]}']
        for c in agg_a.columns.to_flat_index()
    ]

    # agg_df = pd.merge(agg_f, agg_a, left_index = True, right_index = True).fillna(0).sort_index(ascending = True)#.sort_index(ascending = False)
    agg_df = pd.merge(
        pd.merge(agg_f, agg_mf, left_index = True, right_index = True),
        agg_a,
        left_index = True, right_index = True,
    ).fillna(0).sort_index(ascending = True)

    agg_df['control_target_rate'] = agg_df['control_responses'] / agg_df['control_observations']
    agg_df['treatment_target_rate'] = agg_df['treatment_responses'] / agg_df['treatment_observations']
    agg_df['uplift_actual'] = agg_df['treatment_target_rate'] - agg_df['control_target_rate']
    agg_df['all_observations'] = agg_df['treatment_observations'] + agg_df['control_observations']
    agg_df['all_observations_share_cumulative'] = agg_df['all_observations'].cumsum() / agg_df['all_observations'].sum()
    agg_df['all_responses'] = agg_df['treatment_responses'] + agg_df['control_responses']
    agg_df = agg_df[sorted(agg_df.columns)]
    agg_df.reset_index(inplace = True)
    # agg_df['bucket'] =  n_actual_buckets - agg_df['bucket']
    agg_df['control_observations'] = agg_df['control_observations'].astype(int)
    agg_df['control_responses'] = agg_df['control_responses'].astype(int)
    agg_df['treatment_observations'] = agg_df['treatment_observations'].astype(int)
    agg_df['treatment_responses'] = agg_df['treatment_responses'].astype(int)
    agg_df['all_observations'] = agg_df['all_observations'].astype(int)
    agg_df['all_responses'] = agg_df['all_responses'].astype(int)

    ticklabels = agg_df['bucket']
    locs = agg_df.index

    bar_shift_delta = 0.45
    x_ticks_rotation = 'horizontal'
    xlabelfontsize = 15

    plt.style.use(DEFAULT_PLOT_STYLE)
    # fig, ax = plt.subplots(2, 1, figsize = figsize)
    fig = plt.figure(figsize = figsize) #, constrained_layout=True)
    gs = gridspec.GridSpec(ncols = 2, nrows = 4, figure = fig)
    ax_upper = fig.add_subplot(gs[:2, :])
    ax_middle = fig.add_subplot(gs[2:3, :], sharex = ax_upper)
    ax_lower = fig.add_subplot(gs[3:, :], sharex = ax_upper)

    ax = [ax_upper, ax_middle, ax_lower]

    ax[0].bar(x = locs - bar_shift_delta / 2, height = agg_df['uplift_actual'], width = bar_shift_delta, color = 'crimson', label = 'actual')
    ax[0].bar(x = locs + bar_shift_delta / 2, height = agg_df['uplift_forecast'], width = bar_shift_delta, color = 'darkorchid', label = 'forecast')
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_xlabel('Forecast Bucket', fontsize = xlabelfontsize)
    ax[0].set_ylabel('Uplift Value')
    ax[0].set_title('Uplift Chart')
    ax[0].set_xticks(locs)
    ax[0].set_xticklabels(ticklabels, rotation = x_ticks_rotation)

    # assert distr_plot_type in ('bar', 'line')
    # if distr_plot_type == 'bar':
    ax[1].bar(x = locs - bar_shift_delta / 2, height = agg_df['treatment_observations'], width = bar_shift_delta, color = 'palegreen', label = 'treatment')
    ax[1].bar(x = locs + bar_shift_delta / 2, height = agg_df['control_observations'], width = bar_shift_delta, color = 'lightblue', label = 'control')
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_xlabel('Forecast Bucket', fontsize = xlabelfontsize)
    ax[1].set_ylabel('# Observations')
    ax[1].set_title('Treatment & Control Distributions')
    ax[1].set_xticks(locs)
    ax[1].set_xticklabels(ticklabels, rotation = x_ticks_rotation)
    # else:
    ax[2].plot(locs, agg_df['treatment_observations'] / (agg_df['control_observations'] + agg_df['treatment_observations']), linewidth = 5, color = 'teal')
    ax[2].grid(True)
    ax[2].set_xlabel('Forecast Bucket', fontsize = xlabelfontsize)
    ax[2].set_ylabel('Ratio')
    ax[2].set_title('Treatment / (Treatment + Control)')
    ax[2].set_xticks(locs)
    ax[2].set_xticklabels(ticklabels, rotation = x_ticks_rotation)
    leg = ax[2].legend((), (), loc="lower center")
    leg.set_title(f'Total volume: {len(y_true)}', prop = {"size": 15})

    ax[0].set_xlim(-1 + bar_shift_delta, n_actual_buckets - bar_shift_delta)
    ax[1].set_xlim(-1 + bar_shift_delta, n_actual_buckets - bar_shift_delta)
    ax[2].set_xlim(-1 + bar_shift_delta, n_actual_buckets - bar_shift_delta)

    plt.tight_layout()
    plt.ioff()
    plt.close(fig)

    return {'chart': agg_df, 'figure': fig}


def get_uplift_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    treatment: np.ndarray,
    n_buckets: int = 10,
    bucketizer: Callable = base_bucketizer,
    start_bucket_marker: int = 1,
    alpha: float = 0.01,
    n_iterations: int = 1000,
    seed: int = 8,
    t2tc_max_allowed: float = 0.95,
    figsize: Tuple[int, int] = (12, 10),
):

    n_bounds_on_legend_max_allowed = 15
    jitter = 0.1
    x_ticks_rotation = 'horizontal'
    xlabelfontsize = 15
    # interpolate = True

    def _make_bootstrap_uplift(data: pd.DataFrame, n_samples: int, n_iterations: int, seed: int = None):

        distribution = []
        if seed is not None:
            np.random.seed(seed)

        for _ in range(n_iterations):
            indices = np.random.choice(data.index, size = n_samples, replace = True)
            subsample = data.loc[indices, :]
            # assert len(subsample) == n_samples
            gb = subsample.groupby('treatment').agg({'target': 'mean'}).reset_index()
            if len(gb.loc[gb['treatment'] == 1, 'target']) and len(gb.loc[gb['treatment'] == 0, 'target']):
                uplift_actual = gb.loc[gb['treatment'] == 1, 'target'].values[0] - gb.loc[gb['treatment'] == 0, 'target'].values[0]
            else:
                uplift_actual = np.nan
            uplift_forecast = subsample['uplift_forecast'].mean()
            uplift_delta = uplift_actual - uplift_forecast
            distribution.append(uplift_delta)

        return np.array(distribution)

    score_bucket_markers = bucketizer(y_score, n_buckets)
    unique_markers = np.unique(score_bucket_markers)
    n_actual_buckets = len(unique_markers)
    score_bucket_markers = n_actual_buckets - score_bucket_markers + start_bucket_marker  # now inverted and later ascending
    unique_markers = n_actual_buckets - unique_markers + start_bucket_marker

    df = pd.DataFrame({'bucket': score_bucket_markers, 'uplift_forecast': y_score, 'treatment': treatment, 'target': y_true})

    percentile_chart_results = get_percentile_uplift_chart(
        y_true = y_true, y_score = y_score, treatment = treatment,
        n_buckets = n_buckets, bucketizer = bucketizer, start_bucket_marker = start_bucket_marker,
        figsize = figsize
    )
    percentile_chart = percentile_chart_results['chart']
    t2tc_bucket_ratios = percentile_chart['treatment_observations'] / (percentile_chart['treatment_observations'] + percentile_chart['control_observations'])

    lq, uq = (alpha / 2, 1 - alpha / 2)
    bootstrap_iterations_by_bucket = pd.DataFrame({'bucket': list(unique_markers) * n_iterations, 'delta': np.zeros(n_actual_buckets * n_iterations)})
    min_bucket, max_bucket = min(unique_markers), max(unique_markers)

    bounds = {}
    buckets_calibrated = {}
    interpolated = {}

    if seed is not None:
        np.random.seed(seed)

    for bucket, nc, nt, t2tc in zip(
        percentile_chart['bucket'],
        percentile_chart['control_observations'],
        percentile_chart['treatment_observations'],
        t2tc_bucket_ratios
    ):
        bucket_subsample = df.loc[df['bucket'] == bucket]
        n_samples = nc + nt
        ### branches with interpolation are not really tested
        if nc == 0:
            if bucket == min_bucket or bucket == max_bucket:
                warnings.warn(
                    f"No observations from control group found in edge bucket #{bucket} - actual uplift value is not defined, "
                    f"skipping it"
                )
                interpolated[bucket] = -1
            else:
                warnings.warn(
                    f"No observations from control group found in bucket #{bucket} - actual uplift value is not defined, "
                    f"will try to interpolate from neighbor buckets: {bucket - 1} and {bucket + 1}"
                )
                try:
                    left = percentile_chart.loc[percentile_chart['bucket'] == bucket - 1]
                    right = percentile_chart.loc[percentile_chart['bucket'] == bucket + 1]
                    interpolated_uplift_actual = (left['uplift_actual'].values[0] + right['uplift_actual'].values[0]) / 2
                    delta_uplift_distribution = np.array([
                        interpolated_uplift_actual - np.random.choice(bucket_subsample['uplift_forecast'], n_samples, replace = True).mean()
                        for _ in range(n_iterations)
                    ])
                    interpolated[bucket] = 1
                except Exception as _:
                    interpolated[bucket] = -1
        elif nt == 0:
            if bucket == min_bucket or bucket == max_bucket:
                warnings.warn(
                    f"No observations from control group found in edge bucket #{bucket} - actual uplift value is not defined, "
                    f"skipping it"
                )
                interpolated[bucket] = -1
            else:
                warnings.warn(
                    f"No observations from treatment group found in bucket #{bucket} - actual uplift value is not defined, "
                    f"will try to interpolate from neighbor buckets: {bucket - 1} and {bucket + 1}"
                )
                try:
                    left = percentile_chart.loc[percentile_chart['bucket'] == bucket - 1]
                    right = percentile_chart.loc[percentile_chart['bucket'] == bucket + 1]
                    interpolated_uplift_actual = (left['uplift_actual'].values[0] + right['uplift_actual'].values[0]) / 2
                    delta_uplift_distribution = np.array([interpolated_uplift_actual - np.random.choice(bucket_subsample['uplift_forecast'], n_samples, replace = True).mean() for _ in range(n_iterations)])
                    interpolated[bucket] = 1
                except Exception as _:
                    interpolated[bucket] = -1
        elif t2tc > t2tc_max_allowed or (1 - t2tc) > t2tc_max_allowed:
            warnings.warn(
                f"Treatment / (Treatment + Control) ratio for bucket #{bucket} shows it is too imbalanced: {t2tc:.3f}, "
                f"bootstrapping only forecast values"
            )
            interpolated_uplift_actual = percentile_chart.loc[percentile_chart['bucket'] == bucket, 'uplift_actual'].values[0]
            delta_uplift_distribution = np.array([
                interpolated_uplift_actual - np.random.choice(bucket_subsample['uplift_forecast'], n_samples, replace = True).mean()
                for _ in range(n_iterations)
            ])
            interpolated[bucket] = 1
        else:
            delta_uplift_distribution = _make_bootstrap_uplift(data = bucket_subsample, n_samples = n_samples, n_iterations = n_iterations, seed = seed)
            interpolated[bucket] = 0

        if interpolated[bucket] != -1:
            bootstrap_iterations_by_bucket.loc[bootstrap_iterations_by_bucket['bucket'] == bucket, 'delta'] = delta_uplift_distribution
            lb, ub = np.nanquantile(delta_uplift_distribution, q = (lq, uq))
            bounds[bucket] = lb, ub
            buckets_calibrated[bucket] = 1 if 0 >= lb and 0 <= ub else 0
        else:
            bootstrap_iterations_by_bucket.loc[bootstrap_iterations_by_bucket['bucket'] == bucket, 'delta'] = np.zeros(n_iterations)
            lb, ub = 0, 0
            bounds[bucket] = lb, ub
            buckets_calibrated[bucket] = 0

    plt.style.use(DEFAULT_PLOT_STYLE)
    # fig, ax = plt.subplots(1, 1, figsize = figsize)
    fig = plt.figure(figsize = figsize)#, constrained_layout=True)
    gs = gridspec.GridSpec(ncols = 2, nrows = 3, figure = fig)
    ax_upper = fig.add_subplot(gs[:2, :])
    ax_lower = fig.add_subplot(gs[2:, :], sharex = ax_upper)

    ax = ax_upper
    failed_interpolated = [k for k, v in interpolated.items() if v == -1]
    sns.stripplot(
        x = "bucket", y = "delta",
        data = bootstrap_iterations_by_bucket.loc[~bootstrap_iterations_by_bucket['bucket'].isin(failed_interpolated)],
        jitter = jitter, color = 'darkorchid', alpha = 0.5 if n_buckets > 1 else 1,
        ax = ax
    )

    for bucket, (lb, ub) in bounds.items():
        bucket_color = 'forestgreen' if buckets_calibrated[bucket] == 1 else 'crimson' #0 >= lb and 0 <= ub else 'crimson'
        ax.hlines(
            lb, bucket - 1 - 3 * jitter, bucket - 1 + 3 * jitter, linestyle = '--', linewidth = 1, color = bucket_color,
            label = f'bucket {bucket} CI{(1 - alpha):.0%}: ({lb:.4f}, {ub:.4f})'
        )
        ax.hlines(ub, bucket - 1 - 3 * jitter, bucket - 1 + 3 * jitter, linestyle = '--', linewidth = 1, color = bucket_color)

    ax.axhline(y = 0, linestyle = '-.', linewidth = 1, color = 'k', label = f'Ideal calibration')
    ax.grid(True)
    ax.set_xlabel('Forecast Bucket')
    ax.set_ylabel('(Actual - Forecast) Uplift Delta')
    ax.set_title('Bootstrapped calibration analysis')
    if len(bounds) <= n_bounds_on_legend_max_allowed:
        ax.legend(ncol = 2)

    ax = ax_lower

    ticklabels = percentile_chart['bucket']
    locs = range(len(bounds))

    ax.plot(locs, t2tc_bucket_ratios, linewidth = 3, marker = 'o', color = 'teal')
    ax.grid(True)
    ax.set_xlabel('Forecast Bucket', fontsize = xlabelfontsize)
    ax.set_ylabel('Ratio')
    ax.set_title('Treatment / (Treatment + Control)')
    ax.set_xticks(locs)
    ax.set_xticklabels(ticklabels, rotation = x_ticks_rotation)
    leg = ax.legend((), (), loc="lower center")
    leg.set_title(f'Total volume: {len(y_true)}', prop = {"size" : 15})

    plt.tight_layout()
    plt.ioff()
    plt.close(fig)

    calibration_chart = pd.DataFrame([buckets_calibrated, bounds, interpolated]).T.reset_index()
    calibration_chart.columns = ['bucket', 'calibrated', 'CI', 'interpolated']
    calibration_chart = pd.merge(calibration_chart, percentile_chart, on = 'bucket', how = 'left')

    result = {}
    result['calibration_chart'] = calibration_chart
    result['bootstrap'] = bootstrap_iterations_by_bucket
    result['percentile_chart_figure'] = percentile_chart_results['figure']
    result['figure'] = fig

    return result


def get_cumulative_uplift_chart(
    y_true: np.ndarray,
    y_score: np.ndarray,
    treatment: np.ndarray,
    group_by_forecast: bool = True,
) -> Dict[str, Union[pd.DataFrame, plt.Figure, None]]:

    df = pd.DataFrame({'uplift_forecast': y_score, 'treatment': treatment, 'target': y_true})

    if group_by_forecast:
        agg_a = df.groupby(['uplift_forecast', 'treatment']).agg({'target': ['count', 'sum']})
        agg_a = agg_a.unstack(-1)

        _replace_names = {'target_count': 'observations', 'target_sum': 'responses'}
        _replace_treatment = {0: 'control', 1: 'treatment'}
        agg_a.columns = [_replace_treatment[c[2]]+'_'+_replace_names[f'{c[0]}_{c[1]}'] for c in agg_a.columns.to_flat_index()]
        ## already aggregated by score in index
        agg_a = agg_a.sort_index(ascending = False).reset_index().fillna(0)
        agg_cs = agg_a.drop(['uplift_forecast'], axis = 1).cumsum(axis = 0)
        agg_cs.columns = [f'{c}_cumulative' for c in agg_cs.columns]
        agg_cs['uplift_forecast'] = agg_a['uplift_forecast']
        agg_df = agg_cs[sorted(agg_cs.columns)].copy()
    else:
        agg_df = df.sort_values(by = 'uplift_forecast', ascending = False)
        agg_df['treatment_responses_cumulative'] = (agg_df['target'] * (agg_df['treatment'] == 1)).cumsum()
        agg_df['control_responses_cumulative'] = (agg_df['target'] * (agg_df['treatment'] == 0)).cumsum()
        agg_df['treatment_observations_cumulative'] = (agg_df['treatment'] == 1).cumsum()
        agg_df['control_observations_cumulative'] = (agg_df['treatment'] == 0).cumsum()
        agg_df.index = range(len(agg_df))

    agg_df['uplift_cumulative'] = (
        (agg_df['treatment_responses_cumulative'] / agg_df['treatment_observations_cumulative'])
        -
        (agg_df['control_responses_cumulative'] / agg_df['control_observations_cumulative'])
    )
    agg_df['treatment_response_rate'] = agg_df['treatment_responses_cumulative'] / agg_df['treatment_observations_cumulative']
    agg_df['control_response_rate'] = agg_df['control_responses_cumulative'] / agg_df['control_observations_cumulative']

    agg_df['n_all_cumulative'] = agg_df['treatment_observations_cumulative'] + agg_df['control_observations_cumulative']
    agg_df['percentage_cumulative'] = agg_df['n_all_cumulative'] / agg_df['n_all_cumulative'].tail(1).values
    return {
        'chart': agg_df,
        'figure': None
    }


def _check_k_parameter(k: Union[int, float]) -> float:
    if isinstance(k, int) and 0 < k <= 100:
        k /= 100
    elif isinstance(k, float) and 0 < k <= 1:
        pass
    else:
        raise ValueError(f'k should be a percentage of population in (0, 1] or integer in (0, 100], got {k}')
    return k


def _get_uplift_at_k_by_chart(
    k: Union[int, float],
    uplift_chart: pd.DataFrame,
    return_actual_k: bool = False
) -> Union[float, Tuple[float, float]]:
    k = _check_k_parameter(k)
    closest_to_k = np.argmin(np.abs(uplift_chart['percentage_cumulative'] - k))
    xk = uplift_chart.loc[closest_to_k, 'percentage_cumulative']
    yk = uplift_chart.loc[closest_to_k, 'uplift_cumulative']
    if return_actual_k:
        return yk, xk
    return yk


def calc_uplift_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    treatment: np.ndarray,
    k: Union[float, int],
    group_by_forecast: bool = True,
    sort_strategy: str = 'union',
):

    assert sort_strategy in ('union', 'separate'), f"'sort_strategy' should be either 'union' or 'separate', got {sort_strategy}"
    if sort_strategy == 'union':
        uplift_chart = get_cumulative_uplift_chart(
            y_true = y_true, y_score = y_score, treatment = treatment, group_by_forecast = group_by_forecast
        )['chart']
        uplift_at_k = _get_uplift_at_k_by_chart(k, uplift_chart, return_actual_k = False)
    else:
        ### this branch doesn't do grouping by same scores and cuts as is
        k = _check_k_parameter(k)
        y_true, y_score, treatment = (np.array(x) for x in (y_true, y_score, treatment))
        treatment_targets, treatment_scores = y_true[treatment == 1], y_score[treatment == 1]
        control_targets, control_scores = y_true[treatment == 0], y_score[treatment == 0]
        csize, tsize = len(control_scores), len(treatment_scores)
        t_tr_at_k = treatment_targets[np.argsort(-treatment_scores)][:int(k * tsize)].mean()
        c_tr_at_k = control_targets[np.argsort(-control_scores)][:int(k * csize)].mean()
        uplift_at_k = t_tr_at_k - c_tr_at_k

    return uplift_at_k


def get_uplift_grid(
    y_true: np.ndarray,
    y_score: np.ndarray,
    treatment: np.ndarray,
    k_grid: List[int] = None,
):

    k_grid = k_grid or range(10, 101, 10)
    result = pd.DataFrame(index = k_grid, columns = ['union_ungrouped', 'union_grouped', 'separate_ungrouped'])
    for k in k_grid:
        result.loc[k, :] = np.array((
            calc_uplift_at_k(y_true, y_score, treatment, k = k, group_by_forecast = False, sort_strategy = 'union'),
            calc_uplift_at_k(y_true, y_score, treatment, k = k, group_by_forecast = True, sort_strategy = 'union'),
            calc_uplift_at_k(y_true, y_score, treatment, k = k, group_by_forecast = False, sort_strategy = 'separate'),
            # calc_uplift_at_k(y_true, y_score, treatment, group_by_forecast = True, sort_strategy = 'separate')
        ))
    return result


def get_lift_plot(
    y_true: np.ndarray = None,
    y_score: np.ndarray = None,
    treatment: np.ndarray = None,
    group_by_forecast: bool = True,
    uplift_chart: pd.DataFrame = None,
    figsize: Tuple[int, int] = (12, 8),
    ax: plt.Axes = None,
    x_normalize: bool = True,
    x_start_percentage: float = 0,
    show_total: bool = False,
    k: Union[float, int] = None,
    color: str = None,
    label: str = None,
) -> Dict[str, Any]:

    if uplift_chart is None:
        uplift_chart = get_cumulative_uplift_chart(y_true, y_score, treatment, group_by_forecast)['chart']
    ### otherwise ignoring arguments and using precalculated chart

    plt.style.use(DEFAULT_PLOT_STYLE)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = figsize)
    else:
        fig = None
    if color is None:
        color = 'crimson'
    if label is None:
        label = 'actual'

    curves = {}

    if x_normalize:
        x = uplift_chart['percentage_cumulative']
    else:
        x = uplift_chart['n_all_cumulative']

    y = uplift_chart['uplift_cumulative']
    ### there might be nans in the beginning of the chart (as least in the first row if ungrouped)
    ### as not both T and C segments could have appeared so far when ordering
    ### for this chart we're not doing anything with that, as no area metric is being calculated

    # if y_normalize:
    #     y = y / y.tail(1).values
    mask = (uplift_chart['percentage_cumulative'] >= x_start_percentage)

    curves['actual'] = (x, y)
    ax.plot(x.loc[mask], y.loc[mask], color = color, linewidth = 2, label = label)
    ax.set_title('Lift Curve')
    ax.set_xlabel('Population' if not x_normalize else 'Population Share')
    ax.set_ylabel('Cumulative Lift')

    total_uplift = y.tail(1).values[0]
    if show_total:
        ax.axhline(y = total_uplift, linestyle = '-.', linewidth = 1, color = 'k', label = f'total uplift = {total_uplift:.4f}')

    yk = None
    if k:
        yk, xk = _get_uplift_at_k_by_chart(k, uplift_chart, return_actual_k = True)
        lk = xk * 100
        if not x_normalize:
            xk *= uplift_chart['n_all_cumulative'].tail(1).values[0]
        ax.vlines(x = xk, ymin = 0, ymax = yk, linestyle = '--', linewidth = 2, color = 'k', label = f'uplift@{lk:.2f}% = {yk:.4f}')

    ax.legend()
    plt.tight_layout()
    plt.ioff()
    if fig:
        plt.close(fig)

    result = {}
    result['chart'] = uplift_chart
    result['curves'] = curves
    result['uplift_at_k'] = yk
    result['y_final'] = total_uplift
    result['figure'] = fig

    return result


def get_uplift_gain_plot(
    y_true: np.ndarray = None,
    y_score: np.ndarray = None,
    treatment: np.ndarray = None,
    group_by_forecast: bool = True,
    uplift_chart: pd.DataFrame = None,
    figsize: Tuple[int, int] = (12, 8),
    ax: plt.Axes = None,
    x_normalize: bool = True,
    y_normalize: bool = True,
    y_normalize_version: str = 'last',
    show_ideal_Q: bool = False,
    show_random: bool = True,
    show_total: bool = False,
    k: Union[float, int] = None,
    color: str = None,
    label: str = None,
) -> Dict[str, Any]:

    if uplift_chart is None:
        uplift_chart = get_cumulative_uplift_chart(y_true, y_score, treatment, group_by_forecast)['chart']
    ### otherwise ignoring arguments and using precalculated chart

    plt.style.use(DEFAULT_PLOT_STYLE)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = figsize)
    else:
        fig = None
    if not color:
        color = 'crimson'
    if not label:
        label = 'actual'

    curves = {}

    if x_normalize:
        x = uplift_chart['percentage_cumulative']
    else:
        x = uplift_chart['n_all_cumulative']

    y = uplift_chart['uplift_cumulative'] * (uplift_chart['treatment_observations_cumulative'] + uplift_chart['control_observations_cumulative'])

    if y_normalize:
        _assert_y_normalize_version(y_normalize_version)
        if y_normalize_version == 'total':
            y = y / uplift_chart['n_all_cumulative'].tail(1).values
        elif y_normalize_version == 'treatment':
            y = y / uplift_chart['treatment_observations_cumulative'].tail(1).values
        elif y_normalize_version == 'last':
            y = y / y.tail(1).values
        elif y_normalize_version == 'abs_last':
            y = y / abs(y.tail(1).values)

    x, y = _preprocess_xy_uplift_curve(x, y)
    auuc_actual = auc(x, y)
    curves[label] = (x, y)
    ax.plot(x, y, color = color, label = label + f' (AUUC = {auuc_actual:.4f})', linewidth = 2)
    ax.set_title('Uplift Gain Curve')
    ax.set_xlabel('Population' if not x_normalize else 'Population Share')
    ax.set_ylabel('Cumulative Gain' + (' (Normalized)' if y_normalize else ''))

    if show_random:
        x_random, y_random = _get_random_uplift_curve(x, y)
        auuc_random = auc(x_random, y_random)
        curves['random'] = (x_random, y_random)
        ax.plot(x_random, y_random, color = 'black', label = 'random' + f' (AUUC = {auuc_random:.4f})', linestyle = '--', linewidth = 2)

    treatment_rr = uplift_chart['treatment_response_rate'].tail(1).values[0]
    control_rr = uplift_chart['control_response_rate'].tail(1).values[0]
    treatment_control_delta = treatment_rr - control_rr
    # y_final
    total_gain = treatment_control_delta * uplift_chart['n_all_cumulative'].tail(1).values[0]
    if show_total:
        ax.axhline(y = y[-1], linestyle = '-.', linewidth = 1, color = 'gray', label = f'total gain = {total_gain:.2f}')

    yk = None
    gain_at_k = None
    if k:
        yk, xk = _get_uplift_at_k_by_chart(k, uplift_chart, return_actual_k = True)
        eps = 1e-8
        lk = xk * 100
        gain_at_k = np.round(yk * uplift_chart.loc[np.abs(uplift_chart['percentage_cumulative'] - xk) < eps, 'n_all_cumulative'].values[0])
        normalized_gain_at_k = np.round(yk * uplift_chart.loc[np.abs(uplift_chart['percentage_cumulative'] - xk) < eps, 'n_all_cumulative'].values[0]) / total_gain
        if not x_normalize:
            xk *= uplift_chart['n_all_cumulative'].tail(1).values[0]
        if not y_normalize:
            yk = gain_at_k
        else:
            yk = normalized_gain_at_k
        ax.vlines(x = xk, ymin = 0, ymax = yk, linestyle = '-.', linewidth = 1, color = 'k', label = f'gain@{lk:.2f}% = {gain_at_k:.2f}')

    if show_ideal_Q:
        x_ideal, y_ideal = _get_ideal_uplift_curve(y_true, treatment, x_normalize = x_normalize, y_normalize = y_normalize, y_normalize_version = y_normalize_version, group_by_forecast = group_by_forecast)
        auuc_ideal = auc(x_ideal, y_ideal)
        curves['ideal'] = (x_ideal, y_ideal)
        ax.plot(x_ideal, y_ideal, color = 'forestgreen', label = 'ideal' + f' (AUUC = {auuc_ideal:.4f})', linestyle = '--', linewidth = 2)

    ax.legend()
    if fig is not None:
        plt.tight_layout()
        plt.ioff()
        plt.close(fig)

    result = {}
    result['chart'] = uplift_chart
    result['curves'] = curves
    result['gain_at_k'] = gain_at_k
    result['y_final'] = total_gain
    result['ax'] = ax
    result['figure'] = fig

    return result


def get_qini_plot(
    y_true: np.ndarray = None,
    y_score: np.ndarray = None,
    treatment: np.ndarray = None,
    group_by_forecast: bool = True,
    uplift_chart: pd.DataFrame = None,
    figsize: Tuple[int, int] = (12, 8),
    ax: plt.Axes = None,
    x_normalize: bool = True,
    y_normalize: bool = True,
    y_normalize_version: str = 'last',
    show_ideal_Q: bool = False,
    show_ideal_q0: bool = True,
    show_random: bool = True,
    show_total: bool = False,
    k: Union[float, int] = None,
    color: str = None,
    label: str = None,
) -> Dict[str, Any]:

    if uplift_chart is None:
        uplift_chart = get_cumulative_uplift_chart(y_true, y_score, treatment, group_by_forecast)['chart']
    ### otherwise ignoring arguments and using precalculated chart

    plt.style.use(DEFAULT_PLOT_STYLE)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = figsize)
        ax.set_title('Qini Curve')
        ax.set_xlabel('Population' if not x_normalize else 'Population Share')
        ax.set_ylabel('Cumulative Increment in Responses' + (' (Normalized)' if y_normalize else ''))
    else:
        fig = None
    if not color:
        color = 'crimson'
    if not label:
        label = 'actual'

    curves = {}

    ### !?shouldn't we take here only treatment cumulative volume actually..!?
    ### in scikit-lift or gutierrez, gerardy they use total volume for x, but original paper (Radcliffe, 2007) implies we should use treatment volume ..!?
    if x_normalize:
        x = uplift_chart['percentage_cumulative']
    else:
        x = uplift_chart['n_all_cumulative']

    y = uplift_chart['uplift_cumulative'] * uplift_chart['treatment_observations_cumulative']

    if y_normalize:
        _assert_y_normalize_version(y_normalize_version)
        if y_normalize_version == 'total':
            y = y / uplift_chart['n_all_cumulative'].tail(1).values
        elif y_normalize_version == 'treatment':
            y = y / uplift_chart['treatment_observations_cumulative'].tail(1).values
        elif y_normalize_version == 'last':
            y = y / y.tail(1).values
        elif y_normalize_version == 'abs_last':
            y = y / abs(y.tail(1).values)

    x, y = _preprocess_xy_uplift_curve(x, y)
    auqc_actual = auc(x, y)
    curves[label] = (x, y)
    ax.plot(x, y, color = color, label = label + f' (AUQC = {auqc_actual:.4f})', linewidth = 2)

    if show_random:
        x_random, y_random = _get_random_uplift_curve(x, y)
        auqc_random = auc(x_random, y_random)
        curves['random'] = (x_random, y_random)
        ax.plot(x_random, y_random, color = 'black', label = 'random' + f' (AUQC = {auqc_random:.4f})', linestyle = '--', linewidth = 2)

    treatment_rr = uplift_chart['treatment_response_rate'].tail(1).values[0]
    control_rr = uplift_chart['control_response_rate'].tail(1).values[0]
    treatment_control_delta = treatment_rr - control_rr
    # y_final
    total_increment = treatment_control_delta * uplift_chart['treatment_observations_cumulative'].tail(1).values[0]
    if show_total:
        ax.axhline(y = y[-1], linestyle = '-.', linewidth = 1, color = 'gray', label = f'total uplift = {total_increment:.2f}')

    yk = None
    increment_at_k = None
    if k:
        yk, xk = _get_uplift_at_k_by_chart(k, uplift_chart, return_actual_k = True)
        eps = 1e-8
        lk = xk * 100
        increment_at_k = np.round(yk * uplift_chart.loc[np.abs(uplift_chart['percentage_cumulative'] - xk) < eps, 'treatment_observations_cumulative'].values[0])
        normalized_increment_at_k = np.round(yk * uplift_chart.loc[np.abs(uplift_chart['percentage_cumulative'] - xk) < eps, 'treatment_observations_cumulative'].values[0]) / total_increment
        if not x_normalize:
            xk *= uplift_chart['n_all_cumulative'].tail(1).values[0]
        if not y_normalize:
            yk = increment_at_k
        else:
            yk = normalized_increment_at_k
        ax.vlines(x = xk, ymin = 0, ymax = yk, linestyle = '-.', linewidth = 1, color = 'k', label = f'increment@{lk:.2f}% = {increment_at_k:.2f}')

    if show_ideal_Q:
        x_Q, y_Q = _get_ideal_qini_curve(y_true, treatment, x_normalize = x_normalize, y_normalize = y_normalize, y_normalize_version = y_normalize_version, group_by_forecast = group_by_forecast, dnd_effect = True)
        auqc_Q = auc(x_Q, y_Q)
        curves['Q'] = (x_Q, y_Q)
        ax.plot(x_Q, y_Q, color = 'forestgreen', label = 'ideal Q' + f' (AUQC = {auqc_Q:.4f})', linestyle = '--', linewidth = 2)

    if show_ideal_q0:
        if treatment_rr <= control_rr:
            warnings.warn(f"Response rate in treatment group ({treatment_rr:.3f}) is less than or equal to response rate in control group ({control_rr:.3f}) - q0 optimal curve is not defined")
        else:
            x_q0, y_q0 = _get_ideal_qini_curve(y_true, treatment, x_normalize = x_normalize, y_normalize = y_normalize, y_normalize_version = y_normalize_version, group_by_forecast = group_by_forecast, dnd_effect = False)
            auqc_q0 = auc(x_q0, y_q0)
            curves['q0'] = (x_q0, y_q0)
            ax.plot(x_q0, y_q0, color = 'teal', label = 'ideal q0' + f' (AUQC = {auqc_q0:.4f})', linestyle = '--', linewidth = 2)

    ax.legend()
    if fig:
        plt.tight_layout()
        plt.ioff()
        plt.close(fig)

    result = {}
    result['chart'] = uplift_chart
    result['curves'] = curves
    result['increment_at_k'] = increment_at_k
    result['y_final'] = total_increment
    result['ax'] = ax
    result['figure'] = fig

    return result


def calc_auqc(
    y_true: np.ndarray = None,
    y_score: np.ndarray = None,
    treatment: np.ndarray = None,
    group_by_forecast: bool = True,
    uplift_chart: pd.DataFrame = None,
    x_normalize: bool = True,
    y_normalize: bool = True,
    y_normalize_version: str = 'last',
) -> float:

    if uplift_chart is None:
        uplift_chart = get_cumulative_uplift_chart(y_true, y_score, treatment, group_by_forecast)['chart']
    # !? shouldn't we take here only treatment cumulative volume actually..?
    # in scikit-lift or gutierrez, gerardy they use total volume for x,
    # but original paper (Radcliffe, 2007) implies we should use treatment volume.. !?
    if x_normalize:
        x = uplift_chart['percentage_cumulative']
    else:
        x = uplift_chart['n_all_cumulative']

    y = uplift_chart['uplift_cumulative'] * uplift_chart['treatment_observations_cumulative']
    if y_normalize:
        _assert_y_normalize_version(y_normalize_version)
        if y_normalize_version == 'total':
            y = y / uplift_chart['n_all_cumulative'].tail(1).values
        elif y_normalize_version == 'treatment':
            y = y / uplift_chart['treatment_observations_cumulative'].tail(1).values
        elif y_normalize_version == 'last':
            y = y / y.tail(1).values
        elif y_normalize_version == 'abs_last':
            y = y / abs(y.tail(1).values)

    x, y = _preprocess_xy_uplift_curve(x, y)
    auqc_actual = auc(x, y)
    return auqc_actual


def calc_qini(
    y_true: np.ndarray = None,
    y_score: np.ndarray = None,
    treatment: np.ndarray = None,
    group_by_forecast: bool = True,
    uplift_chart: pd.DataFrame = None,
    x_normalize: bool = True,
    y_normalize: bool = True,
    y_normalize_version: str = 'last',
    ideal: str = 'Q',
) -> float:

    qini_results = get_qini_plot(
        y_true = y_true, y_score = y_score, treatment = treatment,
        group_by_forecast = group_by_forecast,
        uplift_chart = uplift_chart,
        x_normalize = x_normalize, y_normalize = y_normalize, y_normalize_version = y_normalize_version,
        show_ideal_Q = True, show_ideal_q0 = True, show_random = True
    )

    AUQC_model = auc(*qini_results['curves']['actual'])
    AUQC_random = auc(*qini_results['curves']['random'])
    assert ideal in ('Q', 'q0')
    AUQC_ideal = auc(*qini_results['curves'][ideal])
    qini = (AUQC_model - AUQC_random) / (AUQC_ideal - AUQC_random)
    return qini


def calc_auuc(
    y_true: np.ndarray = None,
    y_score: np.ndarray = None,
    treatment: np.ndarray = None,
    group_by_forecast: bool = True,
    uplift_chart: pd.DataFrame = None,
    x_normalize: bool = True,
    y_normalize: bool = True,
    y_normalize_version: str = 'last',
) -> float:

    if uplift_chart is None:
        uplift_chart = get_cumulative_uplift_chart(y_true, y_score, treatment, group_by_forecast)['chart']

    if x_normalize:
        x = uplift_chart['percentage_cumulative']
    else:
        x = uplift_chart['n_all_cumulative']

    y = uplift_chart['uplift_cumulative'] * (uplift_chart['treatment_observations_cumulative'] + uplift_chart['control_observations_cumulative'])

    if y_normalize:
        _assert_y_normalize_version(y_normalize_version)
        if y_normalize_version == 'total':
            y = y / uplift_chart['n_all_cumulative'].tail(1).values
        elif y_normalize_version == 'treatment':
            y = y / uplift_chart['treatment_observations_cumulative'].tail(1).values
        elif y_normalize_version == 'last':
            y = y / y.tail(1).values
        elif y_normalize_version == 'abs_last':
            y = y / abs(y.tail(1).values)

    x, y = _preprocess_xy_uplift_curve(x, y)
    auuc_actual = auc(x, y)
    return auuc_actual


class UpliftCalibrator:
    def __init__(self):
        self._mapping_chart = None
        self._isotonic_model = None

    def _get_single_point_adjusted_prediction(self, p: float) -> float:
        if not isinstance(self._mapping_chart, pd.DataFrame):
            raise ValueError("calibrator haven't been yet properly fitted")
        for lb, ub, v in zip(
            self._mapping_chart['uplift_forecast_min'],
            self._mapping_chart['uplift_forecast_max'],
            self._mapping_chart['uplift_group_forecast_extrapolated'],
        ):
            if lb <= p < ub:
                return v
        return np.NAN

    def fit(
        self,
        y_true: ArrayLike1d,
        y_score: ArrayLike1d,
        treatment: ArrayLike1d,
        n_buckets: int = 10,
        bucketizer: Callable = base_bucketizer,
    ):

        start_bucket_marker = 1
        uplift_chart_results = get_percentile_uplift_chart(
            y_true = y_true, y_score = y_score, treatment = treatment,
            n_buckets = n_buckets,
            bucketizer = bucketizer,
            start_bucket_marker = start_bucket_marker,
        )
        # display(uplift_chart_results['figure'])
        # display(uplift_chart_results['chart'].head(n_buckets))
        chart = (
            uplift_chart_results['chart']
            [['uplift_actual', 'uplift_forecast', 'uplift_forecast_min', 'uplift_forecast_max']]
            .sort_values(by = 'uplift_forecast_min')
            .reset_index(drop = True)
        )
        # display(chart)

        expanded_chart = []
        prev_lb = -np.inf
        for lb, ub, v in zip(chart['uplift_forecast_min'], chart['uplift_forecast_max'], chart['uplift_actual']):
            expanded_chart.append((prev_lb, lb, v, 1))
            expanded_chart.append((lb, ub, v, 0))
            prev_lb = ub
        expanded_chart.append((ub, np.inf, v, 1))

        expanded_chart = pd.DataFrame.from_records(
            expanded_chart, columns = ['uplift_forecast_min', 'uplift_forecast_max', 'uplift_actual', 'is_extrapolated']
        )
        expanded_chart.loc[expanded_chart['is_extrapolated'] == 1, 'uplift_actual'] = np.NAN
        expanded_chart['uplift_group_forecast_extrapolated'] = expanded_chart['uplift_actual'].rolling(5, min_periods=1, center=True).mean(ignore_na = True)
        expanded_chart.loc[expanded_chart['is_extrapolated'] == 0, 'uplift_group_forecast_extrapolated'] = expanded_chart.loc[expanded_chart['is_extrapolated'] == 0, 'uplift_actual']
        # display(expanded_chart)
        self._mapping_chart = expanded_chart

        mapped_scores = pd.Series([self._get_single_point_adjusted_prediction(p) for p in y_score], index = y_score.index)
        ir_model = IsotonicRegression(increasing = True, out_of_bounds = "clip")
        ir_model.fit(X = y_score, y = mapped_scores)
        self._isotonic_model = ir_model

    def predict(
        self,
        y_score: ArrayLike1d,
    ) -> pd.Series:

        ir_scores = self._isotonic_model.predict(y_score)
        return ir_scores
