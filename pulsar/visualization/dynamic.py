from typing import Dict, List, Tuple, Union
from collections import OrderedDict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgridspec
from tqdm.auto import tqdm
import inspect
import itertools
import warnings

from pulsar.common.constants import TREATMENT_LABEL, CONTROL_LABEL
from pulsar.common.timeutils import create_grouping_ts_field, get_unix_time_seconds
from pulsar.visualization.constants import (
    DEFAULT_PLOT_STYLE,
    DEFAULT_COLOR,
    VARIANTS_COLORS,
    LINESTYLES,
    MARKERS,
)
from pulsar.visualization.visualization import get_figure_and_axes, place_legend
from pulsar.common.validation import (
    assert_treatment_labels,
    assert_discrete_distribution,
)


def get_metrics_dynamic(
    data: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    score_col: str = None,
    treatment_col: str = None,
    weight_col: str = None,
    control_label: int = CONTROL_LABEL,
    metrics: Dict = None,
    freq: str = "second",
    timestamp_format: str = None,
    verbose: bool = False,
    mode: str = None,
    group_col: str = None,
) -> pd.DataFrame:
    """
    this function aims to aggregate provided `data` in its temporal representation using information from `timestamp`
    it could be used to estimate different metrics/statistics in different scenarios:
        - model's performance metrics on already scored data - you need to provide target and score columns
            if that's an uplift model then treatment column name also needs to be passed along
            `metrics` is a dictionary with callables as dict-values to calculate corresponding aggregation metric
            each should have signature f(target, score, treatment, *args, **kwargs)
            treatment being optional and required for uplift models only
            this version is suitable for regression, classification (multi-class/multi-label), uplift models
            it's more convenient when metrics have approximately the same scale, e.g. in [0, 1] range
        - various statistics calculated on metrics in dynamic on already scored data
            then you need to provide `target` column only and `treatment` column if you want to split it by such groups
            'treatment' then expected to have categorical-like values
            in that scenario `metrics` is a dictionary with callables as values to calculate aggregated statistics
            each should have signature f(target, *args, **kwargs)
    :param data: dataframe with observations
    :param timestamp_col: name of the column with observation timestamp
    :param target_col: name of the target / metric column
    :param score_col: name of the prediction / score column
    :param treatment_col: name of the column containing treatment group or treatment
    :param weight_col: name of the column containing weight for each observation
    :param metrics: dict like {metric_name: metric_func(*args, **kwargs),}
    :param freq: string defining frequency or resolution to which timestamps would be truncated before aggregation
    :param timestamp_format: string defining specific format of a provided timestamp column, when of object/string type
    :param verbose: whether tqdm-wrapper for time-groups should be displayed
    """

    if weight_col is not None:
        raise NotImplementedError("weights are not yet supported")

    def assert_column(data: pd.DataFrame, col: str) -> None:
        if col not in data.columns:
            raise ValueError(f"'{col}' must be provided and present in data")

    # mode = None
    assert_column(data, target_col)
    if mode is None:
        if score_col is not None:
            assert_column(data, score_col)
            if treatment_col is not None:
                assert_column(data, treatment_col)
                mode = "uplift_model"
            else:
                mode = "classic_model"
        else:
            if treatment_col is not None:
                assert_column(data, treatment_col)
                assert_treatment_labels(data[treatment_col])
                assert_discrete_distribution(data[treatment_col])
                mode = "experiment"
    mode = mode or "simple"

    expected_sign, n_args = "f({}, *args, **kwargs)", None
    if mode in ("simple", "experiment"):
        expected_sign, n_args = expected_sign.format("target"), 1
    elif mode == "classic_model":
        expected_sign, n_args = expected_sign.format("target, score"), 2
    elif mode == "uplift_model":
        expected_sign, n_args = expected_sign.format("target, score, treatment"), 3

    metrics = metrics or {}
    if not metrics and mode not in ("simple", "experiment"):
        raise ValueError(
            f"'{metrics}' should have at least one callable metric with signature {expected_sign}-like"
        )
    else:
        for k, v in metrics.items():
            if callable(v):
                sign = inspect.signature(v)
                if len(sign.parameters) >= n_args:
                    continue
            raise ValueError(
                f"metric '{k}' should be callable and have signature {expected_sign}-like, "
                f"got {v}"
            )

    if freq is None:
        # aggregation could be made on initial timestamps, but we're forcing it to be truncated to 1 second frequency
        # which should still be more than enough
        freq = "second"

    if group_col is None:
        # trying to truncate timestamps with predefined frequency to reduce amount of unique time units
        # aggregation then would take place on these larger time units
        data, grouping_field = create_grouping_ts_field(
            data=data,
            timestamp_col=timestamp_col,
            freq=freq,
            timestamp_format=timestamp_format,
            to_str=False,
        )
    else:
        grouping_field = group_col

    grouping_fields = [grouping_field] + (
        [treatment_col] if mode == "experiment" else []
    )
    if verbose:
        group_iterator = tqdm(data.groupby(grouping_fields))
    else:
        group_iterator = data.groupby(grouping_fields)
    aggregates = {}

    for group, group_data in group_iterator:
        group_aggregates = {}
        group_aggregates["n_observations"] = len(group_data)
        default_metric_value = None
        if mode in ("simple", "experiment"):
            target = group_data[target_col]
            args = (target,)
            group_aggregates["average_target"] = np.mean(target)
            if score_col is not None:
                group_aggregates["average_score"] = np.mean(group_data[score_col])
            # group_aggregates["average_score"] = np.nan
        elif mode == "classic_model":
            target, score = group_data[target_col], group_data[score_col]
            args = (target, score)
            group_aggregates["average_target"] = np.mean(target)
            group_aggregates["average_score"] = np.mean(score)
            ### for binary classification problems some metric might fail when target has 0 variance
            if target.nunique() == 1:
                default_metric_value = 0
        elif mode == "uplift_model":
            target, score, treatment = (
                group_data[target_col],
                group_data[score_col],
                group_data[treatment_col],
            )
            args = (target, score, treatment)
            ### calculating ATE within the group as true target
            ### now only supported for binary treatment cases
            group_aggregates["average_target"] = np.mean(
                target[treatment == TREATMENT_LABEL]
            ) - np.mean(target[treatment == CONTROL_LABEL])
            group_aggregates["average_score"] = np.mean(score)

        for metric_name, metric in metrics.items():
            try:
                metric_value = metric(*args)
            except (TypeError, ValueError) as e:
                warnings.warn(
                    f"unable to calculate metric '{metric_name}': {inspect.signature(metric)} "
                    f"on {group} group: {e.__repr__()}, metric value replaced with {default_metric_value}"
                )
                metric_value = default_metric_value
            group_aggregates[metric_name] = metric_value

        aggregates[group] = group_aggregates

    if group_col is None:
        data.drop(columns=grouping_field, inplace=True)
    result = pd.DataFrame(aggregates).transpose()
    if weight_col is None:
        result["n_observations"] = result["n_observations"].astype(int)
    result.index.names = ["timestamp"] + (
        [treatment_col] if mode == "experiment" else []
    )

    if mode == "experiment":
        treatment_labels = sorted(set(result.index.get_level_values(1)))
        unstacked = result.unstack(-1)
        unstacked[('ATE', control_label)] = 0.0
        for t_label in treatment_labels:
            if t_label != control_label:
                unstacked[('ATE', t_label)] = (
                    unstacked[('average_target', t_label)] - unstacked[('average_target', control_label)]
                )
        result = unstacked.stack(1)

    return result


def plot_metrics_dynamic(
    aggregates: pd.DataFrame,
    metrics: List = None,
    metric_volume_key: str = None,
    figsize: Tuple = None,
    title: str = None,
    target_marker: str = None,
    score_marker: str = None,
    y_limits: Tuple = None,
    adjust_lines: bool = False,
    labels_mapping: Dict = None,
    n_reserved_axes: int = 0,
    one_axis_per_metric: bool = False,
) -> Tuple[plt.Figure, List[matplotlib.axes.Axes]]:
    """
    this function is supposed to be using results of `get_metrics_dynamic`

    """

    def adjust_locations(i, n):
        if n % 2 == 0:
            if i < n // 2:
                return ((i - n // 2) + 1 / 2) / n
            else:
                return ((i + 1 - n // 2) - 1 / 2) / n
        else:
            return (i - n // 2) / n

    plt.style.use(DEFAULT_PLOT_STYLE)

    target_marker = target_marker or "^"
    score_marker = score_marker or "v"

    plot_predictions = True if "average_score" in aggregates.columns else False
    n_levels = aggregates.index.nlevels
    n_col_levels = aggregates.columns.nlevels
    ### trying to bring dataframe to the format where
    ### index is single level with pd.Timestamp instances
    ### column-keys are tuples: [('n_observations', 'sample_j', treatment_i), ... , ('metric_k', s_j, treatment_i),]
    d = aggregates.copy()
    if n_levels == 2:
        groups = list(OrderedDict.fromkeys(d.index.get_level_values(1)).keys())
        n_groups = len(groups)
        fake_group_id = None
        if n_col_levels == 1:
            subsets = ['']
            n_subsets = 1
            d.columns = pd.MultiIndex.from_product([d.columns, subsets])
        elif n_col_levels == 2:
            subsets = d[("n_observations",)].columns
            n_subsets = len(subsets)
        else:
            raise ValueError(
                f"unknown aggregates' columns format, expected (metric, ) or (metric, sample), "
                f"got {aggregates.columns}"
            )
    elif n_levels == 1:
        # index is still pd.Timestamp, but we're adding treatment dimension to work in a unified way in a loop further
        fake_group_id, n_groups = -1, 1
        groups = [fake_group_id]
        d = (
            d.assign(treatment=fake_group_id)
            .set_index("treatment", append=True)
        )
        if n_col_levels == 1:
            subsets = ['']
            n_subsets = 1
            d.columns = pd.MultiIndex.from_product([d.columns, subsets])
        elif n_col_levels == 2:
            subsets = d[("n_observations",)].columns
            n_subsets = len(subsets)
        else:
            raise ValueError(
                f"unknown aggregates' columns format, expected (metric, ) or (metric, sample), "
                f"got {aggregates.columns}"
            )
    else:
        raise ValueError(
            f"unknown aggregates' index format, expected (timestamp, ) or (timestamp, treatment), "
            f"got {aggregates.index}"
        )
    dynamic_results = d.unstack(-1)

    metrics = metrics or []

    xticks = pd.Series(dynamic_results.index, index=dynamic_results.index)
    xticklabels = xticks.astype(str)
    while all(xticklabels.str.endswith(":00")):
        xticklabels = xticklabels.str[:-3]
    try:
        xlocations = get_unix_time_seconds(xticks)
        xlabel = "Period"
        xticksrotation = 80
    except Exception:
        xlocations = pd.Series(np.arange(len(xticks)), index = dynamic_results.index)
        xlabel = "Group"
        xticksrotation = 30

    gap = xlocations.diff().min()
    if pd.isnull(gap):
        gap = 1
    width = gap / (n_groups * n_subsets)
    shrinkage = 0.7

    n_rows = 1
    m = len(metrics)
    one_axis_per_metric = one_axis_per_metric and (metric_volume_key is None)
    metric_axes_shift = m if one_axis_per_metric else min(1, m)
    n_rows += metric_axes_shift
    metric_axes = {}
    metric_axes_label = "Metric"

    n_padding_rows = 0
    row_size, padding_size = 10, 0.5
    height_ratios = [row_size] * n_rows

    add_null_ax = False
    if add_null_ax:
        n_padding_rows += 1
        height_ratios = [padding_size] + height_ratios

    n_reserved_axes = max(n_reserved_axes, 0)
    if n_reserved_axes:
        # height_ratios = height_ratios + [padding_size]
        # n_padding_rows += 1
        height_ratios = height_ratios + [row_size] * n_reserved_axes
        n_rows += n_reserved_axes

    if figsize is None:
        figsize = (25, n_rows * 5 + n_padding_rows * 1)

    fig = plt.figure(figsize=figsize)
    gridspec = mgridspec.GridSpec(
        nrows=n_rows + n_padding_rows, ncols=1, figure=fig,
        wspace=0.0, hspace=0.6 if n_reserved_axes else 0.4,
        height_ratios=height_ratios,
    )

    ix_offset = 0
    if add_null_ax:
        ix_offset = 1
        ax_null = fig.add_subplot(gridspec[:1])
        ax_null.set_xticks([])
        ax_null.set_yticks([])

    if m:
        # volume axes and average axes share same subplot and we insert in the lowest slot
        ax_volume = fig.add_subplot(gridspec[ix_offset+metric_axes_shift:ix_offset+metric_axes_shift+1])
        ax_values = ax_volume.twinx()
        if metric_volume_key is not None:
            ax_metric_volume = fig.add_subplot(gridspec[ix_offset:ix_offset+1], sharex=ax_volume)
            ax_metric_volume.grid(False)
            ax_metric_volume.set_ylabel(metric_volume_key)
            ax_metric = ax_metric_volume.twinx()
            ax_metric.set_ylabel(metric_axes_label, rotation=-90, labelpad=20)
            metric_axes[metric_axes_label] = ax_metric
        else:
            if one_axis_per_metric:
                for k, metric_name in enumerate(metrics):
                    ax_metric = fig.add_subplot(gridspec[ix_offset+k:ix_offset+k+1], sharex=ax_volume)
                    ax_metric.set_ylabel(metric_name)
                    ax_metric.grid(True)
                    metric_axes[metric_name] = ax_metric
            else:
                ax_metric = fig.add_subplot(gridspec[ix_offset:ix_offset+1], sharex=ax_volume)
                ax_metric.set_ylabel(metric_axes_label)
                metric_axes[metric_axes_label] = ax_metric

        ix_offset += (metric_axes_shift + 1)

        if y_limits is not None:
            ax_metric.set_ylim(*y_limits)

    else:
        # could enabled only with primary metric axis
        metric_volume_key = None
        # don't have any custom metrics and don't need a separate subplot
        ax_volume = fig.add_subplot(gridspec[ix_offset:ix_offset+1])
        ax_values = ax_volume.twinx()
        # ax_values = fig.add_subplot(gridspec[1:2], sharex = ax_volume)
        ax_metric = ax_values
        ix_offset = ix_offset+1

    reserved_axes = []
    if ix_offset < n_rows + n_padding_rows:
        # ax_volume.xaxis.tick_top()
        # ax_null = fig.add_subplot(gridspec[ix_offset:ix_offset+1])
        # ax_null.set_xticks([])
        # ax_null.set_yticks([])
        reserved_axes = [
            fig.add_subplot(gridspec[i:i+1], sharex=ax_volume)
            for i in range(ix_offset, n_rows + n_padding_rows)
        ]
    else:
        ax_volume.set_xlabel(xlabel)

    ax_volume.grid(False)
    ax_metric.grid(True)
    ax_values.grid(True)
    ax_volume.set_ylabel("Volume")
    ax_values.set_ylabel("Averages", rotation=-90, labelpad=20)

    axes = [ax_volume, ax_values, metric_axes] + [] if metric_volume_key is None else [ax_metric_volume] + reserved_axes
    axes = tuple(axes)

    markersize = 10
    linewidth = 3
    alpha = 0.4
    blank_color = "black"

    volume_patch = mpatches.Patch(color=blank_color, label="Volume", alpha=alpha)
    target_patch = mlines.Line2D(
        [],
        [],
        linewidth=linewidth,
        linestyle="-",
        alpha=alpha,
        marker=target_marker,
        markersize=markersize,
        color=blank_color,
        label="Target",
    )
    if plot_predictions:
        score_patch = mlines.Line2D(
            [],
            [],
            linewidth=linewidth,
            linestyle="-",
            alpha=alpha,
            marker=score_marker,
            markersize=markersize,
            color=blank_color,
            label="Score",
        )
        patches = [volume_patch, target_patch, score_patch]
    else:
        patches = [volume_patch, target_patch]

    selected_colors = [mcolors.CSS4_COLORS.get(col) for col in VARIANTS_COLORS]
    n_colors = len(selected_colors)

    labels_mapping = labels_mapping or {}
    linestyles = itertools.cycle(LINESTYLES)
    for j, (s, ls) in enumerate(zip(subsets, linestyles)):
        for i, g in enumerate(groups):

            # dropping never-met (timestamp + group) pairs, to still show line-segments between existing points
            # and not torn up fragments which would be there when some groups are missing within particular timestamp
            # this incompleteness would be highlighted by marker's absense and zero-height of corresponding volume bar
            subsample = dynamic_results.loc[
                dynamic_results[("n_observations", s, g)].notna(), :
            ]
            xlocs = xlocations.loc[dynamic_results[("n_observations", s, g)].notna()]
            # subsample, xlocs = dynamic_results, xlocations

            volumes = subsample[("n_observations", s, g)]
            targets = subsample[("average_target", s, g)]

            color = selected_colors[i % n_colors]
            xlocs_adjustment = adjust_locations(n_groups * j + i, n_groups * n_subsets)
            xlocs_adjustment = xlocs_adjustment * gap * shrinkage
            xlocs_adjusted_bars = xlocs + xlocs_adjustment
            if adjust_lines or n_subsets > 1:
                xlocs_adjusted = xlocs + xlocs_adjustment
            else:
                xlocs_adjusted = xlocs
            ax_volume.bar(
                xlocs_adjusted_bars,
                volumes,
                width=width * shrinkage,
                color=color,
                alpha=alpha,
            )

            if metric_volume_key is not None:
                ax_metric_volume.bar(
                    xlocs_adjusted_bars,
                    subsample[(metric_volume_key, s, g)],
                    width=width * shrinkage,
                    color=color,
                    alpha=alpha,
                )

            l = labels_mapping.get(g, g)

            # should be reinitialized each sub-loop
            markers = itertools.cycle(MARKERS)
            for mk, metric_name in zip(markers, metrics):
                metric_values = subsample[(metric_name, s, g)]
                if metric_name in metric_axes:
                    ax_metric = metric_axes[metric_name]
                    marker = MARKERS[0]
                else:
                    ax_metric = metric_axes[metric_axes_label]
                    marker = mk
                ax_metric.plot(
                    xlocs_adjusted,
                    metric_values,
                    linewidth=linewidth/2,
                    linestyle=ls,
                    markersize=markersize,
                    marker=marker,
                    label=(
                        f"{metric_name}"
                        + (f" {s}" if s != "" else "")
                        + (f" {l}" if fake_group_id is None else "")
                    ),
                    color=color,
                )

            kws = {}
            if not metrics:
                kws["label"] = (f"{s}" if s != "" else "") + (f" {l}" if fake_group_id is None else "")
            ax_values.plot(
                xlocs_adjusted,
                targets,
                linewidth=linewidth,
                linestyle=ls,
                marker=target_marker,
                markersize=markersize,
                color=color,
                **kws,
            )

            if plot_predictions:
                predictions = subsample[("average_score", s, g)]
                ax_values.plot(
                    xlocs_adjusted,
                    predictions,
                    linewidth=linewidth,
                    linestyle=ls,
                    marker=score_marker,
                    markersize=markersize,
                    color=color,
                    **kws,
                )

    handlelength = 5
    # if metrics:
    #     [label.set_visible(False) for label in ax_metric.get_xticklabels()]
    ax_volume.set_xticks(xlocations)
    ax_volume.set_xticklabels(xticklabels, rotation=xticksrotation)
    for _, ax_metric in metric_axes.items():
        ax_metric.set_xticks(xlocations)
        ax_metric.set_xticklabels(xticklabels, rotation=xticksrotation)
    if metric_volume_key is not None:
        ax_metric_volume.set_xticks(xlocations)
        ax_metric_volume.set_xticklabels(xticklabels, rotation=xticksrotation)
    # ax_metric.xaxis.tick_top()
    fig.suptitle(title, fontsize=30)
    # ax_metric.set_title(title, fontdict = {'fontsize' : 10})
    def make_patch(color, label, linestyle = '-', marker = target_marker):
        patch = mlines.Line2D(
            [],
            [],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            marker=marker,
            markersize=markersize,
            color=color,
            label=label,
        )
        return patch

    patches += [
        make_patch(color = selected_colors[i % n_colors], label = labels_mapping.get(g, g))
        for i, g in enumerate(groups)
    ] if m == 0 and groups != [fake_group_id] else []
    place_legend(
        "",
        ax_volume,
        loc="upper right",
        handles=patches,
        ncol=min(len(patches), 1),
        handlelength=handlelength,
    )
    if fake_group_id is None or n_subsets > 1:
        if len(metric_axes) == 0:
            place_legend(
                "",
                ax_values,
                loc="lower right",
                handles=[
                    make_patch(color = 'black', label = s, linestyle = ls, marker = None)
                    for (s, ls) in zip(subsets, linestyles)
                ],
                ncol=min(len(subsets), 1),
                handlelength=handlelength*4,
            )
        for _, ax_metric in metric_axes.items():
            place_legend(
                "",
                ax_metric,
                loc="upper left" if ax_metric == ax_values else "upper right",
                ncol=min(n_groups, 1),
                handlelength=handlelength * 2,
            )
    x_lim_offset = gap / 2
    x_min, x_max = min(xlocations), max(xlocations)
    ax_volume.set_xlim(x_min - x_lim_offset, x_max + x_lim_offset)
    # plt.tight_layout()
    plt.ioff()
    plt.close()

    return fig, axes


def get_distribution_dynamic(
    data: pd.DataFrame,
    timestamp_col: str,
    metric_col: str,
    quantiles: Union[int, List, np.ndarray] = 20,
    grouping: bool = False,
    freq: str = 'M',
    timestamp_format: str = '%Y-%m-%d',
    figsize: Tuple[int, int] = (25, 8),
    xtickssize: int = 10,
    xticksrotation: int = 80,
    markersize: int = 10,
) -> Dict[str, Union[pd.DataFrame, plt.Figure]]:

    if grouping:
        data, grouping_field = create_grouping_ts_field(
            data,
            timestamp_col = timestamp_col,
            freq = freq,
            timestamp_format = timestamp_format,
        )
    else:
        grouping_field = timestamp_col

    if isinstance(quantiles, int):
        quantiles = np.arange(0, 1.01, 1 / quantiles)

    groups, q_results, means, nans, counts = [], [], [], [], []

    for group, data_grp in data.groupby(grouping_field):
        q_results.append(np.nanquantile(data_grp[metric_col], q = quantiles,))
        nans.append(data_grp[metric_col].isna().mean())
        means.append(np.mean(data_grp[metric_col]))
        counts.append(len(data_grp))
        groups.append(group)

    fig, axes = plt.subplots(2, 1, figsize = figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex = 'col')
    ax = axes[0]
    ax_na = axes[1]
    ax_count = ax_na.twinx()

    xticks = pd.Series(groups)
    if grouping:
        ### means we're using timestamps
        xlocations = get_unix_time_seconds(xticks)
        xticklabels = xticks.astype(str)
        while all(xticklabels.str.endswith(":00")):
            xticklabels = xticklabels.str[:-3]
        gap = xlocations.diff().min()
        width = gap * 0.8
    else:
        xlocations = np.arange(len(xticks))
        xticklabels = xticks.astype(str)
        gap = 1
        width = 0.8

    # palette = sns.color_palette('Spectral_r', n_colors = len(quantiles),)
    palette = sns.diverging_palette(230, 8, l = 55, s = 100, n = len(quantiles))

    results = np.vstack(q_results)

    for i, x in enumerate(xlocations):
        for color, y_coord, q in zip(palette[::-1], results[i][::-1], quantiles[::-1]):
            if i == 0:
                ax.plot(
                    [x], [y_coord], color = color, marker = 'o', markersize = markersize,
                    alpha = 1, label = f'Q {int(q * 100):2d}%', linestyle = None
                )
            else:
                ax.scatter(x, y_coord, color = color, marker = 'o', s = markersize ** 2,)

    results = results.T
    for line in results:
        ax.plot(xlocations, line, linewidth = 0.5, linestyle = '--', color = 'k')

    ax.plot(
        xlocations, means,
        linewidth = 3, linestyle = '--', color = (np.array(palette[0]) + np.array(palette[-1])) / 2,
        markersize = markersize, marker = 'o', label = 'mean'
    )
    ax_count.bar(
        xlocations, counts, color = DEFAULT_COLOR, alpha = 0.3, hatch = '/', edgecolor = 'grey', width = width,
    )
    ax_na.plot(
        xlocations, nans,
        linewidth = 1, linestyle = '--', color = 'k',
        markersize = markersize // 2, marker = 'o', label = 'Missings, %'
    )
    ax_na.set_ylim(-0.01,)
    ax_count.grid(False)
    ax.set_xticks(xlocations)
    ax.set_xticklabels(xticklabels, **{'fontsize': xtickssize, 'rotation': xticksrotation})
    ax_na.set_xticks(xlocations)
    ax_na.set_xticklabels(xticklabels, **{'fontsize': xtickssize, 'rotation': xticksrotation})
    ax.set_ylabel('Feature Value')
    ax_na.set_ylabel('Missings Share')
    ax_na.set_xlabel(timestamp_col)
    ax_na.legend(loc = 'center right', fontsize = 8)
    ax_count.set_ylabel('Volume', rotation = -90, labelpad = 20)
    ax.set_title(f'Numeric distribution')
    ax_na.set_title(f'Total volume and missing values')
    x_lim_offset = gap / 2
    x_min, x_max = min(xlocations), max(xlocations)
    ax_count.set_xlim(x_min - x_lim_offset,)
    ax.legend(loc = 'center right', fontsize = 8)
    fig.suptitle(metric_col)

    plt.tight_layout()
    plt.ioff()
    plt.close()

    if grouping:
        del data[grouping_field]

    results = {
        group: {'q': q, 'count': c, 'mean': m, 'na': n}
        for group, c, q, m, n in zip(groups, counts, q_results, means, nans)
    }

    return {
        'data': pd.DataFrame(results).T,
        'figure': fig,
    }
