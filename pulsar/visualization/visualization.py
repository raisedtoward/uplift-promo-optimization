from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.axes
import statsmodels
import statsmodels.nonparametric
import statsmodels.nonparametric.bandwidths
import warnings

from pulsar.common.constants import TREATMENT_LABEL_NAME, CONTROL_LABEL_NAME
from pulsar.visualization.constants import TREATMENT_COLOR, CONTROL_COLOR
from pulsar.visualization.constants import DEFAULT_COLOR, VARIANTS_COLORS
from pulsar.visualization.constants import DEFAULT_PLOT_STYLE, DIST_KWS
from pulsar.common.validation import validate_contingency_table


def enable_plot_style() -> None:
    plt.plot([1], [1])
    plt.close()
    plt.style.use(DEFAULT_PLOT_STYLE)


def get_figure_and_axes(
    figsize: Tuple = (10, 6),
    ax: matplotlib.axes.Axes = None,
    grid_size: Tuple = (1, 1),
) -> Tuple[plt.Figure, Tuple[Tuple[matplotlib.axes.Axes]]]:
    if ax is None:
        fig, axes = plt.subplots(*grid_size, figsize=figsize, squeeze=False)
        axes[0][0].grid(True)
    else:
        fig = None
        axes = ((ax,),)
    plt.close()
    return fig, axes


def place_legend(
    legend_str: str,
    ax: matplotlib.axes.Axes,
    loc: str = "upper right",
    bbox_to_anchor: Tuple = (1.15, 1),
    **kwargs,
) -> matplotlib.axes.Axes:
    leg = ax.legend(
        loc=loc,  # "best",
        fontsize=12,
        framealpha=0,
        **kwargs,
    )
    leg.set_title(title=legend_str, prop={"size": 14})
    sns.move_legend(obj=ax, loc=loc, bbox_to_anchor=bbox_to_anchor,)
    # plt.tight_layout()
    plt.close()
    return ax


def plot_metric_distributions(
    treatment_data: Union[np.ndarray, pd.Series, List],
    control_data: Union[np.ndarray, pd.Series, List],
    figsize: Tuple = (10, 6),
    ax: matplotlib.axes.Axes = None,
    title: str = None,
    legend: bool = True,
    labels: Tuple = (TREATMENT_LABEL_NAME, CONTROL_LABEL_NAME),
    colors: Tuple = (TREATMENT_COLOR, CONTROL_COLOR),
    **kwargs,
) -> Tuple[plt.Figure, Tuple[Tuple[matplotlib.axes.Axes]]]:

    enable_plot_style()

    dist_kws = {**DIST_KWS, **kwargs}

    fig, axes = get_figure_and_axes(
        figsize=figsize,
        ax=ax,
    )
    ax = axes[0][0]
    treatment_label, control_label = labels
    treatment_color, control_color = colors
    sns.histplot(
        treatment_data,
        label=treatment_label,
        color=treatment_color,
        ax=ax,
        **dist_kws,
    )
    sns.histplot(
        control_data,
        label=control_label,
        color=control_color,
        ax=ax,
        **dist_kws,
    )
    if legend:
        ax.legend()
    ax.set_title(title or "", fontsize=10)

    plt.tight_layout()
    plt.close()
    return fig, axes


def plot_metric_distribution_by_classes(
    data: pd.DataFrame,
    class_name: str,
    metric_name: str,
    label: str = "",
    title: str = None,
    class_label_mapping: Dict = None,
    figsize: Tuple = (10, 6),
    ax: matplotlib.axes.Axes = None,
    plot_margin: float = 0.05,
    frequency: bool = True,
) -> Tuple[plt.Figure, Tuple[Tuple[matplotlib.axes.Axes]]]:

    enable_plot_style()

    selected_colors = [mcolors.CSS4_COLORS.get(col) for col in VARIANTS_COLORS]
    n_colors = len(selected_colors)
    classes = sorted(data[class_name].unique())
    if not class_label_mapping:
        labels = [str(c) for c in classes]
    else:
        labels = [class_label_mapping[c] for c in classes]
    mins, maxs = [], []
    fig, axes = get_figure_and_axes(
        figsize=figsize,
        ax=ax,
    )
    ax = axes[0][0]

    ax.set_title(f"{metric_name} distribution " + label if not title else title)

    for i, c in enumerate(classes):
        subsample = data.loc[data[class_name] == c, metric_name]
        mins.append(min(subsample))
        maxs.append(max(subsample))
        color = selected_colors[i % n_colors]
        _kws = {"kde_kws": {}, "stat": "density"}
        if frequency:
            _kws["stat"] = "count"
        bw = statsmodels.nonparametric.bandwidths.bw_scott(subsample)
        if np.isclose(bw, 0):
            bw = 0.9 * np.std(subsample) * len(subsample) ** (-1 / 5)
            _kws["kde_kws"]["bw_method"] = bw
        sns.histplot(
            subsample,
            ax=ax,
            label=labels[i],
            color=color,
            **{**DIST_KWS, **_kws},
        )

    ax.legend(loc="upper right")
    # xmin, xmax = ax.get_xlim()
    xmin = min((1 + plot_margin) * min(mins), (1 - plot_margin) * min(mins))
    xmax = max((1 + plot_margin) * max(maxs), (1 - plot_margin) * max(maxs))
    if np.isclose(xmin, 0, atol=1e-4):
        xmin = xmin - max(plot_margin, 0.5 * np.std(data[metric_name]))
    elif np.isclose(xmax, 0, atol=1e-4) == 0:
        xmax = xmax + max(plot_margin, 0.5 * np.std(data[metric_name]))
    else:
        pass

    if not np.isnan(xmin) and not np.isnan(xmax):
        ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    plt.close()
    return fig, axes


def plot_bootstrapped_distribution(
    distribution: np.ndarray,
    confidence_level: float,
    lower_bound: float,
    upper_bound: float,
    mu: float = None,
    figsize: Tuple = (8, 5),
    ax: matplotlib.axes.Axes = None,
    color: str = DEFAULT_COLOR,
    legend: bool = True,
    label: str = "",
    mu_label: str = "mu",
) -> Tuple[plt.Figure, Tuple[Tuple[matplotlib.axes.Axes]]]:

    enable_plot_style()

    fig, axes = get_figure_and_axes(figsize, ax)
    ax = axes[0][0]

    label = label + " " if label else ""

    sns.histplot(x=distribution, ax=ax, color=color, **DIST_KWS)
    d_mean = np.mean(distribution)
    ax.axvline(
        x=d_mean,
        linestyle="-",
        linewidth=2,
        c=color,
        label=f"{label} Mean: {d_mean:.4f}",
    )
    if not np.isinf(lower_bound):
        ax.axvline(
            x=lower_bound,
            linestyle="--",
            linewidth=2,
            c=color,
            label=f"{label}{confidence_level * 100:.2f}% CI LB: {lower_bound:.4f}",
        )
    if not np.isinf(upper_bound):
        ax.axvline(
            x=upper_bound,
            linestyle="--",
            linewidth=2,
            c=color,
            label=f"{label}{confidence_level * 100:.2f}% CI UB: {upper_bound:.4f}",
        )
    if mu is not None:
        ax.axvline(
            x=mu,
            linestyle="--",
            linewidth=2,
            c="black",
            label=f"{label}{mu_label}: {mu:.4f}",
        )
    if legend:
        ax.legend(
            loc="best",
            fontsize=12,
            ncol=1,
            framealpha=0,
        )
    plt.tight_layout()
    plt.close()
    return fig, axes


def plot_discrete_distributions(
    data: pd.DataFrame,
    expected_distribution: pd.DataFrame = None,
    title: str = None,
    legend_str: str = None,
    figsize: Tuple = (10, 6),
    ax: matplotlib.axes.Axes = None,
    xticksrotation: str = "horizontal",
    labels_mapping: Dict = None,
) -> Tuple[plt.Figure, Tuple[Tuple[matplotlib.axes.Axes]]]:
    """
    `data` is a contingency table in probability form (normalized by rows) of N x M shape
    where
        - N stands for amount of different variants / distributions
        - M stands for amount of unique realizations of random variable
    `expected_distribution` is an optional contingency table of 1 x M shape
    """

    enable_plot_style()

    def adjust_locations(i, n):
        if n % 2 == 0:
            if i < n // 2:
                return ((i - n // 2) + 1 / 2) / n
            else:
                return ((i + 1 - n // 2) - 1 / 2) / n
        else:
            return (i - n // 2) / n

    data = validate_contingency_table(data, normalized=True)
    if expected_distribution is not None:
        expected_distribution = validate_contingency_table(
            expected_distribution, normalized=True
        )
        if len(expected_distribution) != 1:
            raise ValueError(
                "expected distribution data should have only 1 variant in index"
            )
        unexpected_columns = sorted(
            set(data.columns).difference(expected_distribution.columns)
        )
        if len(unexpected_columns):
            raise ValueError(
                f"the following categories are missing in expected distribution data: "
                f"{unexpected_columns}"
            )
        additional_columns = sorted(
            set(expected_distribution.columns).difference(data.columns)
        )
        if len(additional_columns):
            warnings.warn(
                message=(
                    f"the following categories are missing in data, but are present in expected distribution data: "
                    f"{additional_columns}"
                ),
                category=UserWarning,
            )
        locations_data = pd.Series(
            np.arange(expected_distribution.shape[1]),
            index=expected_distribution.columns,
        )
    else:
        locations_data = pd.Series(np.arange(data.shape[1]), index=data.columns)

    variants = data.index
    n_variants = len(data)
    width = 1 / n_variants
    shrinkage = 0.8

    figsize = (max(figsize[0], len(locations_data)), figsize[1])
    fig, axes = get_figure_and_axes(
        figsize=figsize,
        ax=ax,
    )
    ax = axes[0][0]

    if expected_distribution is not None:
        x_locations = locations_data.values
        y_values = expected_distribution.iloc[0, :]
        ax.bar(
            x_locations,
            y_values,
            # label="Expected",
            color="black",
            alpha=0.1,
            hatch="/",
            edgecolor="black",
            width=0.9,
        )

    # later allow providing palette based on variants as an argument
    # palette could be pre-defined on the experiment level to be used across different subtests consistently
    selected_colors = [mcolors.CSS4_COLORS.get(col) for col in VARIANTS_COLORS]
    n_colors = len(selected_colors)
    labels_mapping = labels_mapping or {}

    for i, v in enumerate(variants):
        color = selected_colors[i % n_colors]
        x_locations = locations_data.loc[data.columns]
        y_values = data.loc[v, :]
        ax.bar(
            x_locations + adjust_locations(i, n_variants) * shrinkage,
            y_values,
            color=color,
            label=labels_mapping.get(v, v),
            width=width * shrinkage,
            alpha=0.8,
        )
    ### adding "Expected" to legend better last, but the bars is already in the background, adding fake 0-height bars
    # ax.bar(
    #     x_locations, np.zeros(len(x_locations)), label="Expected",
    #     color="black", alpha=0.1, hatch="/", edgecolor="black", width=0.9,
    # )

    fontsize = 10
    ax.set_ylabel("Share", fontsize=fontsize)
    ax.set_xlabel("Value", fontsize=fontsize)
    ax.set_title(title or "Distribution", fontsize=fontsize)

    x_locations, x_labels = locations_data.values, locations_data.index
    ax.set_xticks(x_locations)
    ax.set_xticklabels(x_labels, rotation=xticksrotation)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_xlim((-1, x_locations[-1] + 1))

    # to put some additional statistics in the legend maybe
    legend_str = legend_str or ""
    place_legend(legend_str, ax)

    plt.tight_layout()
    plt.close()

    return fig, axes
