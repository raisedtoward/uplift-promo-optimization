from typing import Dict, List, Tuple, Union
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import itertools
from functools import partial
from tqdm.auto import tqdm
import scipy
import scipy.stats
import statsmodels.api as sm


from pulsar.visualization.constants import DEFAULT_PLOT_STYLE, DEFAULT_COLOR, DIST_KWS


SCIPY_DISTRIBUTIONS_CONTINUOUS = tuple(
    nd
    for nd in dir(scipy.stats.distributions)
    ### removing non-distribution objects
    if not nd.startswith("_")
    and not nd.endswith("_gen")
    and nd
    not in (
        "entropy",
        "randint",
        "rv_discrete",
        "rv_frozen",
        "rv_continuous",
        "rv_histogram",
        "studentized_range",
    )
    ### removing discrete distributions since they're only available in scipy 1.9.0
    and not isinstance(
        getattr(scipy.stats.distributions, nd), scipy.stats.distributions.rv_discrete
    )
)


def check_distr(
    feature: Union[np.ndarray, pd.Series, List],
    lower_bound: float = None,
    upper_bound: float = None,
    sample_size: int = None,
    seed: int = None,
    frequency: Union[bool, str] = False,
    ax: matplotlib.axes.Axes = None,
    figsize: Tuple[int, int] = (8, 5),
    color: str = None,
    alpha: float = 0.5,
    label: str = None,
    q_round: int = 4,
    show: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Provides some statistics and visualization about the distribution of a feature.

    :param feature: The feature to check the distribution of.
    :param lower_bound: The lower bound below which distribution should not be under by assumption (optional).
    :param upper_bound: The upper bound above which distribution should not be under by assumption (optional).
    :param sample_size: The size of the sample to use (optional).
    :param seed: The seed value for random number generation (optional).
    :param frequency: Flag indicating whether to show frequency values on histogram instead of density (default: False).
        Could be of string type - then directly substitutes 'stat' parameter in seaborn.histplot.
    :param ax: The axes object to plot histogram on (optional).
    :param figsize: The size of the figure in inches (default: (8, 5)).
    :param color: The color to use for histogram plotting.
    :param alpha: The color intensity to use for histogram plotting.
    :param label: The label to use on the histogram legend.
    :param q_round: The number of decimal places to round the quantile values (default: 4).
    :param show: Flag indicating whether to display the plot (default: False).
    :param verbose: Flag indicating whether to display additional information (default: True).
    :return: Dict
    """
    if not isinstance(feature, pd.Series):
        feature = pd.Series(feature)

    size = len(feature)
    min_v, max_v, mean_v, std_v = np.min(feature), np.max(feature), np.mean(feature), np.std(feature, ddof = 1)
    missings = feature.isna().sum()
    nunique = len(np.unique(feature))
    if verbose:
        print(f"size: {size}")
        print(f"min: {min_v}, max: {max_v}, mean: {mean_v}, std: {std_v}")
        print(f"# missings: {missings}, # unique: {nunique}")

    lower, upper = None, None
    if lower_bound is not None:
        lower = (feature < lower_bound).sum()
        if verbose:
            print(f"# less than lower bound ({lower_bound}): {lower}")
    if upper_bound is not None:
        upper = (feature > upper_bound).sum()
        if verbose:
            print(f"# higher than upper bound ({upper_bound}): {upper}")

    q_grid = np.arange(1.01, step=0.1)
    q = np.nanquantile(feature, q=q_grid)
    if missings == size:
        q = [np.nan] * len(q_grid)

    if seed is not None:
        np.random.seed(seed)
    if sample_size is not None and sample_size < size:
        sampled = np.random.choice(feature, size=sample_size, replace=False)
    else:
        sampled = feature

    plt.style.use(DEFAULT_PLOT_STYLE)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None

    _clr_kws = {"color": DEFAULT_COLOR if color is None else color}
    if isinstance(frequency, str):
        _kws = {"stat": frequency}
    else:
        _kws = {"stat": "count"} if frequency else {}
    if label is not None:
        _kws["label"] = label
    if alpha is not None:
        _kws["alpha"] = alpha
    sns.histplot(sampled, ax=ax, **{**DIST_KWS, **_kws, **_clr_kws})
    if label is not None:
        ax.legend()

    ax.grid(True)
    ax.set_title(f"Qs: {[round(x, q_round) for x in q]}", fontsize=10)
    if not show and fig is not None:
        plt.close()

    result = {
        "size": size,
        "missing": missings,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "lower": lower,
        "upper": upper,
        "figure": fig,
    }

    return result


class EmpiricalDistribution:
    def __init__(self, x: np.ndarray, ecdf_side: str = "right"):
        """
        ecdf_side = 'left' means that F(x) = P(X < x)
        ecdf_side = 'right' means that F(X) = P(X <= x)
        """
        self.ecdf_side = ecdf_side
        self.x_sorted = np.array(sorted(x))
        self.ecdf = sm.distributions.ECDF(self.x_sorted, side=ecdf_side)

    def rvs(
        self,
        size: int,
        quantile_lower: float = 0.0,
        quantile_upper: float = 1.0,
        seed: int = None
    ) -> np.ndarray:
        """
        Generates random variates from the empirical distribution.

        :param size: The number of random variates to generate.
        :param quantile_lower: The lower quantile limit for generating random variates (default: 0.0).
        :param quantile_upper: The upper quantile limit for generating random variates (default: 1.0).
        :param seed: The seed value for random number generation (optional).
        :return: Random variates from the empirical distribution as a NumPy array.
        :rtype: np.ndarray
        """
        if seed is not None:
            np.random.seed(seed)

        x_lower = self.quantile(np.clip(quantile_lower, 0, 1))
        x_upper = self.quantile(np.clip(quantile_upper, 0, 1))
        x = self.x_sorted[(self.x_sorted >= x_lower) & (self.x_sorted <= x_upper)]

        return np.random.choice(x, size=size, replace=True)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the cumulative distribution function (CDF) of the empirical distribution.

        :param x: The input values for which to calculate the CDF.
        :return: The CDF values corresponding to the input values.
        """
        return self.ecdf(x)

    def sf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the survival function (SF(x) == 1 - CDF(x)) of the empirical distribution.

        :param x: The input values for which to calculate the SF.
        :return: The SF values corresponding to the input values.
        """
        return 1 - self.ecdf(x)

    def ppf(self, q: np.ndarray, interpolation: str = "linear") -> np.ndarray:
        """
        Calculates the percent point function (x = PPF(q) <==> q = CDF(x)) of the empirical distribution.

        :param q: The quantile values for which to calculate the PPF.
        :param interpolation: The interpolation method to use (default: "linear").
        :return: The PPF values corresponding to the input quantile values.
        """
        return np.quantile(self.x_sorted, q, method=interpolation)

    def quantile(self, q: np.ndarray, interpolation: str = "linear") -> np.ndarray:
        """
        Calculates the quantile function (percent point function) of the empirical distribution.
        Alias for .ppf

        :param q: The quantile values for which to calculate the PPF.
        :param interpolation: The interpolation method to use (default: "linear").
        :return: The PPF values corresponding to the input quantile values.
        """
        return self.ppf(q, interpolation=interpolation)

    def plot_cdf(
        self,
        ax: matplotlib.axes.Axes = None,
        label: str = None,
        color: str = None,
        title: str = None,
        xlabel: str = None,
        linewidth: int = 3,
        show_quantiles: bool = False,
        scatter: bool = False,
        figsize: Tuple[int] = (10, 6),
    ) -> plt.Figure:
        """
        Plots the cumulative distribution function (CDF) of the empirical distribution.

        :param ax: The matplotlib axes object to plot on (optional).
        :param label: The label for the plot legend (optional).
        :param color: The color of the plot line (optional).
        :param title: The title of the plot (optional).
        :param xlabel: The label for the x-axis (optional).
        :param linewidth: The linewidth of the plot line (default: 3).
        :param show_quantiles: Flag indicating whether to show vertical lines for quantiles (default: False).
        :param scatter: Flag indicating whether to plot cdf as a scatter plot to distinguish breakpoints (default: False).
        :param figsize: The size of the figure in inches (default: (10, 6)).
        :return: None
        """

        plt.style.use(DEFAULT_PLOT_STYLE)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None

        x_sorted = list(self.x_sorted)
        y_sorted = list((self.ecdf(self.x_sorted)))
        if scatter:
            ax.scatter(
                x_sorted,
                y_sorted,
                color='k',#DEFAULT_COLOR if not color else color,
                marker="o",
                s=10,
            )

        if self.ecdf_side == "left":
            x_steps = (
                [2 * x_sorted[0] - x_sorted[1]]
                + x_sorted
                + [x_sorted[-1], 2 * x_sorted[-1] - x_sorted[-2]]
            )
            y_steps = [0] + y_sorted + [1, 1]
            step_where = "pre"
            ### - 'pre': The y value is continued constantly to the left from
            ### every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the value ``y[i]``.

        elif self.ecdf_side == "right":
            x_steps = (
                [2 * x_sorted[0] - x_sorted[1], x_sorted[0]]
                + x_sorted
                + [2 * x_sorted[-1] - x_sorted[-2]]
            )
            y_steps = [0, 0] + y_sorted + [1]
            step_where = "post"
            ### - 'post': The y value is continued constantly to the right from
            ### every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the value ``y[i]``.
        else:
            raise ValueError(f"unknown ecdf_side value: {self.ecdf_side}")

        ax.step(
            x_steps,
            y_steps,
            label="ecdf" if not label else label,
            color=DEFAULT_COLOR if not color else color,
            linewidth=linewidth,
            where=step_where,
        )

        if show_quantiles:
            for q in np.linspace(0, 1, 11):
                ax.vlines(
                    x=self.ppf(q),
                    ymin=0,
                    ymax=q,
                    linestyle="--",
                    linewidth=1,
                    color="k",
                )

        ax.legend(loc="best")
        yspace = np.linspace(0, 1, 21)
        ax.set_yticks(yspace)
        ax.set_yticklabels([f"{t:.2f}" for t in yspace], fontsize=8)
        xspace = np.linspace(min(x_sorted), max(x_sorted), 11)
        ax.set_xticks(xspace)
        ax.set_xticklabels([f"{t:.2f}" for t in xspace], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Cumulative Distribution Function")
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)

        plt.ioff()
        plt.close(fig)
        return fig


def fit_distribution(
    x: Union[np.ndarray, pd.Series, List],
    empirical: bool = False,
    distr_names: List[str] = None,
    plot_all: Union[bool, str] = True,
    verbose: int = 2,
    alpha: float = 0.005,
    sample_size: int = None,
    seed: int = None
) -> Dict:
    """
    Fits probability distributions to the input data.

    :param x: The input data as a NumPy array, pandas Series, or list.
    :param empirical: Flag indicating whether to fit empirical distribution or analytical(default: False).
    :param distr_names: List of distribution names to fit (optional).
    :param plot_all: Whether to generate distribution plots for all fitted distributions or only the best one.
    :param verbose: Verbosity level, higher value means more verbosity (default: 2).
    :param alpha: Significance level for goodness-of-fit test (default: 0.005).
    :param sample_size: Size of the random sample to use for fitting (optional).
    :param seed: Seed value for random number generation (optional).
    :return: A dictionary containing the fitted distribution object, the name of the distribution.
        Also figure with plot and DataFrame with fit statistics in case of non-empirical distribution fit.
    :rtype: dict
    """
    if distr_names is None:
        distr_names = [
            "norm",
            "t",
            "skewnorm",
            "cauchy",
            "gengamma",
            "lognorm",
            "expon",
        ]

    def ks_test(
        x,
        distr,
    ):
        # ks_stat, p_value = scipy.stats.kstest(rvs = x, cdf = distr.cdf, alternative = 'two-sided', N = len(x))
        KS_TEST_SIZE = 1000
        ks_stat, p_value = scipy.stats.kstest(
            rvs=np.random.choice(x, size=min(len(x), KS_TEST_SIZE), replace=False),
            cdf=distr.rvs(size=KS_TEST_SIZE),
            alternative="two-sided",
        )
        return p_value

    if not empirical:
        fitted_distributions, fitted_names, loglikelyhoods, ks_pvalues, fit_times = (
            [],
            [],
            [],
            [],
            [],
        )
        if sample_size is not None:
            if len(x) > sample_size:
                if seed is not None:
                    np.random.seed(seed)
                x_fit = np.random.choice(x, size=sample_size, replace=False)
            else:
                x_fit = x
        else:
            x_fit = x

        for name in tqdm(distr_names):
            distr = getattr(scipy.stats.distributions, name)
            if not hasattr(distr, "fit"):
                if verbose:
                    print(f"{name} skipping")
                continue
            try:
                start_time = datetime.datetime.now()
                if verbose > 1:
                    print(f"{name} starting: {start_time}")
                params = distr.fit(x_fit)
                fit_time = (datetime.datetime.now() - start_time).total_seconds()
                fit_times.append(fit_time)
                if verbose > 1:
                    print(f"{name} fitted in {fit_time} seconds")
                distr = distr(*params)
                ll = distr.logpdf(x_fit).sum()
                # ll = np.sum(np.log(distr.pdf(x)))
                loglikelyhoods.append(ll)
                ks_pvalues.append(
                    ks_test(
                        x_fit,
                        distr,
                    )
                )
                fitted_distributions.append(distr)
                fitted_names.append(name)

            except Exception:
                # raise e
                if verbose:
                    print(f"{name} fit failed")

        fit_stats = pd.DataFrame(
            {
                "name": fitted_names,
                "distribution": fitted_distributions,
                "loglikelyhood": loglikelyhoods,
                "KS_pvalue": ks_pvalues,
                "fit_time": fit_times,
            }
        )
        fit_stats = fit_stats.sort_values(
            by=["KS_pvalue", "loglikelyhood"], ascending=[False, False]
        ).reset_index(drop=True)
        if len(fit_stats):
            # display(fit_stats)
            if alpha:
                good_fits = fit_stats.loc[fit_stats["KS_pvalue"] > alpha, :]
            else:
                good_fits = fit_stats
            if len(good_fits):
                d, name = good_fits[["distribution", "name"]].iloc[0]
            else:
                d, name = None, "NOT_FITTED"
        else:
            d, name = None, "NOT_FITTED"

        fig = None
        if plot_all:
            n = len(fit_stats)
            fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
            for i, row in fit_stats.iterrows():
                name, distr, ll, p_value = (
                    row[k]
                    for k in ("name", "distribution", "loglikelyhood", "KS_pvalue")
                )
                if n > 1:
                    ax = axes[i]
                else:
                    ax = axes
                # print(f'{name} loglikelyhood: {ll:.4f}')
                sns.histplot(x, color=DEFAULT_COLOR, ax=ax, label="true", **DIST_KWS)
                sns.histplot(
                    distr.rvs(size=1000),
                    color="lightblue",
                    ax=ax,
                    label="sampled",
                    **DIST_KWS,
                )
                ax.set_title(
                    f"{name.upper()}, Loglikelyhood: {ll:.4f}, KS_pvalue: {p_value:.4f}",
                    fontsize=10,
                )
                ax.set_xlabel("")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.close()
        else:
            if d is not None:
                fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                sns.histplot(x, color=DEFAULT_COLOR, ax=ax, label="true", **DIST_KWS)
                sns.histplot(
                    d.rvs(size=1000),
                    color="lightblue",
                    ax=ax,
                    label="sampled",
                    **DIST_KWS,
                )
                ll = d.logpdf(x).sum()
                p_value = ks_test(x, d)
                ax.set_title(
                    f"{name.upper()}, Loglikelyhood: {ll:.4f}, KS_pvalue: {p_value:.4f}",
                    fontsize=10,
                )
                ax.set_xlabel("")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.close()

        return {
            "distribution": d,
            "name": name,
            "figure": fig,
            "fit_stats": fit_stats,
        }
    else:
        return {
            "distribution": EmpiricalDistribution(x),
            "name": "empirical",
            "figure": None,
            "fit_stats": None,
        }


def bucketize(
    values: Union[np.ndarray, pd.Series, List],
    n_buckets: int = 20,
    force_even_buckets: bool = False,
    merge_to_right_side: bool = True,
    ecdf: EmpiricalDistribution = None,
    force_consecutive_markers: bool = True,
    start: int = 1,
) -> np.ndarray:
    """
    Bucketizes values into discrete buckets.

    :param values: The input values to be bucketized.
    :param n_buckets: The number of buckets to create (default: 20).
    :param force_even_buckets: Whether to force buckets be approximately of equal size or follow the distribution.
        Former might lead to cases when equal values attributed to different buckets (default: False).
    :param merge_to_right_side: Flag indicating whether to merge values equal to bucket boundaries to the right side
        or to the left (default: True).
    :param ecdf: The empirical cumulative distribution function to use for bucketization (optional).
    :param force_consecutive_markers: Flag indicating whether to force consecutive markers numeration (default: True).
        Takes an effect for some cases where distribution is skewed and some buckets left empty then skipped.
        Setting this flag to True prevents this behaviour.
    :param start: The starting value for bucket enumeration (default: 1).
    :return: Array with numbers of values' corresponding buckets.
    """
    quantile_grid = np.linspace(0, 1, n_buckets + 1)
    x = np.array(values)
    L = len(x)

    if force_even_buckets and ecdf is None:
        argsorted = np.argsort(x)
        pos_in_sorted = np.argsort(argsorted)
        # x_sorted = x[argsorted]
        bucketed_values = np.repeat(
            np.arange(1, n_buckets + 1), np.ceil(L / n_buckets)
        )[:L]
        return bucketed_values[pos_in_sorted]
    else:
        if ecdf is None:
            ecdf = EmpiricalDistribution(
                x, ecdf_side="right" if merge_to_right_side else "left"
            )

        ### not very efficient way, but keep it for now
        def get_bucket(v, bounds):
            for i, (l, r) in enumerate(zip(bounds[:-1], bounds[1:])):
                if merge_to_right_side:
                    if l < v <= r:
                        return i + 1
                else:
                    if l <= v < r:
                        return i + 1
            if v <= bounds[0]:
                return 1
            if v >= bounds[-1]:
                return len(bounds) - 1

        get_quantile_bucket = partial(get_bucket, bounds=quantile_grid)
        bucketed_values = [get_quantile_bucket(v) for v in ecdf.cdf(x)]
        if force_consecutive_markers:
            unique_markers = {
                v: i for i, v in enumerate(sorted(np.unique(bucketed_values)), start)
            }
            bucketed_values = [unique_markers[v] for v in bucketed_values]

        return np.array(bucketed_values)


def custom_bucketize(
    values: Union[np.ndarray, pd.Series, List],
    n_buckets: Union[int, Tuple[int]] = 10,
    bounds: Union[Tuple[float], List[float]] = None,
    eps: float = 1e-5,
    **kwargs,
) -> np.ndarray:
    """
    :param data: Original series of values to be bucketed.
    :param n_buckets: Number of buckets to potentially get within each section divided by bounds.
    :param bounds: Sequence of bounds to create intervals within.
        Each value in a sequence generally used both as left (inclusive) and right (non-inclusive) bound,
        unless falls outside of data distribution.
    :param eps: Addon to make each bound for each of adjoining intervals to be left-inclusive, but right-noninclusive
    :return: Series of the same size as original having bucket id/index for each of original observations.
    """

    if isinstance(values, pd.Series):
        values = values.values
    elif isinstance(values, (list, tuple, np.ndarray)):
        pass
    else:
        raise TypeError("supported types for 'values' argument are pd.Series, list, tuple or np.ndarray")

    bucketed_name = "data_bucketed"
    score_name = 'raw'
    data = pd.DataFrame({score_name: values}, index = range(len(values)))

    if isinstance(n_buckets, int):
        n_buckets = itertools.cycle([n_buckets])

    min_bound, max_bound = data[score_name].min(), data[score_name].max()
    if bounds is None:
        bounds = [-np.inf, np.inf]
    else:
        bounds = sorted(bounds)
        if bounds[0] > min_bound:
            bounds = [min_bound] + bounds
        if bounds[-1] < max_bound:
            bounds = bounds + [max_bound + eps]

    actual_bounds = []
    shift = 0
    for lb, ub, n_b in zip(bounds[:-1], bounds[1:], n_buckets):
        mask = (lb <= data[score_name]) & (data[score_name] < ub)
        subset = data.loc[mask, score_name]
        data.loc[mask, bucketed_name] = bucketize(values = subset, n_buckets = n_b, **kwargs,) + shift
        shift += data.loc[mask, bucketed_name].nunique()
        actual_bounds.append((subset.min(), subset.max()))

    return data[bucketed_name].values


def is_binary_feature(x: pd.Series) -> bool:
    """
    Checks if the input series represents a binary ({0, 1}) feature.

    :param x: The input series to check.
    :return: True if the series represents a binary feature, False otherwise.
    :rtype: bool
    """
    u = sorted(set(pd.Series(x).fillna(-1.0)))
    if len(u) == 2 and all(np.isclose(u, [0.0, 1.0])):
        return True
    return False
