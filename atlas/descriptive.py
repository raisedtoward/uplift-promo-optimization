from typing import Any, List, Dict, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels
import statsmodels.nonparametric
import statsmodels.nonparametric.bandwidths
import itertools
from tqdm.auto import tqdm

from pulsar.visualization.dynamic import get_distribution_dynamic

from .utils import display


def get_null_masks(x: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    isnull_mask = x.isnull()
    none_mask = x.apply(lambda v: v is None)
    npnan_mask = ~none_mask & isnull_mask
    return isnull_mask, none_mask, npnan_mask


def none_to_nan(x: pd.Series) -> pd.Series:
    x = x.copy(deep=True)
    _, none_mask, _ = get_null_masks(x)
    x.loc[none_mask] = np.NaN
    return x


def nan_to_none(x: pd.Series) -> pd.Series:
    x = x.copy(deep=True)
    _, _, npnan_mask = get_null_masks(x)
    x.loc[npnan_mask] = None
    return x


def describe_categorical_feature(
    s: pd.Series,
    t: pd.Series = None,
    normalize_palette: bool = True,
    figsize: Tuple[int, int] = (8, 4),
) -> Dict[str, Any]:

    def _parse_xticklabels(tls):
        return [v.get_position()[0] for v in tls], [v.get_text() for v in tls]

    vc, vc_na = s.value_counts(dropna = True), s.value_counts(dropna = False)
    vc_na.index = list(map(str, vc_na.index))
    most_popular_na = vc_na.index[0]
    pc_na = vc_na / vc_na.sum()
    if len(vc):
        most_popular = vc.index[0]
        # pc = vc / vc.sum()
    else:
        most_popular = most_popular_na
        # pc = pc_na

    fig, ax = plt.subplots(1, 1, figsize = figsize)
    if t is not None:
        gb = pd.concat([s.astype(str), t], axis = 1).groupby(s.name).agg({t.name: 'mean', })
        gb = pd.concat([pc_na.rename('category share'), gb], axis = 1).reset_index().rename({'index': s.name}, axis = 1)
        values = gb[t.name]
        palette = sns.cubehelix_palette(n_colors = 10, start=.8, rot=-.5, dark = 0.25, as_cmap=True)  # n_colors ignored
        if normalize_palette:
            norm = plt.Normalize(vmin = values.min(), vmax = values.max())
        else:
            norm = plt.Normalize(vmin = t.min(), vmax = t.max())
        sm = plt.cm.ScalarMappable(cmap = palette, norm = norm)
        cl_bar = ax.figure.colorbar(sm)
        cl_bar.ax.set_ylabel(t.name, rotation = 90, labelpad = 10, fontsize=10)
        cl_bar.ax.tick_params(labelsize=8)
        colors = dict(zip(gb[s.name], map(lambda x: sm.to_rgba(x), values)))
        sns.barplot(
            data = gb, x = s.name, y = 'category share', hue = s.name, palette = colors,
            dodge = False, orient = 'vertical', ax = ax,
        )
        ax.get_legend().remove()
    else:
        ax.bar(pc_na.index, pc_na, color = 'lightblue')
        ax.set_ylabel('category share')

    plt.draw()
    xticks, xticklabels = _parse_xticklabels(ax.get_xticklabels())
    ax.set_xticks(xticks, xticklabels, rotation = 45)
    ax.set_xlabel(None)
    ax.set_title(s.name)
    plt.tight_layout()
    plt.close()

    return {
        'n_categories': len(vc),
        'n_categories_total': len(vc_na),
        'most_popular': most_popular,
        'most_popular_total': most_popular_na,
        'figure': fig,
    }


def describe_categorical_features(
    data: pd.DataFrame,
    cat_features: List[str] = None,
    target_col: str = None,
    normalize_palette: bool = True,
    figsize: Tuple[int, int] = (8, 4),
    cardinality_threshold = 10,
) -> Dict[str, Any]:

    if not cat_features:
        cat_features = data.columns
    if target_col:
        t = data[target_col]
    else:
        t = None
    cfd = {}
    mixed_types_features = []
    float_convertible_features, float_nonconvertible_features = [], []
    for f in tqdm(cat_features):
        isnull_mask, none_mask, npnan_mask = get_null_masks(data[f])
        if npnan_mask.sum() and none_mask.sum():
            print(f"feature '{f}' has None and np.NaN values")
            mixed_types_features.append(f)
            try:
                _ = data[f].astype(float)
                print(f"feature '{f}' might be converted to float")
                print(f"replacing None for np.NaN in the analysis")
                s = none_to_nan(data[f])
                float_convertible_features.append(f)
            except ValueError:
                print(f"feature '{f}' cannot be converted to float")
                print(f"replacing np.NaN for None in the analysis")
                s = nan_to_none(data[f])
                float_nonconvertible_features.append(f)
        else:
            s = data[f].copy(deep=True)
        cfd[f] = describe_categorical_feature(s, t = t, normalize_palette = normalize_palette, figsize = figsize)

    if mixed_types_features:
        print("mixed types features:", end = '\n\n')
        display(data[mixed_types_features].describe(include = 'all'))

    high_cardinality_features = [f for f in cat_features if cfd[f]['n_categories_total'] > cardinality_threshold]
    low_cardinality_features = [f for f in cat_features if f not in high_cardinality_features]

    return {
        'descriptions': cfd,
        'mixed_types_features': mixed_types_features,
        'float_convertible_features': float_convertible_features,
        'float_nonconvertible_features': float_nonconvertible_features,
        'low_cardinality_features': low_cardinality_features,
        'high_cardinality_features': high_cardinality_features,
    }


def visualize_categorical_pairwise(
    data: pd.DataFrame,
    cat_features: List[str] = None,
    target_col: str = None,
    figsize: Tuple[int, int] = (10, 5),
    plot_type: str = 'heatmap',
) -> Dict[Tuple[str, str], plt.Figure]:

    figures = {}
    n = len(cat_features)
    iterator = tqdm(itertools.combinations(cat_features, 2), total = int(n * (n - 1) / 2))
    if plot_type == 'heatmap':
        for f1, f2 in iterator:
            x, y = data[f1].astype(str), data[f2].astype(str)  # to avoid weird things with nans/Nones
            cross_tab_freq = pd.crosstab(x, y)  # index, columns
            if target_col:
                cross_tab_target = pd.crosstab(x, y, data[target_col], aggfunc=np.mean)

            fig, ax = plt.subplots(1, 2 if target_col else 1, figsize = figsize)
            sns.heatmap(
                cross_tab_freq, mask = cross_tab_freq <= 0,
                cmap=sns.light_palette('seagreen'), ax = ax[0] if target_col else ax,
                xticklabels=False, yticklabels=False,
            )
            if target_col:
                sns.heatmap(
                    cross_tab_target, mask = cross_tab_target.isna(),
                    cmap=sns.cubehelix_palette(start=.8, rot=-.5, as_cmap=True), ax=ax[1],
                    xticklabels=False, yticklabels=False,
                )
                ax[0].set_title("Observations")
                ax[1].set_title("Mean Target")
            else:
                ax.set_title("Observations")
            plt.tight_layout()
            plt.close()
            figures[(f1, f2)] = fig
    elif plot_type == 'scatter':
        def figsize_to_ha(figsize):
            h = figsize[1]
            a = figsize[0] / h
            return h, a
        height, aspect = figsize_to_ha(figsize)
        for f1, f2 in iterator:
            cols = [f1, f2]
            if target_col:
                cols += [target_col]
            data_cat = data[cols].astype(str)
            g = sns.catplot(
                data = data_cat,
                x = f1, y = f2,
                hue = target_col,
                palette = sns.diverging_palette(240, 20, n = 2, l = 60), kind = 'strip',
                height = height, aspect = aspect
            )
            g.set_xticklabels(rotation = 90, fontsize = 10)
            g.set_xlabels(fontsize = 15)
            g.set_yticklabels(rotation = 0, fontsize = 10)
            g.set_ylabels(fontsize = 15)
            plt.close()
            figures[(f1, f2)] = g.ax.figure
    else:
        raise ValueError(f"unknown 'plot_type': '{plot_type}', use 'heatmap' or 'scatter'")
    return figures


def plot_distributions(
    data_list: List[Union[pd.DataFrame, pd.Series]],
    labels: List[str] = None,
    numeric_features: List[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    fixed_color: str = None,
    use_single_axis: bool = False,
    iterate_by_features: bool = False,
    xlim: Tuple[float, float] = None,
    verbose: bool = False,
) -> Union[plt.Figure, Dict[str, plt.Figure]]:

    plt.style.use('fivethirtyeight')
    selected_colors = [
        'springgreen', 'lightskyblue', 'mediumorchid', 'gold', 'salmon', 'darkorchid',
        'crimson', 'dodgerblue', 'forestgreen', 'navy'
    ]
    selected_colors = [mcolors.CSS4_COLORS.get(col) for col in selected_colors]
    n_col = len(selected_colors)

    if labels is None:
        labels = ['data_' + str(i + 1) for i in range(len(data_list))]

    if not use_single_axis:
        figures = {}

    if numeric_features is not None:
        if use_single_axis:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            title = ''
        if verbose:
            iterator = tqdm(enumerate(numeric_features), total = len(numeric_features))
        else:
            iterator = enumerate(numeric_features)

        for j, f in iterator:
            ### every feature on its own figure
            if not use_single_axis:
                fig, ax = plt.subplots(1, 1, figsize = figsize)
                title = f
            for i, df in enumerate(data_list):
                bw = statsmodels.nonparametric.bandwidths.bw_scott(df[f])
                if iterate_by_features:
                    label = labels[i]  # not j
                    if use_single_axis:
                        color = selected_colors[j % n_col]
                    else:
                        color = selected_colors[i % n_col]
                else:
                    label = labels[i]
                    color = selected_colors[i % n_col]
                kde_kws = {}
                if bw == 0:
                    bw = 0.9 * np.std(df[f]) * len(df[f]) ** (-1 / 5)
                    kde_kws['bw'] = bw
                if fixed_color:
                    color = fixed_color
                sns.distplot(df[f], kde_kws = kde_kws, ax = ax, label = label, color = color)

            plt.legend(loc = "upper right")
            plt.title(title + ' distribution')
            plt.grid(True)

            if not use_single_axis:
                plt.ioff()
                plt.close(fig)
                figures[f] = fig
    else:
        use_single_axis = True
        fig, ax = plt.subplots(1, 1, figsize = figsize)
        if verbose:
            iterator = tqdm(enumerate(data_list), total = len(data_list))
        else:
            iterator = enumerate(data_list)

        for i, df in iterator:
            color = selected_colors[i % n_col]
            bw = statsmodels.nonparametric.bandwidths.bw_scott(df)
            kde_kws = {}
            if bw == 0:
                bw = 0.9 * np.std(df) * len(df) ** (-1 / 5)
                kde_kws['bw'] = bw
            if fixed_color:
                color = fixed_color
            sns.distplot(df, kde_kws = kde_kws, ax = ax, label = labels[i], color = color)

        plt.legend(loc = "upper right")
        plt.grid(True)
        plt.ioff()
        plt.close(fig)
    if xlim is not None:
        plt.xlim(xlim)
    # plt.tight_layout()

    if use_single_axis:
        return fig
    else:
        return figures


def seek_nan(
    data: pd.DataFrame,
    missings_threshold: float = 0.99,
) -> Tuple[pd.DataFrame, List, List]:

    nan_rates = data.isnull().mean()
    nan_stats = pd.DataFrame(np.array(nan_rates).reshape(1, -1), columns = data.columns, index = ['missings_rate'])
    remove_nan_columns = list(nan_rates[nan_rates > missings_threshold].index)
    fill_nan_columns = list(nan_rates[(nan_rates <= missings_threshold) & (nan_rates > 0)].index)
    return nan_stats, remove_nan_columns, fill_nan_columns


def seek_types(dataframe) -> Tuple[List, ...]:
    """ 
    returns cat_features, bool_features, float_features, int_features, date_features, other_features
    """

    # to add try except on conversion objects to numerics (for example if decimal was set incorrect) and to dates

    def _get_global_type(t):
        if 'obj' in str(t):
            return 'object'
        elif 'float' in str(t):
            return 'float'
        elif 'int' in str(t):
            return 'int'
        elif 'bool' in str(t):
            return 'bool'
        elif 'date' in str(t):
            return 'date'
        else:
            return 'other'

    found_types = dataframe.dtypes.apply(_get_global_type).reset_index().groupby(0).agg(lambda x: list(x))
    found_types = {k: v for k, v in zip(found_types.index, found_types['index'])}

    cat_features = found_types.get('object', [])
    bool_features = found_types.get('bool', [])
    int_features = found_types.get('int', [])
    float_features = found_types.get('float', [])
    date_features = found_types.get('date', [])
    other_features = found_types.get('other', [])

    return cat_features, bool_features, int_features, float_features, date_features, other_features


def seek_categorical_from_integer_features(
    data: pd.DataFrame,
    int_features: List = None,
    map_binary_to_cat: bool = True,
    n_threshold_low: int = 20,
    n_threshold_high: int = 100,
) -> List[str]:

    cat_recognized = []

    if int_features is None:
        int_features = list(data.columns)

    for f in tqdm(int_features):
        unq = data[f].unique()
        n_unq = len(unq)
        if n_unq > n_threshold_low:
            if sum(unq < 0) == 1 and n_unq <= n_threshold_high:
                ### one specific negative value (-1 or -99 or something else)
                cat_recognized.append(f)
            else:
                ### really just integer feature (age for example) or very high cardinality
                pass
        else:
            if sum(unq > n_unq) >= 10:
                ### clearly doesn't lie in range(N_unq) - may be some city codes or something
                cat_recognized.append(f)
            elif n_unq <= 3:
                ### binary or binary with missings
                if map_binary_to_cat:
                    cat_recognized.append(f)
            else:
                pass
    return cat_recognized


def seek_singletons(data: pd.DataFrame) -> List[str]:
    """
    returns list of columns with only one unique value present (not including missing values)
    """

    unique_counts = data.nunique()
    unique_counts = pd.DataFrame(unique_counts).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
    single_unique = pd.DataFrame(unique_counts[unique_counts['nunique'] <= 1])
    single_unique = list(single_unique['feature'])
    if len(single_unique):
        return single_unique
    else:
        return []


def get_feature_statistics(
    data: pd.DataFrame,
    missings_threshold: float = 0.99,
    verbose: bool = True,
) -> Dict[str, Any]:

    s = data.shape
    cat_features, bool_features, int_features, float_features, date_features, other_features = (
        seek_types(data)
    )
    single_value_features = seek_singletons(data)
    nan_stats, remove_nan_columns, fill_nan_columns = seek_nan(data, missings_threshold = missings_threshold)

    if verbose:
        print(f'{s[0]} observations, {s[1]} columns', end = '\n\n')
        print(f'{data.index.nunique()} unique values in index {data.index}', end = '\n\n')
        print(f'columns where only one value was met are: {single_value_features}', end = '\n\n')
        print(f'almost empty columns which might be dropped: {remove_nan_columns}', end = '\n\n')
        print(f'columns where nans might be filled are: {fill_nan_columns}', end = '\n\n')
        print(f"float_features: {float_features}", end = '\n\n')
        print(f"int_features: {int_features}", end = '\n\n')
        print(f"bool_features: {bool_features}", end = '\n\n')
        print(f"cat_features: {cat_features}", end = '\n\n')
        print(f"date_features: {date_features}", end = '\n\n')
        print(f"other_features: {other_features}", end = '\n\n')

    numeric_description, integers_descriptions, categorical_descriptions = None, None, None

    if float_features + int_features:
        numeric_description = data[float_features + int_features].describe(include = 'all')
        if verbose:
            print('description of numeric features:', end = '\n\n')
            display(numeric_description)
    if int_features:
        integers_descriptions = data[int_features].astype('object').describe(include = 'all')
        if verbose:
            print('description of integer features as categorical:', end = '\n\n')
            display(integers_descriptions)
    if cat_features:
        categorical_descriptions = data[cat_features].describe(include = 'all')
        if verbose:
            print('description of categorical features:', end = '\n\n')
            display(categorical_descriptions)

    return {
        'categorical': cat_features, 'float': float_features, 'integer': int_features,
        'date': date_features, 'other': other_features, 'bool': bool_features,
        'singletons': single_value_features,
        'nan_remove': remove_nan_columns, 'nan_fill': fill_nan_columns,
        'stats_missings': nan_stats,
        'stats_numeric': numeric_description, 'stats_integer': integers_descriptions,
        'stats_categorical': categorical_descriptions
    }


def plot_palettes(n_colors: int = 6) -> None:

    for p in [
        'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
        'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
        'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
        'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',
        'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',
        'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
        'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
        'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r',
        'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
        'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
        'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r',
        'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r',
        'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r',
        'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r',
        'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r',
        'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r',
        'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r',
        'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r',
        'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
        'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
    ]:
        print(p)
        display(sns.color_palette(p, n_colors = n_colors))


def plot_colortable(
    colors: Dict[str, str] = mcolors.CSS4_COLORS,
    title: str = '',
    sort_colors: bool = True,
    emptycols: int = 0
) -> plt.Figure:

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40
    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted(
            (tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
            for name, color in colors.items()
        )
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)
    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)
    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72
    fig, ax = plt.subplots(figsize = (width / dpi, height / dpi), dpi = dpi)
    fig.subplots_adjust(margin / width, margin / height, (width - margin) / width, (height - topmargin) / height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize = 24, loc = "left", pad = 10)
    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height
        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7
        ax.text(text_pos_x, y, name, fontsize = 14, horizontalalignment = 'left', verticalalignment = 'center')
        ax.hlines(y, swatch_start_x, swatch_end_x, color = colors[name], linewidth = 18)
    plt.close()
    return fig


def check_data_overlap(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    key_fields: List[str],
    name1: str,
    name2: str
):
    composite_key_1 = data1[key_fields].copy()
    composite_key_2 = data2[key_fields].copy()
    composite_key_1[name1] = data1.index
    composite_key_2[name2] = data2.index

    overlap = pd.merge(left = composite_key_1, right = composite_key_2, on = key_fields, how = 'inner')
    n = len(overlap[key_fields].drop_duplicates())
    return {
        'n': n,
        'overlap': overlap,
        'summary': f"there are {n} unique data points overlap between {name1} and {name2} by {key_fields}"
    }


def remove_data_overlap(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    key_fields: List[str],
    drop: str = 'second'
):
    """ drop is either 'first', 'second' or 'both' """
    overlap = check_data_overlap(data1, data2, key_fields, 'first', 'second')['overlap']
    if drop == 'first':
        return data1.drop(overlap['first'].unique(), axis = 0), data2
    elif drop == 'second':
        return data1, data2.drop(overlap['second'].unique(), axis = 0)
    else:
        return data1.drop(overlap['first'].unique(), axis = 0), data2.drop(overlap['second'].unique(), axis = 0)


def get_distribution_dynamic_advanced(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    feature_col: Union[str, List[str]],
    timestamp_col: str,
    quantiles: Union[int, List, np.array] = 20,
    grouping: bool = False,
    freq: str = 'M',
    timestamp_format: str = '%Y-%m-%d',
    figsize = (15, 8), xtickssize = 10, xticksrotation = 90,
    markersize = 10,
):

    if isinstance(data, pd.DataFrame):
        data = data
    elif isinstance(data, list):
        data = pd.concat(data, axis = 0)
    else:
        raise ValueError(f"unknown 'data' type: {type(data)}")
    if isinstance(feature_col, str):
        feature_name = [feature_col]
    elif isinstance(feature_col, (list, tuple)):
        pass
    else:
        raise ValueError(f"unknown 'feature_col' type: {type(feature_col)}")

    return {
        f: get_distribution_dynamic(
            data = data,
            metric_col = f,
            timestamp_col = timestamp_col,
            quantiles = quantiles,
            grouping = grouping,
            freq = freq,
            timestamp_format = timestamp_format,
            figsize = figsize, xtickssize = xtickssize, xticksrotation = xticksrotation,
            markersize = markersize,
        ) for f in tqdm(feature_col)
    }
