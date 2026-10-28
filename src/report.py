import logging
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use(backend = 'Agg')
import matplotlib.pyplot as plt

from atlas.interfaces import SampleKeeper
from atlas.uplift.modelling import run_uplift_evaluation_tests
from pulsar import check_distr
from pulsar.common.constants import TREATMENT_LABEL
from pulsar.visualization import plot_metric_distribution_by_classes
from pulsar.visualization.constants import VARIANTS_COLORS
from pulsar.visualization.dynamic import get_metrics_dynamic, plot_metrics_dynamic

from src import config as cfg
from src.models import predict_uplift, convert_uplift_to_target_predictions

figures_path = cfg.output_path / 'figures'
eval_sample_key = 'holdout'
train_sample_key = 'train'


def ps538() -> None:
    plt.plot([0], [0])
    plt.close()
    plt.style.use(style = 'fivethirtyeight')
    np.set_printoptions(legacy = '1.25')


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(fname = figures_path / f'{name}.png', dpi = 150, bbox_inches = 'tight')
    plt.close(fig = fig)
    logging.getLogger(cfg.LOGGER_NAME).info(msg = f'saved figure {name}')


def get_modelling_data(samples: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    ### holdout is a subset of oot
    return pd.concat(objs = [samples['train'], samples['test'], samples['oot']], axis = 0)


def plot_conversion_by_voucher_fd(modelling_data: pd.DataFrame) -> None:
    gb = (
        modelling_data
        .assign(**{cfg.treatment_col: modelling_data[cfg.treatment_col].map(func = cfg.labels_mapping)})
        .groupby(by = [cfg.treatment_col, cfg.aux_treatment_col])
        .agg(func = {cfg.target_col: 'mean'})
        .unstack(level = 1)
    )
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    gb.iloc[:, 0].plot(ax = ax, color = 'dodgerblue', marker = 'o', label = 'free_delivery = 0')
    gb.iloc[:, 1].plot(ax = ax, color = 'red', marker = 'o', label = 'free_delivery = 1')
    ax.set_ylabel(ylabel = f'P({cfg.target_col})')
    ax.legend()
    save_figure(fig = fig, name = 'conversion_by_voucher_fd')


def plot_ate_cost_efficiency(analysis_data: pd.DataFrame) -> None:
    gb = (
        analysis_data
        .groupby(by = cfg.treatment_col)
        .agg(func = {cfg.id_col: 'count', cfg.target_col: 'mean', cfg.cost_col: 'sum'})
    )
    gb['avg_cost_per_user'] = gb[cfg.cost_col] / gb[cfg.id_col]
    gb['ATE'] = 0.0
    gb.loc[1:, 'ATE'] = gb.loc[1:, cfg.target_col] - gb.loc[0, cfg.target_col]
    gb['efficiency'] = gb['ATE'] / gb['avg_cost_per_user']
    gb = gb.iloc[1:, :]

    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (10, 6), sharex = True)
    ax[0].plot(gb.index, gb['ATE'], label = 'Average Treatment Effect', color = 'lightblue', marker = 'o')
    ax[1].plot(gb.index, gb['avg_cost_per_user'], label = 'Average Intervention Cost Per Exposed User', color = 'crimson', marker = 'o')
    ax[2].plot(gb.index, gb['efficiency'], label = 'Efficiency: ATE/AVG_COST', color = 'black', marker = 'o')
    ax[0].set_ylabel(ylabel = 'ATE')
    ax[1].set_ylabel(ylabel = 'Cost')
    ax[2].set_ylabel(ylabel = 'Efficiency')
    ax[2].set_xticks(ticks = gb.index)
    ax[2].set_xticklabels(labels = gb.index.map(mapper = cfg.labels_mapping))
    ax[2].set_xlabel(xlabel = 'Voucher, %')
    for a in ax:
        a.legend()
    save_figure(fig = fig, name = 'ate_cost_efficiency')


def plot_target_rate_dynamic(modelling_data: pd.DataFrame) -> None:
    dynamic_aggregates = get_metrics_dynamic(
        data = modelling_data.loc[:, [cfg.time_col, cfg.target_col, cfg.treatment_col]],
        timestamp_col = None,
        group_col = cfg.time_col,
        target_col = cfg.target_col,
        score_col = None,
        treatment_col = cfg.treatment_col,
        weight_col = None,
        control_label = cfg.CONTROL_LABEL,
        metrics = None,
        freq = None,
        timestamp_format = None,
        verbose = False,
    )
    fig, ax = plot_metrics_dynamic(
        aggregates = dynamic_aggregates,
        figsize = (20, 8),
        metrics = [],
        one_axis_per_metric = True,
        n_reserved_axes = 0,
    )
    ax[0].set_title(label = f"Population Weekly, '{cfg.target_col}' Target Rate")
    ax[0].set_xlabel(xlabel = cfg.time_col)
    save_figure(fig = fig, name = 'target_rate_dynamic')


def plot_cate_example(analysis_data: pd.DataFrame) -> None:
    d = pd.DataFrame(data = {
        'orders_90d_clipped': analysis_data['orders_90d'].clip(upper = 8),
        cfg.treatment_col: analysis_data[cfg.treatment_col],
        cfg.target_col: analysis_data[cfg.target_col],
    })
    gb = (
        d
        .groupby(by = ['orders_90d_clipped', cfg.treatment_col])
        .agg(func = {cfg.target_col: 'mean'})
        .unstack(level = 1)
    )
    fig, ax = plt.subplots(figsize = (10, 6))
    for c, t_label in zip(VARIANTS_COLORS, cfg.treatment_labels):
        ax.plot(gb.index, gb.loc[:, (cfg.target_col, t_label)], marker = 'o', color = c, label = f'{cfg.labels_mapping[t_label]}% group')
    ax.set_xticks(ticks = gb.index)
    ax.set_xticklabels(labels = gb.index)
    ax.set_title(label = 'CATE Example')
    ax.set_xlabel(xlabel = gb.index.name)
    ax.set_ylabel(ylabel = 'Conversion rate')
    ax.legend()
    save_figure(fig = fig, name = 'cate_example')


def plot_score_distributions(test_data: pd.DataFrame, fitted_learners: Dict) -> None:
    test_sample_key = 'test'
    for learner_type, learner in fitted_learners.items():
        uplift_predictions, _ = predict_uplift(learner = learner, data = test_data)
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 5))
        for color, t_label in zip(VARIANTS_COLORS, cfg.treatment_labels):
            if t_label == cfg.CONTROL_LABEL:
                continue
            score_col = f'{learner_type}_uplift_{t_label}'
            check_distr(feature = uplift_predictions[f'uplift_{t_label}'].rename(index = score_col), label = f'{test_sample_key} {score_col}', ax = ax, color = color, verbose = False)
        save_figure(fig = fig, name = f'score_distributions_{learner_type}')


def get_sample_keeper(samples: Dict[str, pd.DataFrame], learner, learner_type: str) -> SampleKeeper:
    s_keeper = SampleKeeper()
    for k in [train_sample_key, 'test', 'oot', 'holdout']:
        data = samples[k].copy()
        uplift_predictions, _ = predict_uplift(learner = learner, data = data)
        uplift_predictions.columns = [f'{learner_type}_{c}' for c in uplift_predictions.columns]
        s_keeper[k] = pd.concat(objs = [data, uplift_predictions], axis = 1)
    s_keeper.set_markup(
        target_name = cfg.target_col,
        treatment_name = cfg.treatment_col,
        date_main_field = cfg.time_col,
        id_field = cfg.id_col,
        all_features = cfg.uplift_features,
        cat_features = cfg.cat_encoded_features,
    )
    return s_keeper


def plot_uplift_evaluation(samples: Dict[str, pd.DataFrame], learner, learner_type: str) -> None:
    """
    the same evaluation as in the notebook: (voucher level t vs control) subsets, treatment temporarily remapped to 0/1
    """
    s_keeper = get_sample_keeper(samples = samples, learner = learner, learner_type = learner_type)
    evaluation_results = {}
    for t_label in cfg.treatment_labels:
        if t_label == cfg.CONTROL_LABEL:
            continue
        filter_query = f'{cfg.treatment_col} == {t_label} or {cfg.treatment_col} == {cfg.CONTROL_LABEL}'
        score_col = f'{learner_type}_uplift_{t_label}'
        temp_treatment_col = '__temp_treatment'
        for k in s_keeper.active:
            s_keeper[k][temp_treatment_col] = s_keeper[k][cfg.treatment_col]
            s_keeper[k].loc[s_keeper[k][cfg.treatment_col] == t_label, temp_treatment_col] = TREATMENT_LABEL
        s_keeper.add_filter(filter_query = filter_query)
        s_keeper.set_markup(treatment_name = temp_treatment_col)
        try:
            evaluation_results[t_label] = run_uplift_evaluation_tests(
                sample_keeper = s_keeper,
                score_name = score_col,
                base_sample_key = train_sample_key,
                k_target_threshold = 20,
                percentile_chart_n_buckets = 10,
                calibration_n_buckets = 10,
                verbose = False,
            )
        finally:
            for k in s_keeper.active:
                s_keeper[k].drop(columns = [temp_treatment_col], inplace = True)
            s_keeper.disable_filter(filter = filter_query)
            s_keeper.set_markup(treatment_name = cfg.treatment_col)

    first_label, last_label = cfg.treatment_labels[1], cfg.treatment_labels[-1]
    save_figure(
        fig = evaluation_results[first_label]['lift_chart']['results'][eval_sample_key]['figure'],
        name = f'{learner_type.lower()}_lift_curve_{cfg.labels_mapping[first_label]}',
    )
    for t_label, results in evaluation_results.items():
        save_figure(fig = results['qini']['figure'], name = f'{learner_type.lower()}_qini_curve_{cfg.labels_mapping[t_label]}')
    save_figure(
        fig = evaluation_results[last_label]['percentile_uplift_chart']['results'][eval_sample_key]['figure'],
        name = f'{learner_type.lower()}_percentile_uplift_{cfg.labels_mapping[last_label]}',
    )
    save_figure(
        fig = evaluation_results[last_label]['calibration']['results'][eval_sample_key]['figure'],
        name = f'{learner_type.lower()}_calibration_{cfg.labels_mapping[last_label]}',
    )


def plot_utilization_rates(analysis_data: pd.DataFrame) -> None:
    plots = {
        'utilization_rate_given_order': (analysis_data.loc[analysis_data[cfg.target_col] == 1, :], 'P(voucher_used = 1|Ordered = 1, D = d, FD = d)'),
        'utilization_rate': (analysis_data, 'P(voucher_used = 1|D = d, FD = d)'),
    }
    for name, (data, title) in plots.items():
        gb = data.groupby(by = [cfg.treatment_col, cfg.aux_treatment_col]).agg(func = {cfg.utilization_col: 'mean'}).unstack(level = -1)
        gb.columns = gb.columns.get_level_values(level = 1)
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
        ax.plot(gb.iloc[:, 0].index.map(mapper = cfg.labels_mapping), gb.iloc[:, 0], marker = 'o', color = 'dodgerblue', label = f'{cfg.aux_treatment_col} == 0')
        ax.plot(gb.iloc[:, 1].index.map(mapper = cfg.labels_mapping), gb.iloc[:, 1], marker = 'o', color = 'red', label = f'{cfg.aux_treatment_col} == 1')
        ax.set_ylabel(ylabel = 'Utilization Rate')
        ax.set_xlabel(xlabel = cfg.treatment_col)
        ax.set_title(label = title)
        ax.legend()
        save_figure(fig = fig, name = name)


def plot_prediction_distributions(oot_data: pd.DataFrame, learner, learner_type: str) -> None:
    uplift_predictions, control_prediction = predict_uplift(learner = learner, data = oot_data)
    fig, ax = plot_metric_distribution_by_classes(
        data = uplift_predictions.melt(), class_name = 'variable', metric_name = 'value',
        figsize = (15, 4),
        title = 'Uplift distributions per treatment label',
    )
    save_figure(fig = fig, name = f'{learner_type.lower()}_uplift_distributions')

    P = convert_uplift_to_target_predictions(uplift_predictions = uplift_predictions, control_prediction = control_prediction)
    fig, ax = plot_metric_distribution_by_classes(
        data = P.melt(), class_name = 'variable', metric_name = 'value',
        figsize = (15, 4),
        title = f'P({cfg.target_col}) distributions per treatment label',
    )
    save_figure(fig = fig, name = f'{learner_type.lower()}_conversion_distributions')


def plot_gain_vs_cost(oot_data: pd.DataFrame, spend_check_solutions: Dict[str, Dict]) -> None:
    """
    cumulative IPW gain vs cumulative cost, users ordered by expected value per expected voucher cost of assigned voucher
    """
    Y = oot_data[cfg.target_col].to_numpy()
    spend_real = oot_data[cfg.cost_col].fillna(value = 0).to_numpy()
    no_treatment = (oot_data[cfg.treatment_col] == cfg.CONTROL_LABEL).astype(dtype = float).to_numpy()
    random_policy = np.ones(shape = len(oot_data))

    fig, axes = plt.subplots(nrows = 1, ncols = len(spend_check_solutions), figsize = (10 * len(spend_check_solutions), 6))
    for ax, (formulation, solution) in zip(axes, spend_check_solutions.items()):
        V, C, A, a = solution['V'], solution['C'], solution['A'], solution['treatment_assignment']
        N = len(a)
        V_a = np.where(a > 0, V[np.arange(N), np.maximum(a - 1, 0)], 0.0)
        C_a = np.where(a > 0, C[np.arange(N), np.maximum(a - 1, 0)], np.nan)
        ratio = np.where(a > 0, V_a / C_a, -np.inf)
        ordering = np.argsort(a = -ratio)

        matched = (oot_data[cfg.treatment_col].to_numpy() == a).astype(dtype = float)
        gain = ((matched - no_treatment) / cfg.propensity_prob) * A * Y
        spend = ((matched - no_treatment) / cfg.propensity_prob) * spend_real
        gain_random = (random_policy - no_treatment / cfg.propensity_prob) * A * Y
        spend_random = (random_policy - no_treatment / cfg.propensity_prob) * spend_real

        n_limit = int((a > 0).sum())
        cumulative_gain = np.cumsum(a = gain[ordering])[:n_limit]
        cumulative_cost = np.cumsum(a = spend[ordering])[:n_limit]
        cumulative_gain_random = np.cumsum(a = gain_random[ordering])[:n_limit]
        cumulative_cost_random = np.cumsum(a = spend_random[ordering])[:n_limit]

        ax.plot(cumulative_cost_random, cumulative_gain_random, color = 'crimson', label = 'randomized experiment policy')
        ax.plot(cumulative_cost, cumulative_gain, color = 'lightblue', label = 'optimized policy')
        ax.set_title(label = f'spend formulation {formulation}')
        ax.set_ylabel(ylabel = 'cumulative gain')
        ax.set_xlabel(xlabel = 'cumulative cost')
        ax.legend()
    save_figure(fig = fig, name = 'gain_vs_cost_week14')


def plot_assignment_distribution(treatment_assignment: np.ndarray) -> None:
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 6))
    vc = pd.Series(data = treatment_assignment).value_counts().sort_index()
    vc = vc.loc[vc.index != cfg.CONTROL_LABEL]
    ax.bar(vc.index, vc)
    ax.set_xticks(ticks = vc.index)
    ax.set_xticklabels(labels = vc.index.map(mapper = cfg.labels_mapping))
    ax.set_title(label = 'Voucher Assignments Policy Distribution')
    save_figure(fig = fig, name = 'week15_assignment_distribution')


def make_figures(samples: Dict[str, pd.DataFrame], models: Dict, spend_check_solutions: Dict[str, Dict], week15_assignment: np.ndarray) -> None:
    ps538()
    figures_path.mkdir(parents = True, exist_ok = True)
    modelling_data = get_modelling_data(samples = samples)
    analysis_data = samples[train_sample_key]
    learner_type = 'URF'
    learner = models['fitted_learners'][learner_type]

    plot_conversion_by_voucher_fd(modelling_data = modelling_data)
    plot_ate_cost_efficiency(analysis_data = analysis_data)
    plot_target_rate_dynamic(modelling_data = modelling_data)
    plot_cate_example(analysis_data = analysis_data)
    plot_score_distributions(test_data = samples['test'], fitted_learners = models['fitted_learners'])
    plot_uplift_evaluation(samples = samples, learner = learner, learner_type = learner_type)
    plot_utilization_rates(analysis_data = analysis_data)
    plot_prediction_distributions(oot_data = samples['oot'], learner = learner, learner_type = learner_type)
    plot_gain_vs_cost(oot_data = samples['oot'], spend_check_solutions = spend_check_solutions)
    plot_assignment_distribution(treatment_assignment = week15_assignment)
