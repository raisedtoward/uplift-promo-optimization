"""
end-to-end pipeline, same steps as in the research notebook:
train -> qini per learner -> week 14 (out-of-time) spend check for 3 spend formulations -> week 15 voucher assignment -> figures

usage:
    uv run python -m src.pipeline            # uses cached models if present
    uv run python -m src.pipeline --retrain  # refits all models
"""
import argparse
import json
import logging
import pickle
from typing import Dict
import numpy as np
import pandas as pd

from atlas.uplift.evaluation import calc_qini

from src import config as cfg
from src.utils import setup_logging
from src.data import load_data, build_all_data, split_samples
from src.models import (
    fit_learners, predict_uplift, convert_uplift_to_target_predictions,
    fit_aov_model, get_aov_estimate,
    fit_utilization_model, get_utilization_estimate,
    get_arm_level_mappings, map_by_free_delivery, get_expected_voucher_cost,
)
from src.policy import solve_assignment_problem, evaluate_spend_ipw
from src.report import make_figures

LEARNER_TYPE = 'URF'


def train(train_data: pd.DataFrame, retrain: bool) -> Dict:
    logger = logging.getLogger(cfg.LOGGER_NAME)
    if cfg.models_cache_path.exists() and not retrain:
        logger.info(msg = f'loading cached models from {cfg.models_cache_path}')
        with open(file = cfg.models_cache_path, mode = 'rb') as f:
            return pickle.load(file = f)
    models = {
        'fitted_learners': fit_learners(train_data = train_data),
        'aov_model': fit_aov_model(train_data = train_data),
        'utilization_model': fit_utilization_model(train_data = train_data),
        'arm_level_mappings': get_arm_level_mappings(train_data = train_data),
    }
    cfg.models_cache_path.parent.mkdir(parents = True, exist_ok = True)
    with open(file = cfg.models_cache_path, mode = 'wb') as f:
        pickle.dump(obj = models, file = f)
    return models


def evaluate_qini(samples: Dict[str, pd.DataFrame], fitted_learners: Dict) -> pd.DataFrame:
    """
    qini of each learner's uplift_t score on (voucher level t vs control) subset of each sample
    """
    qini_results = []
    for learner_type, learner in fitted_learners.items():
        for sample_key in ['train', 'test', 'oot', 'holdout']:
            data = samples[sample_key]
            uplift_predictions, _ = predict_uplift(learner = learner, data = data)
            for t_label in cfg.treatment_labels:
                if t_label == cfg.CONTROL_LABEL:
                    continue
                mask = data[cfg.treatment_col].isin(values = [cfg.CONTROL_LABEL, t_label]).to_numpy()
                qini = calc_qini(
                    y_true = data.loc[mask, cfg.target_col].to_numpy(),
                    y_score = uplift_predictions.loc[mask, f'uplift_{t_label}'].to_numpy(),
                    treatment = (data.loc[mask, cfg.treatment_col] == t_label).astype(dtype = int).to_numpy(),
                )
                qini_results.append({'learner_type': learner_type, cfg.treatment_col: cfg.labels_mapping[t_label], 'sample': sample_key, 'qini': qini})
    return pd.DataFrame(data = qini_results)


def get_problem_inputs(models: Dict, data: pd.DataFrame, formulation: int) -> Dict[str, np.ndarray]:
    """
    V, S: (N, 4) expected incremental value and expected spend for each voucher level under spend formulation:
    1: P(order | d, fd, x) * P(voucher_used | order, d, fd) * E[voucher_cost | voucher_used, d, fd]
    2: P(voucher_used | d, fd) * min(d * AOV, 9)
    3: P(voucher_used | d, fd, x) * min(d * AOV, 9)   <- final
    """
    uplift_predictions, control_prediction = predict_uplift(learner = models['fitted_learners'][LEARNER_TYPE], data = data)
    P = convert_uplift_to_target_predictions(uplift_predictions = uplift_predictions, control_prediction = control_prediction).values
    D = uplift_predictions.values
    A = get_aov_estimate(aov_model = models['aov_model'], data = data).values
    mappings = models['arm_level_mappings']

    if formulation == 1:
        U = map_by_free_delivery(mapping = mappings['utilization_rate_given_order_mapping'], data = data)
        C = map_by_free_delivery(mapping = mappings['avg_cost_mapping'], data = data)
        S = P[:, 1:] * U * C
    elif formulation == 2:
        U = map_by_free_delivery(mapping = mappings['utilization_rate_mapping'], data = data)
        C = get_expected_voucher_cost(aov = pd.Series(data = A))
        S = U * C
    elif formulation == 3:
        U = get_utilization_estimate(utilization_model = models['utilization_model'], data = data).values
        C = get_expected_voucher_cost(aov = pd.Series(data = A))
        S = U * C
    else:
        raise ValueError(f'unknown formulation {formulation}')

    V = A[:, None] * D
    return {'V': V, 'S': S, 'C': C, 'A': A}


def describe_assignment(treatment_assignment: np.ndarray) -> Dict:
    return {
        'n_users': int(len(treatment_assignment)),
        'n_assigned': int((treatment_assignment > 0).sum()),
        'assigned_share': float((treatment_assignment > 0).mean()),
        'assignment_counts': {str(cfg.labels_mapping[t]): int((treatment_assignment == t).sum()) for t in cfg.treatment_labels},
    }


def run_spend_check(models: Dict, oot_data: pd.DataFrame) -> Dict:
    """
    week 14 (out-of-time): solve at the full weekly budget and compare planned spend with realised (IPW) spend
    """
    spend_iterations = []
    spend_check_solutions = {}
    for formulation in [1, 2, 3]:
        problem_inputs = get_problem_inputs(models = models, data = oot_data, formulation = formulation)
        solution = solve_assignment_problem(V = problem_inputs['V'], S = problem_inputs['S'], B = cfg.weekly_budget)
        realised = evaluate_spend_ipw(data = oot_data, treatment_assignment = solution['treatment_assignment'], A = problem_inputs['A'])
        assignment = describe_assignment(treatment_assignment = solution['treatment_assignment'])
        spend_check_solutions[f'v{formulation}'] = {**problem_inputs, 'treatment_assignment': solution['treatment_assignment']}
        spend_iterations.append({
            'formulation': f'v{formulation}',
            'n_assigned': assignment['n_assigned'],
            'assigned_share': assignment['assigned_share'],
            **{f'n_{k}': v for k, v in assignment['assignment_counts'].items() if k != '0'},
            'objective': solution['objective'],
            'planned_spend': solution['projected_spend'],
            'realised_spend': realised['realised_spend'],
            'realised_to_planned': realised['realised_spend'] / solution['projected_spend'],
            'realised_gain': realised['realised_gain'],
        })
    return {'spend_iterations': pd.DataFrame(data = spend_iterations), 'spend_check_solutions': spend_check_solutions}


def assign_week15(models: Dict, weekly_inference_data: pd.DataFrame) -> Dict:
    problem_inputs = get_problem_inputs(models = models, data = weekly_inference_data, formulation = 3)
    ### weekly budget constraint with extra conservative estimate
    B = cfg.weekly_budget / cfg.budget_conservative_coef
    solution = solve_assignment_problem(V = problem_inputs['V'], S = problem_inputs['S'], B = B)

    decisions = pd.DataFrame(data = {
        cfg.id_col: weekly_inference_data[cfg.id_col].to_numpy(),
        cfg.treatment_col: pd.Series(data = solution['treatment_assignment']).map(func = cfg.labels_mapping).to_numpy(),
    })
    summary = {
        cfg.time_col: 15,
        'status': solution['status'],
        **describe_assignment(treatment_assignment = solution['treatment_assignment']),
        'weekly_budget': cfg.weekly_budget,
        'budget_conservative_coef': cfg.budget_conservative_coef,
        'projected_spend_optimistic': solution['projected_spend'],
        'projected_spend_conservative': solution['projected_spend'] * cfg.budget_conservative_coef,
        'expected_incremental_value': solution['objective'],
    }
    return {'decisions': decisions, 'summary': summary, 'treatment_assignment': solution['treatment_assignment']}


def main(retrain: bool) -> None:
    setup_logging(name = cfg.LOGGER_NAME, to_stdout = True)
    logger = logging.getLogger(cfg.LOGGER_NAME)

    raw = load_data()
    all_data = build_all_data(history_data = raw['history_data'], inference_data = raw['inference_data'])
    samples = split_samples(all_data = all_data)
    models = train(train_data = samples['train'], retrain = retrain)
    cfg.output_path.mkdir(parents = True, exist_ok = True)

    qini_results = evaluate_qini(samples = samples, fitted_learners = models['fitted_learners'])
    qini_results.to_csv(path_or_buf = cfg.output_path / 'qini.csv', index = False)
    logger.info(msg = '\n' + qini_results.groupby(by = ['sample', 'learner_type'])['qini'].mean().unstack().to_string())

    spend_check = run_spend_check(models = models, oot_data = samples['oot'])
    spend_iterations = spend_check['spend_iterations']
    spend_iterations.to_csv(path_or_buf = cfg.output_path / 'spend_iterations_week14.csv', index = False)
    logger.info(msg = '\n' + spend_iterations.to_string())

    week15 = assign_week15(models = models, weekly_inference_data = samples['inference'])
    decisions, summary = week15['decisions'], week15['summary']
    assert len(decisions) == len(raw['inference_data'].query(expr = f'{cfg.time_col} == 15'))
    assert summary['projected_spend_conservative'] <= cfg.weekly_budget + 1e-6
    decisions.to_csv(path_or_buf = cfg.output_path / 'decisions.csv', index = False)
    with open(file = cfg.output_path / 'summary.json', mode = 'w') as f:
        json.dump(obj = summary, fp = f, indent = 2)
    logger.info(msg = json.dumps(obj = summary, indent = 2))

    make_figures(
        samples = samples,
        models = models,
        spend_check_solutions = spend_check['spend_check_solutions'],
        week15_assignment = week15['treatment_assignment'],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action = 'store_true')
    main(retrain = parser.parse_args().retrain)
