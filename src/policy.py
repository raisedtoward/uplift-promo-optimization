from typing import Dict
import numpy as np
import pandas as pd
import cvxpy as cp

from src import config as cfg


def solve_assignment_problem(V: np.ndarray, S: np.ndarray, B: float) -> Dict:
    """
    max sum(V * X) s.t. at most 1 voucher per user and sum(S * X) <= B
    V, S: (N, 4) expected incremental value and expected spend under voucher levels 1..4
    """
    N, K = V.shape
    ### promo assignment matrix - technically continuous between 0 and 1, but we'll map to discrete values
    X = cp.Variable(shape = (N, K), nonneg = True)
    constraints = [
        cp.sum(expr = X, axis = 1) <= 1, ### at most 1 promo voucher per user
        cp.sum(expr = cp.multiply(lh_expr = S, rh_expr = X)) <= B, ### budget limitations
    ]
    problem = cp.Problem(objective = cp.Maximize(expr = cp.sum(expr = cp.multiply(lh_expr = V, rh_expr = X))), constraints = constraints)
    problem.solve(solver = cp.HIGHS)

    is_assigned_discount = (X.value.sum(axis = 1) > 0.5).astype(dtype = int)
    treatment_assignment = np.where(
        is_assigned_discount,
        X.value.argmax(axis = 1) + 1, ### shifting because left-most col is not control, but treatment 1
        cfg.CONTROL_LABEL, ### if no selection meaning we're assigning control group
    )
    return {
        'status': problem.status,
        'objective': float(problem.value),
        'projected_spend': float((S * X.value).sum()),
        'treatment_assignment': treatment_assignment,
    }


def evaluate_spend_ipw(data: pd.DataFrame, treatment_assignment: np.ndarray, A: np.ndarray) -> Dict[str, float]:
    """
    realised spend and gain of the policy on randomised data, inverse propensity weighted vs no-voucher control:
    g_i = (1[D_i = policy-based d_i] - 1[D_i = 0]) / e * Y_i * AOV_i, users without assigned voucher contribute 0
    """
    Y = data[cfg.target_col].to_numpy()
    spend_real = data[cfg.cost_col].fillna(value = 0).to_numpy()
    is_assigned_discount = (treatment_assignment > 0)
    matched = (data[cfg.treatment_col].to_numpy() == treatment_assignment).astype(dtype = float)
    no_treatment = (data[cfg.treatment_col].to_numpy() == cfg.CONTROL_LABEL).astype(dtype = float)

    gain = ((matched - no_treatment) / cfg.propensity_prob) * A * Y
    spend = ((matched - no_treatment) / cfg.propensity_prob) * spend_real
    return {
        'realised_spend': float(spend[is_assigned_discount].sum()),
        'realised_gain': float(gain[is_assigned_discount].sum()),
    }
