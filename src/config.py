from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / 'data'
output_path = ROOT / 'outputs'
models_cache_path = ROOT / 'tmp' / 'models.pkl'

seed = 8
p_train = 0.8

id_col = 'user_id'
time_col = 'week'
city_col = 'city'

treatment_col = 'voucher_pct'
aux_treatment_col = 'free_delivery'

target_col = 'ordered'
utilization_col = 'voucher_used'
cost_col = 'voucher_cost'

post_treatment_cols = [
    target_col,
    utilization_col,
    cost_col,
]

num_feature_cols = [
    'orders_90d',
    'orders_30d',
    'days_since_last_order',
    'aov_90d',
    'pct_orders_with_promo_90d',
    'avg_voucher_pct_received_90d',
    'voucher_redemption_rate_90d',
    'manual_voucher_entry_rate_90d',
    'sessions_30d',
    'days_since_last_session',
    'push_opt_in',
    'offers_tab_share_of_sessions_30d',
]

cat_feature_cols = [
    city_col,
]

### columns pulled from previous week's observation of the same user
### post-treatment columns' horizon is strictly 1 week ahead, so for week 15 they are week 14 outcomes - already known
lagged_features = [
    treatment_col,
    aux_treatment_col,
    'aov_90d',
    'orders_90d',
] + post_treatment_cols

additional_features = [f'{c}_lag_1w' for c in lagged_features]
cat_encoded_features = [f'{c}_ENCODED' for c in cat_feature_cols]

uplift_features = num_feature_cols + additional_features + cat_encoded_features + [aux_treatment_col]
utilization_model_features = num_feature_cols + additional_features + [treatment_col, aux_treatment_col]
### engagement-only features, so AOV forecast for non-ordering users is not out-of-distribution
aov_predictive_features = ['sessions_30d', 'days_since_last_session', 'push_opt_in', 'offers_tab_share_of_sessions_30d']

### model labels 0..4 <-> original voucher levels
treatment_levels = [0, 10, 20, 30, 40]
levels_mapping = {v: i for i, v in enumerate(treatment_levels)}
labels_mapping = {i: v for v, i in levels_mapping.items()}
treatment_labels = list(labels_mapping)
CONTROL_LABEL = 0
propensity_prob = 1 / len(treatment_labels)

voucher_cap = 9.0
total_budget_2weeks = 3500
weekly_budget = total_budget_2weeks / 2
### realised / planned spend measured on week 14 (out-of-time) for the final formulation was ~1.28
budget_conservative_coef = 1.25

boost_params = {
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
    'random_state': seed,
}

LOGGER_NAME = 'uplift_model_logger'
