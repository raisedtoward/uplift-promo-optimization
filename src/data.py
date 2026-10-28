from typing import Dict
import numpy as np
import pandas as pd

from atlas.encoders import CategoricalEncoder

from src import config as cfg
from src.utils import set_seed


def load_data() -> Dict[str, pd.DataFrame]:
    history_data = pd.read_parquet(path = cfg.data_path / 'history.parquet')
    inference_data = pd.read_parquet(path = cfg.data_path / 'inference.parquet')

    ### numeric features that are integer as per description are actually floats
    for c in ['orders_90d', 'orders_30d', 'sessions_30d']:
        history_data[c] = history_data[c].astype(dtype = int)
        inference_data[c] = inference_data[c].astype(dtype = int)

    ### placeholder treatment_col with default treatment value for inference data
    inference_data[cfg.treatment_col] = 0
    history_data[cfg.treatment_col] = history_data[cfg.treatment_col].astype(dtype = int)

    ### booleans to 0/1
    for c in [cfg.target_col, cfg.utilization_col, cfg.aux_treatment_col, 'push_opt_in']:
        history_data[c] = history_data[c].astype(dtype = int)
        if c in inference_data.columns:
            inference_data[c] = inference_data[c].astype(dtype = int)

    return {'history_data': history_data, 'inference_data': inference_data}


def build_all_data(history_data: pd.DataFrame, inference_data: pd.DataFrame) -> pd.DataFrame:
    """
    stacks history and inference user-weeks and attaches previous week's readings for the same user
    """
    history_data = history_data.assign(is_inference = 0)
    inference_data = inference_data.assign(is_inference = 1)
    all_data = pd.concat(objs = (history_data, inference_data), axis = 0)

    ### there are no orders with <= 0 aov, so missing aov is filled with 0s
    all_data['aov_90d'] = all_data['aov_90d'].fillna(value = 0)
    ### voucher_cost is missing for user-weeks where voucher wasn't actually used
    all_data.loc[all_data[cfg.utilization_col] == 0, cfg.cost_col] = pd.NA

    all_data['previous_week'] = all_data[cfg.time_col] - 1
    all_data = pd.merge(
        ### left - base data, right - previous week data, w -> pw rename only to join on equal value with base
        left = all_data,
        right = (
            all_data[cfg.lagged_features + [cfg.id_col, cfg.time_col]]
            .assign(is_observed_lag_1w = 1)
            .rename(mapper = {**{c: f'{c}_lag_1w' for c in cfg.lagged_features}, **{cfg.time_col: 'previous_week'}}, axis = 1)
        ),
        on = [cfg.id_col, 'previous_week'], how = 'left',
    )
    all_data['is_observed_lag_1w'] = all_data['is_observed_lag_1w'].fillna(value = 0).astype(dtype = int)

    all_data = (
        all_data
        ### lagged features are completely unavailable for this population slice - no week 8 in the dataset
        .query(expr = f'{cfg.time_col} != 9')
        ### week 16 features already reflect week 15 that hasn't happened yet at decision time
        .query(expr = f'{cfg.time_col} != 16')
        .sort_values(by = [cfg.id_col, cfg.time_col])
        .reset_index(drop = True)
    )
    return all_data


def split_samples(all_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    user-level 80/20 split for weeks 10-13 (train / test), week 14 as out-of-time (oot) with its test-users slice as holdout
    """
    modelling_data = all_data.query(expr = 'is_inference == 0').reset_index(drop = True)
    inference_data = all_data.query(expr = 'is_inference == 1').reset_index(drop = True)
    modelling_data[cfg.treatment_col] = modelling_data[cfg.treatment_col].map(func = cfg.levels_mapping)

    all_ids = sorted(modelling_data[cfg.id_col].unique())
    n_ids_train = int(len(all_ids) * cfg.p_train)
    set_seed(seed = cfg.seed)
    ids_train = set(np.random.choice(a = all_ids, size = n_ids_train, replace = False))
    ids_test = set(all_ids) - ids_train

    train_mask = modelling_data[cfg.id_col].isin(values = ids_train)
    test_mask = modelling_data[cfg.id_col].isin(values = ids_test)
    train_period_mask = (modelling_data[cfg.time_col] < 14)
    oot_period_mask = (modelling_data[cfg.time_col] >= 14)

    samples = {
        'train': modelling_data.loc[train_period_mask & train_mask, :].copy(),
        'test': modelling_data.loc[train_period_mask & test_mask, :].copy(),
        'oot': modelling_data.loc[oot_period_mask, :].copy(),
        'holdout': modelling_data.loc[oot_period_mask & test_mask, :].copy(),
        'inference': inference_data.copy(),
    }
    return encode_categorical(samples = samples)


def encode_categorical(samples: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cat_encoder = CategoricalEncoder(
        cat_na_fill_value = '0_MISSING_VALUE',
        map_unknown_to_na = True,
        seek_categorical_from_int = False,
        verbose = False,
        enforce_string = False,
    )
    cat_encoder.fit(data = samples['train'], cat_features = cfg.cat_feature_cols)
    for k in samples:
        cat_df = cat_encoder.transform(data = samples[k], transform_inplace = False, return_features = False)
        samples[k] = pd.concat(objs = [samples[k], cat_df], axis = 1)
    return samples
