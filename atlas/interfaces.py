from __future__ import annotations

import logging
import warnings
from typing import Callable, Iterable, Dict, List, Optional, Tuple, Union
import itertools
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from copy import deepcopy
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LinearRegression

from pulsar.visualization import get_metrics_dynamic, plot_metrics_dynamic

from atlas.descriptive import get_feature_statistics
from atlas.utils import display, drop_columns_safe, wrap_as_list, BaseModel
from atlas.uplift.learners import (
    BaseLearner,
    SLearner, TLearner, XLearner, CTLearner,
    UpliftTreeLearner, UpliftRandomForestLearner
)

OBJECTIVES = (
    'classification',
    'regression',
    'uplift',
)

SCORING_METHODS = (
    'threshold',
    'proba',
    'classification',
    'class', 'label',
    'regression', 'uplift',
    'predict', 'value',
)

SKLEARN_CLASSIFIERS = (
    RandomForestClassifier, DecisionTreeClassifier, LogisticRegression,
    CalibratedClassifierCV, LogisticRegressionCV,
)

SKLEARN_REGRESSORS = (
    RandomForestRegressor, DecisionTreeRegressor, LinearRegression
)

SKLEARN_MODELS = SKLEARN_CLASSIFIERS + SKLEARN_REGRESSORS


def model_prediction_wrapper(
    model: BaseModel,
    prediction_method: str = 'proba',
    threshold: float = 0.5,
) -> Callable:
    if prediction_method == 'proba':
        predict_function = lambda X: pd.Series(
            model.predict_proba(X)[:, 1],
            index = X.index if isinstance(X, pd.DataFrame) else range(len(X))
        )
    elif prediction_method in ('label', 'value'):
        predict_function = lambda X: pd.Series(
            model.predict(X),
            index = X.index if isinstance(X, pd.DataFrame) else range(len(X))
        )
    elif prediction_method == 'threshold':
        predict_function = lambda X: pd.Series(
            model.predict_proba(X)[:, 1] >= threshold,
            index = X.index if isinstance(X, pd.DataFrame) else range(len(X))
        ).astype(int)
    else:
        raise ValueError('wrong prediction method')
    return predict_function


def model_fit_wrapper(
    model: BaseModel,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cat_features: List[str] = None,
    ignore_model_type: bool = False,
    **kwargs
):

    cat_features = cat_features or []
    if isinstance(model, (LGBMClassifier, LGBMRegressor, BaseLearner)):
        new_kwargs = {}
        for k, v in kwargs.items():
            if k != 'eval_set' and k != 'n_estimators':
                new_kwargs[k] = deepcopy(v)
            elif k == 'n_estimators':
                model.set_params(n_estimators = kwargs[k])
            else:
                pass

        t_fit_keys = ['treatment_kwargs', 'control_kwargs']
        xl_fit_keys = [
            'treatment_kwargs_stage_1', 'control_kwargs_stage_1', 'treatment_kwargs_stage_2', 'control_kwargs_stage_2'
        ]

        if isinstance(model, (LGBMRegressor, LGBMClassifier, SLearner, CTLearner)):
            new_kwargs['categorical_feature'] = cat_features
        elif isinstance(model, TLearner):
            for k in t_fit_keys:
                if k not in new_kwargs:
                    new_kwargs[k] = {}
                new_kwargs[k]['categorical_feature'] = cat_features
        elif isinstance(model, XLearner):
            for k in xl_fit_keys:
                if k not in new_kwargs:
                    new_kwargs[k] = {}
                new_kwargs[k]['categorical_feature'] = cat_features

        ### additionally we could pass eval sets for early stopping
        ### but it needs repacking from [(X,y,t), (X,y,t), ] to [(X,X,), (y,y,), (t,t,)] for lightgbm to accept it
        if 'eval_set' in kwargs:
            if isinstance(model, (LGBMRegressor, LGBMClassifier, SLearner, CTLearner)):
                ### taking X from every elem in eval list
                new_eval_set = [
                    (kwargs['eval_set'][i][0][X.columns], kwargs['eval_set'][i][1])
                    for i in range(len(kwargs['eval_set']))
                ]
                new_kwargs['eval_set'] = new_eval_set

            elif isinstance(model, (TLearner, XLearner)):
                if isinstance(model, TLearner):
                    keys = t_fit_keys
                else:
                    keys = xl_fit_keys
                for k in keys:
                    if k in kwargs.keys():
                        if 'eval_set' in kwargs[k].keys():
                            new_eval_set = [
                                (kwargs[k]['eval_set'][i][0][X.columns], kwargs[k]['eval_set'][i][1])
                                for i in range(len(kwargs[k]['eval_set']))
                            ]
                            new_kwargs[k]['eval_set'] = new_eval_set

        model.fit(X, y, **new_kwargs)

    elif isinstance(model, SKLEARN_MODELS):
        model.fit(X = X, y = y, **kwargs)
    elif ignore_model_type:
        model.fit(X = X, y = y, **kwargs)
    else:
        pass

    return model


def model_n_jobs_wrapper(
    model: Union[LGBMClassifier, LGBMRegressor, BaseLearner],
    n_jobs: Union[int, str],
    reinitialize: bool = False,
) -> Union[LGBMClassifier, LGBMRegressor, BaseLearner]:
    if isinstance(model, (LGBMClassifier, LGBMRegressor)):
        if reinitialize:
            model = type(model)().set_params(**model.get_params())
        model.set_params(n_jobs = n_jobs)
    elif isinstance(model, (SLearner, CTLearner)):
        model.base_model.model = model_n_jobs_wrapper(
            model.base_model.model, n_jobs = n_jobs, reinitialize = reinitialize
        )
    elif isinstance(model, TLearner):
        model.treatment_model.model = model_n_jobs_wrapper(
            model.treatment_model.model, n_jobs = n_jobs, reinitialize = reinitialize
        )
        model.control_model.model = model_n_jobs_wrapper(
            model.control_model.model, n_jobs = n_jobs, reinitialize = reinitialize
        )
    elif isinstance(model, XLearner):
        model.treatment_base_model.model = model_n_jobs_wrapper(
            model.treatment_base_model.model, n_jobs = n_jobs, reinitialize = reinitialize
        )
        model.control_base_model.model = model_n_jobs_wrapper(
            model.control_base_model.model, n_jobs = n_jobs, reinitialize = reinitialize
        )
        model.treatment_imputed_model.model = model_n_jobs_wrapper(
            model.treatment_imputed_model.model, n_jobs = n_jobs, reinitialize = reinitialize
        )
        model.control_imputed_model.model = model_n_jobs_wrapper(
            model.control_imputed_model.model, n_jobs = n_jobs, reinitialize = reinitialize
        )
    elif isinstance(model, UpliftRandomForestLearner):
        model.model.n_jobs = 1
    elif isinstance(model, UpliftTreeLearner):
        pass
    else:
        raise TypeError(f"model type {type(model)} is unsupported")
    return model


def get_metrics_dynamic_advanced(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    timestamp_col: str,
    target_col: str,
    score_col: str = None,
    treatment_col: str = None,
    metrics: Dict[str, Callable] = None,
    freq: str = 'S',
    timestamp_format: str = '%Y-%m-%d',
    labels: List[str] = None,
    weight_col: str = None,
    verbose: bool = False,
):

    kwargs = {
        'timestamp_col': timestamp_col, 'target_col': target_col,
        'score_col': score_col, 'treatment_col': treatment_col,
        'metrics': metrics,
        'freq': freq, 'timestamp_format': timestamp_format,
        'weight_col': weight_col,
        'verbose': verbose,
    }

    if isinstance(data, pd.DataFrame):
        data = [data]

    if not labels:
        if len(data) > 1:
            labels = [f'{i}' for i in range(len(data))]
        else:
            labels = ['value']
    else:
        if not (all(s for s in labels) and len(set(labels)) == len(data)):
            raise ValueError("length of labels does not match number of datasets")

    dynamics = []
    for s, name in zip(data, labels):
        dynamic_results = get_metrics_dynamic(data = s, **kwargs)
        dynamic_results['sample'] = name
        dynamics.append(dynamic_results)
    dynamics = pd.concat(dynamics, axis = 0)
    grouping_fields = ['timestamp'] + (
        [treatment_col] if (treatment_col is not None and score_col is None) else []
    )
    dynamics = dynamics.reset_index().groupby(grouping_fields + ['sample']).sum().unstack(-1)
    ordered_cols = list(
        (fk, sk)
        for fk in dynamics.columns.get_level_values(0).unique()
        for sk in labels
    )
    dynamics = dynamics[[f for f in ordered_cols if f in dynamics.columns]]
    if treatment_col is None:
        ### we could simplify the plot by pretending sample name is a treatment group
        dynamics = dynamics.stack()
    return dynamics


def reduce_memory_usage(df, inplace: bool = False, verbose_f: Callable = None):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep = True).sum() / 1024 ** 2
    if not inplace:
        df = df.copy(deep = True)
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                # elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                #     df[col] = df[col].astype(np.float32)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep = True).sum() / 1024 ** 2
    if verbose_f is not None:
        verbose_f(
            "Memory usage decreased from {:5.2f} mb to {:5.2f} mb with ({:.1f}% reduction)"
            .format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem)
        )
    return df


def save_object(
    obj: object,
    base_path: Union[str, Path],
    file_name: str,
    suffix: str = '.pkl'
):
    save_path = str(Path(base_path) / f'{file_name}{suffix}')
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved {type(obj)} to {save_path}")


def load_object(
    file_name: Union[str, Path],
    suffix: str = '.pkl',
    verbose: bool = False,
):
    file_path = f"{file_name}{suffix}"
    if verbose:
        print(f"Loading from file {file_path}")
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    if verbose:
        print(f"Got object of type {type(obj)}")
    return obj


def pd_hstack(*pd_objects):
    return pd.concat([obj.reset_index(drop = True) for obj in pd_objects], axis = 1)


def pd_vstack(*pd_objects):
    return pd.concat([obj.reset_index(drop = True) for obj in pd_objects], axis = 0, ignore_index = True)


def stack_XY(*xy_tuples):
    return (
        pd_vstack(t[0] for t in xy_tuples), pd_vstack(t[1] for t in xy_tuples)
    )


class ActiveOrdered:
    def __init__(self, special_keys_ordered: Tuple[str, ...] = None):
        self.active = set()
        self.special_keys_ordered = special_keys_ordered or tuple()
        
    def __repr__(self):
        return self.active.__repr__()
    
    def add(self, value):
        self.active.add(value)
        
    def discard(self, value):
        self.active.discard(value)
        
    def remove(self, value):
        self.discard(value)
        
    def __contains__(self, value):
        return value in self.active
        
    def __len__(self):
        return len(self.active)
        
    def __iter__(self):
        for s in self.special_keys_ordered:
            if s in self.active:
                yield s
        for s in self.active:
            if s not in self.special_keys_ordered:
                yield s


class FilterApplicator(object):
    def __init__(self, sample_keeper: SampleKeeper, filter_queries: List[str] = None):
        self.filter_queries = filter_queries or []
        self.sample_keeper = sample_keeper

    def _initialize(self):
        self.added_ids = {}
        self.not_added_ids = {}
        self.existing_ids = {}
        max_existing_id = max(self.sample_keeper.filter_ids.keys()) if len(self.sample_keeper.filter_ids) else 0

        # if len(filter_queries) == 0:
        #     raise ValueError("filter_queries should not be empty")

        for q in self.filter_queries:
            filter_id = self.sample_keeper.add_filter(filter_query = q, enable = False)
            if filter_id is None:
                self.not_added_ids[filter_id] = q
            else:
                if filter_id <= max_existing_id:
                    self.existing_ids[filter_id] = deepcopy(self.sample_keeper.filters[q])
                else:
                    self.added_ids[filter_id] = q

        if len(self.not_added_ids):
            for i, q in self.not_added_ids.items():
                warnings.warn(f"unable to apply query '{q}'")
            self.cleanup()
            raise ValueError("FilterApplicator was not initialized")

        for i, info in self.existing_ids.items():
            self.sample_keeper.enable_filter(i)
        for i, q in self.added_ids.items():
            self.sample_keeper.enable_filter(i)

    def cleanup(self):
        # for i, q in self.not_added_ids.items():
        #     self.sample_keeper.delete_filter(i)
        for i, q in self.added_ids.items():
            self.sample_keeper.delete_filter(i)

        for i, info in self.existing_ids.items():
            if not info['enabled']:
                self.sample_keeper.disable_filter(i)

    def __enter__(self):
        self._initialize()
        return self.sample_keeper

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        if exc_value is not None:
            return False
        return True


class SampleKeeper:
    """
    class to conveniently handling different data samples - usually following train/test/out-of-time protocol
    but actually supporting any arbitrary keys provided through __init__ and keeping their order for report consistency
    implements methods for getting X features, Y targets without tedious indexing each time
    which is done once through setting up the markup but could be change afterwards
    accessing the data could be done through [key] indexing or .get_X(key), .get_XY(key), and so on
    also supports handling data for uplift models with additional treatment column T, like .get_YT(key), .get_XYT(key)
    could create additional data summaries and perform various simple data validation checks
    allows adding some querying filters to test various hypothesis during model training
    could score data samples using ModelKeeper instance with underlying ML-model, supports batch-scoring as well
    use .get_score(key, [score_name]) or get_S(key, [score_name]) to access it afterwards
    able to sample data for faster checks having large data volumes by providing sample_size argument to .get methods
    """
    def __init__(
        self,
        special_keys: Tuple[str, ...] = ('train', 'test', 'oot'),
        logger_name: str = None,
    ):
        self.samples = {}
        self.special_keys = special_keys
        self.active = ActiveOrdered(special_keys)
        self.markup = {}
        self.filters = {}
        self.filter_ids = {}
        self.logger_name = logger_name

    def apply_filters(self, filter_queries: List[str]) -> FilterApplicator:
        """
        :param filter_queries: List of temporary filtering queries to use in a context manager
        :return: context manager holding current sample keeper object with applied temporary filtering queries
        """
        return FilterApplicator(sample_keeper = self, filter_queries = filter_queries)

    def add_filter(self, filter_query: str, enable: bool = True) -> Optional[int]:
        """
        adding logical filter to query pd.DataFrame with
        returns positive integer id of added filter if setup successful, otherwise returns None if some exception was faced,
        or if query already exists, returns its id
        """
        if not isinstance(filter_query, str):
            raise TypeError(f"'filter_query' must be a valid expression in str format, got {type(filter_query)}")
        try:
            for k in self.active:
                self[k].iloc[:10, :].query(filter_query)
            if filter_query in self.filters:
                # warnings.warn(f"{filter_query} already exists, you can check for it by calling .list_filters()")
                filter_id = self.filters[filter_query]['id']
            else:
                max_id = max(self.filter_ids.keys()) if len(self.filter_ids) else 0
                filter_id = max_id + 1
                self.filter_ids[filter_id] = filter_query
                self.filters[filter_query] = {'id': filter_id, 'enabled': False}
            # print(f"saved '{filter_query}' under id = {new_id}")
            if enable:
                self.filters[filter_query]['enabled'] = True
            return filter_id
        except Exception as e:
            warnings.warn(e.__repr__())
            warnings.warn("filter_query could not be set, check the whole warning and fix the query")

    def enable_filter(
        self,
        filter: Union[str, int] = None,
    ):
        """
        enables provided logical filter
        """
        if isinstance(filter, str):
            filter_query = filter
        elif isinstance(filter, int):
            filter_query = self.filter_ids.get(filter)
        else:
            raise TypeError(f"{filter} must be either existing filter_id as integer or filter_query as string")
        if filter_query in self.filters:
            self.filters[filter_query]['enabled'] = True
        else:
            warnings.warn(f"{filter_query} doesn't exists, you can set it up by calling .add_filter()")

    def disable_filter(
        self,
        filter: Union[str, int] = None,
        disable_all: bool = False,
    ):
        """
        disabling logical filter from a given query for it to not be applied anymore
        """
        if disable_all:
            for q in self.filters:
                self.filters[q]['enabled'] = False
        else:
            if isinstance(filter, str):
                filter_query = filter
            elif isinstance(filter, int):
                filter_query = self.filter_ids.get(filter)
            else:
                raise TypeError(f"{filter} must be either existing filter_id as integer or filter query as string")
            if filter_query in self.filters:
                self.filters[filter_query]['enabled'] = False

    def delete_filter(
        self,
        filter: Union[str, int],
    ):
        """
        completely deletes logical filter
        """
        if isinstance(filter, str):
            filter_query = filter
        elif isinstance(filter, int):
            filter_query = self.filter_ids.get(filter)
        else:
            raise TypeError(f"{filter} must be either existing filter_id as integer or filter query as string")
        if filter_query in self.filters:
            filter_id = self.filters[filter_query]['id']
            del self.filters[filter_query]
            del self.filter_ids[filter_id]
        
    def list_filters(self):
        """
        lists all filters - both enabled and disabled
        """
        pprint.pprint(self.filters)
        
    def _filter(self, df: pd.DataFrame):
        """
        querying dataframe using conjunction of all active filters
        """
        active_filters = ' and '.join([f"({query})" for query, info in self.filters.items() if info['enabled']])
        if active_filters:
            return df.query(active_filters)
        return df

    def _sample(self, df: pd.DataFrame, sample_size: int = None, seed: int = 8, replace: bool = False):
        n = len(df)
        if sample_size is None:
            return df
        elif not replace and sample_size > n:
            return df
        else:
            return df.sample(n = sample_size, random_state = seed, replace = replace)

    def _special_key(self, key: str):
        if key.lower() in self.special_keys:
            return key.lower()
        return key

    def _check_key(self, key: str):
        key = self._special_key(key)
        if key not in self.active:
            raise ValueError(f'{key} sample is not present')
        return key

    def _check_sample_name(self, sample_name: str):
        return self._check_key(sample_name)

    def _vstack(self, keys: List[str]):
        """
        allows out-of-box sample stacking for get_ methods
        """
        if isinstance(keys, (list, tuple)):
            return pd_vstack(*(self[self._check_key(k)] for k in keys))
        elif isinstance(keys, str):
            return self[self._check_key(keys)]
        else:
            raise ValueError("'sample_name' should be either string or list/tuple of strings")

    def __delitem__(self, key):
        key = self._check_key(key)
        self.active.remove(key)
        del self.samples[key]

    def __getitem__(self, key):
        key = self._check_key(key)
        return self.samples[key]

    def __setitem__(self, key, value):
        if not isinstance(value, pd.DataFrame):
            try:
                value = pd.DataFrame(value)
            except ValueError:
                raise ValueError(f'provide dataframe-like value, got {type(value)}')
        self.active.add(key)
        self.samples[key] = value

    def drop_columns(
        self,
        col_name: Union[str, List[str]] = None,
        sample_name: Union[str, List[str]] = None,
    ):
        """
        by providing 'col_name' explicitly you can actually drop any feature
        """
        if col_name is None or not len(col_name):
            return
        col_name = wrap_as_list(col_name)
        if self.markup.get('score_name') is not None and col_name[0] == self.markup['score_name']:
            self.markup['score_name'] = None

        key_iterator = wrap_as_list(self.active if sample_name is None else sample_name)
        for k in key_iterator:
            drop_columns_safe(self[k], col_name, inplace = True)

    def drop_duplicated_columns(self, sample_name: Union[str, List[str]] = None,):
        """
        only name-check is performed, not the actual data full-match
        """
        key_iterator = self.active if sample_name is None else [sample_name]
        for k in key_iterator:
            self[k] = self[k].loc[:, ~self[k].columns.duplicated(keep='first')]

    def reduce_memory_usage(self, safe: bool = True, verbose: bool = True):
        """
        tries to compress dataframes by lowering float and int types if feature range fits in it
        """
        if safe:
            if verbose:
                logger = logging.getLogger(self.logger_name)
                logger.info("checking compressed dtypes consistency..")
                verbose_f = logger.info
            else:
                verbose_f = None
            dts = {}
            for k in self.active:
                dts[k] = reduce_memory_usage(self[k], inplace = False, verbose_f = verbose_f).dtypes
            dts = pd.DataFrame(dts).T
            cols_dtypes_nunique = dts.agg('nunique')
            cols_dtypes_not_unique = cols_dtypes_nunique[cols_dtypes_nunique > 1]
            if len(cols_dtypes_not_unique):
                logger.warning("unable to preserve same dtypes after compression across all samples for following columns:")
                display(dts[cols_dtypes_not_unique])
                return
        for k in tqdm(self.active):
            self[k] = reduce_memory_usage(self[k], verbose = verbose)

    def get_dynamic_statistics(
        self,
        samples: List[str] = None,
        target_col: str = None,
        treatment_col: str = None,
        score_col: str = None,
        timestamp_col = None,
        freq: str = 'D',
        timestamp_format: str = '%Y-%m-%d',
        metrics: Dict[str, Callable] = None,
        one_axis_per_metric: bool = True,
        filtered: bool = False,
        verbose: bool = False,
    ):
        timestamp_col = timestamp_col or self.date_main_field
        target_col = target_col or self.target_name
        score_col = score_col or self.markup.get('score_name')
        if metrics is None or not len(metrics):
            score_col = None
        treatment_col = treatment_col or self.markup.get('treatment_name')
        keys = samples or self.active

        if filtered:
            data = [self._filter(self[k]).copy() for k in keys]
        else:
            data = [self[k] for k in keys]

        dynamic_results = get_metrics_dynamic_advanced(
            data = data,
            labels = [k for k in keys],
            target_col = target_col,
            timestamp_col = timestamp_col,
            score_col = score_col,
            treatment_col = treatment_col,
            metrics = metrics,
            freq = freq,
            timestamp_format = timestamp_format,
            weight_col = None,
            verbose = verbose,
        )

        fig, axes = plot_metrics_dynamic(
            dynamic_results,
            metrics = list(metrics) if metrics else [],
            title = 'Dynamic aggregates',
            one_axis_per_metric = one_axis_per_metric,
        )
        return {
            'data': dynamic_results,
            'figure': fig,
        }

    def get_features_descriptive_statistics(
        self,
        sample_name: str,
        filtered: bool = False,
        features: List[str] = None,
        missings_threshold: float = 0.99,
        verbose: bool = False,
    ):
        """
        `features` should be string-keyword like 'numeric', 'cat' or 'all' or list of actual feature names in the data
        """
        sample_name = self._check_key(sample_name)
        if filtered:
            df = self._filter(self[sample_name])
        else:
            df = self[sample_name]
        if features:
            features = self._get_features_list(features)
            df = df[features]
        return get_feature_statistics(data = df, missings_threshold = missings_threshold, verbose = verbose)

    def get_aggregate_statistics(
        self,
        target_col: str = None,
        treatment_col: str = None,
        id_col: str = None,
        date_fields: List[str] = None,
        filtered: bool = False,
    ) -> pd.DataFrame:

        treatment_col = treatment_col or self.markup.get('treatment_name')
        target_col = target_col or self.markup.get('target_name')
        id_col = id_col or self.markup.get('id_field')
        date_fields = date_fields or self.markup.get('date_fields')

        gb_cols = ['sample', treatment_col] if treatment_col else ['sample']

        stats_results = []
        for k in self.active:
            if filtered:
                df = self._filter(self[k])
            else:
                df = self[k]
            gb = (
                df
                .assign(sample = k)
                .groupby(gb_cols)
                .agg({
                    **{
                        id_col: ['count', 'nunique'],
                        target_col: ['mean'],
                    },
                    **{c: ['min', 'max'] for c in date_fields}
                })
            )
            stats_results.append(gb)

        stats_results = pd.concat(stats_results, axis = 0)
        return stats_results

    def set_markup(
        self,
        target_name: str = None,
        score_name: str = None,
        treatment_name: str = None,
        all_features: List[str] = None,
        cat_features: List[str] = None,
        disable_features: List[str] = None,
        id_field: str = None,
        date_fields: List[str] = None,
        date_main_field: str = None,
        weight_field: str = None,
    ):
        if target_name is not None:
            self.markup['target_name'] = target_name
        if score_name is not None:
            self.markup['score_name'] = score_name
        if treatment_name is not None:
            self.markup['treatment_name'] = treatment_name            
        if all_features is not None:
            self.markup['all_features'] = all_features
            self.markup['cat_features'] = []
            self.markup['numeric_features'] = all_features
        if cat_features is not None:
            self.markup['cat_features'] = cat_features
            if all_features is None:
                all_features = self.all_features
            add_cat_features = [f for f in cat_features if f not in all_features]
            all_features += add_cat_features
            self.markup['all_features'] = all_features
            self.markup['numeric_features'] = [f for f in all_features if f not in cat_features]
        if disable_features is not None:
            for k in ('all', 'cat', 'numeric'):
                self.markup[f'{k}_features'] = [f for f in self.markup[f'{k}_features'] if f not in disable_features]
        if id_field is not None:
            self.markup['id_field'] = id_field
        if date_main_field is not None:
            self.markup['date_main_field'] = date_main_field
        if date_fields is not None:
            self.markup['date_fields'] = date_fields
        elif self.markup.get('date_fields', []):
            pass
        else:
            self.markup['date_fields'] = []
        if (
            self.markup.get('date_main_field') is not None
            and self.markup.get('date_main_field') not in self.markup['date_fields']
        ):
            self.markup['date_fields'] += [self.markup['date_main_field']]
        if weight_field is not None:
            self.markup['weight_field'] = weight_field

    def get_markup(self):
        return deepcopy(self.markup)

    def set_markup_from_dict(self, markup: Dict):
        markup = deepcopy(markup)
        if 'numeric_features' in markup:
            markup.pop('numeric_features')
        self.set_markup(**markup)

    def show_features_info(self):
        separator = '-'*80
        print(separator)
        print(f"current numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(separator)
        print(f"current categorical features ({len(self.cat_features)}): {self.cat_features}")
        print(separator)

    @property
    def all_features(self):
        return deepcopy(self.markup['all_features'])

    @all_features.setter
    def all_features(self, value):
        raise AttributeError('can be reset only through `set_markup`')

    @all_features.deleter
    def all_features(self):
        raise AttributeError('can be reset only through `set_markup`')

    @property
    def cat_features(self):
        return deepcopy(self.markup['cat_features'])

    @cat_features.setter
    def cat_features(self, value):
        raise AttributeError('can be reset only through `set_markup`')

    @cat_features.deleter
    def cat_features(self):
        raise AttributeError('can be reset only through `set_markup`')

    @property
    def numeric_features(self):
        return deepcopy(self.markup['numeric_features'])

    @numeric_features.setter
    def numeric_features(self, value):
        raise AttributeError('can be reset only through `set_markup`')

    @numeric_features.deleter
    def numeric_features(self):
        raise AttributeError('can be reset only through `set_markup`')

    @property
    def target_name(self):
        return self.markup['target_name']

    @target_name.setter
    def target_name(self, value):
        raise AttributeError('can be reset only through `set_markup`')

    @target_name.deleter
    def target_name(self):
        raise AttributeError('can be reset only through `set_markup`')

    @property
    def treatment_name(self):
        return self.markup['treatment_name']

    @treatment_name.setter
    def treatment_name(self, value):
        raise AttributeError('can be reset only through `set_markup`')

    @treatment_name.deleter
    def treatment_name(self):
        raise AttributeError('can be reset only through `set_markup`')

    @property
    def score_name(self):
        return self.markup['score_name']

    @score_name.setter
    def score_name(self, value):
        raise AttributeError('can be reset only through `set_markup`')

    @score_name.deleter
    def score_name(self):
        raise AttributeError('can be reset only through `set_markup`')

    @property
    def id_field(self):
        return self.markup['id_field']

    @id_field.setter
    def id_field(self, value):
        raise AttributeError('can be reset only through `set_markup`')

    @id_field.deleter
    def id_field(self):
        raise AttributeError('can be reset only through `set_markup`')

    @property
    def date_main_field(self):
        return self.markup['date_main_field']

    @date_main_field.setter
    def date_main_field(self, value):
        raise AttributeError('can be reset only through `set_markup`')

    @date_main_field.deleter
    def date_main_field(self):
        raise AttributeError('can be reset only through `set_markup`')

    def _get_features_list(self, features: Union[str, List[str]] = 'all'):
        if isinstance(features, str):
            if features not in ('all', 'cat', 'numeric'):
                raise ValueError(f"unknown features type: '{features}'")
            else:
                features_list = self.markup[f'{features}_features']
            return features_list
        elif isinstance(features, list):
            return features
        else:
            raise ValueError(f"unknown features type: {features}")

    def _get_base(self, sample_name: Union[str, List[str]], subset = None, sample_size = None):
        if subset:
            return self._sample(self._filter(self._vstack(sample_name))[subset], sample_size = sample_size)
        else:
            return self._sample(self._filter(self._vstack(sample_name)), sample_size = sample_size)

    def _dedup_list_entries(self, values: List) -> List:
        seen = set()
        result = []
        for v in values:
            if v not in seen:
                result.append(v)
                seen.add(v)
        return result

    def get_X(
        self,
        sample_name: Union[str, List[str]],
        features: Union[str, List[str]] = 'all',
        sample_size: int = None,
    ) -> pd.DataFrame:
        """features is "all", "numeric" or "cat" or list of columns"""
        features_list = self._dedup_list_entries(self._get_features_list(features))
        return self._get_base(sample_name, subset = features_list, sample_size = sample_size)

    def get_Y(self, sample_name: Union[str, List[str]], sample_size: int = None):
        return self._get_base(sample_name, subset = self.target_name, sample_size = sample_size)
            
    def get_T(self, sample_name: Union[str, List[str]], sample_size: int = None):
        return self._get_base(sample_name, subset = self.treatment_name, sample_size = sample_size)

    def get_id(self, sample_name: Union[str, List[str]], sample_size: int = None):
        return self._get_base(sample_name, subset = self.markup['id_field'], sample_size = sample_size)

    def get_XY(
        self,
        sample_name: Union[str, List[str]],
        features: Union[str, List[str]] = 'all',
        sample_size: int = None,
        stacked: bool = False,
    ):
        keys = [self.target_name]
        features_list = self._dedup_list_entries(self._get_features_list(features))
        subset = [f for f in features_list if f not in keys] + keys
        data = self._get_base(sample_name, subset = subset, sample_size = sample_size)
        if stacked:
            return data
        else:
            return (
                data[features_list], data[self.target_name]
            )

    def get_XY_id(
        self,
        sample_name: Union[str, List[str]],
        features: Union[str, List[str]] = 'all',
        sample_size: int = None,
        stacked: bool = False,
    ):
        id_field = self.markup['id_field']
        keys = [self.target_name, id_field]
        features_list = self._dedup_list_entries(self._get_features_list(features))
        subset = [f for f in features_list if f not in keys] + keys
        data = self._get_base(sample_name, subset = subset, sample_size = sample_size)
        if stacked:
            return data
        else:
            return (
                data[features_list], data[self.target_name], data[id_field]
            )

    def get_XYS(
        self,
        sample_name: Union[str, List[str]],
        features: Union[str, List[str]] = 'all',
        score_name: str = None,
        sample_size: int = None,
        stacked: bool = False,
    ):
        score_name = self._get_score_name(score_name)
        keys = [self.target_name, score_name]
        features_list = self._dedup_list_entries(self._get_features_list(features))
        subset = [f for f in features_list if f not in keys] + keys
        data = self._get_base(sample_name, subset = subset, sample_size = sample_size)
        if stacked:
            return data
        else:
            return (
                data[features_list], data[self.target_name], data[score_name]
            )

    def get_XYT(
        self,
        sample_name: Union[str, List[str]],
        features: Union[str, List[str]] = 'all',
        sample_size: int = None,
        stacked: bool = False,
    ):
        keys = [self.target_name, self.treatment_name]
        features_list = self._dedup_list_entries(self._get_features_list(features))
        subset = [f for f in features_list if f not in keys] + keys
        data = self._get_base(sample_name, subset = subset, sample_size = sample_size)
        if stacked:
            return data
        else:
            return (
                data[features_list], data[self.target_name], data[self.treatment_name]
            )

    def get_TT(self, sample_name: Union[str, List[str]], sample_size: int = None, stacked: bool = False):
        subset = [self.target_name, self.treatment_name]
        data = self._get_base(sample_name, subset = subset, sample_size = sample_size)
        if stacked:
            return data
        return (
            data[self.target_name], data[self.treatment_name]
        )

    def get_target_and_treatment(
        self,
        sample_name: Union[str, List[str]],
        sample_size: int = None,
        stacked: bool = False,
    ):
        return self.get_TT(sample_name, sample_size, stacked)

    def _get_score_name(self, score_name: str = None):
        if score_name is None:
            if self.markup['score_name'] is None:
                raise ValueError('sample has not been scored yet')
            else:
                return self.markup['score_name']
        else:
            return score_name

    def get_score(self, sample_name: Union[str, List[str]], score_name: str = None, sample_size: int = None):
        """
        by providing 'score_name' explicitly you actually can get any feature
        """
        score_name = self._get_score_name(score_name)
        return self._get_base(sample_name, subset = score_name, sample_size = sample_size)

    def get_S(self, sample_name: Union[str, List[str]], score_name: str = None, sample_size: int = None):
        """
        by providing 'score_name' explicitly you actually can get any feature
        """
        return self.get_score(sample_name = sample_name, score_name = score_name, sample_size = sample_size)
        
    def get_TS(
        self,
        sample_name: Union[str, List[str]],
        score_name: str = None,
        sample_size: int = None,
        stacked: bool = False,
    ):
        """
        by providing 'score_name' you actually can get any feature
        """
        score_name = self._get_score_name(score_name)
        subset = [self.target_name, score_name]
        data = self._get_base(sample_name, subset = subset, sample_size = sample_size)
        if stacked:
            return data
        else:
            return (
                data[self.target_name], data[score_name]
            )

    def get_target_and_score(
        self,
        sample_name: Union[str, List[str]],
        score_name: str = None,
        sample_size: int = None,
        stacked: bool = False,
    ):
        return self.get_TS(sample_name, score_name, sample_size, stacked)

    def get_true_and_predicted(
        self,
        sample_name: Union[str, List[str]],
        score_name: str = None,
        sample_size: int = None,
        stacked: bool = False,
    ):
        return self.get_TS(sample_name, score_name, sample_size, stacked)

    def get_TST(
        self,
        sample_name: Union[str, List[str]],
        score_name: str = None,
        sample_size: int = None,
        stacked: bool = False,
    ):
        """
        by providing 'score_name' you actually can get any feature
        """
        score_name = self._get_score_name(score_name)
        subset = [self.target_name, score_name, self.treatment_name]
        data = self._get_base(sample_name, subset = subset, sample_size = sample_size)
        if stacked:
            return data
        else:
            return (
                data[self.target_name], data[score_name], data[self.treatment_name]
            )
            
    def get_target_score_treatment(
        self,
        sample_name: Union[str, List[str]],
        score_name: str = None,
        sample_size: int = None,
        stacked: bool = False,
    ):
        return self.get_TST(sample_name, score_name, sample_size, stacked)

    @staticmethod
    def score_by_batches(
        sample: pd.DataFrame,
        model_keeper: ModelKeeper,
        method: str = 'predict',
        batch_size: int = 100000,
        **kwargs,
    ) -> pd.Series:
    
        n = len(sample)
        indices = sample.index

        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size parameter should be an integer, got {type(batch_size)}")
        if n != len(set(indices)):
            raise ValueError("indices in the sample dataset are not unique")
        if not (min(indices) == 0 and max(indices) == n - 1):
            raise ValueError("indices in the sample dataset are not sequential, should be [0, n-1]")
        
        n_batches = n // batch_size + int(n % batch_size > 0)

        batches = []
        for i in tqdm(range(n_batches)):
            batch_result = model_keeper.score(
                sample.iloc[(i * batch_size): ((i+1) * batch_size), :],
                method = method,
                **kwargs,
            )
            batches.append(pd.Series(batch_result))

        score = pd.concat(batches, axis = 0)
        score.index = indices
        return score

    def score_samples(
        self,
        model_keeper: ModelKeeper,
        samples: str = 'active',
        features: Union[str, List[str]] = 'all',
        score_name: str = None,
        method: str = 'proba',
        batch_size: int = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        method is either 'predict', 'proba' or 'threshold'
        in the last two cases arguments might be provided accordingly through **kwargs
        when `score_name` is None, the one stated in markup will be used, or you will be asked to provide it
        """
        logger = logging.getLogger(self.logger_name)

        if self.markup.get('score_name') is not None:
            if score_name is None:
                ### reusing existing name
                score_name = self.markup['score_name']
                do_score = input(
                    f"another score name was set in markup before and will be used as no other name was provided, "
                    "this might overwrite existing values - want to continue? (y/n): "
                )
                if do_score.lower() == 'y':
                    pass
                else:
                    logger.info('exiting...')
            else:
                if self.markup['score_name'] != score_name:
                    if verbose:
                        logger.info(
                            f"another score name was set in markup before: '{self.markup['score_name']}', "
                            f"but using provided name '{score_name}' instead, "
                            f"init score name in markup will not be changed"
                        )
        else:
            if score_name is None:
                score_name = str(input("provide a name for score column (this will be set in markup): "))
                self.set_markup(score_name = score_name)
        
        if isinstance(samples, str):
            if samples == 'active':
                samples = self.active
            else:
                samples = [samples] if samples in self.active else []
        elif isinstance(samples, (list, tuple, set)):
            samples = [s for s in samples if s in self.active]
        else:
            raise ValueError("unknown 'samples' argument")
        
        for sample_name in samples:
            if verbose:
                logger.info(f"scoring '{sample_name}' sample with '{method}' method under '{score_name}' name")

            index = self._get_base(sample_name = sample_name).index
            if batch_size is not None:
                # self[sample_name][score_name] = self.score_by_batches(
                self[sample_name].loc[index, score_name] = self.score_by_batches(
                    sample = self[sample_name].loc[index, :],
                    model_keeper = model_keeper,
                    method = method,
                    batch_size = batch_size,
                    **kwargs
                )
            else:
                self[sample_name].loc[index, score_name] = model_keeper.score(
                    self.get_X(sample_name, features),
                    method = method,
                    **kwargs
                )

    def apply_func_to_samples(
        self,
        func: Callable,
        use_name: bool = False,
        samples: List[Union[str, pd.DataFrame]] = None,
        **kwargs
    ):
        """
        set `use_name` = True e.g. when func is this class's method or does something like .get_XY(sample_name)
        """
        results = {}
        if not samples:
            samples = self.active
        for sample_name in samples:
            sample_name = self._check_sample_name(sample_name)
            if use_name:
                results[sample_name] = func(sample_name, **kwargs)
            else:
                results[sample_name] = func(self[sample_name], **kwargs)
        return results    

    @staticmethod
    def _metric_function_wrapper_for_saving(
        metric: Callable,
        y_true: Iterable,
        y_score: Iterable,
        **kwargs
    ):
        """ this method tries to work not only with get_ metrics functions """
        results = metric(y_true, y_score, **kwargs)

        if isinstance(results, tuple):
            numeric_results = results[0]
            plotted_results = results[1]
        else:
            if isinstance(results, plt.Figure):
                numeric_results = None
                plotted_results = results
            else:
                numeric_results = results
                plotted_results = None                

        return numeric_results, plotted_results

    def calc_metrics_from_samples(
        self,
        metrics: Dict[str, Callable],
        samples: List[str] = None,
        score_name: str = None,
        save_path: Union[str, Path] = None,
        **kwargs
    ):
        
        results = {}
        if not samples:
            samples = self.active
        for metric_name, metric in metrics.items():
            metrics_results = {}
            for sample_name in samples:
                sample_name = self._check_sample_name(sample_name)
                numeric_results, plotted_results = self._metric_function_wrapper_for_saving(
                    metric, *self.get_true_and_predicted(sample_name, score_name), **kwargs
                )
                metrics_results[sample_name] = {
                    'numeric_results': numeric_results,
                    'plotted_results': plotted_results
                }
                if save_path is not None:
                    if plotted_results is not None:
                        plotted_results.savefig(Path(save_path) / f"{metric_name}_{sample_name}.png")
                    if isinstance(numeric_results, pd.DataFrame):
                        numeric_results.to_parquet(
                            Path(save_path) / f"{metric_name}_{sample_name}.pq",
                            engine = 'pyarrow', compression = 'snappy',
                        )
                    elif numeric_results is None:
                        pass
                    else:
                        save_object(numeric_results, save_path, f"{metric_name}_{sample_name}")
            results[metric_name] = metrics_results
        return results

    def get_weights(self, sample_name: str):
        sample_name = self._check_sample_name(sample_name)
        if self.markup.get('weight_field'):
            w = self._filter(self[sample_name])[self.markup.get('weight_field')]
        else:
            w = np.ones(self._filter(self[sample_name]).shape[0])
        return w


class ModelKeeper:
    """
    Wrapper around tabular ML-model with sklearn-like api
    implements some features like optional pre- and post- processing provided as callables,
    thresholding for probability-based models and rearranging features to match model schema if available
    """
    def __init__(
        self,
        model,
        features: List[str] = None,
        preprocessing: Callable = None,
        postprocessing: Callable = None,
    ):

        self.objective = None
        if isinstance(model, (LGBMClassifier, ) + SKLEARN_CLASSIFIERS):
            self.objective = 'classification'
        elif isinstance(model, (LGBMRegressor, ) + SKLEARN_REGRESSORS):
            self.objective = 'regression'
        elif isinstance(model, BaseLearner):
            self.objective = 'uplift'
        else:
            warnings.warn(f"unknown model type {type(model)}, some methods might not work as expected, use with caution")
        self.model = model

        self.features_ordered = features
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, data: pd.DataFrame):
        if self.preprocessing is not None:
            return self.preprocessing(data)
        return data
        
    def postprocess(self, score: pd.Series):
        if self.postprocessing is not None:
            return self.postprocessing(score)
        return score

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        *args,
        **kwargs,
    ) -> Union[pd.Series, np.ndarray]:
        if isinstance(data, pd.DataFrame):
            features = list(data.columns)
        else:
            features = None
        fit_result = self.model.fit(data, *args, **kwargs)
        ### if fit is successful, remember the feature order
        self.features_ordered = features
        return fit_result

    def try_rearrange(self, data: Union[pd.DataFrame, np.ndarray]):
        if isinstance(data, np.ndarray):
            return self.preprocess(data)
        elif isinstance(data, pd.DataFrame):
            if self.features_ordered is None:
                return data
            else:
                return self.preprocess(data[self.features_ordered])
        else:
            raise TypeError("only pd.DataFrame and np.ndarray types are supported")

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        *args,
        **kwargs,
    ) -> Union[pd.Series, np.ndarray]:
        return self.postprocess(self.model.predict(self.try_rearrange(data), *args, **kwargs))

    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        binary: bool = True,
        *args,
        **kwargs,
    ) -> Union[pd.Series, np.ndarray]:
        """
        if binary - returns only results for second class
        """
        if binary:
            return self.postprocess(self.model.predict_proba(self.try_rearrange(data), *args, **kwargs)[:, 1])
        else:
            return self.postprocess(self.model.predict_proba(self.try_rearrange(data), *args, **kwargs))

    def predict_by_threshold(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        threshold: Union[float, np.ndarray, pd.Series] = 0.5,
        binary: bool = True,
        single_class_prediction: bool = True,
        list_predicted_classes: bool = False,
        *args,
        **kwargs,
    ) -> Union[pd.Series, np.ndarray]:
        """
        when probability is more or equal to threshold == positive prediction for class (1), else negative (0)
        
        if binary, returns only results for second class
        if not binary:
            if single_class_prediction is False, then provided thresholds must match the number of classes by dimensions

            if classes are supposed to be mutually exclusive (softmax-like paradigm) then
            it's possible that highest proba would be below given threshold
            even if single_class_prediction is True then -1 is returned

            when classes are not mutually exclusive - every probability in prediction can be any value from [0, 1]
            then for every observation:
                a binary mask for classes on every observation is returned if list_predicted_classes is False
                otherwise result is a list (N_observations) of lists and each inner list could be length of [0, N_classes]

        Note that postprocessing does not apply here
        """
        
        if binary:
            return (self.model.predict_proba(self.try_rearrange(data), *args, **kwargs)[:,1] >= threshold).astype(int)
        elif single_class_prediction:
            x = self.model.predict_proba(self.try_rearrange(data)) # shape (N_obs x N_classes)
            ## first finding the class with maximum probability
            icx = np.argmax(x, axis = 1)  # shape (N_obs, )
            ## then finding the maximum probability itself
            pvx = np.max(x, axis = 1)  # shape (N_obs, )
            ## returning class index where maximum probability above the threshold
            return np.where(pvx >= threshold, icx, -1)
        else:
            x = self.model.predict_proba(self.try_rearrange(data), *args, **kwargs)
            m = x.shape[1]
            l = len(threshold)
            if m != l:
                raise ValueError(f"length of thresholds array ({l}) and number of predicted classes ({m}) do not match")
            else:
                threshold = np.asarray(threshold)
                if not list_predicted_classes:
                    return np.where(x >= threshold, 1, 0)
                else:
                    ### row-col indices for nonzero elements
                    inzx = pd.DataFrame(np.transpose((x >= threshold).nonzero()))
                    ### list of predicted classes for every obs where N_predicted_classes > 0
                    inzx = inzx.groupby(0).aggregate(lambda s: list(s.unique()))
                    return [inzx.loc[i, 1] if i in inzx.index else [] for i in range(x.shape[0])]

    def score(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        method: str = None,
        **kwargs,
    ) -> Union[pd.Series, np.ndarray]:
        if method is None and self.objective is not None:
            ### we'll try to fallback onto what makes the most sense in most cases
            if self.objective == 'classification':
                method = 'proba'
            elif self.objective in ('regression', 'uplift'):
                method = 'predict'

        if method == 'proba':
            return self.predict_proba(data = data, **kwargs)
        elif method == 'threshold':
            return self.predict_by_threshold(data = data, **kwargs)
        elif method in SCORING_METHODS:
            return self.predict(data = data, **kwargs)
        else:
            raise ValueError(
                f"got unknown 'method' value: '{method}' - could be 'predict' for any objective, "
                "'value', 'regression' for regression models, "
                "'proba', 'classification', 'class' or 'threshold' for classification models, "
                "'uplift' for uplift models"
            )
