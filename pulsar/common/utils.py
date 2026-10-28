from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
from functools import partial
import inspect
import warnings
import pathos.multiprocessing
import pandas as pd
import numpy as np


try:
    from IPython.display import display
except ModuleNotFoundError:
    def display(*args, **kwargs):
        pass


def change_docstring(func_with_doc: Callable, mode: str = "append") -> Callable:
    """
    Decorator that changes the docstring of a function using docstring of function provided as first argument.

    :param func_with_doc: Function with the desired docstring.
    :param mode: Mode of changing the docstring: "append", "prepend", or "replace" (default: "append").
    :return: Decorator function that changes the docstring.
    :rtype: Callable
    """

    def wrapper(func: Callable) -> Callable:
        if func.__doc__ is None:
            func.__doc__ = ""
        if mode == "append":
            func.__doc__ = func.__doc__ + func_with_doc.__doc__
        elif mode == "prepend":
            func.__doc__ = func_with_doc.__doc__ + func.__doc__
        elif mode == "replace":
            func.__doc__ = func_with_doc.__doc__
        else:
            raise ValueError("Invalid mode. Mode must be 'append', 'prepend', or 'replace'.")
        return func

    return wrapper


def filter_kwargs(f: Callable, kwargs: Dict) -> Dict:
    """
    Filters keyword arguments based on the parameters of the given function.

    :param f: Function to filter keyword arguments for.
    :param kwargs: Keyword arguments to filter.
    :return: Filtered keyword arguments.
    :rtype: Dict

    :Example:
    >>> def f(a, b, c=0):
    ...     pass
    >>> kwargs = {'a': 1, 'b': 2, 'd': 3}
    >>> filtered_kwargs = filter_kwargs(f, kwargs)
    >>> filtered_kwargs
    {'a': 1, 'b': 2}
    """
    sig = inspect.signature(f)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    filtered_kwargs = {k: kwargs[k] for k in filter_keys if k in kwargs}
    return filtered_kwargs


def safe_kwargs_partial(f: Callable, **kwargs):
    """
    Creates a partial function with keyword arguments filtered based on the parameters of the given function.

    :param f: Function to create a partial function from.
    :param kwargs: Keyword arguments to filter.
    :return: Partial function with filtered keyword arguments.
    :rtype: Callable

    :Example:
    >>> def f(a, b, c=0):
    ...     pass
    >>> kwargs = {'a': 1, 'b': 2, 'd': 3}
    >>> partial_f = safe_kwargs_partial(f, **kwargs)
    >>> partial_f
    functools.partial(<function f at 0x...>, a=1, b=2)
    """
    return partial(f, **filter_kwargs(f, kwargs))


def mp_tqdm_wrapper(f: Callable, verbose: bool = True) -> Callable:
    """
    Wrapper function that enables tqdm-like progress bars for multiprocess functions being displayed in Jupyter.

    :param f: Function to wrap.
    :param verbose: Flag for backward compatibility, when False default behaviour not changed and stdout not affected
    :return: Wrapped function.
    :rtype: Callable
    """
    def inner(*args, **kwargs):
        if verbose:
            # this line is the strange hack for tqdm/mp in jupyter
            print(" ", end="", flush=True)
        return f(*args, **kwargs)

    return inner


def make_star_function(function: Callable, verbose: bool = True) -> Callable:
    def starred(args: Tuple):
        if verbose:
            # this line is the strange hack for tqdm/mp in jupyter
            print(" ", end="", flush=True)
        return function(*args)
    return starred


def resolve_target_metric_arg(
    target_col: str = None,
    metric_col: str = None,
):
    """
    Allows some methods use both named arguments but verifies only one is used.

    :param target_col: Name of the column containing the target metric variable.
    :param metric_col: Alias for target_col.
    """
    if target_col is None and metric_col is None:
        raise ValueError(
            f"either one of the aliases 'target_col' and 'metric_col' should be provided"
        )
    elif target_col is not None and metric_col is not None:
        raise ValueError(
            f"only one of the aliases 'target_col' and 'metric_col' should be provided"
        )
    target_col = target_col or metric_col
    if not isinstance(target_col, str):
        raise TypeError(
            f"'target_col' or 'metric_col' must be a string"
        )
    return target_col


def split_iterations_per_jobs(n_iterations: int, n_jobs: int) -> List[int]:
    # for the most equal split of an array of length l into n sections
    # we return l % n sub-arrays of size l//n + 1 and the rest of size l//n
    r = n_iterations % n_jobs
    s = n_iterations // n_jobs
    n_iterations_per_job = [s + 1] * r + [s] * (n_jobs - r)
    return n_iterations_per_job


def adjust_n_jobs_for_mpc(
    n_iterations: int,
    n_jobs: Union[int, str] = "auto",
    min_iterations_parallel: int = 10,
    pool: Union[pathos.multiprocessing.Pool, pathos.multiprocessing.ProcessPool, None] = None,
) -> Tuple[int, Union[pathos.multiprocessing.Pool, pathos.multiprocessing.ProcessPool, None]]:
    """
    :param n_iterations: The number of bootstrap iterations to perform.
    :param n_jobs: The number of parallel jobs to run for bootstrap. Can be an integer >= 1 or 'auto' (default: 'auto').
    :param min_iterations_parallel: The number of iterations to actually use multiprocessing with.
    :param pool: Pool of processes (not threads) to parallel upon, should come from pathos or multiprocessing libraries.
    """
    # -2 for a safer option to not take all resources
    n_cpu_available = pathos.multiprocessing.cpu_count() - 2
    if n_jobs == "auto":
        n_jobs = max(n_cpu_available, 1)
    elif n_jobs > 1:
        n_jobs = max(min(n_jobs, n_cpu_available), 1)
    else:
        n_jobs = 1
    if n_iterations <= min_iterations_parallel:
        n_jobs = 1
    n_jobs = min(n_jobs, n_iterations)

    if pool is None:
        if n_jobs > 1:
            # pool = pathos.multiprocessing.ProcessPool(nodes=n_jobs)
            pool = pathos.multiprocessing.Pool(processes=n_jobs)
    else:
        ### try to split into how many workers are there in the pool already
        if hasattr(pool, '_processes'):
            n_pool_workers = getattr(pool, '_processes')
        elif hasattr(pool, '_ProcessPool__nodes'):
            n_pool_workers = getattr(pool, '_ProcessPool__nodes')
        else:
            warnings.warn("unable to detect number of workers in the pool, pool will not be used")
            n_pool_workers = 1
            pool = None
        n_jobs = min(n_jobs, n_pool_workers)

    return n_jobs, pool


def pool_execute(
    pool: Union[pathos.multiprocessing.Pool, pathos.multiprocessing.ProcessPool, None],
    func: Callable,
    iterator: Iterable,
    verbose: bool = False,
) -> Iterable:
    with pool as _pool:
        f = mp_tqdm_wrapper(func, verbose=verbose)
        do_starmap = hasattr(_pool, 'starmap')
        if do_starmap:
            results = _pool.starmap(f, iterator)
        else:
            ### pathos' ProcessPool (which is a wrapper around multiprocessing.Pool) doesn't have .starmap
            ### but somehow with this additional wrapping for .map it suppresses
            ### tqdm progress bars from subprocesses to be displayed in jupyter even with mp_tqdm_wrapper
            # f_star = lambda x: f(*x)
            f_star = make_star_function(f)
            results = _pool.map(f_star, iterator)

    return results


def make_sampling_iteration(
    data: pd.DataFrame,
    randomization_unit_col: str,
    t_size: int,
    c_size: int,
    aggregating_function: Union[Callable, None],
    replace: bool,
    seed: int,
    unique_index_units: set = None,
    enable_relabeling: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    `aggregating_function` is supposed to both accept and return pd.DataFrame, rather than a scalar value
    `enable_relabeling` when set to True allows to create new ids for units under `randomization_unit_col`
        after sampling with replacement is done - to make multiple identical but distinct observations.
        This makes sense when `aggregating_function` is actually creating units on `randomization_unit_col` level,
        e.g. sampling by user and calculating user-level CTR rather global CTR, but it makes things much slower.
        When sampling without replacement this flag has no effect.
    """

    def identity(x: Any) -> Any:
        return x

    if aggregating_function is None:
        aggregating_function = identity

    if unique_index_units is None:
        unique_index_units = data[randomization_unit_col].unique()

    data = data.set_index(randomization_unit_col)

    ### it's really important to set different seeds within each subprocesses, otherwise we'd get identical results
    if seed is not None:
        np.random.seed(seed)

    relabel = enable_relabeling and replace

    if replace:
        t_index = list(np.random.choice(unique_index_units, size = t_size, replace = replace))
        c_index = list(set(unique_index_units).difference(t_index))
        c_index = list(np.random.choice(list(c_index), size = c_size, replace = replace))
    else:
        if t_size + c_size > len(unique_index_units):
            raise ValueError("t_size + c_size should not exceed amount of unique_index_units")
        index = np.random.permutation(unique_index_units)
        t_index, c_index = index[:t_size], index[t_size: t_size + c_size]

    ### t_indexer, c_indexer = data.index.isin(t_index), data.index.isin(c_index)
    ### above is incorrect indexation for sampling when analysis unit is smaller that randomization unit
    ### cause it omits duplicated index values reducing sample size
    t_indexer, c_indexer = t_index, c_index

    ### this is needed for cases when aggregating_function is doing aggregation on randomization_unit_col level
    ### when sampling with replacement we need to distinguish each new synthetic units from each other
    ### this is to avoid them being aggregated into the same single unit
    if relabel:
        data["__pseudoindex__"] = range(len(data))
        data["__r__"] = 1

    def subset_relabel_duplicates(data: pd.DataFrame, indexer: List) -> pd.DataFrame:
        data = data.loc[indexer, :].reset_index(drop = False)
        data["__r__"] = (
            data
            .groupby([randomization_unit_col, "__pseudoindex__"])
            ["__r__"].rank(method = "first", ascending = True)
        )
        data["__randomization_unit_col__"] = data[randomization_unit_col]
        separator = '+++'
        data[randomization_unit_col] = (
            data[[randomization_unit_col, "__r__"]]
            .apply(lambda row: separator.join(row.values.astype(str)), axis = 1)
        )
        return data.set_index(randomization_unit_col, inplace = False)

    def subset_identity(data: pd.DataFrame, indexer: List) -> pd.DataFrame:
        return data.loc[indexer, :]

    def aggregate(data: pd.DataFrame, indexer: List) -> pd.DataFrame:
        f = subset_relabel_duplicates if relabel else subset_identity
        return aggregating_function(
            f(data, indexer)
            .reset_index(drop = False)
            ### resetting cause aggregating_function might be somehow relying on randomization_unit_col
        )

    t_data = aggregate(data, t_indexer)
    c_data = aggregate(data, c_indexer)

    if relabel:
        for d in (data, t_data, c_data):
            d.drop(["__pseudoindex__", "__r__"], axis = 1, inplace = True)
    data.reset_index(drop = False, inplace = True)

    return t_data, c_data
