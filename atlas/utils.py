from typing import Any, Dict, Iterable, List, Union
from abc import ABC
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from collections import defaultdict


try:
    from IPython.display import display
except ModuleNotFoundError:
    def display(*args, **kwargs):
        """
        this lets avoid adding Jupyter dependencies in production code
        but keeps functionality for local debugging without changing another source code
        """
        # return print(*args, **kwargs)
        pass


class ArrayLike1d(ABC):
    """ArrayLike1d type that encompasses lists, tuples, Pandas Series, and NumPy arrays."""
    pass


ArrayLikeTypes = (list, tuple, pd.Series, np.ndarray)
for tp in ArrayLikeTypes:
    ArrayLike1d.register(tp)


class BaseModel:
    pass


def ps538():
    plt.plot([], []);
    plt.close()
    plt.style.use('fivethirtyeight')


def filter_kwargs(f, kwargs):
    sig = inspect.signature(f)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_kwargs = {k:kwargs[k] for k in filter_keys if k in kwargs}
    return filtered_kwargs


def safe_kwargs_partial(f, **kwargs):
    return partial(f, **filter_kwargs(f, kwargs))


def drop_columns_safe(data: pd.DataFrame, columns: List[str], inplace: bool = False):
    drop_cols = [f for f in columns if f in data.columns]
    return data.drop(drop_cols, axis = 1, inplace = inplace)


def defaultdict_to_dict(d: defaultdict) -> Dict:
    plain_d = {}
    for k, v in d.items():
        if isinstance(v, defaultdict):
            plain_d[k] = defaultdict_to_dict(v)
        else:
            plain_d[k] = v
    return plain_d


def wrap_as_list(v: Union[Any, Iterable[Any]]) -> List[Any]:
    if not isinstance(v, str) and isinstance(v, Iterable):
        v = list(v)
    else:
        v = [v]
    return v
