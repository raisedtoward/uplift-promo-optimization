from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from pulsar.common.constants import CONTROL_LABEL, TREATMENT_LABEL


def assert_treatment_labels(
    labels: Union[np.ndarray, pd.Series, List],
    force_binary: bool = False,
    force_consecutive: bool = False,
) -> None:
    """
    Asserts that the treatment labels meet the required protocol: subset of integer encoded 0, 1, ... n-1.
    Sometimes it is used to check just protocol compliance and sometimes to check that all labels are present.

    :param labels: Treatment labels as a NumPy array, pandas Series, or list.
    :param force_binary: Flag indicating whether to force binary treatment labels check (default: False).
    :param force_consecutive: Flag indicating whether to check if all treatment labels are present (default: False).
    :raises ValueError: If the treatment labels are missing, have only one unique label, or do not match the protocol.
    :return: None
    """
    x_unique = set(labels)
    n_unique = len(x_unique)
    if n_unique == 0:
        raise ValueError(f"treatment labels data are missing")
    elif n_unique == 1:
        raise ValueError(
            f"only 1 unique treatment label is present in the data: {x_unique}, should be at least 2"
        )
    elif force_binary:
        expected_labels = set([CONTROL_LABEL, TREATMENT_LABEL])
    else:
        expected_labels = set(range(n_unique))
    if force_consecutive and x_unique != expected_labels:
        raise ValueError(
            f"{n_unique} unique labels are present in the data: {x_unique}, "
            f"but they do not match expected protocol: {expected_labels}"
        )
    elif len(x_unique.difference(expected_labels)) > 0:
        raise ValueError(
            f"{n_unique} unique labels are present in the data: {x_unique}, "
            f"but they do not match expected protocol: {expected_labels}"
        )


def assert_array_container_type(
    x: Union[np.ndarray, pd.Series, List],
) -> None:
    """
    Asserts that the input array is one of the allowed container types.

    :param x: Input array as a NumPy array, pandas Series, or list.
    :raises TypeError: If the input array is not one of the allowed container types.
    :return: None
    """
    if not isinstance(x, (np.ndarray, pd.Series, List)):
        raise TypeError(
            f"input should be one of the following types: [np.ndarray, pd.Series, list], got {type(x)}"
        )


def validate_input_array(
    x: Union[np.ndarray, pd.Series, List],
    dtype: type = None,
) -> np.ndarray:
    """
    Validates and processes the input array.

    :param x: Input array as a NumPy array, pandas Series, or list.
    :param dtype: Optional data type for the processed array to be casted to (default: None).
    :raises TypeError: If the input array container is not one of the allowed types.
    :raises ValueError: If the array contains NaN or None values or has less than 2 observations.
    :return: Processed and validated array as a NumPy array.
    :rtype: np.ndarray
    """
    assert_array_container_type(x)
    if dtype is not None:
        if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating):
            assert_non_string(x)
        x = np.asarray(x, dtype=dtype)
    else:
        x = np.asarray(x)
    ### pd.isna() also works with str types and None, while np.isnan doesn't
    nan_flags = pd.isna(x)
    if any(nan_flags):
        raise ValueError(
            f"provided array contains {sum(nan_flags)} np.nan and/or None values"
        )
    if len(x) <= 1:
        raise ValueError(
            f"provided array contains less than 2 observations in total: {x}"
        )
    return x


def validate_contingency_table(
    data: pd.DataFrame,
    normalized: bool,
) -> pd.DataFrame:
    """
    Validates and processes a contingency table DataFrame.

    :param data: Contingency table as a pandas DataFrame.
    :param normalized: Flag indicating whether the resulting contingency table should be normalized.
    :raises ValueError: If the index or columns of the contingency table are not unique,
                        if the contingency table contains missing values,
                        or if the contingency table contains negative values.
    :return: Validated and sorted contingency table as a pandas DataFrame.
    :rtype: pd.DataFrame
    """
    if len(data.index) != data.index.nunique():
        raise ValueError("all labels in contingency table index should be unique")
    if len(data.columns) != data.columns.nunique():
        raise ValueError("all categories in contingency table columns should be unique")
    if normalized:
        # in case it's already normalized this shouldn't do any harm
        data = data.div(data.sum(axis=1), axis="index")
    if data.isna().values.any():
        raise ValueError("there should be no missing values in contingency table")
    if (data.values < 0).any():
        raise ValueError("there should be no negative values in contingency table")
    # already should be sorted by both axis, but to be sure we're consistent everywhere
    return data.sort_index(axis=0).sort_index(axis=1)


def assert_discrete_distribution(
    x: Union[np.ndarray, pd.Series, List],
    continuity_cutoff: int = 30,
) -> None:
    """
    Asserts that the input distribution is discrete.

    :param x: Input distribution as a NumPy array, pandas Series, or list.
    :param continuity_cutoff: Max number of unique values for the distribution to be considered discrete (default: 30).
    :raises ValueError: If the number of unique values in the sample exceeds the continuity cutoff.
    :return: None
    """
    n_unique = len(set(x))
    if n_unique > continuity_cutoff:
        raise ValueError(
            f"number of unique values in the sample ({n_unique}) is higher than allowed by continuity cutoff: "
            f"{continuity_cutoff}"
        )


def assert_non_singleton(
    x: Union[np.ndarray, pd.Series, List],
) -> None:
    """
    Asserts that the input sample has more than one unique value.

    :param x: Input sample as a NumPy array, pandas Series, or list.
    :raises ValueError: If the number of unique values in the sample is less than 2.
    :return: None
    """
    n_unique = len(set(x))
    if n_unique <= 1:
        raise ValueError(
            f"number of unique values in the sample ({n_unique}) is less than expected minimum of 2"
        )


def assert_non_string(
    x: Union[np.ndarray, pd.Series, List],
) -> None:
    """
    Asserts that the input array does not contain string values.

    :param x: Input array as a NumPy array, pandas Series, or list.
    :raises TypeError: If the input array contains string values.
    :return: None
    """
    assert_array_container_type(x)

    if isinstance(x, np.ndarray):
        if x.dtype.type is np.object_ or x.dtype.type is np.str_:
            raise TypeError(
                f"array of integer or float type was expected, got {x.dtype.type}"
            )
    elif isinstance(x, pd.Series):
        x = x.values
        if x.dtype.type is np.object_ or x.dtype.type is np.str_:
            raise TypeError(
                f"array of integer or float type was expected, got {x.dtype.type}"
            )
    elif isinstance(x, list):
        if any(type(v) is str for v in x):
            raise TypeError(
                f"array of integer or float type was expected, got at least 1 string value in list"
            )


def convert_to_dataframes(
    treatment_data: Union[pd.DataFrame, pd.Series, np.ndarray, List, Tuple],
    control_data: Union[pd.DataFrame, pd.Series, np.ndarray, List, Tuple],
    metric_col: str,
    treatment_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if (
        (not isinstance(treatment_data, pd.DataFrame) and isinstance(control_data, pd.DataFrame))
        or (isinstance(treatment_data, pd.DataFrame) and not isinstance(control_data, pd.DataFrame))
    ):
        raise TypeError("'treatment_data' and 'control_data' should be both of pd.DataFrame type or both iterables")
    elif not isinstance(treatment_data, pd.DataFrame) and not isinstance(control_data, pd.DataFrame):
        treatment_data = pd.DataFrame({
            metric_col: validate_input_array(treatment_data, dtype = None),
            treatment_col: TREATMENT_LABEL,
        })
        control_data = pd.DataFrame({
            metric_col: validate_input_array(control_data, dtype = None),
            treatment_col: CONTROL_LABEL,
        })

    return treatment_data, control_data
