from typing import List, Tuple, Union
import numpy as np


def calc_weighted_sum(
    x: Union[List, np.ndarray],
    w: Union[List, np.ndarray]
) -> float:
    """
    Calculates the weighted sum of values.

    :param x: Input values as a list or NumPy array.
    :param w: Input weights as a list or NumPy array.
    :raises ValueError: If the input arrays are empty or have incompatible shapes.
    :return: The weighted sum.
    :rtype: float

    :Example:
    >>> x = [1, 2, 3, 4, 5]
    >>> w = [0.25, 0.25, 0.1, 0.1, 0.3]
    >>> calc_weighted_sum(x, w)
    2.95
    """

    if len(x) == 0 or len(w) == 0:
        raise ValueError("Input arrays must not be empty.")
    elif len(x) != len(w):
        raise ValueError("Input arrays must have compatible shapes.")

    return float(np.dot(x, w))


def calc_weighted_mean(
    x: Union[List, np.ndarray],
    w: Union[List, np.ndarray]
) -> float:
    """
    Calculates the weighted mean of values.

    :param x: Input values as a list or NumPy array.
    :param w: Input weights as a list or NumPy array.
    :raises ValueError: If the input arrays are empty or have incompatible shapes.
    :return: The weighted mean.
    :rtype: float

    :Example:
    >>> x = [1, 2, 3, 4, 5]
    >>> w = [25, 25, 10, 10, 30]
    >>> calc_weighted_mean(x, w)
    2.95
    """
    return float(calc_weighted_sum(x, w) / np.sum(w))


def calc_weighted_cov(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    w: Union[List, np.ndarray],
    denominator_correction: float = 0.0,
) -> float:
    """
    Calculates the weighted covariance between two arrays of random variables.

    :param x: Input values for the first variable as a list or NumPy array.
    :param y: Input values for the second variable as a list or NumPy array.
    :param w: Input weights as a list or NumPy array.
    :param denominator_correction: Value to subtract from the denominator for correction (default: 0.0).
    :raises ValueError: If the input arrays are empty or have incompatible shapes.
    :return: The weighted covariance.
    :rtype: float

    :Example:
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [0, 1, 2, -1, 4]
    >>> w = [25, 25, 10, 10, 30]
    >>> calc_weighted_cov(x, y, w)
    2.1275
    """

    if len(x) == 0 or len(y) == 0 or len(w) == 0:
        raise ValueError("Input arrays must not be empty.")
    elif len(x) != len(y) or len(x) != len(w):
        raise ValueError("Input arrays must have compatible shapes.")
    x, y, w = np.array(x), np.array(y), np.array(w)
    return np.sum(
        w * (x - calc_weighted_mean(x, w)) * (y - calc_weighted_mean(y, w))
    ) / (np.sum(w) - denominator_correction)


def calc_weighted_corr(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    w: Union[List, np.ndarray],
) -> float:
    """
    Calculates the weighted correlation coefficient between two variables.

    :param x: Input values for the first variable as a list or NumPy array.
    :param y: Input values for the second variable as a list or NumPy array.
    :param w: Input weights as a list or NumPy array.
    :raises ValueError: If the input arrays are empty or have incompatible shapes.
    :return: The weighted correlation coefficient.
    :rtype: float

    :Example:
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [0, 1, 2, -1, 4]
    >>> w = [25, 25, 10, 10, 30]
    >>> round(calc_weighted_corr(x, y, w), 2)
    0.75
    """

    return calc_weighted_cov(x, y, w) / np.sqrt(
        calc_weighted_cov(x, x, w) * calc_weighted_cov(y, y, w)
    )


def calc_abs_and_rel_change(
    x: Union[int, float],
    y: Union[int, float]
) -> Tuple[float, Union[float, None]]:
    """
    Calculates the absolute and relative change between two values.

    :param x: First value as an integer or float.
    :param y: Second value as an integer or float.
    :return: Tuple containing the absolute change as float and relative change as float or None if y is close to 0.
    :rtype: Tuple[float, Union[float, None]]

    :Example:
    >>> x = 3
    >>> y = 2
    >>> calc_abs_and_rel_change(x, y)
    (1, 0.5)

    >>> x = 1
    >>> y = 0
    >>> calc_abs_and_rel_change(x, y)
    (1, None)
    """
    abs_change = x - y
    if not np.isclose(y, 0.0, atol=1e-8):
        rel_change = (x - y) / y
    else:
        rel_change = None
    return abs_change, rel_change
