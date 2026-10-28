from typing import Tuple, Union
import pandas as pd
import numpy as np
import datetime

STRFTIME_DEFAULT = "%Y-%m-%d %H:%M:%S"

TIME_RESOLUTIONS = {
    "second": "S",
    "s": "S",
    "minute": "MINUTE",
    "min": "MINUTE",
    "mt": "MINUTE",
    "t": "MINUTE",
    "hour": "H",
    "h": "H",
    "day": "D",
    "d": "D",
    "week": "W",
    "w": "W",
    "month": "MONTH",
    "m": "MONTH",
    "year": "Y",
    "y": "Y",
}


def get_unix_time_s(
    utc_timestamp: Union[str, pd.Timestamp, datetime.datetime, pd.Series],
    rounding: bool = False,
) -> Union[float, int, pd.Series]:
    """
    converts utc timestamp value into offset (in seconds) from 0 utc timestamp
    """
    tp = None
    if isinstance(utc_timestamp, (list, tuple)):
        tp = type(utc_timestamp)
        utc_timestamp = pd.Series(utc_timestamp)

    epoch = datetime.datetime.utcfromtimestamp(0)
    if isinstance(utc_timestamp, pd.Series):
        s = (utc_timestamp - epoch).dt.total_seconds()
        if rounding:
            s = s.round().astype(int)
        if tp is not None:
            s = tp(s)
        return s
    elif isinstance(utc_timestamp, str):
        ### attempt for blind conversion
        utc_timestamp = pd.to_datetime(utc_timestamp)
    s = (utc_timestamp - epoch).total_seconds()
    if rounding:
        return int(np.round(s))
    return s


def get_unix_time_ms(
    utc_timestamp: Union[str, pd.Timestamp, datetime.datetime, pd.Series],
    rounding: bool = False,
) -> Union[float, int, pd.Series]:
    """
    converts utc timestamp value into offset (in milliseconds) from 0 utc timestamp
    """
    s = get_unix_time_s(utc_timestamp, rounding = False) * 1000.0
    if rounding:
        if isinstance(s, pd.Series):
            return s.round().astype(int)
        else:
            return int(np.round(s))
    return s


def get_unix_time_seconds(
    utc_timestamp: Union[str, pd.Timestamp, datetime.datetime, pd.Series],
    rounding: bool = False,
) -> Union[float, int, pd.Series]:
    """
    Converts a UTC timestamp value into the offset (in seconds) from the Unix epoch (0 UTC timestamp).

    :param utc_timestamp: Input UTC timestamp value as a string, pandas Timestamp, datetime.datetime, or pandas Series.
    :param rounding: Flag indicating whether to round the result to the nearest second and return int (default: False).
    :return: Offset from the Unix epoch in seconds as a float, int or pandas Series.
    :rtype: Union[str, pd.Timestamp, datetime.datetime, pd.Series]

    :Example:
    >>> utc_timestamp = '2007-08-09 12:34:56'
    >>> get_unix_time_seconds(utc_timestamp, rounding=True)
    1186662896

    >>> utc_timestamp = pd.Timestamp('2007-08-09 12:34:56')
    >>> get_unix_time_seconds(utc_timestamp, rounding=False)
    1695923696.0

    >>> utc_timestamp = datetime.datetime(2007, 8, 9, 12, 34, 56)
    >>> get_unix_time_seconds(utc_timestamp)
    1695923696.0

    >>> utc_timestamp = pd.Series(['2007-08-09 12:34:56', '2001-04-05 10:20:30'])
    >>> get_unix_time_seconds(utc_timestamp, rounding=True)
    0    1186662896
    1     986466030
    dtype: int64
    """

    return get_unix_time_s(utc_timestamp = utc_timestamp, rounding = rounding)


def truncate_timestamp_minutes(
    ts: Union[str, pd.Timestamp, datetime.datetime, pd.Series],
    window_minutes: int,
    shift_minutes: int = 0,
) -> Union[pd.Timestamp, pd.Series]:
    """
    Truncates a timestamp to the nearest minute interval and applies an optional shift.

    :param ts: Input timestamp as a string, pandas Timestamp, datetime.datetime, or pandas Series.
    :param window_minutes: Size of the minute window to truncate to.
    :param shift_minutes: Optional shift in minutes to apply after truncation (default: 0).
    :return: Truncated and shifted timestamp as a pandas Timestamp, datetime.datetime, or pandas Series.
    :rtype: Union[pd.Timestamp, datetime.datetime, pd.Series]

    :Example:
    >>> ts = '2007-08-09 12:34:56'
    >>> truncate_timestamp_minutes(ts, 15, shift_minutes=0)
    Timestamp('2007-08-09 12:30:00')

    >>> ts = datetime.datetime(2007, 8, 9, 12, 34, 56)
    >>> truncate_timestamp_minutes(ts, 15)
    Timestamp('2007-08-09 12:30:00')

    >>> ts = pd.Series(['2007-08-09 12:34:56', '2001-04-05 10:20:30'])
    >>> truncate_timestamp_minutes(ts, 15, shift_minutes=3)
    0   2007-08-09 12:33:00
    1   2001-04-05 10:18:00
    dtype: datetime64[ns]
    """

    unix_time_offset = get_unix_time_seconds(ts)
    window_size_seconds = int(np.round(60 * window_minutes))
    unix_time_offset_div = unix_time_offset // window_size_seconds
    unix_time_offset_truncated = unix_time_offset_div * window_size_seconds
    ts_truncated = pd.to_datetime(unix_time_offset_truncated, unit="s")
    ts_truncated_shifted = ts_truncated + pd.Timedelta(shift_minutes, unit="m")
    return ts_truncated_shifted


def get_week_bounds(
    ts: Union[datetime.datetime, pd.Timestamp, pd.Series]
) -> Union[Tuple[pd.Timestamp, pd.Timestamp], Tuple[pd.Series, pd.Series]]:
    """
    Returns the beginning and the end of the week for provided timestamp(s) as Monday/Sunday both at 00:00:00.
    """

    if not isinstance(ts, (str, datetime.datetime, pd.Timestamp, pd.DatetimeIndex, pd.Series, list, tuple)):
        raise TypeError("unknown argument type")

    tp = type(ts)
    ts = pd.to_datetime(ts)

    if isinstance(ts, pd.DatetimeIndex):
        ### to enable usage of .dt
        ts = pd.Series(ts)

    if isinstance(ts, pd.Series):
        day_of_week = ts.dt.weekday
        to_beginning_of_week = pd.to_timedelta(day_of_week, unit='days')
        beginning_of_week = ts - to_beginning_of_week
        beginning_of_week = beginning_of_week.dt.normalize()
        to_end_of_week = pd.to_timedelta(6 - day_of_week, unit='days')
        end_of_week = ts + to_end_of_week
        end_of_week = end_of_week.dt.normalize()
    else:
        day_of_week = ts.weekday()
        to_beginning_of_week = pd.Timedelta(days = day_of_week)
        beginning_of_week = ts - to_beginning_of_week
        beginning_of_week = beginning_of_week.replace(minute=0, hour=0, second=0, microsecond=0)
        to_end_of_week = pd.Timedelta(days = 6 - day_of_week)
        end_of_week = ts + to_end_of_week
        end_of_week = end_of_week.replace(minute=0, hour=0, second=0, microsecond=0)

    beginning_of_week, end_of_week = tp(beginning_of_week), tp(end_of_week)

    return beginning_of_week, end_of_week


def get_week_beginning(ts: pd.Series) -> pd.Series:
    """
    Returns the end of the week for provided timestamp(s) as Monday of such week at 00:00:00.

    :Example:
    >>> >>> ts = pd.Series(['2007-08-09 12:34:56', '2001-04-05 10:20:30'])
    >>> get_week_beginning(ts)
    0   2007-08-06
    1   2001-04-02
    dtype: datetime64[ns]
    """
    beginning_of_week, end_of_week = get_week_bounds(ts)
    return beginning_of_week


def get_week_end(ts: pd.Series) -> pd.Series:
    """
    Returns the end of the week for provided timestamp(s) as Sunday of such week at 00:00:00.

    :Example:
    >>> >>> ts = pd.Series(['2007-08-09 12:34:56', '2001-04-05 10:20:30'])
    >>> get_week_end(ts)
    0   2007-08-12
    1   2001-04-08
    dtype: datetime64[ns]
    """
    beginning_of_week, end_of_week = get_week_bounds(ts)
    return end_of_week


def create_grouping_ts_field(
    data: pd.DataFrame,
    timestamp_col: str,
    freq: str = "s",
    timestamp_format: str = None,
    to_str: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Creates a grouping column in the input DataFrame (inplace) based on a specified frequency.

    :param data: Input pandas DataFrame.
    :param timestamp_col: Name of the column containing the timestamps.
    :param freq: Frequency for grouping ("mt" or "t" for minute, "m" for month, "h" for hour, etc.) (default: "s").
    :param timestamp_format: Optional format for parsing the timestamps (default: None).
    :param to_str: Flag indicating whether to store grouping values as strings instead of pd.timestamps (default: False).
    :return: Tuple containing the modified DataFrame and the name of the grouping column.
    :rtype: Tuple[pd.DataFrame, str]

    :Example:
    >>> data = pd.DataFrame({
    ...     'timestamp': ['2007-08-09 12:34:56', '2007-08-09 23:45:00'],
    ...     'value': [0, 1]
    ... })
    >>> data, grouping_column = create_grouping_ts_field(data, 'timestamp', freq='D', to_str=True)
    >>> data[['timestamp', grouping_column]]
                 timestamp __tsfield__
    0  2007-08-09 12:34:56  2007-08-09
    1  2007-08-09 23:45:00  2007-08-09
    """
    if not isinstance(freq, str) or freq.lower() not in TIME_RESOLUTIONS:
        raise TypeError(
            f"'freq' parameter must be one of {sorted(TIME_RESOLUTIONS.keys())}, got {freq}"
        )
    freq = TIME_RESOLUTIONS[freq.lower()]
    ts_group_field = "__tsfield__"
    while ts_group_field in data.columns:
        ts_group_field = "_" + ts_group_field + "_"
    kws = {"format": timestamp_format} if timestamp_format is not None else {}
    data[ts_group_field] = pd.to_datetime(data[timestamp_col], **kws)
    # data[ts_group_field] = data[ts_group_field].dt.floor(freq=freq)
    if freq == "W":
        if not to_str:
            data[ts_group_field] = get_week_beginning(data[ts_group_field])
        else:
            data[ts_group_field] = data[ts_group_field].dt.strftime("%Y-%W")
    else:
        if freq == "Y":
            data[ts_group_field] = data[ts_group_field].dt.strftime("%Y")
            suffix = "-01-01"
        elif freq == "MONTH":
            data[ts_group_field] = data[ts_group_field].dt.strftime("%Y-%m")
            suffix = "-01"
        elif freq == "D":
            data[ts_group_field] = data[ts_group_field].dt.strftime("%Y-%m-%d")
            suffix = ""
        elif freq == "H":
            data[ts_group_field] = data[ts_group_field].dt.strftime("%Y-%m-%d %H")
            suffix = ":00"
        elif freq == "MINUTE":
            data[ts_group_field] = data[ts_group_field].dt.strftime("%Y-%m-%d %H:%M")
            suffix = ":00"
        elif freq == "S":
            data[ts_group_field] = data[ts_group_field].dt.strftime("%Y-%m-%d %H:%M:%S")
            suffix = ""
        if not to_str:
            data[ts_group_field] = pd.to_datetime(data[ts_group_field] + suffix)

    return data, ts_group_field


def normalize_to_date(
    ts: Union[int, float, str, pd.Timestamp, pd.Series, pd.DatetimeIndex],
    to_str: bool = False,
) -> str:
    if isinstance(ts, (int, float)):
        dt = pd.to_datetime(ts, unit = 's').floor(freq = 'd')
    elif isinstance(ts, (str, pd.Timestamp)):
        dt = pd.to_datetime(ts).floor(freq = 'd')
    elif isinstance(ts, (pd.Series, pd.DatetimeIndex)):
        dt = pd.to_datetime(ts).dt.floor(freq = 'd')
    else:
        raise TypeError("received unexpected timestamp type")

    if to_str:
        return dt.strftime(STRFTIME_DEFAULT)[:10]
    return dt


def get_long_ago() -> pd.Timestamp:
    return pd.to_datetime(0, unit = 'ms')


def get_utc_unix_time_ms_now() -> int:
    return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp() * 1000)


def get_n_days_from_today(n_days: int, utc: bool = False) -> pd.Timestamp:
    if utc:
        #### daystart utc unix seconds
        utc_today = pd.to_datetime(get_utc_unix_time_ms_now(), unit = 'ms')
        return pd.to_datetime(utc_today + datetime.timedelta(days = n_days)).normalize()
    return pd.to_datetime(datetime.datetime.today() + datetime.timedelta(days = n_days)).normalize()


def get_yesterday(utc: bool = False) -> pd.Timestamp:
    return get_n_days_from_today(-1, utc = utc)


def get_today(utc: bool = False) -> pd.Timestamp:
    return get_n_days_from_today(0, utc = utc)


def get_tomorrow(utc: bool = False) -> pd.Timestamp:
    return get_n_days_from_today(1, utc = utc)
