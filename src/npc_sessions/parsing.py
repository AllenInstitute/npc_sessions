import re

ID = r"^(_?[a-zA-Z0-9-:]_?)+$"

YEAR = r"(20[1-2][0-9])"
MONTH = r"(0[1-9]|10|11|12)"
DAY = r"(0[1-9]|[1-2][0-9]|3[0-1])"
DATE_SEP = r"[-/]?"
PARSE_DATE = rf"{YEAR}{DATE_SEP}{MONTH}{DATE_SEP}{DAY}"

HOUR = r"([0-1][0-9]|[2][0-3])"
MINUTE = r"([0-5][0-9])"
SECOND = r"([0-5][0-9])"
SUBSECOND = r"([0-9]{1,6})"
TIME_SEP = r"[-:.]?"
_TIME = rf"{HOUR}{TIME_SEP}{MINUTE}{TIME_SEP}{SECOND}(\.{SUBSECOND})?"
# avoid parsing time alone without a preceding date:
# if seperators not present will falsely match 8-digit numbers with low values

PARSE_DATETIME = rf"{PARSE_DATE}\D{_TIME}"
PARSE_DATE_OPTIONAL_TIME = rf"{PARSE_DATE}(\D{_TIME})?"

SUBJECT = r"([0-9]{6,7})"
PARSE_SUBJECT = rf"(?<![0-9]){SUBJECT}(?![0-9])"
PARSE_SESSION_INDEX = r"_[0-9]+$"
PARSE_SESSION_ID = rf"{PARSE_SUBJECT}[_ ]+{PARSE_DATE_OPTIONAL_TIME}[_ ]+({PARSE_SESSION_INDEX})?"  # does not allow time after date

VALID_DATE = rf"^{YEAR}-{MONTH}-{DAY}$"
VALID_TIME = rf"^{HOUR}:{MINUTE}:{SECOND}$"
VALID_DATETIME = rf"^{VALID_DATE.strip('^$')}\s{VALID_TIME.strip('^$')}$"
VALID_SUBJECT = rf"^{SUBJECT}$"
VALID_SESSION_ID = (
    rf"^{VALID_SUBJECT.strip('^$')}_{VALID_DATE.strip('^$')}({PARSE_SESSION_INDEX})?$"
)


def strip_non_numeric(s: str) -> str:
    """Remove all non-numeric characters from a string.

    >>> strip_non_numeric('2021-06-01_10:12/34.0')
    '202106011012340'
    """
    return re.sub("[^0-9]", "", s)


def extract_isoformat_date(s: str) -> str | None:
    """Extract and normalize date from a string.
    Return None if no date found.

    >>> extract_isoformat_date('2021-06-01_10-00-00')
    '2021-06-01'
    >>> extract_isoformat_date('20210601_100000')
    '2021-06-01'
    """
    match = re.search(PARSE_DATE, s)
    if not match:
        return None
    value = strip_non_numeric(match.group(0))
    return value[:4] + "-" + value[4:6] + "-" + value[6:8]


def extract_isoformat_time(s: str) -> str | None:
    """Extract and normalize time from a string.
    Return None if no time found.

    >>> extract_isoformat_time('2021-06-01_10:12:03')
    '10:12:03'
    >>> extract_isoformat_time('20210601_101203')
    '10:12:03'
    >>> extract_isoformat_time('20209900_251203') == None
    True
    """
    match = re.search(PARSE_DATETIME, s)
    if not match:
        return None
    value = re.sub(PARSE_DATE, "", match.group(0))
    value = strip_non_numeric(value)
    return value[:2] + ":" + value[2:4] + ":" + value[4:6]


def extract_subject(s: str) -> int | None:
    """Extract subject ID from a string.
    Return None if no subject ID found.

    >>> extract_subject('366122_2021-06-01_10:12:03_3')
    366122
    >>> extract_subject('366122_20210601_101203')
    366122
    >>> extract_subject('366122_20210601')
    366122
    >>> extract_subject('0123456789')

    """
    # remove date and time to narrow down search
    s = re.sub(PARSE_DATE_OPTIONAL_TIME, "", s)
    match = re.search(PARSE_SUBJECT, s)
    if not match:
        return None
    return int(strip_non_numeric(match.group(0)))


def extract_session_index(s: str) -> int | None:
    """Extract appended session index from a string.

    >>> extract_session_index('366122_2021-06-01_10:12:03_3')
    3
    >>> extract_session_index('366122_2021-06-01')

    """
    # remove other components to narrow down search
    s = re.sub(PARSE_DATE_OPTIONAL_TIME, "", s)
    s = re.sub(PARSE_SUBJECT, "", s)
    match = re.search(PARSE_SESSION_INDEX, s)
    if not match:
        return None
    return int(match.group(0).strip("_"))


def extract_session_id(s: str, include_null_index: bool = False) -> str:
    """Extract session ID from a string.
    Raises ValueError if no session ID found.

    >>> extract_session_id('2021-06-01_10:12:03_366122')
    '366122_2021-06-01'
    >>> extract_session_id('366122_20210601_101203')
    '366122_2021-06-01'
    >>> extract_session_id('366122_20210601_1')
    '366122_2021-06-01_1'
    >>> extract_session_id('DRPilot_366122_20210601')
    '366122_2021-06-01'
    >>> extract_session_id('3661_12345678_1')
    Traceback (most recent call last):
    ...
    ValueError: Could not extract date and subject from 3661_12345678_1
    """
    date = extract_isoformat_date(s)
    subject = extract_subject(s)
    index = extract_session_index(s) or 0
    if not date or not subject:
        raise ValueError(f"Could not extract date and subject from {s}")
    return f"{subject}_{date}" + (f"_{index}" if index or include_null_index else "")


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
