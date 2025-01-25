import QuantLib as ql
import pandas as pd
from datetime import datetime

def pydatetime_to_quantlib_date(py_datetime: datetime) -> ql.Date:
    """Convert Python datetime to QuantLib Date."""
    return ql.Date(py_datetime.day, py_datetime.month, py_datetime.year)

def quantlib_date_to_pydatetime(ql_date: ql.Date):
    """Convert QuantLib Date to Python datetime."""
    return datetime(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())