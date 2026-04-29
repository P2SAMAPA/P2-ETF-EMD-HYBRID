"""
US trading calendar utilities using pandas.
Provides next trading day (NYSE) handling weekends and holidays.
"""

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pandas.tseries.offsets import CustomBusinessDay

# Define NYSE calendar: US federal holidays + Good Friday
class NYSECalendar(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

# Custom business day frequency
nyse_bd = CustomBusinessDay(calendar=NYSECalendar())

def next_trading_day(from_date=None):
    """
    Returns the next NYSE trading day as a datetime.date object.
    If from_date is None, uses today's date.
    """
    if from_date is None:
        from_date = pd.Timestamp.today()
    else:
        from_date = pd.Timestamp(from_date)
    next_date = from_date + nyse_bd
    return next_date.date()

def last_trading_day(from_date=None):
    """
    Returns the most recent NYSE trading day (including from_date if it is a trading day).
    """
    if from_date is None:
        from_date = pd.Timestamp.today()
    else:
        from_date = pd.Timestamp(from_date)
    last_date = from_date - nyse_bd
    return last_date.date()

def is_trading_day(date):
    """Check if a given date is a NYSE trading day."""
    date = pd.Timestamp(date)
    return bool(pd.bdate_range(start=date, periods=1, freq=nyse_bd).size == 1)
