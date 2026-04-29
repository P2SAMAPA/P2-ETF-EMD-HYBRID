"""
US trading calendar utilities.
Provides functions to get the next/last NYSE trading day.
Handles weekends and US market holidays (including Good Friday).
"""

from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday, GoodFriday
from pandas.tseries.offsets import CustomBusinessDay

# Define NYSE calendar: US federal holidays + Good Friday
class NYSECalendar(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

# Custom business day frequency using the NYSE calendar
nyse_business_day = CustomBusinessDay(calendar=NYSECalendar())

def next_trading_day(from_date=None):
    """
    Returns the next NYSE trading day as a datetime.date object.
    If from_date is None, uses today's date.
    If from_date is a trading day, returns the following trading day.
    """
    if from_date is None:
        from_date = datetime.today()
    # Ensure from_date is a datetime object
    if isinstance(from_date, datetime.date) and not isinstance(from_date, datetime):
        from_date = datetime.combine(from_date, datetime.min.time())
    # Next trading day: add one business day
    next_date = from_date + nyse_business_day
    return next_date.date()

def last_trading_day(from_date=None):
    """
    Returns the most recent NYSE trading day (including from_date if it is a trading day).
    """
    if from_date is None:
        from_date = datetime.today()
    if isinstance(from_date, datetime.date) and not isinstance(from_date, datetime):
        from_date = datetime.combine(from_date, datetime.min.time())
    # Subtract one business day (rolls back to previous trading day)
    last_date = from_date - nyse_business_day
    return last_date.date()

def is_trading_day(date):
    """Check if a given date is a NYSE trading day."""
    if isinstance(date, datetime.date) and not isinstance(date, datetime):
        date = datetime.combine(date, datetime.min.time())
    # Create a date range of one day; if it contains a business day, it's a trading day
    return bool(pd.date_range(start=date, periods=1, freq=nyse_business_day).size == 1)
