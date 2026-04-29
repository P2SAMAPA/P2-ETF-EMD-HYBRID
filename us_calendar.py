US trading calendar utilities.
Provides function to get the next NYSE trading day (excluding weekends and holidays).
"""

from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Create a custom business day calendar for NYSE
# NYSE holidays: standard US market holidays (we use pandas built‑in)
# Additional specific closures (e.g., Good Friday, early close) can be added.
# For simplicity, we use USFederalHolidayCalendar which covers most major holidays.
# Note: Good Friday is not a federal holiday but NYSE is closed – we add it.
class NYSECalendar(USFederalHolidayCalendar):
    # Add Good Friday (floating, but pandas holiday calendar can handle via rule)
    # We'll manually define Good Friday as a Holiday instance
    from pandas.tseries.holiday import Holiday, GoodFriday
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

us_bd = CustomBusinessDay(calendar=NYSECalendar())

def next_trading_day(from_date=None):
    """
    Returns the next NYSE trading day as a datetime.date object.
    If from_date is None, uses today's date.
    """
    if from_date is None:
        from_date = datetime.today()
    # If from_date itself is a trading day, the next trading day is the following one.
    # We want the next available trading day after today (or after given date).
    # Offset by 1 business day.
    next_date = from_date + us_bd
    return next_date.date()

def is_trading_day(date):
    """Check if given date is a NYSE trading day."""
    return bool(pd.bdate_range(start=date, end=date, freq=us_bd).size == 1)

def last_trading_day(from_date=None):
    """Returns the most recent trading day (including from_date if it's a trading day)."""
    if from_date is None:
        from_date = datetime.today()
    # Roll backward to the last trading day
    # Use custom business day offset with negative sign
    last_date = from_date - us_bd
    return last_date.date()

# Example usage in a Streamlit display:
# from us_calendar import next_trading_day
# st.write(f"Next trading day: {next_trading_day()}")
