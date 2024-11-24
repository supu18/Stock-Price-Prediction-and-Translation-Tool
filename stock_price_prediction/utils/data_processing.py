"""
The get_stock_data function retrieves stock data for a given ticker symbol within a specified date range using the yf.download method from the yfinance library. It takes three parameters: ticker_name, start, and end. The start and end parameters default to START_DATE and END_DATE respectively. The function returns a pandas DataFrame containing the downloaded stock data.
"""

from utils.imports import *
from .config import *


def get_stock_data(ticker_name, start=START_DATE, end=END_DATE):
    """
    Retrieves stock data for a given ticker symbol and date range.

    Args:
        ticker (str): The ticker symbol of the stock.
        start_date (str): The start date of the data range in the
            format 'YYYY-MM-DD'.
        end_date (str): The end date of the data range in the
            format 'YYYY-MM-DD'.

    Returns:
        pandas.DataFrame: The stock data for the specified ticker
            and date range.
    """
    stock_data = yf.download(ticker_name, start, end)
    return stock_data