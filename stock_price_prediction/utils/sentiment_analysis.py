"""
The analyze_sentiment function performs sentiment analysis on financial news for given stock tickers. It fetches news for each ticker using yfinance, extracts the text, and calculates sentiment scores using TextBlob. The function returns a DataFrame with the ticker, news publish time, and sentiment score for each news item.
"""

from utils.imports import *
from .config import *


def analyze_sentiment(ticker):
    """
    Perform sentiment analysis on financial news for a given ticker.

    Args:
        ticker (str): Ticker symbol of the stock.

    Returns:
        pandas.DataFrame: DataFrame containing sentiment analysis results.
    """
    news_sentiment = []

    for ticker in TICKERS:
        news = yf.Ticker(ticker).news

        if news:
            for item in news:
                combined_text = item.get('summary') or item.get('title')
                if not combined_text:
                    print(
                        f"No suitable text found for sentiment "
                        f"analysis in news item for {ticker}.")
                    continue
                blob = TextBlob(combined_text)
                sentiment_score = blob.sentiment.polarity
                news_sentiment.append((
                    ticker,
                    datetime.utcfromtimestamp(item['providerPublishTime']),
                    sentiment_score))
                print(
                    f"Sentiment score for {ticker} on "
                    f"{item['providerPublishTime']}: {sentiment_score:.2f}")
        else:
            print(f"No news available for {ticker} for sentiment analysis.")

    df_sentiment = pd.DataFrame(
        news_sentiment,
        columns=['Company', 'Date', 'Sentiment Score'])
    # Convert 'Date' column to datetime
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])

    # Convert 'Date' column to time format
    df_sentiment['Time'] = df_sentiment['Date'].dt.strftime('%I:%M %p')

    return df_sentiment