"""
This Python script sets up configurations for a stock prediction project. It defines the seaborn theme, color palette, date ranges for data and predictions, number of training epochs, stock tickers, company names, a Germany-specific ticker, risk tolerance, metrics, and initializes a StandardScaler. It also sets a path for saving images and creates the directory if it doesn't exist. These configurations are used throughout the project to ensure consistency.
"""


from utils.imports import *


sns.set_theme(style='darkgrid')

COLORBLIND_COLORS = sns.color_palette("colorblind", n_colors=20)
START_DATE = '2022-01-01'
END_DATE = datetime.now()
START_DATE_PREDICTION = '2023-01-02'
END_DATE_PREDICTION = '2023-02-01'
NUM_EPOCHS = 1000  # Train the model
TICKERS = ['TSLA', 'META', 'AMD']
COMPANY_NAME = ['TESLA', 'META', 'Advanced Micro Devices']
GERMANY_TICKER = '^GDAXI'  # Select Germany-specific stock data (e.g., using DAX index)
RISK_TOLERANCE = 0.1
NUM_EPOCHS = 1000
METRICS = ['Volume', 'Daily Return']
NUM_METRICS = len(METRICS)
NUM_COMPANIES = len(TICKERS)
SAVE_PATH = "images/"
scaler = StandardScaler()
# Use colorblind-friendly colors from seaborn
colorblind_colors = sns.color_palette("colorblind", n_colors=20)
# Define the path to the images folder

# Ensure the images folder exists, if not, create it
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
