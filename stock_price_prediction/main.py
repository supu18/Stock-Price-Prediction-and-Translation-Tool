"""
    Author: Supriya Jayarj
    Date: 2024-02-10

    This script is the main entry point for the stock price prediction project. It fetches historical stock data, performs sentiment analysis on financial news, and visualizes the data using various plots. It also trains a deep learning model to predict stock returns and evaluates the model's performance. Finally, it simulates trades and calculates trading metrics based on the model's predictions.
"""
from utils.imports import *
from utils.data_processing import get_stock_data
from utils.sentiment_analysis import analyze_sentiment
from utils.visualizations import plot_graphs
from utils.optimize_portfolio_analysis import optimize_portfolio
from utils.config import *
from models.models import *

# Fetch historical stock data
stock_prices_all = pd.concat({
    ticker: get_stock_data(ticker) for ticker in TICKERS
}, axis=1)
stock_prices_all = stock_prices_all.stack(level=0)
stock_prices_all.reset_index(inplace=True)

# Convert the 'Date' column to datetime type
stock_prices_all['Date'] = pd.to_datetime(stock_prices_all['Date'])

# Select Germany-specific stock data (e.g., using DAX index)
germany_data = get_stock_data(GERMANY_TICKER)

# Merge Germany data with individual stock data
stock_prices_all = pd.merge(
    stock_prices_all,
    germany_data['Adj Close'].rename('Germany'),
    left_on='Date',
    right_index=True,
    how='left'
)

# Drop rows with missing values (if any)
stock_prices_all.dropna(inplace=True)

# Calculate daily returns
returns = stock_prices_all.pivot(
    index='Date',
    columns='level_1',
    values='Adj Close'
).pct_change().dropna()


# Plot separate graphs for each metric (Volume, Closing Price, 
# and Daily Return) and each company
for metric in METRICS:
    plt.figure(figsize=(15, 8))  # Increase the figure height
    plt.subplots_adjust(top=0.9, bottom=0.2, hspace=0.6, wspace=0.2)

    for i, ticker in enumerate(TICKERS):
        plt.subplot(2, 2, i + 1)

        if metric == 'Daily Return':
            # Calculate daily returns for the current company
            returns_ticker = returns[ticker]
            sns.histplot(
                returns_ticker,
                bins=30,
                kde=True,
                element="step",
                fill=False,
                color=colorblind_colors[i]
            )
            plt.title(f'{COMPANY_NAME[i]} - {metric} Histogram')
        else:
            # Plots the current metric for the current company
            sns.lineplot(
                x='Date',
                y=metric,
                data=stock_prices_all[stock_prices_all['level_1'] == ticker],
                color=colorblind_colors[i]
            )
            plt.title(f'{COMPANY_NAME[i]} - {metric} over Time')

        # Set xlabel based on the metric
        if metric == 'Volume':
            plt.xlabel('Date')
            plt.ylabel('Volume of Sales')
        elif metric == 'Daily Return':
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')
        # Rotate x-axis labels to 45 degrees
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}{metric.replace(' ', '_')}_histogram.png")
    # Show the plot for each metric
    plt.show()

optimal_weights = optimize_portfolio(returns, RISK_TOLERANCE)
print("Optimal Weights:", optimal_weights)

# Plot stock prices over time using seaborn style
plt.figure(figsize=(15, 8))
for i, ticker in enumerate(TICKERS):
    sns.lineplot(x='Date', y='Adj Close',
                data=stock_prices_all[stock_prices_all['level_1'] == ticker],
                label=f'{COMPANY_NAME[i]} - Stock Price')

plot_graphs('Stock Prices Over Time', 'Date', 'Stock Price', 'colorblind')


# Create subplots with one row and three columns
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
sns.set_palette('Dark2')
for i, ticker in enumerate(TICKERS):
    # Calculate Moving Averages
    ma_30 = stock_prices_all[stock_prices_all['level_1'] == ticker]['Adj Close'].rolling(window=30).mean()
    ma_60 = stock_prices_all[stock_prices_all['level_1'] == ticker]['Adj Close'].rolling(window=60).mean()
    ma_90 = stock_prices_all[stock_prices_all['level_1'] == ticker]['Adj Close'].rolling(window=90).mean()

    # Plot Adjusted Close
    axes[i].plot(stock_prices_all[stock_prices_all['level_1'] == ticker]['Date'],
                stock_prices_all[stock_prices_all['level_1'] == ticker]['Adj Close'],
                label='Adj Close', color='m', alpha=0.8)

    # Plot 30-day Moving Average
    axes[i].plot(stock_prices_all[stock_prices_all['level_1'] == ticker]['Date'], ma_30, label='MA 30', color='b')

    # Plot 60-day Moving Average
    axes[i].plot(stock_prices_all[stock_prices_all['level_1'] == ticker]['Date'], ma_60, label='MA 60', color='y')

    # Plot 90-day Moving Average
    axes[i].plot(stock_prices_all[stock_prices_all['level_1'] == ticker]['Date'], ma_90, label='MA 90', color='r')

    # Set y-label and title
    axes[i].set_ylabel('Stock Price')
    axes[i].set_title(f'{COMPANY_NAME[i]} - Stock Price with Moving Averages')

    # Add legend on the right side
    axes[i].legend(loc='upper right', bbox_to_anchor=(0.91, 1.01))
    # Rotate x-axis labels to 45 degrees
    axes[i].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
plt.savefig(f"{SAVE_PATH} Stock_Price_with_Moving_Averages.png")
plt.show()

# Plot Relative Strength Index (RSI) over time
plt.figure(figsize=(15, 8))


for ticker, name in zip(TICKERS, COMPANY_NAME):
    data = stock_prices_all[stock_prices_all['level_1'] == ticker]
    rsi = RSIIndicator(data['Adj Close'])
    sns.lineplot(x=data['Date'], y=rsi.rsi().values, label=f'{name} - RSI')

plot_graphs('Relative Strength Index (RSI) Over Time', 'Date', 'RSI', 'tab20')

plt.style.use('fivethirtyeight')

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))

# Iterate through the tickers and corresponding subplots
for i, (ticker, ax) in enumerate(zip(TICKERS, axes.flatten())):
    bb = BollingerBands(stock_prices_all[stock_prices_all['level_1'] == ticker]['Adj Close'])

    # Plot Adjusted Closing Price
    ax.plot(stock_prices_all[stock_prices_all['level_1'] == ticker]['Date'], stock_prices_all[stock_prices_all['level_1'] == ticker]['Adj Close'], label=f'{COMPANY_NAME[i]} - Close', color='red', lw=2)

    # Plot Bollinger Bands
    ax.plot(stock_prices_all[stock_prices_all['level_1'] == ticker]['Date'], bb.bollinger_mavg().values, label=f'{ticker} - Middle Band', color='black', lw=2)
    ax.fill_between(stock_prices_all[stock_prices_all['level_1'] == ticker]['Date'], bb.bollinger_hband().values, bb.bollinger_lband().values, color='grey', label="Band Range", alpha=0.3)

    # Set title and labels
    ax.set_title(f'Bollinger Bands for {COMPANY_NAME[i]}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prices')
    ax.legend()

    # Rotate x-axis labels to 45 degrees for better visibility
    ax.tick_params(axis='x', rotation=45)

# Remove the empty subplot in the last row
fig.delaxes(axes[1, 1])
plt.legend(loc='upper right', bbox_to_anchor=(0.91, 1.01))
# Adjust layout to move the third graph to the center
fig.subplots_adjust(top=0.9, bottom=0.2, hspace=0.6, wspace=0.2)
plt.tight_layout()  # Adjust the rect parameter to fit the title
plt.savefig(f"{SAVE_PATH} Bollinger_Bands.png")
# Show the plot
plt.show()


plt.style.use('seaborn')
# Plot Money Flow Index (MFI) over time
plt.figure(figsize=(15, 8))
for ticker, name in zip(TICKERS, COMPANY_NAME):
    data = stock_prices_all[stock_prices_all['level_1'] == ticker]
    mfi = money_flow_index(
        data['High'],
        data['Low'],
        data['Adj Close'],
        data['Volume']
    )
    sns.lineplot(x=data['Date'], y=mfi.values, label=f'{name} - MFI')

plot_graphs('Money Flow Index (MFI) Over Time', 'Date', 'MFI', 'deep')

df_sentiment = analyze_sentiment(TICKERS)

plt.figure(figsize=(15, 8))
sns.set_palette('Set2')
sns.scatterplot(
    data=df_sentiment, x='Time', y='Sentiment Score',
    hue='Company', hue_order=TICKERS,
    palette=sns.color_palette('bright', len(TICKERS)), style='Company',
    markers=['o'] * len(TICKERS), legend='full')

plot_graphs('Sentiment Scores Over Time for Companies', 'Time',
            'Sentiment Score', 'muted')

# Calculate correlations with stock prices
correlations = returns.corr()

plt.figure(figsize=(15, 8))
sns.set_palette('viridis')
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap with Stock Prices')
plt.tight_layout()
plt.savefig(f"{SAVE_PATH} Correlation_Heatmap_with_Stock_Prices.png")
plt.show()

# Log transformation of returns
log_returns = np.log(1 + returns)

# Standardization
standardized_returns = (log_returns - log_returns.mean()) / \
    log_returns.std()

# Min-Max normalization
normalized_returns = (log_returns - log_returns.min()) / \
                        (log_returns.max() - log_returns.min())

print("Original Returns DataFrame:")
print(returns.head())
print("\nLog-transformed Returns DataFrame:")
print(log_returns.head())
print("\nStandardized Returns DataFrame:")
print(standardized_returns.head())
print("\nNormalized Returns DataFrame:")
print(normalized_returns.head())

# Add 'Germany' to the daily returns DataFrame
returns['Germany'] = germany_data['Adj Close'].pct_change().dropna()

# Prepare the features (X) and target variable (y) for the model
X = returns[['TSLA', 'META', 'AMD', 'Germany']].values
y = returns['TSLA'].values  # Predicting Apple's stock returns

# Split the data into training and test sets
train_size = int(len(stock_prices_all) * 0.8)
train_data = stock_prices_all[:train_size]
test_data = stock_prices_all[train_size:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the training and test data
plt.figure(figsize=(15, 8))
sns.lineplot(x='Date', y='Adj Close', data=train_data, label='Training Data')
sns.lineplot(x='Date', y='Adj Close', data=test_data, label='Test Data')
plot_graphs(
    'Training and Test Data for Closing Price',
    'Date',
    'stock price',
    'Dark2'
)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Instantiate the model
input_size = X_train_tensor.shape[1]
model = StockPredictor(input_size)

# Train the model
train_model(model, X_train_tensor, y_train_tensor, num_epochs=NUM_EPOCHS)

# Evaluate the model on the test set
test_predictions = evaluate_model(model, X_test_tensor)

common_length = min(len(test_data), len(y_test), len(test_predictions.flatten()))

# Slice the arrays to the common length
test_data_slice = test_data.iloc[-common_length:]

# Create the DataFrame
results = pd.DataFrame({
    'Date': test_data_slice['Date'].values,
    'Actual Returns': y_test[:common_length],
    'Predicted Returns': test_predictions.flatten()[:common_length],
    'Adj Close': test_data_slice['Adj Close'].values
})


plt.figure(figsize=(15, 8))
plt.plot(results['Date'], y_test, label='Actual Returns')
plt.plot(results['Date'], test_predictions, label='Predicted Returns')
plot_graphs(
    'Actual vs. Predicted Stock Returns for TESLA',
    'Date',
    'Stock Price (%)',
    'husl'

)
# Calculate metrics
mae, mse, rmse = calculate_metrics(y_test, test_predictions)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Flatten the arrays
test_predictions_flat = test_predictions.flatten()
y_test_flat = y_test.flatten()


# Calculate volatility of actual and predicted returns
actual_volatility = np.std(y_test_flat)
predicted_volatility = np.std(test_predictions_flat)

print("Actual Volatility:", actual_volatility)
print("Predicted Volatility:", predicted_volatility)

date_values = stock_prices_all['Date']

# Truncate 'date_values' to match the length of 'y_test_flat'
date_values = date_values[:len(y_test_flat)]

# Plot the actual vs predicted graph with volatility
plt.figure(figsize=(15, 8))
plt.plot(date_values, y_test_flat, label='Actual Returns')
plt.plot(date_values, test_predictions_flat, label='Predicted Returns')
plt.fill_between(date_values, test_predictions_flat - predicted_volatility, test_predictions_flat + predicted_volatility, color='gray', alpha=0.2, label='Predicted Volatility')
plt.fill_between(date_values, y_test_flat - actual_volatility, y_test_flat + actual_volatility, color='yellow', alpha=0.2, label='Actual Volatility')
plot_graphs('Actual vs Predicted Stock Returns with Volatility', 'Date', 'Returns(%)', 'Set2')


# Simulate trades and calculate trading metrics
total_trades, positive_trades, negative_trades, win_ratio, profit_factor, cumulative_returns, max_drawdown, sharpe_ratio, test_trades = simulate_trades(y_test, test_predictions)

# Print trading metrics
print("Total Trades:", total_trades)
print("Positive Trades:", positive_trades)
print("Negative Trades:", negative_trades)
print("Win Ratio:", win_ratio)
print("Profit Factor:", profit_factor)
print("Max Drawdown:", max_drawdown)
print("Sharpe Ratio:", sharpe_ratio)


plt.figure(num="Actual vs Predicted with Simulated Trades", figsize=(15, 8))  # Specify a unique identifier for the figure

date_values = stock_prices_all['Date']

# Ensure that 'y_test', 'test_predictions', and 'test_trades' have the same length
assert len(y_test) == len(test_predictions) == len(test_trades), "Lengths of arrays must match"

# Plot actual and predicted returns
plt.plot(date_values[:len(y_test)], y_test, label='Actual Returns')  # Truncate date_values to match the length of y_test
plt.plot(date_values[:len(test_predictions)], test_predictions, label='Predicted Returns')

# Plot buy signals
buy_indices = np.where(test_trades == 1)[0]
plt.scatter(date_values[buy_indices], y_test[buy_indices], color='green', marker='^', label='Buy Signal')

# Plot sell signals
sell_indices = np.where(test_trades == -1)[0]
plt.scatter(date_values[sell_indices], y_test[sell_indices], color='red', marker='v', label='Sell Signal')

plot_graphs(
    'Actual vs Predicted Stock Returns with Simulated Trades',
    'Date',
    'Stock Price',
    'bright'
)
