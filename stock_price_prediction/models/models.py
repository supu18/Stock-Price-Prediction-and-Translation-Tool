"""
This Python code defines a simple trading strategy and a neural network model for stock prediction. It includes functions to train and evaluate the model, calculate error metrics, and simulate trades based on the model's predictions. The model is trained using mean squared error loss and the Adam optimizer.
"""

from utils.imports import *
from utils.config import *


# Define a simple trading strategy
def simple_trading_strategy(predicted_returns):
    """
    Simple trading strategy: Buy if predicted return is positive, Sell if negative.
    
    Parameters:
    - predicted_returns (numpy.ndarray): Array of predicted returns.
    
    Returns:
    - trades (numpy.ndarray): Array of trading decisions. 1 represents buying, -1 represents selling.
    """
    trades = np.where(predicted_returns > 0, 1, -1)
    return trades


# Define the neural network model
class StockPredictor(nn.Module):
    """
    StockPredictor is a neural network model for stock prediction.

    Args:
        input_size (int): The size of the input features.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        relu (nn.ReLU): The ReLU activation function.
        fc2 (nn.Linear): The second fully connected layer.

    Methods:
        forward(x): Performs forward pass through the network.

    """

    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Performs forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Function to train the model
def train_model(model, X_train_tensor, y_train_tensor, num_epochs=1000):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Function to evaluate the model
def evaluate_model(model, X_test_tensor):
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
    return test_predictions.numpy()


# Function to calculate metrics
def calculate_metrics(y_test, test_predictions):
    mae = mean_absolute_error(y_test, test_predictions)
    mse = mean_squared_error(y_test, test_predictions)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


# Function to simulate trades and calculate trading metrics
def simulate_trades(y_test, test_predictions):
    test_trades = simple_trading_strategy(test_predictions)
    test_returns = test_trades * y_test

    total_trades = len(test_trades)
    positive_trades = np.sum(test_trades[test_trades > 0])
    negative_trades = total_trades - positive_trades
    win_ratio = positive_trades / total_trades
    profit_factor = np.sum(test_returns[test_returns > 0]) / np.abs(np.sum(test_returns[test_returns < 0]))
    cumulative_returns = np.cumsum(test_returns)
    max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
    sharpe_ratio = np.mean(test_returns) / np.std(test_returns)

    return total_trades, positive_trades, negative_trades, win_ratio, profit_factor, cumulative_returns, max_drawdown, sharpe_ratio,test_trades