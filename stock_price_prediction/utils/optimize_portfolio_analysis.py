"""
The optimize_portfolio function optimizes portfolio weights based on expected returns and risk tolerance. It formulates an optimization problem using cvxpy, where the objective is to maximize the expected return minus the risk tolerance times the risk, subject to the constraints that the weights sum to 1 and are non-negative.
"""

from utils.imports import *
from .config import *


def optimize_portfolio(returns, risk_tolerance):
    """
    Optimize portfolio weights based on expected returns and risk tolerance.

    Args:
        returns (pandas.DataFrame): DataFrame of historical returns.
        risk_tolerance (float): Desired level of risk tolerance.

    Returns:
        numpy.ndarray: Optimal portfolio weights.
    """
    # Ensure expected_returns is a 1D array
    expected_returns = np.ravel(np.mean(returns, axis=0))

    # Define the optimization problem
    weights = cp.Variable(len(expected_returns))
    risk = cp.quad_form(weights, np.cov(returns, rowvar=False))
    objective = cp.Maximize(expected_returns @ weights - risk_tolerance * risk)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem
    problem.solve()

    # Get the optimal weights
    optimal_weights = weights.value

    # Ensure optimal weights are non-negative
    optimal_weights[optimal_weights < 0] = 0

    return optimal_weights
