"""
This Python script imports necessary libraries for a stock prediction project. It includes libraries for data manipulation (pandas, numpy), machine learning (sklearn, torch), financial data (yfinance, ta), sentiment analysis (TextBlob), optimization (cvxpy), visualization (matplotlib, seaborn), and others (os, datetime).
"""

import math
import seaborn as sns
from textblob import TextBlob
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import money_flow_index
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import money_flow_index
from textblob import TextBlob
import yfinance as yf
import cvxpy as cp
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os