# region imports
from AlgorithmImports import *
# endregion
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data.Market import *
from QuantConnect.Indicators import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import timedelta
import math

class NQTradingStrategy(QCAlgorithm):
    def Initialize(self):
        self.set_start_date(2020, 1, 1)  
        self.set_cash(100000)  
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)  

        self.nq = self.add_future(Futures.Indices.NASDAQ_100_E_MINI, Resolution.MINUTE)
        self.nq.SetFilter(timedelta(0), timedelta(90))

        self.target_vol = 0.30

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleDNN(input_size=30).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.schedule.on(
            self.date_rules.every_day(self.nq.Symbol),
            self.time_rules.at(0, 0),
            self.DailyBiasAndRebalance
        )

        self.set_warm_up(30)

        self.bias = 0
        self.vol_window = 20
        self.contract_multiplier = 20  
        self.recent_vol = 0.0  

    def OnData(self, data):
        if self.nq.Mapped not in data.Bars:
            return
        price = data.Bars[self.nq.Mapped].Close
        self.debug(f"Price: {price}")

    def DailyBiasAndRebalance(self):
        try:
            # Get history - returns pandas DataFrame
            history = self.history([self.nq.Symbol], 31, Resolution.DAILY)
            
            if history.empty:
                self.debug("History is empty")
                return

            # Debug the DataFrame structure
            self.debug(f"History shape: {history.shape}")
            self.debug(f"History columns: {list(history.columns)}")
            self.debug(f"History index levels: {history.index.names}")
            
            # Extract close prices - try multiple approaches
            closes = []
            
            # Method 1: Try direct column access if single symbol
            if 'close' in history.columns:
                closes = history['close'].dropna().values
            elif 'Close' in history.columns:
                closes = history['Close'].dropna().values
            else:
                # Method 2: Multi-index DataFrame - get symbol data
                try:
                    # Try different symbol representations
                    for symbol_key in [self.nq.Symbol, str(self.nq.Symbol), self.nq.Symbol.Value]:
                        try:
                            symbol_data = history.loc[symbol_key]
                            if 'close' in symbol_data.columns:
                                closes = symbol_data['close'].dropna().values
                                break
                            elif 'Close' in symbol_data.columns:
                                closes = symbol_data['Close'].dropna().values
                                break
                        except:
                            continue
                            
                    # Method 3: If still no data, try xs method
                    if len(closes) == 0:
                        symbol_data = history.xs(self.nq.Symbol, level=0)
                        if 'close' in symbol_data.columns:
                            closes = symbol_data['close'].dropna().values
                        elif 'Close' in symbol_data.columns:
                            closes = symbol_data['Close'].dropna().values
                            
                except Exception as e:
                    self.debug(f"Error accessing symbol data: {e}")
                    return
            
            if len(closes) < 30:
                self.debug(f"Insufficient price data: {len(closes)} bars")
                return
            
            # Calculate returns using numpy
            closes_array = np.array(closes, dtype=float)
            returns = np.diff(closes_array) / closes_array[:-1]
            
            # Get last 30 returns for DNN
            if len(returns) < 30:
                self.debug(f"Insufficient returns data: {len(returns)} returns")
                return
                
            recent_returns = returns[-30:]
            features = torch.tensor(recent_returns, dtype=torch.float32).unsqueeze(0).to(self.device)

            # DNN inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(features)
                self.bias = 1 if output.item() > 0 else -1
                self.debug(f"DNN bias: {self.bias}, output: {output.item():.4f}")

            # Calculate volatility from returns array
            if len(returns) >= self.vol_window:
                vol_returns = returns[-self.vol_window:]
                self.recent_vol = float(np.std(vol_returns) * math.sqrt(252))
            else:
                self.debug("Insufficient data for volatility calculation")
                return

            if self.recent_vol == 0:
                self.debug("Zero volatility - skipping rebalance")
                return

            # Position sizing
            capital = self.portfolio.total_portfolio_value
            target_exposure = (self.target_vol / self.recent_vol) * capital

            # Get current price
            current_price = self.securities[self.nq.Mapped].price
            contract_value = current_price * self.contract_multiplier

            if contract_value == 0:
                self.debug("Contract value is zero — skipping rebalance")
                return

            target_contracts = int(target_exposure / contract_value) * self.bias

            # Execute rebalancing
            current_quantity = self.portfolio[self.nq.Mapped].quantity
            if target_contracts != current_quantity:
                order_quantity = target_contracts - current_quantity
                self.market_order(self.nq.Mapped, order_quantity)
                self.debug(f"Rebalanced: {current_quantity} -> {target_contracts} contracts (order: {order_quantity})")
            else:
                self.debug(f"No rebalancing needed. Current: {current_quantity}, Target: {target_contracts}")
                
        except Exception as e:
            self.debug(f"Error in DailyBiasAndRebalance: {str(e)}")
            
class SimpleDNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
