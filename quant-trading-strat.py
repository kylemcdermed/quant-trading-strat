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
        history = self.History(self.nq.mapped, 31, Resolution.DAILY)
        if history is None or len(history) == 0:
            return

        closes = history.close.unstack(level=0)
        returns = closes.pct_change().dropna().values[-30:]
        features = torch.tensor(returns, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(features)
            self.bias = 1 if output.item() > 0 else -1

        vol_returns = closes.pct_change().dropna().values.flatten()[-self.vol_window:]
        self.recent_vol = float(np.std(vol_returns) * math.sqrt(252))

        if self.recent_vol == 0:
            return

        capital = self.Portfolio.TotalPortfolioValue  # type: ignore
        target_exposure = (self.target_vol / self.recent_vol) * capital

        current_price = self.Securities[self.nq.Mapped].Price  # type: ignore
        contract_value = current_price * self.contract_multiplier
        target_contracts = int(target_exposure / contract_value) * self.bias

        current_quantity = self.Portfolio[self.nq.Mapped].Quantity  # type: ignore
        if target_contracts != current_quantity:
            self.MarketOrder(self.nq.Mapped, target_contracts - current_quantity)  # type: ignore

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
