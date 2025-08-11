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
from datetime import timedelta, time
import math

class OpenRangeGap:
    def __init__(self):
        self.open_price = None
        self.close_price = None
        self.open_time = None
        self.close_time = None
        self.is_valid = False
        self.ce_price = None  # Consequent Encroachment (50% level)
        self.q1_price = None  # 25% level
        self.q3_price = None  # 75% level
        self.size = 0
        self.direction = 0  # 1 for bullish gap, -1 for bearish gap

class FairValueGap:
    def __init__(self):
        self.high = None
        self.low = None
        self.time = None
        self.is_bullish = False
        self.ce_price = None  # Consequent Encroachment (50% level)
        self.is_filled = False
        self.is_first_of_day = False


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

        # schedule daily DNN bias calculation
        self.schedule.on(
            self.date_rules.every_day(self.nq.Symbol),
            self.time_rules.at(0, 0),
            self.DailyBiasAndRebalance
        )

        # schedule ORG detection at market open (9:30 AM ET)
        self.schedule.on(
            self.date_rules.every_day(self.nq.Symbol),
            self.time_rules.at(9, 30),
            self.DailyBiasAndRebalance
        )

        # Schedule ORG detection at market open (9:30 AM ET)
        self.schedule.on(
            self.date_rules.every_day(self.nq.Symbol),
            self.time_rules.at(16, 14),
            self.DailyBiasAndRebalance
        )


        self.set_warm_up(30) 

        # DNN variables
        self.bias = 0
        self.vol_window = 20
        self.contract_multiplier = 20  
        self.recent_vol = 0.0

        # ORG and FVG variables
        self.current_org = OpenRangeGap()
        self.current_fvg = None
        self.daily_fvgs = []
        self.first_fvg_detected = False
        self.session_started = False
        
        # Market session times (Eastern Time)
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 14)
        
        # Store minute bars for FVG detection
        self.minute_bars = []
        self.max_minute_bars = 500  # Keep last 500 minute bars


    def OnData(self, data):
        if self.nq.Mapped not in data.Bars:
            return
            
        current_bar = data.Bars[self.nq.Mapped]
        current_time = self.time
        
        # Store minute bar for FVG analysis
        self.StoreMinuteBar(current_bar, current_time)
        
        # Detect FVG on new minute bars
        if self.IsMarketSession(current_time):
            self.DetectFVG()



    def IsMarketSession(self, time_stamp):
        """Check if current time is within market session (9:30 AM - 4:00 PM ET)"""
        current_time = time_stamp.time()
        return self.market_open_time <= current_time <= self.market_close_time

    def StoreMinuteBar(self, bar, time_stamp):
        """Store minute bars for FVG detection"""
        bar_data = {
            'time': time_stamp,
            'open': float(bar.Open),
            'high': float(bar.High),
            'low': float(bar.Low),
            'close': float(bar.Close),
            'volume': int(bar.Volume)
        }
        
        self.minute_bars.append(bar_data)
        
        # Keep only recent bars
        if len(self.minute_bars) > self.max_minute_bars:
            self.minute_bars.pop(0)

    def CaptureOpeningRange(self):
        """Capture the opening price for ORG calculation"""
        try:
            if self.nq.Mapped in self.securities:
                current_price = self.securities[self.nq.Mapped].price
                self.current_org.open_price = float(current_price)
                self.current_org.open_time = self.time
                self.session_started = True
                self.first_fvg_detected = False
                self.daily_fvgs = []  # Reset daily FVGs
                
                self.debug(f"ORG Open captured: {self.current_org.open_price} at {self.current_org.open_time}")
        except Exception as e:
            self.debug(f"Error capturing opening range: {e}")

    def CaptureClosingPrice(self):
        """Capture the closing price for ORG calculation"""
        try:
            if self.nq.Mapped in self.securities:
                current_price = self.securities[self.nq.Mapped].Price
                self.current_org.close_price = float(current_price)
                self.current_org.close_time = self.time
                
                # Calculate ORG levels
                self.CalculateORGLevels()
                
                self.debug(f"ORG Close captured: {self.current_org.close_price} at {self.current_org.close_time}")
        except Exception as e:
            self.debug(f"Error capturing closing price: {e}")

    def CalculateORGLevels(self):
        """Calculate ORG levels (CE, Q1, Q3) based on open and close"""
        if self.current_org.open_price is None or self.current_org.close_price is None:
            return
            
        try:
            open_price = self.current_org.open_price
            close_price = self.current_org.close_price
            
            # Calculate gap size and direction
            self.current_org.size = abs(close_price - open_price)
            self.current_org.direction = 1 if close_price > open_price else -1
            
            # Calculate key levels
            gap_range = close_price - open_price
            
            # Consequent Encroachment (50% level)
            self.current_org.ce_price = open_price + (gap_range * 0.5)
            
            # Quarter levels
            self.current_org.q1_price = open_price + (gap_range * 0.25)
            self.current_org.q3_price = open_price + (gap_range * 0.75)
            
            self.current_org.is_valid = True
            
            self.debug(f"ORG Levels - Open: {open_price:.2f}, Close: {close_price:.2f}")
            self.debug(f"ORG Levels - CE: {self.current_org.ce_price:.2f}, Q1: {self.current_org.q1_price:.2f}, Q3: {self.current_org.q3_price:.2f}")
            self.debug(f"ORG Size: {self.current_org.size:.2f}, Direction: {self.current_org.direction}")
            
        except Exception as e:
            self.debug(f"Error calculating ORG levels: {e}")

    def DetectFVG(self):
        """Detect Fair Value Gaps using 3-candle pattern"""
        if len(self.minute_bars) < 3:
            return
            
        try:
            # Get last 3 bars
            bar1 = self.minute_bars[-3]  # 2 bars ago
            bar2 = self.minute_bars[-2]  # 1 bar ago  
            bar3 = self.minute_bars[-1]  # Current bar
            
            # Check for bullish FVG: bar1.low > bar3.high
            if bar1['low'] > bar3['high']:
                fvg = FairValueGap()
                fvg.high = bar1['low']
                fvg.low = bar3['high']
                fvg.time = bar3['time']
                fvg.is_bullish = True
                fvg.ce_price = (fvg.high + fvg.low) / 2
                
                # Check if this is the first FVG of the day
                if self.session_started and not self.first_fvg_detected:
                    fvg.is_first_of_day = True
                    self.first_fvg_detected = True
                    self.current_fvg = fvg
                    self.debug(f"First Bullish FVG detected: {fvg.low:.2f} - {fvg.high:.2f} at {fvg.time}")
                
                self.daily_fvgs.append(fvg)
                
            # Check for bearish FVG: bar1.high < bar3.low
            elif bar1['high'] < bar3['low']:
                fvg = FairValueGap()
                fvg.high = bar3['low']
                fvg.low = bar1['high']
                fvg.time = bar3['time']
                fvg.is_bullish = False
                fvg.ce_price = (fvg.high + fvg.low) / 2
                
                # Check if this is the first FVG of the day
                if self.session_started and not self.first_fvg_detected:
                    fvg.is_first_of_day = True
                    self.first_fvg_detected = True
                    self.current_fvg = fvg
                    self.debug(f"First Bearish FVG detected: {fvg.low:.2f} - {fvg.high:.2f} at {fvg.time}")
                
                self.daily_fvgs.append(fvg)
                
        except Exception as e:
            self.debug(f"Error detecting FVG: {e}")

    def GetORGTarget(self):
        """Get 50% ORG target level"""
        if self.current_org.is_valid:
            return self.current_org.ce_price
        return None

    def GetFirstFVG(self):
        """Get the first FVG of the day"""
        return self.current_fvg

    def CheckORGDirection(self):
        """Get ORG directional bias"""
        if self.current_org.is_valid:
            return self.current_org.direction
        return 0

    

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
