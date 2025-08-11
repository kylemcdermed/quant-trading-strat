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

# Import our custom modules
from models.dnn_model import SimpleDNN
from indicators.market_structure import OpenRangeGap, FairValueGap, DetectFVG, CalculateORGLevels  
from utils.helpers import StoreMinuteBar, IsMarketSession, CheckORGDirection, GetFirstFVG, GetORGTarget
from trading.conflicting_signals import HandleConflictingSignals
from trading.aligned_signals import HandleAlignedSignals
from trading.entry_logic import GetFVGStopLoss, GetFVGStopLossLevels



class NQTradingStrategy(QCAlgorithm):
    def Initialize(self):
        self.set_start_date(2020, 1, 1)  
        self.set_cash(10000000)  
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)  

        self.nq = self.add_future(Futures.Indices.NASDAQ_100_E_MINI, Resolution.MINUTE)
        self.nq.SetFilter(timedelta(0), timedelta(90))

        self.target_vol = 0.30

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleDNN(input_size=30).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # schedule daily DNN bias calculation -- should just calculate bias
        self.schedule.on(
            self.date_rules.every_day(self.nq.Symbol),
            self.time_rules.at(0, 0),
            self.CalculateDailyBias
        )
        
        # Capture opening price for ORG at market open (9:30 AM ET)
        self.schedule.on(
            self.date_rules.every_day(self.nq.Symbol),
            self.time_rules.at(9, 30),
            self.CaptureOpeningRange
        )

        # Capture closing price for ORG at market close (4:14 PM ET)
        self.schedule.on(
            self.date_rules.every_day(self.nq.Symbol),
            self.time_rules.at(16, 14),
            self.CaptureClosingPrice
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
        self.trade_taken_today = False
        
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
        StoreMinuteBar(current_bar, current_time, self.minute_bars, self.max_minute_bars)
        
        # Detect FVG on new minute bars
        if IsMarketSession(current_time, self.market_open_time, self.market_close_time):
            detected_fvg = DetectFVG(self.minute_bars, self.session_started, self.first_fvg_detected, self.current_fvg, self.daily_fvgs, self.debug)
            
            # If FVG was detected and it's the first one of the day
            if detected_fvg is not None:
                # Check if this is the first FVG of the day
                if self.session_started and not self.first_fvg_detected:
                    detected_fvg.is_first_of_day = True
                    self.first_fvg_detected = True
                    self.current_fvg = detected_fvg
                    self.debug(f"First {'Bullish' if detected_fvg.is_bullish else 'Bearish'} FVG detected: {detected_fvg.low:.2f} - {detected_fvg.high:.2f} at {detected_fvg.time}")
                    
                    # **TRIGGER TRADING LOGIC** when first FVG is detected
                    if not self.trade_taken_today:
                        self.debug("Triggering trading logic for first FVG...")
                        self.TradingLogic()


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
                self.trade_taken_today = False  # Reset daily
                
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
                CalculateORGLevels(self.current_org, self.debug)
                
                self.debug(f"ORG Close captured: {self.current_org.close_price} at {self.current_org.close_time}")
        except Exception as e:
            self.debug(f"Error capturing closing price: {e}")


    def TradingLogic(self):
        """
        Main trading logic that evaluates DNN vs ORG conditions for entry
        
        This function is called after market open (9:30 AM) when we have:
        - DNN daily bias prediction (+1 bullish, -1 bearish)
        - ORG direction from overnight gap (+1 gap up, -1 gap down)
        
        Entry conditions based on DNN vs ORG alignment:
        1. Conflicting signals (DNN bullish + ORG bearish, or DNN bearish + ORG bullish)
        2. Aligned signals (DNN bullish + ORG bullish, or DNN bearish + ORG bearish)
        """
        try:
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: CHECK IF WE'VE ALREADY TRADED TODAY
            # ═══════════════════════════════════════════════════════════════
            if self.trade_taken_today:
                self.debug("Trade already taken today - no more entries allowed")
                return
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 2: GET DNN PREDICTION AND ORG DIRECTION
            # ═══════════════════════════════════════════════════════════════
            dnn_bias = self.bias  # +1 bullish, -1 bearish from DNN daily prediction
            org_direction = CheckORGDirection(self.current_org)  # +1 gap up, -1 gap down
            
            # Validate we have both signals
            if dnn_bias == 0 or org_direction == 0:
                self.debug("Missing DNN bias or ORG direction - skipping trade logic")
                return
            
            self.debug(f"Trading Logic - DNN Bias: {dnn_bias}, ORG Direction: {org_direction}")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 3: EVALUATE TRADING CONDITIONS
            # ═══════════════════════════════════════════════════════════════
            
            # use these for all conditions 
            first_fvg = GetFirstFVG(self.current_fvg)
            org_target = GetORGTarget(self.current_org)

            # CONDITION 1: DNN BULLISH (+1) + ORG BEARISH (-1) = CONFLICTING SIGNALS
            if dnn_bias == 1 and org_direction == -1:
                self.debug("CONFLICT: DNN Bullish vs ORG Bearish")
                # TODO: Implement momentum entry logic with FVG
                HandleConflictingSignals(self, direction=1, entry_type="1FVG", first_fvg=first_fvg, org_target=org_target)

            # CONDITION 2: DNN BEARISH (-1) + ORG BULLISH (+1) = CONFLICTING SIGNALS  
            elif dnn_bias == -1 and org_direction == 1:
                self.debug("CONFLICT: DNN Bearish vs ORG Bullish")
                # TODO: Implement momentum entry logic with FVG
                HandleConflictingSignals(self, direction=-1, entry_type="1FVG", first_fvg=first_fvg, org_target=org_target)

            # CONDITION 3: DNN BULLISH (+1) + ORG BULLISH (+1) = ALIGNED BULLISH
            elif dnn_bias == 1 and org_direction == 1:
                self.debug("ALIGNED BULLISH: DNN Bullish + ORG Bullish")
                # TODO: Implement standard FVG entry logic
                HandleConflictingSignals(self, direction=1, entry_type="momentum", first_fvg=first_fvg, org_target=org_target)

            # CONDITION 4: DNN BEARISH (-1) + ORG BEARISH (-1) = ALIGNED BEARISH
            elif dnn_bias == -1 and org_direction == -1:
                self.debug("ALIGNED BEARISH: DNN Bearish + ORG Bearish") 
                # TODO: Implement standard FVG entry logic
                HandleConflictingSignals(self, direction=-1, entry_type="momentum", first_fvg=first_fvg, org_target=org_target)

            else:
                self.debug("Unexpected condition - no trading logic triggered")
                
        except Exception as e:
            self.debug(f"Error in TradingLogic: {e}")

    

    def CalculateDailyBias(self):
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

            '''
            # do not use DNN to execute
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
            '''
        except Exception as e:
            self.debug(f"Error in CalculateDailyBias: {str(e)}")
