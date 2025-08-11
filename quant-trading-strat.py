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

    

    def CalculateKellyCriterion(self, win_rate, avg_win, avg_loss):
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Kelly = (bp - q) / b
        Where:
        - b = odds received (avg_win / avg_loss)
        - p = probability of winning (win_rate)
        - q = probability of losing (1 - win_rate)
        
        Returns: fraction of capital to risk (0 to 1)
        """
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            # Calculate odds
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            # Kelly formula
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly at reasonable levels (prevent over-leverage)
            max_kelly = 0.25  # Never risk more than 25% of capital
            kelly_fraction = max(0.0, min(kelly_fraction, max_kelly))
            
            self.debug(f"Kelly Calculation - Win Rate: {win_rate:.2%}, Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}")
            self.debug(f"Kelly Fraction: {kelly_fraction:.3f} ({kelly_fraction*100:.1f}% of capital)")
            
            return kelly_fraction
            
        except Exception as e:
            self.debug(f"Error calculating Kelly: {e}")
            return 0.0

    def GetHistoricalPerformance(self, lookback_days=60):
        """
        Analyze recent trade performance for Kelly Criterion inputs
        
        Returns: (win_rate, avg_win, avg_loss)
        """
        try:
            # Get recent history for performance analysis
            history = self.history([self.nq.Symbol], 31, Resolution.DAILY)
            
            if history.empty:
                self.debug("History is empty")
                return
            
            # Extract price data
            closes = []
            if 'close' in history.columns:
                closes = history['close'].dropna().values
            elif 'Close' in history.columns:
                closes = history['Close'].dropna().values
            else:
                # Try multi-index access
                for symbol_key in [self.nq.Symbol, str(self.nq.Symbol)]:
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
            
            if len(closes) < 20:
                return 0.55, 1.0, 1.0
            
            # Calculate daily returns
            returns = np.diff(closes) / closes[:-1]
            
            # Simulate strategy performance based on DNN predictions
            wins = []
            losses = []
            
            for i in range(len(returns)):
                daily_return = returns[i]
                
                # Simple simulation: assume our DNN would have predicted correctly 55% of the time
                # In reality, you'd track actual trade results
                if daily_return > 0:
                    wins.append(abs(daily_return))
                else:
                    losses.append(abs(daily_return))
            
            # Calculate performance metrics
            total_trades = len(wins) + len(losses)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0.55
            avg_win = np.mean(wins) if wins else 0.01
            avg_loss = np.mean(losses) if losses else 0.01
            
            # Apply some realism adjustments
            win_rate = max(0.45, min(0.65, win_rate))  # Keep between 45-65%
            
            return win_rate, avg_win, avg_loss
            
        except Exception as e:
            self.debug(f"Error calculating historical performance: {e}")
            return 0.55, 1.0, 1.0  # Conservative defaults

    def CalculateKellyPositionSize(self):
        """
        Calculate position size using Kelly Criterion
        
        Returns: number of contracts to trade
        """
        try:
            # Get historical performance for Kelly inputs
            win_rate, avg_win, avg_loss = self.GetHistoricalPerformance()
            
            # Calculate Kelly fraction
            kelly_fraction = self.CalculateKellyCriterion(win_rate, avg_win, avg_loss)
            
            if kelly_fraction <= 0:
                return 0
            
            # Calculate position size
            capital = self.portfolio.total_portfolio_value
            risk_capital = capital * kelly_fraction
            
            # Get current price
            current_price = self.securities[self.nq.Mapped].Price
            contract_value = current_price * self.contract_multiplier
            
            if contract_value <= 0:
                return 0
            
            # Calculate number of contracts
            contracts = int(risk_capital / contract_value)
            
            # Apply reasonable limits
            max_contracts = int(capital * 0.25 / contract_value)  # Never more than 25% of capital
            contracts = min(contracts, max_contracts)
            
            self.debug(f"Kelly Position Sizing - Capital: ${capital:,.0f}, Risk: ${risk_capital:,.0f}, Contracts: {contracts}")
            
            return contracts
            
        except Exception as e:
            self.debug(f"Error calculating Kelly position size: {e}")
            return 0


    def GetFVGStopLossLevels(self, fvg_time):
        """
        Get stop loss levels from the 3-candle FVG pattern
        
        Args:
            fvg_time: The timestamp when the FVG was detected
            
        Returns:
            tuple: (highest_price, lowest_price) from the 3 candles that formed the FVG
        """
        try:
            if not self.minute_bars or len(self.minute_bars) < 3:
                self.Debug("Insufficient minute bars for stop loss calculation")
                return None, None
            
            # Find the 3 candles that formed the FVG
            fvg_candles = []
            
            # Look for the candles around the FVG time
            for i in range(len(self.minute_bars) - 2):
                if (self.minute_bars[i+2]['time'] == fvg_time or 
                    abs((self.minute_bars[i+2]['time'] - fvg_time).total_seconds()) < 60):
                    
                    # Found the FVG formation - get the 3 candles
                    fvg_candles = [
                        self.minute_bars[i],     # Candle 1 (2 bars ago)
                        self.minute_bars[i+1],   # Candle 2 (1 bar ago)
                        self.minute_bars[i+2]    # Candle 3 (current when FVG formed)
                    ]
                    break
            
            if len(fvg_candles) != 3:
                self.Debug("Could not find the 3 candles that formed the FVG")
                return None, None
            
            # Get all highs and lows from the 3 candles
            all_highs = [candle['high'] for candle in fvg_candles]
            all_lows = [candle['low'] for candle in fvg_candles]
            
            # Find the extreme levels
            highest_price = max(all_highs)
            lowest_price = min(all_lows)
            
            self.debug(f"FVG Stop Loss Levels - Highest: {highest_price:.2f}, Lowest: {lowest_price:.2f}")
            self.debug(f"Candle Details:")
            for i, candle in enumerate(fvg_candles, 1):
                self.debug(f"  Candle {i}: H:{candle['high']:.2f} L:{candle['low']:.2f} O:{candle['open']:.2f} C:{candle['close']:.2f}")
            
            return highest_price, lowest_price
            
        except Exception as e:
            self.debug(f"Error calculating FVG stop loss levels: {e}")
            return None, None

    def GetFVGStopLoss(self, fvg_object, trade_direction):
        """
        Get the appropriate stop loss for a trade based on FVG and trade direction
        
        Args:
            fvg_object: The FairValueGap object containing FVG data
            trade_direction: +1 for long trades, -1 for short trades
            
        Returns:
            float: Stop loss price level
        """
        try:
            if fvg_object is None:
                self.debug("No FVG object provided for stop loss calculation")
                return None
            
            # Get the extreme levels from the FVG formation
            highest_price, lowest_price = self.GetFVGStopLossLevels(fvg_object.time)
            
            if highest_price is None or lowest_price is None:
                self.debug("Could not determine FVG stop loss levels")
                return None
            
            # Determine stop loss based on trade direction
            if trade_direction == 1:  # Long trade
                stop_loss = lowest_price
                self.debug(f"Long trade stop loss: {stop_loss:.2f} (lowest of 3 FVG candles)")
            elif trade_direction == -1:  # Short trade
                stop_loss = highest_price
                self.debug(f"Short trade stop loss: {stop_loss:.2f} (highest of 3 FVG candles)")
            else:
                self.debug("Invalid trade direction for stop loss calculation")
                return None
            
            return stop_loss
            
        except Exception as e:
            self.debug(f"Error getting FVG stop loss: {e}")
            return None



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
            org_direction = self.CheckORGDirection()  # +1 gap up, -1 gap down
            
            # Validate we have both signals
            if dnn_bias == 0 or org_direction == 0:
                self.debug("Missing DNN bias or ORG direction - skipping trade logic")
                return
            
            self.debug(f"Trading Logic - DNN Bias: {dnn_bias}, ORG Direction: {org_direction}")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 3: EVALUATE TRADING CONDITIONS
            # ═══════════════════════════════════════════════════════════════
            
            # CONDITION 1: DNN BULLISH (+1) + ORG BEARISH (-1) = CONFLICTING SIGNALS
            if dnn_bias == 1 and org_direction == -1:
                self.Debug("CONFLICT: DNN Bullish vs ORG Bearish")
                self.Debug("Strategy: Use momentum-based entry within 1st FVG")
                self.Debug("Target: 50% ORG level")
                # TODO: Implement momentum entry logic with FVG
                self.HandleConflictingSignals(direction=1, entry_type="momentum")
                
            # CONDITION 2: DNN BEARISH (-1) + ORG BULLISH (+1) = CONFLICTING SIGNALS  
            elif dnn_bias == -1 and org_direction == 1:
                self.Debug("CONFLICT: DNN Bearish vs ORG Bullish")
                self.Debug("Strategy: Use momentum-based entry within 1st FVG")
                self.Debug("Target: 50% ORG level")
                # TODO: Implement momentum entry logic with FVG
                self.HandleConflictingSignals(direction=-1, entry_type="momentum")
                
            # CONDITION 3: DNN BULLISH (+1) + ORG BULLISH (+1) = ALIGNED BULLISH
            elif dnn_bias == 1 and org_direction == 1:
                self.Debug("ALIGNED BULLISH: DNN Bullish + ORG Bullish")
                self.Debug("Strategy: Standard FVG entry (both signals agree)")
                self.Debug("Target: 50% ORG level")
                # TODO: Implement standard FVG entry logic
                self.HandleAlignedSignals(direction=1, entry_type="standard")
                
            # CONDITION 4: DNN BEARISH (-1) + ORG BEARISH (-1) = ALIGNED BEARISH
            elif dnn_bias == -1 and org_direction == -1:
                self.Debug("ALIGNED BEARISH: DNN Bearish + ORG Bearish") 
                self.Debug("Strategy: Standard FVG entry (both signals agree)")
                self.Debug("Target: 50% ORG level")
                # TODO: Implement standard FVG entry logic
                self.HandleAlignedSignals(direction=-1, entry_type="standard")
                
            else:
                self.Debug("Unexpected condition - no trading logic triggered")
                
        except Exception as e:
            self.Debug(f"Error in TradingLogic: {e}")

    def HandleConflictingSignals(self, direction, entry_type):
        """
        Handle conflicting DNN vs ORG signals
        
        Args:
            direction: +1 for bullish bias, -1 for bearish bias (from DNN)
            entry_type: "momentum" for momentum-based entry
        """
        self.Debug(f"Handling conflicting signals - Direction: {direction}, Type: {entry_type}")
        
        # Get first FVG for entry timing
        first_fvg = self.GetFirstFVG()
        org_target = self.GetORGTarget()
        
        if first_fvg is None:
            self.Debug("Waiting for first FVG to develop...")
            return
            
        if org_target is None:
            self.Debug("No valid ORG target available")
            return
            
        self.Debug(f"First FVG: {first_fvg.low:.2f} - {first_fvg.high:.2f}")
        self.Debug(f"ORG Target (50%): {org_target:.2f}")
        
        # TODO: Implement momentum-based entry within FVG
        # - Wait for price to enter FVG zone
        # - Look for momentum confirmation
        # - Enter with Kelly position sizing
        # - Target 50% ORG level
        
    def HandleAlignedSignals(self, direction, entry_type):
        """
        Handle aligned DNN and ORG signals
        
        Args:
            direction: +1 for bullish, -1 for bearish (both DNN and ORG agree)
            entry_type: "standard" for standard FVG entry
        """
        self.Debug(f"Handling aligned signals - Direction: {direction}, Type: {entry_type}")
        
        # Get first FVG for entry timing
        first_fvg = self.GetFirstFVG()
        org_target = self.GetORGTarget()
        
        if first_fvg is None:
            self.Debug("Waiting for first FVG to develop...")
            return
            
        if org_target is None:
            self.Debug("No valid ORG target available")
            return
            
        self.Debug(f"First FVG: {first_fvg.low:.2f} - {first_fvg.high:.2f}")
        self.Debug(f"ORG Target (50%): {org_target:.2f}")
        
        # TODO: Implement standard FVG entry
        # - Wait for price to enter FVG zone
        # - Enter immediately when FVG is hit
        # - Enter with Kelly position sizing  
        # - Target 50% ORG level



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
