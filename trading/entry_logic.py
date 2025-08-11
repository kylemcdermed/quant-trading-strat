


def GetFVGStopLossLevels(fvg_time):
        """
        Get stop loss levels from the 3-candle FVG pattern
        
        Args:
            fvg_time: The timestamp when the FVG was detected
            
        Returns:
            tuple: (highest_price, lowest_price) from the 3 candles that formed the FVG
        """
        try:
            if not minute_bars or len(minute_bars) < 3:
                debug("Insufficient minute bars for stop loss calculation")
                return None, None
            
            # Find the 3 candles that formed the FVG
            fvg_candles = []
            
            # Look for the candles around the FVG time
            for i in range(len(minute_bars) - 2):
                if (minute_bars[i+2]['time'] == fvg_time or 
                    abs((minute_bars[i+2]['time'] - fvg_time).total_seconds()) < 60):
                    
                    # Found the FVG formation - get the 3 candles
                    fvg_candles = [
                        minute_bars[i],     # Candle 1 (2 bars ago)
                        minute_bars[i+1],   # Candle 2 (1 bar ago)
                        minute_bars[i+2]    # Candle 3 (current when FVG formed)
                    ]
                    break
            
            if len(fvg_candles) != 3:
                debug("Could not find the 3 candles that formed the FVG")
                return None, None
            
            # Get all highs and lows from the 3 candles
            all_highs = [candle['high'] for candle in fvg_candles]
            all_lows = [candle['low'] for candle in fvg_candles]
            
            # Find the extreme levels
            highest_price = max(all_highs)
            lowest_price = min(all_lows)
            
            debug(f"FVG Stop Loss Levels - Highest: {highest_price:.2f}, Lowest: {lowest_price:.2f}")
            debug(f"Candle Details:")
            for i, candle in enumerate(fvg_candles, 1):
                debug(f"  Candle {i}: H:{candle['high']:.2f} L:{candle['low']:.2f} O:{candle['open']:.2f} C:{candle['close']:.2f}")
            
            return highest_price, lowest_price
            
        except Exception as e:
            debug(f"Error calculating FVG stop loss levels: {e}")
            return None, None



def GetFVGStopLoss(fvg_object, trade_direction):
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
                debug("No FVG object provided for stop loss calculation")
                return None
            
            # Get the extreme levels from the FVG formation
            highest_price, lowest_price = self.GetFVGStopLossLevels(fvg_object.time)
            
            if highest_price is None or lowest_price is None:
                debug("Could not determine FVG stop loss levels")
                return None
            
            # Determine stop loss based on trade direction
            if trade_direction == 1:  # Long trade
                stop_loss = lowest_price
                debug(f"Long trade stop loss: {stop_loss:.2f} (lowest of 3 FVG candles)")
            elif trade_direction == -1:  # Short trade
                stop_loss = highest_price
                debug(f"Short trade stop loss: {stop_loss:.2f} (highest of 3 FVG candles)")
            else:
                debug("Invalid trade direction for stop loss calculation")
                return None
            
            return stop_loss
            
        except Exception as e:
            debug(f"Error getting FVG stop loss: {e}")
            return None



def HandleConflictingSignals(direction, entry_type):
        """
        Handle conflicting DNN vs ORG signals
        
        Args:
            direction: +1 for bullish bias, -1 for bearish bias (from DNN)
            entry_type: "momentum" for momentum-based entry
        """
        debug(f"Handling conflicting signals - Direction: {direction}, Type: {entry_type}")
        
        # Get first FVG for entry timing
        first_fvg = GetFirstFVG()
        org_target = GetORGTarget()
        
        if first_fvg is None:
            debug("Waiting for first FVG to develop...")
            return
            
        if org_target is None:
            debug("No valid ORG target available")
            return
            
        debug(f"First FVG: {first_fvg.low:.2f} - {first_fvg.high:.2f}")
        debug(f"ORG Target (50%): {org_target:.2f}")
        
        # TODO: Implement momentum-based entry within FVG
        # - Wait for price to enter FVG zone
        # - Look for momentum confirmation
        # - Enter with Kelly position sizing
        # - Target 50% ORG level
      


def HandleAlignedSignals(direction, entry_type):
        """
        Handle aligned DNN and ORG signals
        
        Args:
            direction: +1 for bullish, -1 for bearish (both DNN and ORG agree)
            entry_type: "standard" for standard FVG entry
        """
        debug(f"Handling aligned signals - Direction: {direction}, Type: {entry_type}")
        
        # Get first FVG for entry timing
        first_fvg = GetFirstFVG()
        org_target = GetORGTarget()
        
        if first_fvg is None:
            debug("Waiting for first FVG to develop...")
            return
            
        if org_target is None:
            debug("No valid ORG target available")
            return
            
        debug(f"First FVG: {first_fvg.low:.2f} - {first_fvg.high:.2f}")
        debug(f"ORG Target (50%): {org_target:.2f}")
        
        # TODO: Implement standard FVG entry
        # - Wait for price to enter FVG zone
        # - Enter immediately when FVG is hit
        # - Enter with Kelly position sizing  
        # - Target 50% ORG level

