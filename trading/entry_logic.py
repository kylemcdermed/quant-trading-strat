



def GetFVGFormationCandles(minute_bars, fvg_time):
    """Get the 3 candles that formed the FVG"""
    if not minute_bars or len(minute_bars) < 3:
        return []
    
    # Find the candles around the FVG time
    for i in range(len(minute_bars) - 2):
        if (minute_bars[i+2]['time'] == fvg_time or 
            abs((minute_bars[i+2]['time'] - fvg_time).total_seconds()) < 60):
            
            return [
                minute_bars[i],     # Candle 1 (2 bars ago)
                minute_bars[i+1],   # Candle 2 (1 bar ago)
                minute_bars[i+2]    # Candle 3 (current when FVG formed)
            ]
    return []

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



def HandleConflictingSignals(algorithm_instance, direction, entry_type, first_fvg, org_target):
    """Handle conflicting DNN vs ORG signals"""
    algorithm_instance.debug(f"Handling conflicting signals - Direction: {direction}, Type: {entry_type}")
    
    if first_fvg is None:
        algorithm_instance.debug("Waiting for first FVG to develop...")
        return
        
    if org_target is None:
        algorithm_instance.debug("No valid ORG target available")
        return
        
    algorithm_instance.debug(f"First FVG: {first_fvg.low:.2f} - {first_fvg.high:.2f}")
    algorithm_instance.debug(f"ORG Target (50%): {org_target:.2f}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW SCENARIO 1: DNN BULLISH (+1) + ORG BEARISH (-1) = CONFLICTING SIGNALS
    # Strategy: Enter long when price action confirms bullish bias despite bearish ORG
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if direction == 1:  # DNN is bullish, ORG is bearish (conflicting)
        
        # Find the 3 candles that formed the FVG (we need candle 1 for entry levels)
        fvg_formation_candles = None
        fvg_formation_index = None
        
        if len(algorithm_instance.minute_bars) >= 3:
            for i in range(len(algorithm_instance.minute_bars) - 2):
                if algorithm_instance.minute_bars[i+2]['time'] == first_fvg.time:
                    fvg_formation_candles = [
                        algorithm_instance.minute_bars[i],     # Candle 1 (2 bars ago when FVG formed)
                        algorithm_instance.minute_bars[i+1],   # Candle 2 (1 bar ago when FVG formed)  
                        algorithm_instance.minute_bars[i+2]    # Candle 3 (current bar when FVG formed)
                    ]
                    fvg_formation_index = i + 2  # Index of candle 3 in minute_bars
                    break
        
        if fvg_formation_candles is None:
            algorithm_instance.debug("Could not find FVG formation candles - skipping entry logic")
            return
        
        # Get current candle (must be AFTER the 3 FVG formation candles)
        current_candle_index = len(algorithm_instance.minute_bars) - 1
        
        # Ensure we're looking at a candle AFTER the FVG formation (not one of the 3 formation candles)
        if current_candle_index <= fvg_formation_index:
            algorithm_instance.debug("Current candle is part of FVG formation - waiting for next candle")
            return
        
        current_candle = algorithm_instance.minute_bars[current_candle_index]
        current_price = current_candle['close']  # Use candle close, not live price
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CASE 1: BEARISH FVG (Discount/Lower Prices) + BULLISH DNN
        # Entry: Wait for price to close BELOW candle 1 high (showing rejection/reversal)
        # Logic: FVG shows selling pressure, but DNN is bullish, so we buy the dip
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if not first_fvg.is_bullish:  # 1st FVG is bearish (gap down/selling)
            algorithm_instance.debug("BEARISH FVG DETECTED: Looking for bullish reversal entry")
            
            candle1_high = fvg_formation_candles[0]['high']  # Candle 1 high = entry level
            algorithm_instance.debug(f"Candle 1 High: {candle1_high:.2f} (entry level for bearish FVG)")
            
            # Check if current candle closes BELOW candle 1 high (corrected logic)
            if current_price < candle1_high:
                algorithm_instance.debug(f"ENTRY SIGNAL: Current close {current_price:.2f} < Candle 1 High {candle1_high:.2f}")
                algorithm_instance.debug("Setting BUY LIMIT at Candle 1 High for bearish FVG reversal")
                
                # Entry Setup
                entry_price = candle1_high
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, 1)  # 1 = long trade direction
                take_profit = org_target  # 50% ORG level
                
                algorithm_instance.debug(f"TRADE SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                # TODO: Execute BUY limit order logic here
                
            else:
                algorithm_instance.debug(f"NO ENTRY: Current close {current_price:.2f} >= Candle 1 High {candle1_high:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CASE 2: BULLISH FVG (Premium/Higher Prices) + BULLISH DNN  
        # Entry: Wait for price to close ABOVE candle 1 low (showing continuation)
        # Logic: FVG shows buying pressure, DNN is bullish, so we buy the breakout
        # ═══════════════════════════════════════════════════════════════════════════════
        
        elif first_fvg.is_bullish:  # 1st FVG is bullish (gap up/buying)
            algorithm_instance.debug("BULLISH FVG DETECTED: Looking for bullish continuation entry")
            
            candle1_low = fvg_formation_candles[0]['low']  # Candle 1 low = entry level
            algorithm_instance.debug(f"Candle 1 Low: {candle1_low:.2f} (entry level for bullish FVG)")
            
            # Check if current candle closes ABOVE candle 1 low (showing strength)
            if current_price > candle1_low:
                algorithm_instance.debug(f"ENTRY SIGNAL: Current close {current_price:.2f} > Candle 1 Low {candle1_low:.2f}")
                algorithm_instance.debug("Setting BUY LIMIT at Candle 1 Low for bullish FVG continuation")
                
                # Entry Setup
                entry_price = candle1_low
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, 1)  # 1 = long trade direction
                take_profit = org_target  # 50% ORG level
                
                algorithm_instance.debug(f"TRADE SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                # TODO: Execute BUY limit order logic here
                
            else:
                algorithm_instance.debug(f"NO ENTRY: Current close {current_price:.2f} <= Candle 1 Low {candle1_low:.2f}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW SCENARIO 2: DNN BEARISH (-1) + ORG BULLISH (+1) = CONFLICTING SIGNALS
    # Strategy: Enter short when price action confirms bearish bias despite bullish ORG
    # ═══════════════════════════════════════════════════════════════════════════════
    
    elif direction == -1:  # DNN is bearish, ORG is bullish (conflicting)
        
        # Find the 3 candles that formed the FVG (we need candle 1 for entry levels)
        fvg_formation_candles = None
        fvg_formation_index = None
        
        if len(algorithm_instance.minute_bars) >= 3:
            for i in range(len(algorithm_instance.minute_bars) - 2):
                if algorithm_instance.minute_bars[i+2]['time'] == first_fvg.time:
                    fvg_formation_candles = [
                        algorithm_instance.minute_bars[i],     # Candle 1 (2 bars ago when FVG formed)
                        algorithm_instance.minute_bars[i+1],   # Candle 2 (1 bar ago when FVG formed)  
                        algorithm_instance.minute_bars[i+2]    # Candle 3 (current bar when FVG formed)
                    ]
                    fvg_formation_index = i + 2  # Index of candle 3 in minute_bars
                    break
        
        if fvg_formation_candles is None:
            algorithm_instance.debug("Could not find FVG formation candles - skipping entry logic")
            return
        
        # Get current candle (must be AFTER the 3 FVG formation candles)
        current_candle_index = len(algorithm_instance.minute_bars) - 1
        
        # Ensure we're looking at a candle AFTER the FVG formation (not one of the 3 formation candles)
        if current_candle_index <= fvg_formation_index:
            algorithm_instance.debug("Current candle is part of FVG formation - waiting for next candle")
            return
        
        current_candle = algorithm_instance.minute_bars[current_candle_index]
        current_price = current_candle['close']  # Use candle close, not live price
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CASE 1: BEARISH FVG + BEARISH DNN - SHORT CONTINUATION
        # Entry: Wait for price to close BELOW candle 1 high (showing continued selling)
        # Logic: FVG shows selling pressure, DNN agrees bearish, so we sell the breakdown
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if not first_fvg.is_bullish:  # 1st FVG is bearish (gap down/selling)
            algorithm_instance.debug("BEARISH FVG DETECTED: Looking for bearish continuation entry")
            
            candle1_high = fvg_formation_candles[0]['high']  # Candle 1 high = entry level
            algorithm_instance.debug(f"Candle 1 High: {candle1_high:.2f} (entry level for bearish FVG)")
            
            # Check if current candle closes BELOW candle 1 high (showing continued weakness)
            if current_price < candle1_high:
                algorithm_instance.debug(f"SHORT ENTRY SIGNAL: Current close {current_price:.2f} < Candle 1 High {candle1_high:.2f}")
                algorithm_instance.debug("Setting SELL LIMIT at Candle 1 High for bearish FVG continuation")
                
                # Entry Setup for SHORT trade
                entry_price = candle1_high
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, -1)  # -1 = short trade direction
                take_profit = org_target  # 50% ORG level
                
                algorithm_instance.debug(f"SHORT TRADE SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                # TODO: Execute SELL limit order logic here
                
            else:
                algorithm_instance.debug(f"NO SHORT ENTRY: Current close {current_price:.2f} >= Candle 1 High {candle1_high:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CASE 2: BULLISH FVG + BEARISH DNN - SHORT REVERSAL
        # Entry: Wait for price to close BELOW candle 1 low (showing rejection/reversal)
        # Logic: FVG shows buying pressure, but DNN is bearish, so we sell the rejection
        # ═══════════════════════════════════════════════════════════════════════════════
        
        elif first_fvg.is_bullish:  # 1st FVG is bullish (gap up/buying)
            algorithm_instance.debug("BULLISH FVG DETECTED: Looking for bearish reversal entry")
            
            candle1_low = fvg_formation_candles[0]['low']  # Candle 1 low = entry level
            algorithm_instance.debug(f"Candle 1 Low: {candle1_low:.2f} (entry level for bullish FVG)")
            
            # Check if current candle closes BELOW candle 1 low (showing rejection/reversal)
            if current_price < candle1_low:
                algorithm_instance.debug(f"SHORT ENTRY SIGNAL: Current close {current_price:.2f} < Candle 1 Low {candle1_low:.2f}")
                algorithm_instance.debug("Setting SELL LIMIT at Candle 1 Low for bullish FVG reversal")
                
                # Entry Setup for SHORT trade
                entry_price = candle1_low
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, -1)  # -1 = short trade direction
                take_profit = org_target  # 50% ORG level
                
                algorithm_instance.debug(f"SHORT TRADE SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                # TODO: Execute SELL limit order logic here
                
            else:
                algorithm_instance.debug(f"NO SHORT ENTRY: Current close {current_price:.2f} >= Candle 1 Low {candle1_low:.2f}")



def HandleAlignedSignals(algorithm_instance, direction, entry_type, first_fvg, org_target):
    """
    Handle aligned DNN and ORG signals
    
    Args:
        algorithm_instance: Reference to main algorithm
        direction: +1 for bullish, -1 for bearish (both DNN and ORG agree)
        entry_type: "standard" for standard FVG entry
        first_fvg: First FVG of the day
        org_target: ORG target level
    """
    algorithm_instance.debug(f"Handling aligned signals - Direction: {direction}, Type: {entry_type}")
    
    if first_fvg is None:
        algorithm_instance.debug("Waiting for first FVG to develop...")
        return
        
    if org_target is None:
        algorithm_instance.debug("No valid ORG target available")
        return
        
    algorithm_instance.debug(f"First FVG: {first_fvg.low:.2f} - {first_fvg.high:.2f}")
    algorithm_instance.debug(f"ORG Target (50%): {org_target:.2f}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SCENARIO 3: DNN BULLISH (+1) + ORG BULLISH (+1) = ALIGNED BULLISH SIGNALS
    # Strategy: "Momentum Entry with Daily Time Frame" - Both signals agree bullish
    # Take Profit: 50 points from entry (NOT ORG target)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if direction == 1:  # Both DNN and ORG are bullish (aligned)
        
        # Find the 3 candles that formed the FVG (we need candle 1 and candle 3 for entry levels)
        fvg_formation_candles = None
        fvg_formation_index = None
        
        if len(algorithm_instance.minute_bars) >= 3:
            for i in range(len(algorithm_instance.minute_bars) - 2):
                if algorithm_instance.minute_bars[i+2]['time'] == first_fvg.time:
                    fvg_formation_candles = [
                        algorithm_instance.minute_bars[i],     # Candle 1 (2 bars ago when FVG formed)
                        algorithm_instance.minute_bars[i+1],   # Candle 2 (1 bar ago when FVG formed)  
                        algorithm_instance.minute_bars[i+2]    # Candle 3 (current bar when FVG formed)
                    ]
                    fvg_formation_index = i + 2  # Index of candle 3 in minute_bars
                    break
        
        if fvg_formation_candles is None:
            algorithm_instance.debug("Could not find FVG formation candles - skipping entry logic")
            return
        
        # Get current candle (must be AFTER the 3 FVG formation candles)
        current_candle_index = len(algorithm_instance.minute_bars) - 1
        
        # Ensure we're looking at a candle AFTER the FVG formation (not one of the 3 formation candles)
        if current_candle_index <= fvg_formation_index:
            algorithm_instance.debug("Current candle is part of FVG formation - waiting for next candle")
            return
        
        current_candle = algorithm_instance.minute_bars[current_candle_index]
        current_price = current_candle['close']  # Use candle close, not live price
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CASE 1: BEARISH FVG + ALIGNED BULLISH SIGNALS - MOMENTUM SHORT ENTRY
        # Entry: Wait for price to close ABOVE candle 3 high (showing momentum breakout)
        # Logic: Despite bullish alignment, take short when price breaks above bearish FVG
        # Trade: SELL LIMIT at candle 3 high with 50 point profit target
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if not first_fvg.is_bullish:  # 1st FVG is bearish (gap down/discount)
            algorithm_instance.debug("SCENARIO 3A: BEARISH FVG + ALIGNED BULLISH - Looking for momentum SHORT entry")
            
            candle3_high = fvg_formation_candles[2]['high']  # Candle 3 high = SHORT entry level
            algorithm_instance.debug(f"Candle 3 High: {candle3_high:.2f} (SHORT entry level for bearish FVG momentum)")
            
            # Check if current candle closes ABOVE candle 3 high (momentum breakout for short)
            if current_price > candle3_high:
                algorithm_instance.debug(f"✅ MOMENTUM SHORT SIGNAL: Current close {current_price:.2f} > Candle 3 High {candle3_high:.2f}")
                algorithm_instance.debug("Setting SELL LIMIT at Candle 3 High for momentum short entry")
                
                # Entry Setup for MOMENTUM SHORT trade
                entry_price = candle3_high
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, -1)  # -1 = short trade direction
                take_profit = entry_price - 50  # 50 points profit target from entry
                
                algorithm_instance.debug(f"MOMENTUM SHORT SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                algorithm_instance.debug("NOTE: Momentum short trade with 50 point profit target")
                # TODO: Execute SELL limit order logic here
                
            else:
                algorithm_instance.debug(f"❌ NO MOMENTUM SHORT: Current close {current_price:.2f} <= Candle 3 High {candle3_high:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CASE 2: BULLISH FVG + ALIGNED BULLISH SIGNALS - MOMENTUM LONG ENTRY
        # Entry: Wait for price to close ABOVE candle 1 low (showing momentum continuation)
        # Logic: All signals bullish, enter long when price breaks above bullish FVG support
        # Trade: BUY LIMIT at candle 1 low with 50 point profit target
        # ═══════════════════════════════════════════════════════════════════════════════
        
        elif first_fvg.is_bullish:  # 1st FVG is bullish (gap up/premium)
            algorithm_instance.debug("SCENARIO 3B: BULLISH FVG + ALIGNED BULLISH - Looking for momentum LONG entry")
            
            candle1_low = fvg_formation_candles[0]['low']  # Candle 1 low = LONG entry level
            algorithm_instance.debug(f"Candle 1 Low: {candle1_low:.2f} (LONG entry level for bullish FVG momentum)")
            
            # Check if current candle closes ABOVE candle 1 low (momentum continuation for long)
            if current_price > candle1_low:
                algorithm_instance.debug(f"✅ MOMENTUM LONG SIGNAL: Current close {current_price:.2f} > Candle 1 Low {candle1_low:.2f}")
                algorithm_instance.debug("Setting BUY LIMIT at Candle 1 Low for momentum long entry")
                
                # Entry Setup for MOMENTUM LONG trade
                entry_price = candle1_low
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, 1)  # 1 = long trade direction
                take_profit = entry_price + 50  # 50 points profit target from entry
                
                algorithm_instance.debug(f"MOMENTUM LONG SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                algorithm_instance.debug("NOTE: Momentum long trade with 50 point profit target")
                # TODO: Execute BUY limit order logic here
                
            else:
                algorithm_instance.debug(f"❌ NO MOMENTUM LONG: Current close {current_price:.2f} <= Candle 1 Low {candle1_low:.2f}")
    
    # Scenario 4 will be added here when you provide the requirements
