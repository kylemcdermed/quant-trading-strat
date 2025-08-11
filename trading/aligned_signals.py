from trading.entry_logic import GetFVGStopLoss, GetFVGStopLossLevels





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
                algorithm_instance.debug(f"MOMENTUM SHORT SIGNAL: Current close {current_price:.2f} > Candle 3 High {candle3_high:.2f}")
                algorithm_instance.debug("Setting SELL LIMIT at Candle 3 High for momentum short entry")
                
                # Entry Setup for MOMENTUM SHORT trade
                entry_price = candle3_high
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, -1)  # -1 = short trade direction
                take_profit = entry_price - 50  # 50 points profit target from entry
                
                algorithm_instance.debug(f"MOMENTUM SHORT SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                algorithm_instance.debug("NOTE: Momentum short trade with 50 point profit target")
                # TODO: Execute SELL limit order logic here
                
            else:
                algorithm_instance.debug(f"NO MOMENTUM SHORT: Current close {current_price:.2f} <= Candle 3 High {candle3_high:.2f}")
        
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
                algorithm_instance.debug(f"MOMENTUM LONG SIGNAL: Current close {current_price:.2f} > Candle 1 Low {candle1_low:.2f}")
                algorithm_instance.debug("Setting BUY LIMIT at Candle 1 Low for momentum long entry")
                
                # Entry Setup for MOMENTUM LONG trade
                entry_price = candle1_low
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, 1)  # 1 = long trade direction
                take_profit = entry_price + 50  # 50 points profit target from entry
                
                algorithm_instance.debug(f"MOMENTUM LONG SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                algorithm_instance.debug("NOTE: Momentum long trade with 50 point profit target")
                # TODO: Execute BUY limit order logic here
                
            else:
                algorithm_instance.debug(f"NO MOMENTUM LONG: Current close {current_price:.2f} <= Candle 1 Low {candle1_low:.2f}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SCENARIO 4: DNN BEARISH (-1) + ORG BEARISH (-1) = ALIGNED BEARISH SIGNALS
    # Strategy: "Momentum Entry with Daily Time Frame" - Both signals agree bearish
    # Take Profit: 50 points from entry (NOT ORG target)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    elif direction == -1:  # Both DNN and ORG are bearish (aligned)
        
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
        # CASE 4A: BEARISH FVG + ALIGNED BEARISH SIGNALS - MOMENTUM SHORT CONTINUATION
        # Entry: Wait for price to close BELOW candle 1 high (showing momentum continuation)
        # Logic: All signals bearish (DNN + ORG + FVG), enter short on momentum breakdown
        # Trade: SELL LIMIT at candle 1 high with 50 point profit target
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if not first_fvg.is_bullish:  # 1st FVG is bearish (gap down/discount)
            algorithm_instance.debug("SCENARIO 4A: BEARISH FVG + ALIGNED BEARISH - Looking for momentum SHORT continuation")
            
            candle1_high = fvg_formation_candles[0]['high']  # Candle 1 high = SHORT entry level
            algorithm_instance.debug(f"Candle 1 High: {candle1_high:.2f} (SHORT entry level for bearish FVG momentum)")
            
            # Check if current candle closes BELOW candle 1 high (momentum continuation for short)
            if current_price < candle1_high:
                algorithm_instance.debug(f"MOMENTUM SHORT CONTINUATION: Current close {current_price:.2f} < Candle 1 High {candle1_high:.2f}")
                algorithm_instance.debug("Setting SELL LIMIT at Candle 1 High for momentum short continuation")
                
                # Entry Setup for MOMENTUM SHORT CONTINUATION trade
                entry_price = candle1_high
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, -1)  # -1 = short trade direction
                take_profit = entry_price - 50  # 50 points profit target from entry
                
                algorithm_instance.debug(f"MOMENTUM SHORT SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                algorithm_instance.debug("NOTE: Bearish momentum continuation trade with 50 point profit target")
                # TODO: Execute SELL limit order logic here
                
            else:
                algorithm_instance.debug(f"NO MOMENTUM SHORT: Current close {current_price:.2f} >= Candle 1 High {candle1_high:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # CASE 4B: BULLISH FVG + ALIGNED BEARISH SIGNALS - MOMENTUM SHORT REVERSAL
        # Entry: Wait for price to close BELOW candle 3 high (showing momentum reversal)
        # Logic: Despite bullish FVG, aligned bearish signals call for short on reversal
        # Trade: SELL LIMIT at candle 3 high with 50 point profit target
        # ═══════════════════════════════════════════════════════════════════════════════
        
        elif first_fvg.is_bullish:  # 1st FVG is bullish (gap up/premium)
            algorithm_instance.debug("SCENARIO 4B: BULLISH FVG + ALIGNED BEARISH - Looking for momentum SHORT reversal")
            
            candle3_high = fvg_formation_candles[2]['high']  # Candle 3 high = SHORT entry level
            algorithm_instance.debug(f"Candle 3 High: {candle3_high:.2f} (SHORT entry level for bullish FVG reversal)")
            
            # Check if current candle closes BELOW candle 3 high (momentum reversal for short)
            if current_price < candle3_high:
                algorithm_instance.debug(f"MOMENTUM SHORT REVERSAL: Current close {current_price:.2f} < Candle 3 High {candle3_high:.2f}")
                algorithm_instance.debug("Setting SELL LIMIT at Candle 3 High for momentum short reversal")
                
                # Entry Setup for MOMENTUM SHORT REVERSAL trade
                entry_price = candle3_high
                stop_loss = algorithm_instance.GetFVGStopLoss(first_fvg, -1)  # -1 = short trade direction
                take_profit = entry_price - 50  # 50 points profit target from entry
                
                algorithm_instance.debug(f"MOMENTUM SHORT SETUP - Entry: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                algorithm_instance.debug("NOTE: Bearish momentum reversal trade with 50 point profit target")
                # TODO: Execute SELL limit order logic here
                
            else:
                algorithm_instance.debug(f"NO MOMENTUM SHORT: Current close {current_price:.2f} >= Candle 3 High {candle3_high:.2f}")
