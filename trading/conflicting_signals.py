# region imports
from AlgorithmImports import *
# endregion


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
                
                # EXECUTE BUY LIMIT ORDER
                quantity = 1  # For NQ futures, typically 1 contract
                order_ticket = algorithm_instance.LimitOrder(algorithm_instance.nq.mapped, quantity, entry_price)
                algorithm_instance.debug(f"BUY LIMIT order placed: {quantity} contracts at {entry_price}")
                
                # Set stop loss and take profit
                if order_ticket:
                    algorithm_instance.StopMarketOrder(algorithm_instance.nq.mapped, -quantity, stop_loss)
                    algorithm_instance.LimitOrder(algorithm_instance.nq.mapped, -quantity, take_profit)
                    algorithm_instance.debug(f"Stop Loss set at {stop_loss}, Take Profit at {take_profit}")
                    algorithm_instance.trade_taken_today = True
                
                
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
                
                # EXECUTE BUY LIMIT ORDER
                quantity = 1  # For NQ futures, typically 1 contract
                order_ticket = algorithm_instance.LimitOrder(algorithm_instance.nq.mapped, quantity, entry_price)
                algorithm_instance.debug(f"BUY LIMIT order placed: {quantity} contracts at {entry_price}")
                
                # Set stop loss and take profit
                if order_ticket:
                    algorithm_instance.StopMarketOrder(algorithm_instance.nq.mapped, -quantity, stop_loss)
                    algorithm_instance.LimitOrder(algorithm_instance.nq.mapped, -quantity, take_profit)
                    algorithm_instance.debug(f"Stop Loss set at {stop_loss}, Take Profit at {take_profit}")
                    algorithm_instance.trade_taken_today = True
                
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
                
                # EXECUTE SELL LIMIT ORDER (SHORT)
                quantity = 1  # For NQ futures, typically 1 contract
                order_ticket = algorithm_instance.LimitOrder(algorithm_instance.nq.mapped, -quantity, entry_price)
                algorithm_instance.debug(f"SELL LIMIT order placed: {quantity} contracts at {entry_price}")
                
                # Set stop loss and take profit for SHORT
                if order_ticket:
                    algorithm_instance.StopMarketOrder(algorithm_instance.nq.mapped, quantity, stop_loss)
                    algorithm_instance.LimitOrder(algorithm_instance.nq.mapped, quantity, take_profit)
                    algorithm_instance.debug(f"Stop Loss set at {stop_loss}, Take Profit at {take_profit}")
                    algorithm_instance.trade_taken_today = True
                
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
                
                # EXECUTE SELL LIMIT ORDER (SHORT)
                quantity = 1  # For NQ futures, typically 1 contract
                order_ticket = algorithm_instance.LimitOrder(algorithm_instance.nq.mapped, -quantity, entry_price)
                algorithm_instance.debug(f"SELL LIMIT order placed: {quantity} contracts at {entry_price}")
                
                # Set stop loss and take profit for SHORT
                if order_ticket:
                    algorithm_instance.StopMarketOrder(algorithm_instance.nq.mapped, quantity, stop_loss)
                    algorithm_instance.LimitOrder(algorithm_instance.nq.mapped, quantity, take_profit)
                    algorithm_instance.debug(f"Stop Loss set at {stop_loss}, Take Profit at {take_profit}")
                    algorithm_instance.trade_taken_today = True
                
            else:
                algorithm_instance.debug(f"NO SHORT ENTRY: Current close {current_price:.2f} >= Candle 1 Low {candle1_low:.2f}")


