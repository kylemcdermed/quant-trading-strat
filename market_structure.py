def DetectFVG(minute_bars, session_started, first_fvg_detected, current_fvg, daily_fvgs, debug):
    """Detect Fair Value Gaps using 3-candle pattern"""
    if len(minute_bars) < 3 or (session_started and first_fvg_detected):
        return None  # Return None if already detected first FVG

    try:
        # Get last 3 bars
        bar1 = minute_bars[-3]  # 2 bars ago
        bar2 = minute_bars[-2]  # 1 bar ago  
        bar3 = minute_bars[-1]  # Current bar
        
        # Check for bullish FVG: bar1.low > bar3.high
        if bar1['low'] > bar3['high']:
            fvg = FairValueGap()
            fvg.high = bar1['low']
            fvg.low = bar3['high']
            fvg.time = bar3['time']
            fvg.is_bullish = True
            fvg.ce_price = (fvg.high + fvg.low) / 2
            
            if session_started:
                fvg.is_first_of_day = True
                debug(f"First Bullish FVG detected: {fvg.low:.2f} - {fvg.high:.2f} at {fvg.time}")
            
            daily_fvgs.append(fvg)
            return fvg
            
        # Check for bearish FVG: bar1.high < bar3.low
        elif bar1['high'] < bar3['low']:
            fvg = FairValueGap()
            fvg.high = bar3['low']
            fvg.low = bar1['high']
            fvg.time = bar3['time']
            fvg.is_bullish = False
            fvg.ce_price = (fvg.high + fvg.low) / 2
            
            if session_started:
                fvg.is_first_of_day = True
                debug(f"First Bearish FVG detected: {fvg.low:.2f} - {fvg.high:.2f} at {fvg.time}")
            
            daily_fvgs.append(fvg)
            return fvg
            
        return None
        
    except Exception as e:
        debug(f"Error detecting FVG: {e}")
        return None
