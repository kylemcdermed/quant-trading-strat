# region imports
from AlgorithmImports import *
# endregion
from datetime import time



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



def CalculateORGLevels(current_org, debug):
        """Calculate ORG levels (CE, Q1, Q3) based on open and close"""
        if current_org.open_price is None or current_org.close_price is None:
            return
            
        try:
            open_price = current_org.open_price
            close_price = current_org.close_price
            
            # Calculate gap size and direction
            current_org.size = abs(close_price - open_price)
            current_org.direction = 1 if close_price > open_price else -1
            
            # Calculate key levels
            gap_range = close_price - open_price
            
            # Consequent Encroachment (50% level)
            current_org.ce_price = open_price + (gap_range * 0.5)
            
            # Quarter levels
            current_org.q1_price = open_price + (gap_range * 0.25)
            current_org.q3_price = open_price + (gap_range * 0.75)
            
            current_org.is_valid = True
            
            debug(f"ORG Levels - Open: {open_price:.2f}, Close: {close_price:.2f}")
            debug(f"ORG Levels - CE: {current_org.ce_price:.2f}, Q1: {current_org.q1_price:.2f}, Q3: {current_org.q3_price:.2f}")
            debug(f"ORG Size: {current_org.size:.2f}, Direction: {current_org.direction}")
            
        except Exception as e:
            debug(f"Error calculating ORG levels: {e}")



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
