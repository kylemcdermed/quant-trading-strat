




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

