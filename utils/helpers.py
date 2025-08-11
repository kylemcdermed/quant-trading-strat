from datetime import time



def IsMarketSession(time_stamp, market_open_time, market_close_time):
        """Check if current time is within market session (9:30 AM - 4:00 PM ET)"""
        current_time = time_stamp.time()
        return market_open_time <= current_time <= market_close_time



def StoreMinuteBar(bar, time_stamp, minute_bars, max_minute_bars):
        """Store minute bars for FVG detection"""
        bar_data = {
            'time': time_stamp,
            'open': float(bar.Open),
            'high': float(bar.High),
            'low': float(bar.Low),
            'close': float(bar.Close),
            'volume': int(bar.Volume)
        }
        
        minute_bars.append(bar_data)
        
        # Keep only recent bars
        if len(minute_bars) > max_minute_bars:
            minute_bars.pop(0)



# ORG Functions
def GetORGTarget(current_org):
    """Get 50% ORG target level"""
    if hasattr(current_org, 'is_valid') and current_org.is_valid:
        if hasattr(current_org, 'ce_price'):
            return current_org.ce_price
    return None

def GetFirstFVG(current_fvg):
    """Get the first FVG of the day"""
    return current_fvg

def CheckORGDirection(current_org):
    """Get ORG directional bias"""
    if hasattr(current_org, 'is_valid') and current_org.is_valid:
        if hasattr(current_org, 'direction'):
            return current_org.direction
    return 0
