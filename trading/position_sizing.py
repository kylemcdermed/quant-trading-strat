import numpy as np
import math




def CalculateKellyCriterion(win_rate, avg_win, avg_loss):
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
            
            debug(f"Kelly Calculation - Win Rate: {win_rate:.2%}, Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}")
            debug(f"Kelly Fraction: {kelly_fraction:.3f} ({kelly_fraction*100:.1f}% of capital)")
            
            return kelly_fraction
            
        except Exception as e:
            debug(f"Error calculating Kelly: {e}")
            return 0.0




def GetHistoricalPerformance(lookback_days=60):
        """
        Analyze recent trade performance for Kelly Criterion inputs
        
        Returns: (win_rate, avg_win, avg_loss)
        """
        try:
            # Get recent history for performance analysis
            history = history([nq.Symbol], 31, Resolution.DAILY)
            
            if history.empty:
                debug("History is empty")
                return
            
            # Extract price data
            closes = []
            if 'close' in history.columns:
                closes = history['close'].dropna().values
            elif 'Close' in history.columns:
                closes = history['Close'].dropna().values
            else:
                # Try multi-index access
                for symbol_key in [nq.Symbol, str(nq.Symbol)]:
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
            debug(f"Error calculating historical performance: {e}")
            return 0.55, 1.0, 1.0  # Conservative defaults


def CalculateKellyPositionSize():
        """
        Calculate position size using Kelly Criterion
        
        Returns: number of contracts to trade
        """
        try:
            # Get historical performance for Kelly inputs
            win_rate, avg_win, avg_loss = GetHistoricalPerformance()
            
            # Calculate Kelly fraction
            kelly_fraction = CalculateKellyCriterion(win_rate, avg_win, avg_loss)
            
            if kelly_fraction <= 0:
                return 0
            
            # Calculate position size
            capital = portfolio.total_portfolio_value
            risk_capital = capital * kelly_fraction
            
            # Get current price
            current_price = securities[nq.Mapped].Price
            contract_value = current_price * contract_multiplier
            
            if contract_value <= 0:
                return 0
            
            # Calculate number of contracts
            contracts = int(risk_capital / contract_value)
            
            # Apply reasonable limits
            max_contracts = int(capital * 0.25 / contract_value)  # Never more than 25% of capital
            contracts = min(contracts, max_contracts)
            
            debug(f"Kelly Position Sizing - Capital: ${capital:,.0f}, Risk: ${risk_capital:,.0f}, Contracts: {contracts}")
            
            return contracts
            
        except Exception as e:
            debug(f"Error calculating Kelly position size: {e}")
            return 0
