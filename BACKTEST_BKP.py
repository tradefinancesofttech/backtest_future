import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==================================================
# OPTIMIZED CONFIG FOR 18%+ RETURNS
# ==================================================
# Timeframes (in minutes) - OPTIMIZED: Wider spread for better trend confirmation
HIGHER_TF = 30   # Increased from 30 for stronger trend signals
LOWER_TF = 5     # Keep for micro confirmation
CURRENT_TF = 15  # Your CSV timeframe

# Indicator Settings - OPTIMIZED
EMA_LEN = 21     # Increased from 20 (Fibonacci number, better trend capture)
RSI_LEN = 14     # Standard, proven effective
ATR_LEN = 14     # Standard

# Entry Thresholds - OPTIMIZED: More selective = better quality trades
BUY_ENTRY_PCT = 80   # Increased from 70 - only take high-confidence setups
SELL_ENTRY_PCT = 80  # Increased from 70

# ATR-based TP/SL - OPTIMIZED: Better risk/reward ratio
TP_ATR_MULTIPLIER = 2.2   # Increased from 1.8 - let winners run more
SL_ATR_MULTIPLIER = 0.5  # Decreased from 1.0 - tighter stops
# R:R Ratio = 2.59:1 (excellent!)

# Trailing Stop Settings - OPTIMIZED: Smarter trailing
ENABLE_TRAILING = True
TRAIL_ACTIVATION_PCT = 10    # Start trailing at 50% of TP (earlier protection)
TRAIL_STEP_PCT = 4.5         # Smaller steps = smoother trailing
SL_MOVE_PER_STEP = 5.0       # Move SL by 4 points per step

# ENHANCED: Partial Profit Taking - Lock in gains early
ENABLE_PARTIAL_PROFITS = True
PARTIAL_PROFIT_AT_PCT = 60   # Take partial at 60% of full TP
PARTIAL_SIZE_PCT = 40        # Exit 40% of position

# ENHANCED: Multi-level profit taking
ENABLE_MULTI_LEVEL = True
LEVEL1_TP_PCT = 50          # First level at 50% of TP
LEVEL1_EXIT_PCT = 25        # Exit 25% at level 1
LEVEL2_TP_PCT = 75          # Second level at 75% of TP
LEVEL2_EXIT_PCT = 30        # Exit 30% at level 2
# Remaining 45% rides to full TP or trailing

# ENHANCED: Volatility-based position sizing
ENABLE_VOL_FILTER = True
MIN_ATR = 15        # Avoid very low volatility (choppy)
MAX_ATR = 400       # Avoid extreme volatility (risky)
OPTIMAL_ATR_LOW = 25   # Sweet spot range for best trades
OPTIMAL_ATR_HIGH = 150

# ENHANCED: Momentum filter
ENABLE_MOMENTUM_FILTER = True
MOMENTUM_PERIOD = 10
MIN_MOMENTUM_STRENGTH = 5  # Minimum momentum for entry

# ENHANCED: Volume filter
ENABLE_VOLUME_FILTER = True
VOLUME_MA_PERIOD = 20
MIN_VOLUME_RATIO = 0.8  # Current volume should be 80% of average

# Backtest Settings
INITIAL_CAPITAL = 100000
TRADING_DAYS_PER_YEAR = 252

# POSITION SIZING SETTINGS
POSITION_SIZE_MODE = "FIXED"     # Options: "FIXED", "RISK_BASED", "PERCENT_CAPITAL"
FIXED_QUANTITY = 1               # Number of contracts/lots per trade (for FIXED mode)
RISK_PER_TRADE_PCT = 2.0         # % of capital to risk per trade (for RISK_BASED mode)
CAPITAL_PER_TRADE_PCT = 10.0     # % of capital to allocate per trade (for PERCENT_CAPITAL mode)
POINT_VALUE = 20                 # Dollar value per point (e.g., NQ = $20/point, ES = $50/point)

# ==================================================
# PERFORMANCE METRICS
# ==================================================
def sharpe_ratio(returns):
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def sortino_ratio(returns):
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return returns.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = (equity / roll_max - 1.0).min()
    return drawdown if not np.isnan(drawdown) else 0.0

def calmar_ratio(returns, max_dd):
    if max_dd == 0:
        return 0.0
    annualized_return = returns.mean() * TRADING_DAYS_PER_YEAR
    return abs(annualized_return / max_dd)

# ==================================================
# TECHNICAL INDICATORS
# ==================================================
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_momentum(close, period):
    return close.diff(period)

def calculate_volume_ratio(volume, period):
    vol_ma = volume.rolling(window=period).mean()
    return volume / vol_ma

# ==================================================
# POSITION SIZING
# ==================================================
def calculate_position_size(mode, equity, entry_price, stop_loss, atr):
    """
    Calculate position size based on selected mode
    
    Returns: (quantity, max_loss_dollars)
    """
    if mode == "FIXED":
        return FIXED_QUANTITY, abs(entry_price - stop_loss) * FIXED_QUANTITY * POINT_VALUE
    
    elif mode == "RISK_BASED":
        # Risk-based: Risk X% of capital per trade
        risk_amount = equity * (RISK_PER_TRADE_PCT / 100.0)
        risk_per_contract = abs(entry_price - stop_loss) * POINT_VALUE
        
        if risk_per_contract > 0:
            quantity = int(risk_amount / risk_per_contract)
            quantity = max(1, quantity)  # At least 1 contract
            return quantity, risk_per_contract * quantity
        else:
            return 1, abs(entry_price - stop_loss) * POINT_VALUE
    
    elif mode == "PERCENT_CAPITAL":
        # Allocate X% of capital per trade
        allocation = equity * (CAPITAL_PER_TRADE_PCT / 100.0)
        cost_per_contract = entry_price * POINT_VALUE  # Approximate margin requirement
        
        if cost_per_contract > 0:
            quantity = int(allocation / cost_per_contract)
            quantity = max(1, quantity)  # At least 1 contract
            return quantity, abs(entry_price - stop_loss) * quantity * POINT_VALUE
        else:
            return 1, abs(entry_price - stop_loss) * POINT_VALUE
    
    else:
        return FIXED_QUANTITY, abs(entry_price - stop_loss) * FIXED_QUANTITY * POINT_VALUE

# ==================================================
# RESAMPLE TO DIFFERENT TIMEFRAMES
# ==================================================
def resample_to_timeframe(df, timeframe_minutes):
    """Resample dataframe to a different timeframe"""
    df_copy = df.copy()
    df_copy.set_index('datetime', inplace=True)
    
    resampled = df_copy.resample(f'{timeframe_minutes}T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    return resampled

# ==================================================
# ENHANCED MTF PROBABILITY CALCULATION
# ==================================================
def calculate_mtf_signals(df, higher_tf_df, lower_tf_df):
    """Calculate buy/sell probability scores with ENHANCED MTF analysis"""
    
    # Current timeframe indicators
    df['ema'] = calculate_ema(df['close'], EMA_LEN)
    df['rsi'] = calculate_rsi(df['close'], RSI_LEN)
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], ATR_LEN)
    
    if ENABLE_MOMENTUM_FILTER:
        df['momentum'] = calculate_momentum(df['close'], MOMENTUM_PERIOD)
    
    if ENABLE_VOLUME_FILTER:
        df['volume_ratio'] = calculate_volume_ratio(df['volume'], VOLUME_MA_PERIOD)
    
    # Higher timeframe indicators - ENHANCED
    higher_tf_df['ema_htf'] = calculate_ema(higher_tf_df['close'], EMA_LEN)
    higher_tf_df['rsi_htf'] = calculate_rsi(higher_tf_df['close'], RSI_LEN)
    higher_tf_df['momentum_htf'] = calculate_momentum(higher_tf_df['close'], MOMENTUM_PERIOD)
    
    # Lower timeframe indicators
    lower_tf_df['ema_ltf'] = calculate_ema(lower_tf_df['close'], EMA_LEN)
    lower_tf_df['rsi_ltf'] = calculate_rsi(lower_tf_df['close'], RSI_LEN)
    
    # Initialize arrays
    buy_score = np.zeros(len(df))
    sell_score = np.zeros(len(df))
    buy_pct = np.zeros(len(df))
    sell_pct = np.zeros(len(df))
    signal = ['WAIT'] * len(df)
    market_state = ['RANGING'] * len(df)
    signal_quality = np.zeros(len(df))  # NEW: Track signal quality
    
    for i in range(max(EMA_LEN, RSI_LEN, ATR_LEN, MOMENTUM_PERIOD, VOLUME_MA_PERIOD) + 5, len(df)):
        current_time = df.at[i, 'datetime']
        
        # ENHANCED scoring system with weighted components
        b_score = 0.0
        s_score = 0.0
        quality = 0.0
        
        # === CURRENT TIMEFRAME (Weight: 3.5) ===
        
        # Price vs EMA with slope confirmation (weight: 2.0)
        price_above_ema = df.at[i, 'close'] > df.at[i, 'ema']
        ema_slope = df.at[i, 'ema'] - df.at[i-1, 'ema']
        
        if price_above_ema and ema_slope > 0:
            b_score += 2.0
            quality += 1.0
        elif price_above_ema:
            b_score += 1.0
        
        if not price_above_ema and ema_slope < 0:
            s_score += 2.0
            quality += 1.0
        elif not price_above_ema:
            s_score += 1.0
        
        # RSI with ENHANCED zones (weight: 1.5)
        rsi_val = df.at[i, 'rsi']
        if rsi_val > 58:  # Strong bullish
            b_score += 1.5
            quality += 0.5
        elif rsi_val > 52:  # Mild bullish
            b_score += 0.5
        
        if rsi_val < 50:  # Strong bearish
            s_score += 1.5
            quality += 0.5
        elif rsi_val < 48:  # Mild bearish
            s_score += 0.5
        
        # === MOMENTUM FILTER (Weight: 1.0) ===
        if ENABLE_MOMENTUM_FILTER:
            momentum = df.at[i, 'momentum']
            if not np.isnan(momentum):
                if momentum > MIN_MOMENTUM_STRENGTH:
                    b_score += 1.0
                    quality += 0.5
                elif momentum < -MIN_MOMENTUM_STRENGTH:
                    s_score += 1.0
                    quality += 0.5
        
        # === HIGHER TIMEFRAME (Weight: 2.5 - INCREASED) ===
        htf_row = higher_tf_df[higher_tf_df['datetime'] <= current_time].tail(1)
        if not htf_row.empty:
            htf_idx = htf_row.index[0]
            htf_bullish = htf_row.at[htf_idx, 'close'] > htf_row.at[htf_idx, 'ema_htf']
            htf_rsi = htf_row.at[htf_idx, 'rsi_htf']
            htf_momentum = htf_row.at[htf_idx, 'momentum_htf']
            
            # Strong HTF alignment (all factors agree)
            if htf_bullish and htf_rsi > 50 and htf_momentum > 0:
                b_score += 2.5
                quality += 1.5
            elif htf_bullish:
                b_score += 1.0
            
            if not htf_bullish and htf_rsi < 50 and htf_momentum < 0:
                s_score += 2.5
                quality += 1.5
            elif not htf_bullish:
                s_score += 1.0
        
        # === LOWER TIMEFRAME (Weight: 0.8 - REDUCED) ===
        ltf_row = lower_tf_df[lower_tf_df['datetime'] <= current_time].tail(1)
        if not ltf_row.empty:
            ltf_idx = ltf_row.index[0]
            ltf_rsi = ltf_row.at[ltf_idx, 'rsi_ltf']
            
            if ltf_row.at[ltf_idx, 'close'] > ltf_row.at[ltf_idx, 'ema_ltf'] and ltf_rsi > 50:
                b_score += 0.8
            elif ltf_row.at[ltf_idx, 'close'] < ltf_row.at[ltf_idx, 'ema_ltf'] and ltf_rsi < 50:
                s_score += 0.8
        
        # === VOLUME CONFIRMATION (Weight: 0.7) ===
        if ENABLE_VOLUME_FILTER:
            vol_ratio = df.at[i, 'volume_ratio']
            if not np.isnan(vol_ratio) and vol_ratio >= MIN_VOLUME_RATIO:
                quality += 0.7
        
        # === MARKET STATE ===
        atr_val = df.at[i, 'atr']
        atr_ratio = (df.at[i, 'high'] - df.at[i, 'low']) / atr_val if atr_val > 0 else 0
        ema_change = abs(ema_slope) if not np.isnan(ema_slope) else 0
        
        is_trending = (
            atr_ratio > 1.1 and
            (rsi_val > 58 or rsi_val < 50) and
            ema_change > atr_val * 0.06
        )
        
        market_state[i] = "TRENDING" if is_trending else "RANGING"
        
        # Bonus for trending markets
        if is_trending:
            quality += 1.0
        
        # Normalize to percentages
        total = b_score + s_score
        if total > 0:
            b_pct = (b_score / total) * 100
            s_pct = 100 - b_pct
        else:
            b_pct = 50
            s_pct = 50
        
        buy_score[i] = b_score
        sell_score[i] = s_score
        buy_pct[i] = b_pct
        sell_pct[i] = s_pct
        signal_quality[i] = quality
        
        # Signal determination with QUALITY filter
        if b_pct >= BUY_ENTRY_PCT and quality >= 3.0:  # Require minimum quality
            signal[i] = "BUY"
        elif s_pct >= SELL_ENTRY_PCT and quality >= 3.0:
            signal[i] = "SELL"
        else:
            signal[i] = "WAIT"
    
    df['buy_score'] = buy_score
    df['sell_score'] = sell_score
    df['buy_pct'] = buy_pct
    df['sell_pct'] = sell_pct
    df['signal'] = signal
    df['market_state'] = market_state
    df['signal_quality'] = signal_quality
    
    return df

# ==================================================
# ENHANCED BACKTEST ENGINE
# ==================================================
def backtest_csv(csv_path, starting_capital=INITIAL_CAPITAL, file_name=""):
    """OPTIMIZED backtesting with multi-level exits and smart filtering"""
    
    # Load data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    
    # Datetime handling
    time_col = None
    for col in ['datetime', 'date', 'time', 'timestamp']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise Exception(f"No datetime column found in {csv_path}")
    
    df['datetime'] = pd.to_datetime(df[time_col].astype(str).str.strip(), errors='coerce', utc=True)
    df = df.dropna(subset=['datetime'])
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    df = df.sort_values('datetime').reset_index(drop=True)
    df['source_file'] = file_name
    
    # Validate columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            raise Exception(f"Missing column '{col}' in {csv_path}")
    
    # Resample to different timeframes
    higher_tf_df = resample_to_timeframe(df, HIGHER_TF)
    lower_tf_df = resample_to_timeframe(df, LOWER_TF)
    
    # Calculate MTF signals
    df = calculate_mtf_signals(df, higher_tf_df, lower_tf_df)
    
    # ==================================================
    # ENHANCED TRADING LOGIC
    # ==================================================
    equity = starting_capital
    equity_curve = []
    trades = []
    trade_log = []
    
    position = None
    entry_price = 0.0
    entry_time = None
    position_size = 1.0  # Track remaining position size (as percentage)
    quantity = 0         # Number of contracts
    take_profit = 0.0
    stop_loss = 0.0
    
    # Multi-level targets
    level1_tp = 0.0
    level2_tp = 0.0
    level1_taken = False
    level2_taken = False
    
    # Trailing variables
    trailing_active = False
    trail_activation_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    prev_signal = "WAIT"
    filtered_count = 0
    
    for i in range(len(df)):
        current_signal = df.at[i, 'signal']
        current_time = df.at[i, 'datetime']
        current_close = df.at[i, 'close']
        current_high = df.at[i, 'high']
        current_low = df.at[i, 'low']
        current_atr = df.at[i, 'atr']
        
        if np.isnan(current_atr):
            equity_curve.append(equity)
            prev_signal = current_signal
            continue
        
        # === VOLATILITY FILTER ===
        if ENABLE_VOL_FILTER:
            if current_atr < MIN_ATR or current_atr > MAX_ATR:
                equity_curve.append(equity)
                prev_signal = "WAIT"
                filtered_count += 1
                continue
        
        # === POSITION MANAGEMENT ===
        if position == 'LONG':
            if current_high > highest_price:
                highest_price = current_high
            
            # Multi-level profit taking
            if ENABLE_MULTI_LEVEL:
                if not level1_taken and current_high >= level1_tp:
                    contracts_to_exit = quantity * (LEVEL1_EXIT_PCT / 100.0)
                    partial_pnl = (level1_tp - entry_price) * contracts_to_exit * POINT_VALUE
                    trades.append(partial_pnl)
                    equity += partial_pnl
                    trade_log.append({
                        'File': file_name,
                        'Entry Time': entry_time,
                        'Exit Time': current_time,
                        'Type': 'LONG (L1)',
                        'Quantity': contracts_to_exit,
                        'Entry Price': entry_price,
                        'Exit Price': level1_tp,
                        'Exit Reason': 'Level 1 TP',
                        'PnL': partial_pnl,
                        'Equity': equity
                    })
                    level1_taken = True
                    position_size -= (LEVEL1_EXIT_PCT / 100.0)
                
                if not level2_taken and current_high >= level2_tp and level1_taken:
                    contracts_to_exit = quantity * (LEVEL2_EXIT_PCT / 100.0)
                    partial_pnl = (level2_tp - entry_price) * contracts_to_exit * POINT_VALUE
                    trades.append(partial_pnl)
                    equity += partial_pnl
                    trade_log.append({
                        'File': file_name,
                        'Entry Time': entry_time,
                        'Exit Time': current_time,
                        'Type': 'LONG (L2)',
                        'Quantity': contracts_to_exit,
                        'Entry Price': entry_price,
                        'Exit Price': level2_tp,
                        'Exit Reason': 'Level 2 TP',
                        'PnL': partial_pnl,
                        'Equity': equity
                    })
                    level2_taken = True
                    position_size -= (LEVEL2_EXIT_PCT / 100.0)
            
            # Trailing activation
            if ENABLE_TRAILING and not trailing_active and current_high >= trail_activation_price:
                trailing_active = True
            
            # Update trailing stop
            if trailing_active:
                profit_above_activation = highest_price - trail_activation_price
                num_steps = int(profit_above_activation / (current_atr * TRAIL_STEP_PCT / 100.0))
                if num_steps > 0:
                    new_stop = trail_activation_price + (num_steps * SL_MOVE_PER_STEP)
                    stop_loss = max(stop_loss, new_stop)
            
            # Check stop loss
            if current_low <= stop_loss:
                remaining_contracts = quantity * position_size
                pnl = (stop_loss - entry_price) * remaining_contracts * POINT_VALUE
                trades.append(pnl)
                equity += pnl
                exit_reason = 'Trailing Stop' if trailing_active else 'Stop Loss'
                trade_log.append({
                    'File': file_name,
                    'Entry Time': entry_time,
                    'Exit Time': current_time,
                    'Type': 'LONG',
                    'Quantity': remaining_contracts,
                    'Entry Price': entry_price,
                    'Exit Price': stop_loss,
                    'Exit Reason': exit_reason,
                    'PnL': pnl,
                    'Equity': equity
                })
                position = None
                position_size = 1.0
                quantity = 0
                trailing_active = False
                level1_taken = False
                level2_taken = False
            # Check full TP
            elif not trailing_active and current_high >= take_profit:
                remaining_contracts = quantity * position_size
                pnl = (take_profit - entry_price) * remaining_contracts * POINT_VALUE
                trades.append(pnl)
                equity += pnl
                trade_log.append({
                    'File': file_name,
                    'Entry Time': entry_time,
                    'Exit Time': current_time,
                    'Type': 'LONG',
                    'Quantity': remaining_contracts,
                    'Entry Price': entry_price,
                    'Exit Price': take_profit,
                    'Exit Reason': 'Take Profit',
                    'PnL': pnl,
                    'Equity': equity
                })
                position = None
                position_size = 1.0
                quantity = 0
                trailing_active = False
                level1_taken = False
                level2_taken = False
        
        elif position == 'SHORT':
            if current_low < lowest_price:
                lowest_price = current_low
            
            # Multi-level profit taking
            if ENABLE_MULTI_LEVEL:
                if not level1_taken and current_low <= level1_tp:
                    contracts_to_exit = quantity * (LEVEL1_EXIT_PCT / 100.0)
                    partial_pnl = (entry_price - level1_tp) * contracts_to_exit * POINT_VALUE
                    trades.append(partial_pnl)
                    equity += partial_pnl
                    trade_log.append({
                        'File': file_name,
                        'Entry Time': entry_time,
                        'Exit Time': current_time,
                        'Type': 'SHORT (L1)',
                        'Quantity': contracts_to_exit,
                        'Entry Price': entry_price,
                        'Exit Price': level1_tp,
                        'Exit Reason': 'Level 1 TP',
                        'PnL': partial_pnl,
                        'Equity': equity
                    })
                    level1_taken = True
                    position_size -= (LEVEL1_EXIT_PCT / 100.0)
                
                if not level2_taken and current_low <= level2_tp and level1_taken:
                    contracts_to_exit = quantity * (LEVEL2_EXIT_PCT / 100.0)
                    partial_pnl = (entry_price - level2_tp) * contracts_to_exit * POINT_VALUE
                    trades.append(partial_pnl)
                    equity += partial_pnl
                    trade_log.append({
                        'File': file_name,
                        'Entry Time': entry_time,
                        'Exit Time': current_time,
                        'Type': 'SHORT (L2)',
                        'Quantity': contracts_to_exit,
                        'Entry Price': entry_price,
                        'Exit Price': level2_tp,
                        'Exit Reason': 'Level 2 TP',
                        'PnL': partial_pnl,
                        'Equity': equity
                    })
                    level2_taken = True
                    position_size -= (LEVEL2_EXIT_PCT / 100.0)
            
            # Trailing activation
            if ENABLE_TRAILING and not trailing_active and current_low <= trail_activation_price:
                trailing_active = True
            
            # Update trailing stop
            if trailing_active:
                profit_below_activation = trail_activation_price - lowest_price
                num_steps = int(profit_below_activation / (current_atr * TRAIL_STEP_PCT / 100.0))
                if num_steps > 0:
                    new_stop = trail_activation_price - (num_steps * SL_MOVE_PER_STEP)
                    stop_loss = min(stop_loss, new_stop)
            
            # Check stop loss
            if current_high >= stop_loss:
                remaining_contracts = quantity * position_size
                pnl = (entry_price - stop_loss) * remaining_contracts * POINT_VALUE
                trades.append(pnl)
                equity += pnl
                exit_reason = 'Trailing Stop' if trailing_active else 'Stop Loss'
                trade_log.append({
                    'File': file_name,
                    'Entry Time': entry_time,
                    'Exit Time': current_time,
                    'Type': 'SHORT',
                    'Quantity': remaining_contracts,
                    'Entry Price': entry_price,
                    'Exit Price': stop_loss,
                    'Exit Reason': exit_reason,
                    'PnL': pnl,
                    'Equity': equity
                })
                position = None
                position_size = 1.0
                quantity = 0
                trailing_active = False
                level1_taken = False
                level2_taken = False
            # Check full TP
            elif not trailing_active and current_low <= take_profit:
                remaining_contracts = quantity * position_size
                pnl = (entry_price - take_profit) * remaining_contracts * POINT_VALUE
                trades.append(pnl)
                equity += pnl
                trade_log.append({
                    'File': file_name,
                    'Entry Time': entry_time,
                    'Exit Time': current_time,
                    'Type': 'SHORT',
                    'Quantity': remaining_contracts,
                    'Entry Price': entry_price,
                    'Exit Price': take_profit,
                    'Exit Reason': 'Take Profit',
                    'PnL': pnl,
                    'Equity': equity
                })
                position = None
                position_size = 1.0
                quantity = 0
                trailing_active = False
                level1_taken = False
                level2_taken = False
        
        # === SIGNAL FLIP DETECTION ===
        if position is None and prev_signal != "WAIT" and current_signal != "WAIT" and prev_signal != current_signal:
            entry_price = current_close
            entry_time = current_time
            position_size = 1.0
            trailing_active = False
            level1_taken = False
            level2_taken = False
            
            if current_signal == "BUY":
                position = 'LONG'
                profit_distance = current_atr * TP_ATR_MULTIPLIER
                take_profit = entry_price + profit_distance
                stop_loss = entry_price - (current_atr * SL_ATR_MULTIPLIER)
                
                # Calculate position size
                quantity, max_risk = calculate_position_size(
                    POSITION_SIZE_MODE, 
                    equity, 
                    entry_price, 
                    stop_loss, 
                    current_atr
                )
                
                # Set multi-level targets
                level1_tp = entry_price + (profit_distance * LEVEL1_TP_PCT / 100.0)
                level2_tp = entry_price + (profit_distance * LEVEL2_TP_PCT / 100.0)
                trail_activation_price = entry_price + (profit_distance * TRAIL_ACTIVATION_PCT / 100.0)
                
                highest_price = entry_price
                
            elif current_signal == "SELL":
                position = 'SHORT'
                profit_distance = current_atr * TP_ATR_MULTIPLIER
                take_profit = entry_price - profit_distance
                stop_loss = entry_price + (current_atr * SL_ATR_MULTIPLIER)
                
                # Calculate position size
                quantity, max_risk = calculate_position_size(
                    POSITION_SIZE_MODE, 
                    equity, 
                    entry_price, 
                    stop_loss, 
                    current_atr
                )
                
                # Set multi-level targets
                level1_tp = entry_price - (profit_distance * LEVEL1_TP_PCT / 100.0)
                level2_tp = entry_price - (profit_distance * LEVEL2_TP_PCT / 100.0)
                trail_activation_price = entry_price - (profit_distance * TRAIL_ACTIVATION_PCT / 100.0)
                
                lowest_price = entry_price
        
        equity_curve.append(equity)
        prev_signal = current_signal
    
    # Close open position
    if position is not None:
        final_close = df.at[len(df)-1, 'close']
        remaining_contracts = quantity * position_size
        
        if position == 'LONG':
            pnl = (final_close - entry_price) * remaining_contracts * POINT_VALUE
        else:
            pnl = (entry_price - final_close) * remaining_contracts * POINT_VALUE
        
        trades.append(pnl)
        equity += pnl
        trade_log.append({
            'File': file_name,
            'Entry Time': entry_time,
            'Exit Time': df.at[len(df)-1, 'datetime'],
            'Type': position,
            'Quantity': remaining_contracts,
            'Entry Price': entry_price,
            'Exit Price': final_close,
            'Exit Reason': 'End of Data',
            'PnL': pnl,
            'Equity': equity
        })
    
    # Calculate metrics
    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    
    total_return = ((equity - starting_capital) / starting_capital) * 100
    win_rate = (len(wins) / len(trades) * 100) if len(trades) > 0 else 0.0
    max_dd = max_drawdown(equity_series)
    
    if len(losses) and losses.sum() != 0:
        profit_factor = abs(wins.sum() / losses.sum())
    else:
        profit_factor = 0.0
    
    tp_exits = sum(1 for t in trade_log if 'Take Profit' in t['Exit Reason'])
    sl_exits = sum(1 for t in trade_log if t['Exit Reason'] == 'Stop Loss')
    trail_exits = sum(1 for t in trade_log if t['Exit Reason'] == 'Trailing Stop')
    partial_exits = sum(1 for t in trade_log if 'L1' in t['Type'] or 'L2' in t['Type'])
    
    results = {
        "Initial Capital": starting_capital,
        "Final Equity": equity,
        "Total Return": total_return,
        "Total Trades": len(trades),
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Win Rate": win_rate,
        "Gross Profit": wins.sum() if len(wins) else 0.0,
        "Gross Loss": losses.sum() if len(losses) else 0.0,
        "Net Profit": equity - starting_capital,
        "Profit Factor": profit_factor,
        "Average Win": wins.mean() if len(wins) else 0.0,
        "Average Loss": losses.mean() if len(losses) else 0.0,
        "Largest Win": wins.max() if len(wins) else 0.0,
        "Largest Loss": losses.min() if len(losses) else 0.0,
        "Max Drawdown": max_dd * 100,
        "Sharpe Ratio": sharpe_ratio(returns),
        "Sortino Ratio": sortino_ratio(returns),
        "Calmar Ratio": calmar_ratio(returns, max_dd),
        "TP Exits": tp_exits,
        "SL Exits": sl_exits,
        "Trailing Exits": trail_exits,
        "Partial Exits": partial_exits,
        "Filtered Signals": filtered_count,
    }
    
    return results, trade_log, equity

# ==================================================
# FORMAT AND RUN
# ==================================================
def format_results(results):
    return {
        "Initial Capital": f"${results['Initial Capital']:,.2f}",
        "Final Equity": f"${results['Final Equity']:,.2f}",
        "Total Return": f"{results['Total Return']:.2f}%",
        "Total Trades": results['Total Trades'],
        "Winning Trades": results['Winning Trades'],
        "Losing Trades": results['Losing Trades'],
        "Win Rate": f"{results['Win Rate']:.2f}%",
        "Gross Profit": f"${results['Gross Profit']:,.2f}",
        "Gross Loss": f"${results['Gross Loss']:,.2f}",
        "Net Profit": f"${results['Net Profit']:,.2f}",
        "Profit Factor": f"{results['Profit Factor']:.2f}" if results['Profit Factor'] > 0 else "N/A",
        "Average Win": f"${results['Average Win']:,.2f}",
        "Average Loss": f"${results['Average Loss']:,.2f}",
        "Largest Win": f"${results['Largest Win']:,.2f}",
        "Largest Loss": f"${results['Largest Loss']:,.2f}",
        "Max Drawdown": f"{results['Max Drawdown']:.2f}%",
        "Sharpe Ratio": f"{results['Sharpe Ratio']:.2f}",
        "Sortino Ratio": f"{results['Sortino Ratio']:.2f}",
        "Calmar Ratio": f"{results['Calmar Ratio']:.2f}",
        "TP Exits": results['TP Exits'],
        "SL Exits": results['SL Exits'],
        "Trailing Exits": results['Trailing Exits'],
        "Partial Exits": results['Partial Exits'],
        "Filtered Signals": results['Filtered Signals'],
    }

def run_backtest_from_folder(csv_folder):
    """
    Run backtest on ALL CSV files in a folder
    Automatically reads only .csv files and ignores other files
    Creates a unique timestamped folder for each run
    """
    # Create timestamped folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"15min_nq_{timestamp}"
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if folder exists
    if not os.path.exists(csv_folder):
        print(f"Error: Folder not found - {csv_folder}")
        return
    
    # Get all CSV files from folder (automatically filters only .csv files)
    csv_files = sorted([f for f in os.listdir(csv_folder) if f.lower().endswith('.csv')])
    
    if not csv_files:
        print(f"No CSV files found in {csv_folder}")
        return
    
    print(f"{'='*80}")
    print(f"OPTIMIZED MTF STRATEGY - TARGETING 18%+ RETURNS")
    print(f"{'='*80}")
    print(f"CSV Folder: {csv_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Found {len(csv_files)} CSV file(s)\n")
    print(f"KEY OPTIMIZATIONS:")
    print(f"  • Higher TF: {HIGHER_TF}min (wider spread for better signals)")
    print(f"  • Entry Threshold: {BUY_ENTRY_PCT}% (more selective)")
    print(f"  • TP/SL: {TP_ATR_MULTIPLIER}x / {SL_ATR_MULTIPLIER}x (R:R = {TP_ATR_MULTIPLIER/SL_ATR_MULTIPLIER:.2f}:1)")
    print(f"  • Position Sizing: {POSITION_SIZE_MODE}")
    if POSITION_SIZE_MODE == "FIXED":
        print(f"    - Fixed Quantity: {FIXED_QUANTITY} contracts")
    elif POSITION_SIZE_MODE == "RISK_BASED":
        print(f"    - Risk Per Trade: {RISK_PER_TRADE_PCT}% of capital")
    elif POSITION_SIZE_MODE == "PERCENT_CAPITAL":
        print(f"    - Capital Per Trade: {CAPITAL_PER_TRADE_PCT}% allocation")
    print(f"  • Point Value: ${POINT_VALUE}/point")
    print(f"  • Multi-level exits: {LEVEL1_EXIT_PCT}%@{LEVEL1_TP_PCT}% + {LEVEL2_EXIT_PCT}%@{LEVEL2_TP_PCT}%")
    print(f"  • Trail activation: {TRAIL_ACTIVATION_PCT}% of TP")
    print(f"  • Quality filter: Minimum 3.0 score required\n")
    
    all_results = []
    all_trades = []
    current_capital = INITIAL_CAPITAL
    
    for idx, file_name in enumerate(csv_files):
        csv_path = os.path.join(csv_folder, file_name)
        print(f"[{idx+1}/{len(csv_files)}] {file_name}")
        
        try:
            results, trade_log, ending_equity = backtest_csv(
                csv_path,
                starting_capital=current_capital,
                file_name=file_name
            )
            
            all_results.append({'File': file_name, **results})
            all_trades.extend(trade_log)
            current_capital = ending_equity
            
            # Save individual file reports in timestamped folder
            report_file = os.path.join(output_folder, f"{file_name.replace('.csv','')}_report.txt")
            trades_file = os.path.join(output_folder, f"{file_name.replace('.csv','')}_trades.csv")
            
            with open(report_file, "w", encoding='utf-8') as f:
                f.write(f"OPTIMIZED MTF BACKTEST\n")
                f.write(f"File: {file_name}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"R:R Ratio: {TP_ATR_MULTIPLIER/SL_ATR_MULTIPLIER:.2f}:1\n")
                f.write(f"Entry Threshold: {BUY_ENTRY_PCT}%\n\n")
                
                formatted = format_results(results)
                for k, v in formatted.items():
                    f.write(f"{k:.<50} {v}\n")
            
            if trade_log:
                pd.DataFrame(trade_log).to_csv(trades_file, index=False)
            
            print(f"  ${results['Initial Capital']:,.0f} → ${ending_equity:,.0f} ({results['Total Return']:+.2f}%)")
            print(f"  Trades: {results['Total Trades']} | Win: {results['Win Rate']:.1f}% | PF: {results['Profit Factor']:.2f}\n")
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
    
    # Combined report in timestamped folder
    if all_results:
        combined_report = os.path.join(output_folder, "ALL_NQ_COMBINED.txt")
        
        total_initial = INITIAL_CAPITAL
        total_final = current_capital
        total_return = ((total_final - total_initial) / total_initial) * 100
        
        all_trades_array = np.array([t['PnL'] for t in all_trades])
        all_wins = all_trades_array[all_trades_array > 0]
        all_losses = all_trades_array[all_trades_array < 0]
        
        combined_equity = [INITIAL_CAPITAL]
        for trade in all_trades:
            combined_equity.append(trade['Equity'])
        
        combined_equity_series = pd.Series(combined_equity)
        combined_returns = combined_equity_series.pct_change().dropna()
        combined_max_dd = max_drawdown(combined_equity_series)
        
        with open(combined_report, "w", encoding='utf-8') as f:
            f.write(f"OPTIMIZED MTF - COMBINED RESULTS\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Initial: ${total_initial:,.0f}\n")
            f.write(f"Final: ${total_final:,.0f}\n")
            f.write(f"Return: {total_return:.2f}%\n")
            f.write(f"Trades: {len(all_trades_array)}\n")
            f.write(f"Win Rate: {(len(all_wins)/len(all_trades_array)*100):.1f}%\n")
            f.write(f"Profit Factor: {abs(all_wins.sum()/all_losses.sum()):.2f}\n")
            f.write(f"Sharpe: {sharpe_ratio(combined_returns):.2f}\n")
            f.write(f"Max DD: {combined_max_dd*100:.2f}%\n\n")
            
            for res in all_results:
                f.write(f"{res['File']}: ${res['Initial Capital']:,.0f} → ${res['Final Equity']:,.0f} ({res['Total Return']:+.2f}%)\n")
        
        pd.DataFrame(all_trades).to_csv(os.path.join(output_folder, "ALL_NQ_trades.csv"), index=False)
        pd.DataFrame(all_results).to_csv(os.path.join(output_folder, "ALL_NQ_summary.csv"), index=False)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"${total_initial:,.0f} → ${total_final:,.0f} = {total_return:.2f}%")
        print(f"Max Drawdown: {combined_max_dd*100:.2f}%")
        print(f"Target: 38%+ | Status: {'✓ ACHIEVED' if total_return >= 38 else '✗ MISSED'}")
        print(f"\nAll reports saved in: {output_folder}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR CSV FOLDER
    csv_folder = "E:/SHALINI_BACKUP/Advanced_trading_systems/Fut-Bot/back_test/BACKTEST/15mint_csvs"
    
    run_backtest_from_folder(csv_folder)