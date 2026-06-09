import pandas as pd
import numpy as np

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates the Average True Range (ATR) based on hidden env_ prices."""
    high_low = df['env_high'] - df['env_low']
    high_close = np.abs(df['env_high'] - df['env_close'].shift())
    low_close = np.abs(df['env_low'] - df['env_close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def generate_labels(df: pd.DataFrame, max_hold: int = 32, rr_ratio: float = 2.0, spread: float = 0.15) -> pd.DataFrame:
    """
    Sweeps the dataset and labels future price action.
    0 = Hold/Noise, 1 = Long 2R Hit, 2 = Short 2R Hit.
    """
    print("Calculating ATR and generating forward-looking labels...")
    df = df.copy()
    df['env_atr'] = calculate_atr(df)
    
    # Extract to fast numpy arrays
    close_p = df['env_close'].values
    high_p = df['env_high'].values
    low_p = df['env_low'].values
    atr_v = df['env_atr'].values
    
    targets = np.zeros(len(df), dtype=int)
    
    # Iterate through all rows (except the very end where we can't look ahead)
    for i in range(len(df) - max_hold):
        if np.isnan(atr_v[i]):
            continue
            
        entry_price = close_p[i]
        atr = atr_v[i]
        
        # Avoid zero-volatility errors
        if atr < 0.1: 
            atr = 0.5
            
        # Define strict brackets
        long_tp = entry_price + (atr * rr_ratio) + spread
        long_sl = entry_price - atr - spread
        
        short_tp = entry_price - (atr * rr_ratio) - spread
        short_sl = entry_price + atr + spread
        
        long_valid = True
        short_valid = True
        target = 0
        
        # Look ahead into the future
        for j in range(1, max_hold + 1):
            future_idx = i + j
            f_high = high_p[future_idx]
            f_low = low_p[future_idx]
            
            # Check Long
            if long_valid:
                if f_low <= long_sl:
                    long_valid = False # Stopped out
                elif f_high >= long_tp:
                    target = 1 # TP Hit!
                    break # Break out of look-ahead loop
                    
            # Check Short
            if short_valid:
                if f_high >= short_sl:
                    short_valid = False # Stopped out
                elif f_low <= short_tp:
                    target = 2 # TP Hit!
                    break
                    
            if not long_valid and not short_valid:
                break # Both stopped out, target remains 0
                
        targets[i] = target
        
    df['target'] = targets
    
    # Drop rows where we couldn't calculate ATR or look ahead
    df = df.dropna().iloc[:-max_hold]
    print(f"Labeling complete. Found {np.sum(targets == 1)} Longs, {np.sum(targets == 2)} Shorts, and {np.sum(targets == 0)} Noise/Losses.")
    return df

if __name__ == "__main__":
    print("Loading master features...")
    df = pd.read_csv('../data/processed/master_features_15m.csv', index_col='time')
    
    df_labeled = generate_labels(df, max_hold=32, rr_ratio=2.0)
    
    save_path = '../data/processed/labeled_features_15m.csv'
    df_labeled.to_csv(save_path)
    print(f"Labeled dataset saved to {save_path}")