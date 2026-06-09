# features/label_targets.py
import pandas as pd
import numpy as np
import os

def generate_labels(df: pd.DataFrame, atr_multiplier: float = 2.0, max_lookahead: int = 48, spread: float = 0.15) -> pd.DataFrame:
    """
    Sweeps the dataframe to simulate an ATR-based 1:2 RR bracket.
    Assigns targets: 0 (Hold/Loss), 1 (Long Win), 2 (Short Win).
    """
    print(f"Scanning {len(df)} rows for deterministic bracket resolutions...")
    targets = np.zeros(len(df), dtype=int)
    
    # Extract to numpy arrays for significantly faster iteration than pandas .iterrows()
    highs = df['env_high'].values
    lows = df['env_low'].values
    closes = df['env_close'].values
    atrs = df['env_atr'].values

    # We subtract max_lookahead to prevent index out of bounds at the end of the file
    for i in range(len(df) - max_lookahead):
        entry_price = closes[i]
        atr = atrs[i]
        
        if pd.isna(atr) or atr == 0:
            continue

        # Mathematical bracket definitions
        long_entry = entry_price + spread
        long_sl = long_entry - atr
        long_tp = long_entry + (atr * atr_multiplier)
        
        short_entry = entry_price - spread
        short_sl = short_entry + atr
        short_tp = short_entry - (atr * atr_multiplier)
        
        long_outcome = 0   # 0 = Pending, 1 = Hit TP, -1 = Hit SL
        short_outcome = 0

        # Scan the horizon (forward time modeling)
        for j in range(i + 1, i + max_lookahead):
            future_high = highs[j]
            future_low = lows[j]
            
            # Evaluate Long
            if long_outcome == 0:
                if future_low <= long_sl:
                    long_outcome = -1
                elif future_high >= long_tp:
                    long_outcome = 1
                    
            # Evaluate Short
            if short_outcome == 0:
                if future_high >= short_sl:
                    short_outcome = -1
                elif future_low <= short_tp:
                    short_outcome = 1
                    
            # Break early if both directions have resolved to save compute
            if long_outcome != 0 and short_outcome != 0:
                break
                
        # Resolve the classification
        # If both miraculously won (massive identical wick in both directions), we label 0 to avoid confusing the network
        if long_outcome == 1 and short_outcome != 1:
            targets[i] = 1
        elif short_outcome == 1 and long_outcome != 1:
            targets[i] = 2
        else:
            targets[i] = 0

    df_out = df.copy()
    df_out['target'] = targets
    return df_out

if __name__ == "__main__":
    input_path = '../data/processed/master_features_15m.csv'
    output_path = '../data/processed/labeled_features_15m.csv'
    
    if os.path.exists(input_path):
        df = pd.read_csv(input_path, index_col=0)
        labeled_df = generate_labels(df)
        
        # Verify distribution
        counts = labeled_df['target'].value_counts()
        print("\nTarget Distribution:")
        print(f"0 (Hold/Fail): {counts.get(0, 0)}")
        print(f"1 (Long Win):  {counts.get(1, 0)}")
        print(f"2 (Short Win): {counts.get(2, 0)}")
        
        labeled_df.to_csv(output_path)
        print(f"\nSaved labeled dataset to {output_path}")
    else:
        print(f"File not found: {input_path}")