# 🧠 SYSTEM PROMPT & PROJECT MEMORY: HYBRID ORACLE-MANAGER ENGINE

**Core Objective:** Develop, train, and forward-test a hybrid Algorithmic Trading Engine for Gold (XAUUSD) that decouples Pattern Recognition from Risk Management. 

## 🏛️ The "Oracle-Manager" Architecture

### 1. The Oracle (Supervised Deep Sequence Model)
* **Goal:** Eliminate the RL "Credit Assignment Problem" by predicting deterministic mathematical outcomes based on 30m/4H structural zones and DXY momentum.
* **Network:** PyTorch LSTM (`models/oracle_lstm.py`).
* **Input Space:** A rolling 30-candle window of strictly stationary features (Z-score scaled).
* **Target Output:** Multiclass Softmax `[P_Hold, P_Long, P_Short]`. 
* **State:** Currently undergoing "Goldilocks" regularization tuning. We are balancing `dropout=0.3` and `weight_decay=5e-5` to prevent both training-data memorization (overfitting) and structural blindness (underfitting). Outputs are stable, and it accurately exports its scaling statistics (`oracle_scaler.npz`) for live environment normalization.

### 2. The Manager (Continuous RL Agent)
* **Goal:** Optimize the portfolio equity curve by managing capital deployment based on the Oracle's confidence.
* **Network:** Stable-Baselines3 `SAC` (Soft Actor-Critic) (`models/manager_sac.py`).
* **Observation Space (7D):** Current Account Balance, Drawdown %, Oracle Probabilities (x3), Current 15m ATR, and Bars Held.
* **Action Space:** Continuous `Box` space outputting two values: `Direction` (-1.0 to 1.0) and `TP_Multiplier` (-1.0 to 1.0).
* **State:** Core psychology is cured. The agent has successfully learned to execute strict Risk-to-Reward parameters (achieving an out-of-sample R:R of 1.62). 

## 🛠️ Diagnostic Infrastructure
The project utilizes `models/diagnostic_tester.py`, generating a `master_diagnostic_log.txt`. This file unifies the Oracle's internal probabilities, the Manager's raw continuous output, and the Environment's physical step outcomes into a single readable log. This, combined with the `hybrid_forward_journal_london.csv`, is the primary tool for auditing agent psychology.

---

## 📜 HISTORY OF ITERATIONS & FIXES

**Phase 1: The "Infinite Hold & Deadzone Trap"**
* **Symptom:** The SAC agent executed 0 trades over 1,000+ candles, despite 90%+ confidence setups. 
* **Fix:** Discovered the environment lacked exit logic (TP/SL triggers) and was rewarding the agent `+0.01` for sitting in cash. Lowered the entry threshold to `0.25`, implemented strict high/low exit evaluations, and removed the idle cash reward.
* **Result:** The agent began trading, but fell into a "Risk-Averse Equilibrium," surviving 20,000+ steps but refusing to take high-probability trades to protect capital.

**Phase 2: Asymmetric Reward Shaping**
* **Symptom:** The Manager ignored 91% Oracle signals because the mathematical pain of losing outweighed the reward of winning.
* **Fix:** Introduced a `2.0x` multiplier for winning trades and a "Missed Opportunity Sting" (penalty) for holding cash when the Oracle screamed >85% confidence.
* **Result:** The Manager's fear was cured. It executed 374 trades and achieved a professional 1.62 R:R (Average Win: $132, Average Loss: $81).

**Phase 3: The "Forced Trading Paradox"**
* **Symptom:** Account bled out rapidly (15% win rate, terminating at 2,366 steps). 
* **Fix/Diagnosis:** The Oracle was over-regularized (`dropout=0.4`) and went blind, spitting out erratic probabilities. Because of the "Missed Opportunity Sting" added in Phase 2, the Manager was being penalized for ignoring these erratic signals, forcing it into terrible, rapid-fire trades. 

---

## 📍 EXACT CURRENT STATUS & ACTIVE GOAL

**Current Status:** The structural physics of the engine are mathematically bulletproof. The Manager knows how to size risk and cut losses. We are currently executing the **"Goldilocks Fix"** to dial in the final equilibrium.

**The Math to Profitability:** With an average Risk-to-Reward ratio of 1.62, the strategy requires a **38.1% win rate** to mathematically break even. The current out-of-sample win rate sits at 36.9%. We are 1.2% away from a profitable out-of-sample forward test. 

## 🎯 IMMEDIATE NEXT ACTIONS 
1. **Remove the Cash Veto Penalty:** Update `hybrid_env.py` to remove the cash penalty, restoring the Manager's right to veto the Oracle if the market feels too choppy, without getting financially stung. 
2. **Widen the Stop Loss Breathing Room:** Increase the Stop Loss distance in `hybrid_env.py` to `1.5 * ATR` to prevent the Manager from getting chopped out by 15m market noise before the 4H structural pushes can realize.
3. **Restore Oracle Vision:** Retrain `oracle_lstm.py` with dialed-back regularization (`dropout=0.3`, `weight_decay=5e-5`).
4. **Final Cloud Run:** Execute a 1,000,000 timestep SAC training run on Colab with the newly sighted Oracle and the newly un-forced Manager.