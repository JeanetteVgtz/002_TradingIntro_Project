# opt.py
import optuna
import numpy as np
import pandas as pd
import json, os

from signals import craft_signals            # señales (RSI+MACD+BB)
from backtest import execute_backtest        # backtest (df_bt, capital) — usa 'comision'
from metrics import calculate_all_metrics    # métricas
from clean_data import df as data            # datos limpios



# Split 
def split_train_test(df, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    n = len(df)
    if n < 10:
        raise ValueError("DataFrame demasiado pequeño para split.")
    if train_ratio + test_ratio + val_ratio != 1.0:
        val_ratio = 1.0 - train_ratio - test_ratio
        if val_ratio <= 0:
            raise ValueError("Ratios inválidos: deben sumar 1.0.")

    i_tr_end = int(n * train_ratio)
    i_te_end = int(n * (train_ratio + test_ratio))
    df_tr = df.iloc[:i_tr_end].copy()
    df_te = df.iloc[i_tr_end:i_te_end].copy()
    df_va = df.iloc[i_te_end:].copy()
    return df_tr, df_te, df_va


# Objective (maximize Calmar)

def objective(trial, df):
    # --- Hyperparams ---
    rsi_period      = trial.suggest_int('rsi_period', 20, 76)
    rsi_overbought  = trial.suggest_int('rsi_overbought', 65, 80)
    rsi_oversold    = trial.suggest_int('rsi_oversold', 20, 26)

    macd_fast       = trial.suggest_int('macd_fast', 10, 14)
    macd_slow       = trial.suggest_int('macd_slow', 15, 25)
    macd_signal     = trial.suggest_int('macd_signal', 5, 7)

    bb_window       = trial.suggest_int('bb_window', 10, 25)
    bb_std          = trial.suggest_float('bb_std', 1.5, 3.0)

    n_shares        = trial.suggest_float('n_shares', 0.5, 4.0)
    stop_loss_pct   = trial.suggest_float('stop_loss_pct', 0.03, 0.06)
    take_profit_pct = trial.suggest_float('take_profit_pct', 0.04, 0.15)

    # Señales
    try:
        sig = craft_signals(
            df,
            price_col="close",
            rsi_len=rsi_period, rsi_hi=rsi_overbought, rsi_lo=rsi_oversold,
            macd_fast=macd_fast, macd_slow=macd_slow, macd_sig=macd_signal,
            bb_len=bb_window, bb_dev=bb_std
        )
    except Exception:
        return -1e6

    try:
        bt_df, _final_cap = execute_backtest(
            sig,
            stop_thr=stop_loss_pct,
            tp_thr=take_profit_pct,
            lot_size=n_shares,
            comision=0.125/100,      
            col_price="close",
            start_cap=1_000_000
        )
    except Exception:
        return -1e6

    # Métricas
    mets = calculate_all_metrics(bt_df, risk_free_rate=0.0, bars_per_year=24*365)
    calmar = mets.get("calmar_ratio", np.nan)

    # mínimo de trades cerrados (contar no-cero)
    if "trade_pnl" in bt_df.columns:
        closed = int((bt_df["trade_pnl"] != 0).sum())
    else:
        closed = 0

    if closed < 5 or calmar is None or np.isnan(calmar):
        return -1e6

    # info útil
    trial.set_user_attr("closed_trades", closed)
    trial.set_user_attr("total_return", mets.get("total_return"))
    trial.set_user_attr("sharpe_ratio", mets.get("sharpe_ratio"))
    trial.set_user_attr("max_drawdown", mets.get("max_drawdown"))
    trial.set_user_attr("win_rate", mets.get("win_rate"))

    return float(calmar)



# Evaluar con params ganadores

def evaluate_on_df(df, params):
    sig = craft_signals(
        df,
        price_col="close",
        rsi_len=params['rsi_period'],
        rsi_hi=params['rsi_overbought'],
        rsi_lo=params['rsi_oversold'],
        macd_fast=params['macd_fast'],
        macd_slow=params['macd_slow'],
        macd_sig=params['macd_signal'],
        bb_len=params['bb_window'],
        bb_dev=params['bb_std'],
    )
    bt_df, cash_end = execute_backtest(
        sig,
        stop_thr=params['stop_loss_pct'],
        tp_thr=params['take_profit_pct'],
        lot_size=params['n_shares'],
        comision=0.125/100,           # << alineado al backtest
        col_price="close",
        start_cap=1_000_000,
    )
    m = calculate_all_metrics(bt_df, risk_free_rate=0.0, bars_per_year=24*365)
    return bt_df, cash_end, m



# Resumen y entrenamiento

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)

    tr, te, va = split_train_test(data)
    target = tr  # optimizar en train

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, target), n_trials=200, show_progress_bar=True)

    print("\n=== OPTIMIZATION RESULTS (maximize Calmar) ===")
    print(f"Best Calmar: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")

    bt_test, cash_test, m_test = evaluate_on_df(te, study.best_params)
    print("\n=== TEST METRICS (holdout) ===")
    for k, v in m_test.items():
        try:
            if k in ("total_return", "max_drawdown", "win_rate"):
                print(f"{k:>14}: {float(v)*100:.2f}%")
            else:
                print(f"{k:>14}: {float(v):.4f}")
        except Exception:
            print(f"{k:>14}: {v}")
    print(f"Final portfolio (test): {cash_test:,.2f}")



# Guardar / cargar resumen

def save_optuna_summary(study, cash_test, metrics_test, path="outputs/optuna_summary.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safe_metrics = {}
    for k, v in metrics_test.items():
        try:
            safe_metrics[k] = float(v)
        except Exception:
            safe_metrics[k] = v
    payload = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "cash_test": float(cash_test),
        "metrics_test": safe_metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_optuna_summary(path="outputs/optuna_summary.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
