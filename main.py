"""
Main pipeline orchestrator - BTC strategy end-to-end
- Carga datos limpios
- Genera señales (RSI + MACD + Bollinger, 2/3)
- Ejecuta backtest con parametros optimizados
- Calcula métricas
- Genera gráficas
"""

from __future__ import annotations
import pandas as pd

# --- data / core ---
from clean_data import df 
from signals import craft_signals
from backtest import execute_backtest
from metrics import calculate_all_metrics

# --- plotting ---
from plotting import (
    plot_portfolio_vs_benchmark,
    plot_drawdown,
    plot_price_with_signals,
    plot_returns_hist,
)


def _print_metrics(title: str, metrics: dict):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        try:
            if k in ("total_return", "max_drawdown", "win_rate"):
                print(f"{k:>14}: {float(v)*100:.2f}%")
            else:
                print(f"{k:>14}: {float(v):.4f}")
        except Exception:
            print(f"{k:>14}: {v}")


def main():

    # =========================
    #  SIGNALS
    # =========================
    df_sig = craft_signals(df)
    print("[SIGNALS] columnas añadidas:", [c for c in ["rsi", "macd_line", "macd_signal", "bb_lower", "bb_upper", "signal"] if c in df_sig.columns])

    # =========================
    # BACKTEST (OPTIMIZADO)
    # =========================
    bt_df, capital_final = execute_backtest(
        df_sig,
        stop_thr=0.04414069466687109,      
        tp_thr=0.13703959474772304,       
        lot_size=3,    # n_shares   
        comision=0.125/100, 
        col_price="close",
        start_cap=1_000_000,
    )

    print(f"[OPTIMIZADO] Capital final: {capital_final:,.2f}")

    # =========================
    # 4) MÉTRICAS (base)
    # =========================
    mets = calculate_all_metrics(bt_df, risk_free_rate=0.0, bars_per_year=24*365)
    _print_metrics("PERFORMANCE (base)", mets)

    # =========================
    # 5) PLOTS (base)
    # =========================
    plot_portfolio_vs_benchmark(
    portfolio_history=bt_df,
    df=df,                    
    benchmark_col="close",
    normalize=True,
    title="Estrategia (Optimizada) vs Buy & Hold (BTC/USDT)",
    save_path="outputs/opt_equity_vs_benchmark.png",
)

    plot_drawdown(
        equity=bt_df,
        title="Drawdown de la Estrategia (BASE)",
        save_path="outputs/base_drawdown.png",
    )

    plot_price_with_signals(
        df=df_sig,
        price_col="close",
        signal_col="signal",
        title="BTC Precio + Señales (BASE)",
        save_path="outputs/base_price_signals.png",
    )


    plot_returns_hist(
        bt_df=bt_df,
        bins=60,
        title="Distribución de Retornos (BASE)",
        save_path="outputs/base_returns_hist.png",
    )


if __name__ == "__main__":
    main()
