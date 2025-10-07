# plotting.py
from __future__ import annotations
import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------
def _asegura_dir(path: Optional[str]) -> None:
    if path:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

def _save_fig(save_path: Optional[str]) -> None:
    """
    Guarda la figura (si hay save_path) y muestra la ruta absoluta en consola.
    Luego intenta mostrar y cierra la figura para evitar fugas.
    """
    if save_path:
        out = os.path.abspath(save_path)
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(out, dpi=140, bbox_inches="tight")
        print(f"[PLOT] saved -> {out}")
    else:
        print("[PLOT] no save_path provided; not saving.")
    try:
        plt.show(block=False)
    except TypeError:
        plt.show()
    plt.close()

def _serie_equity(x: pd.DataFrame | pd.Series) -> pd.Series:
    """Devuelve una Serie numérica con el equity (soporta DataFrame con 'portfolio_value')."""
    if isinstance(x, pd.DataFrame):
        if "portfolio_value" not in x.columns:
            raise KeyError("No encuentro 'portfolio_value' en el DataFrame.")
        s = x["portfolio_value"]
    else:
        s = x
    return pd.to_numeric(s, errors="coerce").dropna()

# ---------------------------------------------------------------------
# 1) Estrategia vs Benchmark (buy & hold)
# ---------------------------------------------------------------------
def plot_portfolio_vs_benchmark(
    portfolio_history: pd.DataFrame | pd.Series,
    df: pd.DataFrame,
    benchmark_col: str = "close",
    normalize: bool = True,
    title: str = "Estrategia vs Buy & Hold",
    save_path: Optional[str] = None,
) -> None:
    equity = _serie_equity(portfolio_history)
    if benchmark_col not in df.columns:
        raise KeyError(f"No encuentro columna '{benchmark_col}' en df.")
    bench = pd.to_numeric(df[benchmark_col], errors="coerce").dropna()
    # Alinear por índice
    bench = bench.reindex(equity.index, method="nearest").dropna()
    equity = equity.reindex(bench.index)

    if normalize:
        equity_plot = equity / float(equity.iloc[0])
        bench_plot  = bench / float(bench.iloc[0])
        ylbl = "Valor normalizado"
    else:
        equity_plot, bench_plot = equity, bench
        ylbl = "Valor"

    plt.figure(figsize=(11, 5))
    plt.plot(equity_plot.index, equity_plot.values, label="Estrategia", linewidth=1.5)
    plt.plot(bench_plot.index, bench_plot.values, label="Buy & Hold", alpha=0.9)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel(ylbl)
    plt.legend()
    plt.grid(True, alpha=0.25)

    _save_fig(save_path)

# ---------------------------------------------------------------------
# 2) Curva de drawdown
# ---------------------------------------------------------------------
def plot_drawdown(
    equity: pd.DataFrame | pd.Series,
    title: str = "Drawdown de la estrategia",
    save_path: Optional[str] = None,
) -> None:
    eq = _serie_equity(equity)
    pico = eq.cummax()
    dd = eq / pico - 1.0

    plt.figure(figsize=(11, 3.5))
    plt.fill_between(dd.index, dd.values, 0, step="pre", alpha=0.35)
    plt.plot(dd.index, dd.values, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.25)

    _save_fig(save_path)

# ---------------------------------------------------------------------
# 3) Precio con señales (flechas de compra/venta)
# ---------------------------------------------------------------------
def plot_price_with_signals(
    df: pd.DataFrame,
    price_col: str = "close",
    signal_col: str = "signal",
    title: str = "Precio con señales",
    save_path: Optional[str] = None,
) -> None:
    if price_col not in df.columns or signal_col not in df.columns:
        raise KeyError(f"Faltan columnas '{price_col}' o '{signal_col}'.")
    px = pd.to_numeric(df[price_col], errors="coerce")
    sig = pd.to_numeric(df[signal_col], errors="coerce").fillna(0).astype(int)

    # Solo marcamos cambios hacia 1 o -1 para evitar saturación
    prev = sig.shift(1, fill_value=0)
    buys_idx  = sig.eq(1)  & prev.ne(1)
    sells_idx = sig.eq(-1) & prev.ne(-1)

    plt.figure(figsize=(11, 5))
    plt.plot(px.index, px.values, label="Precio", linewidth=1.2)
    plt.scatter(px.index[buys_idx],  px[buys_idx],  marker="^", s=40, label="Buy",  zorder=3)
    plt.scatter(px.index[sells_idx], px[sells_idx], marker="v", s=40, label="Sell", zorder=3)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True, alpha=0.25)

    _save_fig(save_path)


# ---------------------------------------------------------------------
# 5) Histograma de retornos por barra
# ---------------------------------------------------------------------
def plot_returns_hist(
    bt_df: pd.DataFrame,
    bins: int = 50,
    title: str = "Distribución de retornos (barra a barra)",
    save_path: Optional[str] = None,
) -> None:
    if "portfolio_value" not in bt_df.columns:
        raise KeyError("Falta 'portfolio_value' en el DataFrame del backtest.")
    r = pd.to_numeric(bt_df["portfolio_value"], errors="coerce").pct_change().dropna()

    plt.figure(figsize=(8, 4))
    plt.hist(r.values, bins=bins, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel("Retorno")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.25)

    _save_fig(save_path)

