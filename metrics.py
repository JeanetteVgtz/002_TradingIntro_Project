# metrics.py
# Cálculo de métricas de desempeño para una curva de portafolio
from __future__ import annotations
import numpy as np
import pandas as pd

# Datos horarios 24/7 (BTC)
HOURS_PER_YEAR = 365 * 24


# Utilidades internas

def _as_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    s = x if isinstance(x, pd.Series) else x.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return s

def _bar_returns(equity_df: pd.DataFrame) -> pd.Series:
    if "portfolio_value" not in equity_df.columns:
        raise KeyError("Falta la columna 'portfolio_value'.")
    return equity_df["portfolio_value"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()

def _gross_return(equity_df: pd.DataFrame) -> float:
    eq = _as_series(equity_df["portfolio_value"])
    if len(eq) < 2:
        return np.nan
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0)


# Metricas individuales

def drawdown_stats(equity: pd.Series) -> tuple[float, pd.Series]:
    """Devuelve (mdd_negativo, serie_drawdown)."""
    eq = _as_series(equity)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan, dd

def annual_growth(equity: pd.Series, bars_per_year: int) -> float:
    eq = _as_series(equity)
    if len(eq) < 2:
        return np.nan
    years = len(eq) / float(bars_per_year)
    if years <= 0:
        return np.nan
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    return (1.0 + total) ** (1.0 / years) - 1.0

def ratio_sharpe(ret: pd.Series, bars_per_year: int, rf_annual: float = 0.0) -> float:
    r = pd.to_numeric(ret, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return np.nan
    # rf anual a por-barra (aprox lineal para simplicidad)
    rf_bar = rf_annual / float(bars_per_year)
    excess = r - rf_bar
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return np.sqrt(bars_per_year) * excess.mean() / vol

def ratio_sortino(ret: pd.Series, bars_per_year: int, rf_annual: float = 0.0) -> float:
    r = pd.to_numeric(ret, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return np.nan
    rf_bar = rf_annual / float(bars_per_year)
    excess = r - rf_bar
    downside = excess[excess < 0]
    dvol = downside.std(ddof=0)
    if dvol == 0 or np.isnan(dvol) or len(downside) == 0:
        return np.nan
    return np.sqrt(bars_per_year) * excess.mean() / dvol

def ratio_calmar(equity: pd.Series, bars_per_year: int) -> float:
    cg = annual_growth(equity, bars_per_year)
    mdd, _ = drawdown_stats(equity)
    denom = abs(mdd)
    if denom == 0 or np.isnan(denom) or np.isnan(cg):
        return np.nan
    return cg / denom

def hit_rate(port_hist: pd.DataFrame) -> float:
    """% de trades ganadores usando la columna 'trade_pnl'."""
    if "trade_pnl" not in port_hist.columns:
        return np.nan
    pnl = pd.to_numeric(port_hist["trade_pnl"], errors="coerce").dropna()
    if pnl.empty:
        return np.nan
    w = int((pnl > 0).sum())
    l = int((pnl < 0).sum())
    tot = w + l
    return (w / tot) if tot > 0 else np.nan


# Métricas completas

def calculate_all_metrics(
    portfolio_hist: pd.DataFrame,
    risk_free_rate: float = 0.0,
    bars_per_year: int = HOURS_PER_YEAR
) -> dict:
    """
    Calcula: total_return, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, win_rate
    a partir de 'portfolio_value' (y 'trade_pnl' para win_rate).
    """
    ret = _bar_returns(portfolio_hist)
    total_ret = _gross_return(portfolio_hist)
    mdd, _dd_series = drawdown_stats(portfolio_hist["portfolio_value"])

    sh = ratio_sharpe(ret, bars_per_year, rf_annual=risk_free_rate)
    so = ratio_sortino(ret, bars_per_year, rf_annual=risk_free_rate)
    cal = ratio_calmar(portfolio_hist["portfolio_value"], bars_per_year)
    wr = hit_rate(portfolio_hist)

    return {
        "total_return": total_ret,
        "sharpe_ratio": sh,
        "sortino_ratio": so,
        "max_drawdown": mdd,
        "calmar_ratio": cal,
        "win_rate": wr,
    }