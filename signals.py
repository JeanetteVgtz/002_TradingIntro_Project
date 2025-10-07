# signals.py
import pandas as pd
import numpy as np
from clean_data import df as data
import ta


def craft_signals(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    # RSI
    rsi_len: int = 10,
    rsi_hi: float = 76,
    rsi_lo: float = 26,
    # MACD
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig: int = 9,
    # Bollinger
    bb_len: int = 25,
    bb_dev: float = 2.032004,
) -> pd.DataFrame:
    """
    Construye señales discretas (-1, 0, 1) a partir de RSI, MACD y Bandas de Bollinger
    aplicando confirmación por mayoría (2 de 3).

    Parámetros
    ----------
    df : pd.DataFrame
        Debe contener al menos la columna `price_col` (por defecto 'close').
    price_col : str
        Nombre de la columna de precios a usar.
    rsi_len, rsi_hi, rsi_lo : int/float
        Parámetros del RSI (periodo y zonas sobrecompra/sobreventa).
    macd_fast, macd_slow, macd_sig : int
        Parámetros del MACD (rápida, lenta y línea de señal).
    bb_len, bb_dev : int/float
        Parámetros de Bandas de Bollinger (ventana y desviaciones estándar).

    Reglas de voto
    --------------
    - RSI:    rsi < rsi_lo  -> +1 ;  rsi > rsi_hi -> -1 ; en otro caso 0
    - MACD:   macd_line > macd_signal -> +1 ; macd_line < macd_signal -> -1 ; si iguales 0
    - BBands: price < bb_lower -> +1 ; price > bb_upper -> -1 ; en otro caso 0

    Señal final
    -----------
    signal = 1  si (votos >= +2)
    signal = -1 si (votos <= -2)
    signal = 0  en otro caso

    Returns
    -------
    pd.DataFrame
        Misma tabla con columnas adicionales:
        ['rsi_val', 'macd_val', 'macd_sig', 'bb_mid', 'bb_hi', 'bb_lo',
         'vote_rsi', 'vote_macd', 'vote_bb', 'signal']
    """
    if price_col not in df.columns:
        raise ValueError(f"No encuentro la columna de precio '{price_col}' en el DataFrame.")

    out = df.copy()

    # ======================
    # Indicador: RSI
    # ======================
    rsi_obj = ta.momentum.RSIIndicator(close=out[price_col], window=rsi_len)
    out["rsi_val"] = rsi_obj.rsi()

    out["vote_rsi"] = 0
    out.loc[out["rsi_val"] < rsi_lo, "vote_rsi"]  = 1
    out.loc[out["rsi_val"] > rsi_hi, "vote_rsi"]  = -1

    # ======================
    # Indicador: MACD
    # ======================
    macd_obj = ta.trend.MACD(
        close=out[price_col],
        window_fast=macd_fast,
        window_slow=macd_slow,
        window_sign=macd_sig,
    )
    out["macd_val"] = macd_obj.macd()
    out["macd_sig"] = macd_obj.macd_signal()

    out["vote_macd"] = 0
    out.loc[out["macd_val"] > out["macd_sig"], "vote_macd"] = 1
    out.loc[out["macd_val"] < out["macd_sig"], "vote_macd"] = -1

    # ======================
    # Indicador: Bollinger Bands
    # ======================
    bb = ta.volatility.BollingerBands(close=out[price_col], window=bb_len, window_dev=bb_dev)
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_hi"]  = bb.bollinger_hband()
    out["bb_lo"]  = bb.bollinger_lband()

    out["vote_bb"] = 0
    out.loc[out[price_col] < out["bb_lo"],  "vote_bb"] = 1
    out.loc[out[price_col] > out["bb_hi"],  "vote_bb"] = -1

    # ======================
    # 2-de-3
    # ======================
    votes = out["vote_rsi"] + out["vote_macd"] + out["vote_bb"]

    out["signal"] = 0
    out.loc[votes >= 2,  "signal"] = 1
    out.loc[votes <= -2, "signal"] = -1

    # Opcional: asegurar tipos enteros en señales
    out["vote_rsi"]  = out["vote_rsi"].astype(int)
    out["vote_macd"] = out["vote_macd"].astype(int)
    out["vote_bb"]   = out["vote_bb"].astype(int)
    out["signal"]    = out["signal"].astype(int)

    return out


# =========================
# Helpers de conteo de señales
# =========================
def signal_summary(df: pd.DataFrame, signal_col: str = "signal") -> dict:
    """
    Devuelve:
      - 'totals': nº de barras con señal short/hold/long
      - 'entries': nº de ENTRADAS (transiciones a 1 o a -1)
    """
    s = pd.to_numeric(df[signal_col], errors="coerce").fillna(0).astype(int)

    # barras con cada estado
    totals = s.value_counts().reindex([-1, 0, 1], fill_value=0).to_dict()

    # entradas (evita contar repeticiones consecutivas)
    prev = s.shift(1, fill_value=0)
    long_entries  = int(((s == 1)  & (prev != 1)).sum())
    short_entries = int(((s == -1) & (prev != -1)).sum())

    return {
        "totals": {
            "short": int(totals.get(-1, 0)),
            "hold":  int(totals.get(0, 0)),
            "long":  int(totals.get(1, 0)),
        },
        "entries": {
            "long": long_entries,
            "short": short_entries,
        },
    }

def print_signal_summary(df: pd.DataFrame, signal_col: str = "signal") -> None:
    info = signal_summary(df, signal_col)
    t, e = info["totals"], info["entries"]
    print(f"Señales (barras) -> short: {t['short']}, hold: {t['hold']}, long: {t['long']}")
    print(f"Entradas         -> long: {e['long']}, short: {e['short']}")

# =========================
# 
# =========================
if __name__ == "__main__":
    # Usa el df que ya importaste arriba: `from clean_data import df as data`
    sig = craft_signals(data)
    print_signal_summary(sig)
