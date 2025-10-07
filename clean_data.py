import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load Bitcoin hourly price data from CSV.
    - Skips first junk row (Binance format).
    - Keeps only OHLCV columns.
    - Converts timestamp to datetime.
    - Drops NaNs.
    - Sorts by date ascending.
    - Saves clean data to 'data/BTCUSDT_hourly_ASC.csv'.
    """
    # Leer el CSV, saltando la primera fila basura
    df = pd.read_csv(filepath, skiprows=1)

    # Normalizar encabezados
    df.columns = [c.strip() for c in df.columns]

    # Detectar columna de volumen (BTC o USDT)
    volume_col = None
    for cand in ["Volume USDT", "Volume BTC", "Volume"]:
        if cand in df.columns:
            volume_col = cand
            break

    if volume_col is None:
        raise ValueError(f"No encontré columna de volumen en {df.columns}")

    # Seleccionar y renombrar columnas
    df = df[["Date", "Open", "High", "Low", "Close", volume_col]].rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        volume_col: "volume"
    })

    # Convertir a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Eliminar filas con NaN
    df = df.dropna()

    # Ordenar ascendente por fecha
    df = df.sort_values("timestamp")

    # Usar timestamp como índice
    df = df.set_index("timestamp")

    return df

# Ejemplo de uso:
df = load_data("data/Binance_BTCUSDT_1h.csv")
data = df
print(data.head())