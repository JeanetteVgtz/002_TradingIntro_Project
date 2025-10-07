import pandas as pd

# Leer el archivo saltando la primera fila basura
df = pd.read_csv("data/Binance_BTCUSDT_1h.csv", skiprows=1)

# Nos quedamos solo con las columnas que ocupamos
df = df[["Date","Open","High","Low","Close","Volume BTC"]]

# Renombrar columnas
df = df.rename(columns={
    "Date": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume BTC": "volume"
})

# Convertir a datetime con el formato correcto
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

# Eliminar filas inválidas (si alguna no se pudo convertir)
df = df.dropna(subset=["timestamp"])

# Guardar ASC
df.sort_values("timestamp", ascending=True).to_csv("data/BTCUSDT_hourly_ASC.csv", index=False)

print("Filas:", len(df))
print("Rango de fechas:", df['timestamp'].min(), "→", df['timestamp'].max())

asc = pd.read_csv("data/BTCUSDT_hourly_ASC.csv")

print("DATA")
print(asc.head(5))
print("...")
print(asc.tail(5))

