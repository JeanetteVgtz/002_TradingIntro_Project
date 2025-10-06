import pandas as pd

asc = pd.read_csv("data/BTCUSDT_hourly_ASC.csv")
desc = pd.read_csv("data/BTCUSDT_hourly.csv")

print("=== ASC (viejo→nuevo) ===")
print(asc.head(3))
print("...")
print(asc.tail(3))

print("\n=== DESC (nuevo→viejo) ===")
print(desc.head(3))
print("...")
print(desc.tail(3))

