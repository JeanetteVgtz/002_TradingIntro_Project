# backtest.py
import pandas as pd
from dataclasses import dataclass
from clean_data import df

@dataclass
class Trade:
    side: str          # "long" o "short"
    qty: float         # n_shares fijos
    entry: float       # precio de entrada
    sl: float          # stop loss (precio)
    tp: float          # take profit (precio)

def execute_backtest(
    data: pd.DataFrame,
    stop_thr: float = 0.02,          # 2% SL
    tp_thr: float = 0.04,            # 4% TP
    lot_size: float = 1.0,           # n_shares fijos
    comision: float = 0.125 / 100,   # 0.125%
    col_price: str = "close",
    start_cap: float = 1_000_000,
):
    """
    Backtest según el esquema solicitado:
    - CASH inicial
    - Por barra: revisar señal; abrir LONG/SHORT si hay efectivo suficiente
    - Cerrar posiciones por SL/TP
    - Aplicar comisión al abrir y cerrar
    - Valuar portafolio tras los cierres: cash + valor longs + valor shorts
    Retorna (DataFrame con 'portfolio_value' y 'trade_pnl', capital_final).
    Requiere columna 'signal' (1=buy, 0=hold, -1=sell) y el precio en `col_price`.
    """
    df = data.copy()

    cash = float(start_cap)
    active_long: list[Trade] = []
    active_short: list[Trade] = []
    portfolio_values: list[float] = []
    trade_pnls: list[float] = []

    for _, row in df.iterrows():
        price = float(row[col_price])
        signal = int(row["signal"])  

        pnl_this_step = 0.0
        closed_any = False

        # =========================
        # CERRAR POSICIONES (SL / TP)
        # =========================
        # LONGS
        for pos in active_long.copy():
            if price >= pos.tp or price <= pos.sl:
                # PnL (solo informativo): incluye comisiones de entrada/salida
                entry_fee = pos.entry * pos.qty * comision
                exit_fee  = price * pos.qty * comision
                pnl_realized = (price - pos.entry) * pos.qty - entry_fee - exit_fee

                # Flujo de caja al cerrar long: venta neta de comisión
                cash += price * pos.qty * (1 - comision)
                active_long.remove(pos)

                pnl_this_step += pnl_realized
                closed_any = True

        # SHORTS
        for pos in active_short.copy():
            if price <= pos.tp or price >= pos.sl:
                # PnL bruto de un short
                pnl_gross = (pos.entry - price) * pos.qty
                entry_fee = pos.entry * pos.qty * comision
                exit_fee  = price * pos.qty * comision
                pnl_realized = pnl_gross - entry_fee - exit_fee

                # Flujo de caja al cerrar short (según el esquema dado)
                cash += (pnl_gross * (1 - comision)) + (pos.entry * pos.qty)
                active_short.remove(pos)

                pnl_this_step += pnl_realized
                closed_any = True

        # =========================
        # ABRIR OPERACIONES
        # =========================
        # LONG
        if signal == 1:
            cost = price * lot_size * (1 + comision)
            if cash > cost:
                cash -= cost
                active_long.append(
                    Trade(
                        side="long",
                        qty=float(lot_size),
                        entry=price,
                        sl=price * (1 - stop_thr),
                        tp=price * (1 + tp_thr),
                    )
                )

        # SHORT
        if signal == -1:
            cost = price * lot_size * (1 + comision)
            if cash > cost:
                cash -= cost
                active_short.append(
                    Trade(
                        side="short",
                        qty=float(lot_size),
                        entry=price,
                        sl=price * (1 + stop_thr),   # SL arriba
                        tp=price * (1 - tp_thr),     # TP abajo
                    )
                )

        # =========================
        # VALUACIÓN DEL PORTAFOLIO
        # =========================
        val = cash
        # valor de longs abiertos al precio actual
        for pos in active_long:
            val += pos.qty * price
        # valor de shorts abiertos (marcado)
        for pos in active_short:
            val += (pos.entry - price) * pos.qty + (pos.entry * pos.qty)

        portfolio_values.append(val)
        trade_pnls.append(pnl_this_step if closed_any else 0.0)

    # =========================
    # CIERRE FORZADO FINAL
    # =========================
    if len(df) > 0:
        last_price = float(df.iloc[-1][col_price])

        # cerrar longs restantes: venta neta comisión
        if active_long:
            total_qty = sum(p.qty for p in active_long)
            cash += last_price * total_qty * (1 - comision)
            active_long.clear()

        # cerrar shorts restantes: pnl + devolver entrada
        if active_short:
            for p in active_short:
                pnl_gross = (p.entry - last_price) * p.qty
                cash += (pnl_gross * (1 - comision)) + (p.entry * p.qty)
            active_short.clear()

        # la última valuación refleja el cash final
        portfolio_values[-1] = cash

    df["portfolio_value"] = portfolio_values
    df["trade_pnl"] = trade_pnls

    return df, cash


