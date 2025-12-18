#!/usr/bin/env python3
"""Analyze backtest trades with SL/TP calculations."""

import pandas as pd
import numpy as np

# Wczytaj trades
trades = pd.read_csv('ml/outputs/backtests/trades_20251217_232318.csv')
trades['timestamp'] = pd.to_datetime(trades['timestamp'])

# Parametry z modelu (ATR 1.0/1.8)
ATR_SL_MULTIPLIER = 1.0
ATR_TP_MULTIPLIER = 1.8

print('=' * 80)
print('SZCZEGÓŁOWA ANALIZA TRADÓW - MODEL ATR 1.0/1.8')
print('=' * 80)
print()

for idx, trade in trades.iterrows():
    print(f'TRADE #{idx + 1}')
    print('-' * 80)
    
    # Podstawowe info
    entry_price = trade['entry_price']
    position_size = trade['position_size']
    gross_pnl = trade['gross_pnl']
    probability = trade['probability']
    
    print(f'Czas wejścia:        {trade["timestamp"]}')
    print(f'Cena wejścia:        ${entry_price:.2f}')
    print(f'Position size:       {position_size} lots')
    print(f'Prawdopodobieństwo:  {probability*100:.2f}%')
    print()
    
    # Oblicz ATR na podstawie gross P&L
    # gross_pnl = position_size * price_move
    # price_move = TP = ATR * 1.8
    # ATR = gross_pnl / (position_size * 1.8)
    estimated_atr = gross_pnl / (position_size * ATR_TP_MULTIPLIER)
    
    # Oblicz SL i TP
    sl_distance = ATR_SL_MULTIPLIER * estimated_atr
    tp_distance = ATR_TP_MULTIPLIER * estimated_atr
    
    sl_price = entry_price - sl_distance
    tp_price = entry_price + tp_distance
    
    print(f'Szacowany ATR:       ${estimated_atr:.4f}')
    print(f'Stop Loss:           ${sl_price:.2f} (dystans: ${sl_distance:.4f}, {sl_distance/entry_price*100:.2f}%)')
    print(f'Take Profit:         ${tp_price:.2f} (dystans: ${tp_distance:.4f}, {tp_distance/entry_price*100:.2f}%)')
    print(f'Risk/Reward:         1:{ATR_TP_MULTIPLIER/ATR_SL_MULTIPLIER:.1f}')
    print()
    
    # Kapitał zainwestowany
    invested = position_size * entry_price
    print(f'Kapitał wejścia:     ${invested:.2f}')
    print()
    
    # P&L
    print(f'Gross P&L:           ${gross_pnl:.4f}')
    print(f'Transaction cost:    ${trade["transaction_cost"]:.2f}')
    print(f'Net P&L:             ${trade["pnl"]:.4f}')
    print(f'ROI (net):           {trade["pnl"]/invested*100:.2f}%')
    print()
    
    # Czas trwania (max_horizon = 60 minut według parametrów)
    print(f'Max czas trade:      60 minut (max_horizon)')
    print(f'Rzeczywisty czas:    Nieznany (brak danych exit time w backtecie)')
    print(f'Status:              {"WIN ✅" if trade["is_win"] else "LOSS ❌"}')
    print()
    print()

# Podsumowanie
print('=' * 80)
print('PODSUMOWANIE')
print('=' * 80)
avg_atr = trades['gross_pnl'].mean() / (trades['position_size'].mean() * ATR_TP_MULTIPLIER)
print(f'Średni ATR:          ${avg_atr:.4f}')
print(f'Średni SL dystans:   ${avg_atr * ATR_SL_MULTIPLIER:.4f} ({avg_atr * ATR_SL_MULTIPLIER / trades["entry_price"].mean() * 100:.2f}%)')
print(f'Średni TP dystans:   ${avg_atr * ATR_TP_MULTIPLIER:.4f} ({avg_atr * ATR_TP_MULTIPLIER / trades["entry_price"].mean() * 100:.2f}%)')
print(f'Średni kapitał:      ${(trades["position_size"] * trades["entry_price"]).mean():.2f}')
print(f'Średni Net P&L:      ${trades["pnl"].mean():.4f}')
print(f'Średni ROI:          {(trades["pnl"] / (trades["position_size"] * trades["entry_price"])).mean() * 100:.2f}%')
