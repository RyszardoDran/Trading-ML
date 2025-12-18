"""
Analiza tradów z nowymi kolumnami: exit_time, duration, SL/TP
"""
import pandas as pd

trades = pd.read_csv('ml/outputs/backtests/trades_20251217_233406.csv')

print('=' * 80)
print('ANALIZA TRADÓW Z REALISTYCZNYM EXIT SIMULATION')
print('=' * 80)
print()

for i, t in trades.iterrows():
    print(f'TRADE #{i+1}')
    print('-' * 80)
    print(f'Entry Time:      {t.timestamp}')
    print(f'Entry Price:     ${t.entry_price:.2f}')
    print(f'Position Size:   {t.position_size} lots')
    print(f'Probability:     {t.probability*100:.2f}%')
    print()
    print(f'Exit Time:       {t.exit_time}')
    print(f'Exit Price:      ${t.exit_price:.2f}')
    print(f'Duration:        {t.duration_minutes:.0f} minutes')
    print()
    print(f'Stop Loss:       ${t.sl_price:.2f}')
    print(f'Take Profit:     ${t.tp_price:.2f}')
    print()
    
    result = 'TP HIT ✅' if t.is_win else 'SL HIT ❌'
    print(f'Result:          {result}')
    print(f'Gross P&L:       ${t.gross_pnl:.4f}')
    print(f'Transaction:     ${t.transaction_cost:.2f}')
    print(f'Net P&L:         ${t.pnl:.4f}')
    print()

print('=' * 80)
print('PODSUMOWANIE')
print('=' * 80)
wins = trades[trades['is_win'] == True]
losses = trades[trades['is_win'] == False]

print(f'Total Trades:    {len(trades)}')
print(f'Wins:            {len(wins)} ({len(wins)/len(trades)*100:.1f}%)')
print(f'Losses:          {len(losses)} ({len(losses)/len(trades)*100:.1f}%)')
print()
print(f'Avg Duration:    {trades["duration_minutes"].mean():.1f} minutes')
print(f'Total P&L:       ${trades["pnl"].sum():.2f}')
print(f'Avg P&L/trade:   ${trades["pnl"].mean():.2f}')
