import json
from statistics import mean, median
from pathlib import Path
p = Path(__file__).resolve().parents[3] / 'backend' / 'outputs' / 'prediction_2025.json'
with p.open('r', encoding='utf-8') as f:
    j = json.load(f)
sig = j['Signals']
win = [float(s['Probability'].replace(',','.').replace('%',''))/100.0 for s in sig if s['Outcome']=='Win']
loss = [float(s['Probability'].replace(',','.').replace('%',''))/100.0 for s in sig if s['Outcome']=='Loss']
pending = [float(s['Probability'].replace(',','.').replace('%',''))/100.0 for s in sig if s['Outcome']=='Pending']
print('Counts: total',len(sig),'win',len(win),'loss',len(loss),'pending',len(pending))
print('Avg prob win',mean(win) if win else None,'avg prob loss',mean(loss) if loss else None)
print('Median prob win',median(win) if win else None,'median prob loss',median(loss) if loss else None)
high_loss = [s for s in sig if s['Outcome']=='Loss' and float(s['Probability'].replace(',','.').replace('%',''))>70]
print('High-prob losses count >70%:',len(high_loss))
for s in high_loss[:5]: print(s['EntryTime'],s['Probability'],s['ProfitLoss'])
# ATR analysis
import statistics
loss_atr = [float(s['AtrM5'].replace(',','.')) for s in sig if s['Outcome']=='Loss' and s.get('AtrM5')]
win_atr = [float(s['AtrM5'].replace(',','.')) for s in sig if s['Outcome']=='Win' and s.get('AtrM5')]
print('Avg ATR loss',statistics.mean(loss_atr) if loss_atr else None,'avg ATR win',statistics.mean(win_atr) if win_atr else None)
# Time of day analysis (hour)
from collections import Counter
ct = Counter([int(s['EntryTime'][11:13]) for s in sig])
print('Top hours:',ct.most_common(5))
