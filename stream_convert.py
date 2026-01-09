#!/usr/bin/env python3
"""Stream-based converter dla dużych plików."""
from datetime import datetime

input_file = 'ml/src/data/XAUUSD_M1 (2).csv'
output_file = 'ml/src/data/XAU_1m_data_2026_additional.csv'

rows_written = 0

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Write header
    outfile.write('Date;Open;High;Low;Close;Volume\n')
    
    # Skip input header
    next(infile)
    
    # Stream lines
    for line in infile:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 6:
            continue
        
        try:
            time_str = parts[0]  # '2025-09-25 15:51:00'
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            date_formatted = dt.strftime('%Y.%m.%d %H:%M')
            
            open_val = float(parts[1])
            high_val = float(parts[2])
            low_val = float(parts[3])
            close_val = float(parts[4])
            volume_val = int(float(parts[5]))
            
            outfile.write(f'{date_formatted};{open_val:.6f};{high_val:.6f};{low_val:.6f};{close_val:.6f};{volume_val}\n')
            rows_written += 1
        except Exception as e:
            print(f"Error on line: {line[:50]} - {e}")

print(f"✓ Konwertowano {rows_written} wierszy do {output_file}")
