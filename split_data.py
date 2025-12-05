#!/usr/bin/env python3
"""Split XAU_1m_data.csv by year"""

import os
from pathlib import Path

# Define paths
data_dir = r"c:\Users\Arek\Documents\Repos\Traiding\TradnigML-1\data"
input_file = os.path.join(data_dir, "XAU_1m_data.csv")

# Check if file exists
if not os.path.exists(input_file):
    print(f"Error: File not found: {input_file}")
    exit(1)

print(f"Reading file: {input_file}")

# Read the file
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Extract header
header = lines[0]
print(f"Header: {header.strip()}")

# Group by year
year_data = {}

for i, line in enumerate(lines[1:], start=2):
    if i % 10000 == 0:
        print(f"Processing line {i}...")
    
    # Extract year from date (format: YYYY.MM.DD HH:MM)
    date_str = line.split(';')[0]
    year = date_str.split('.')[0]
    
    if year not in year_data:
        year_data[year] = []
    
    year_data[year].append(line)

print(f"\nYear distribution:")
for year in sorted(year_data.keys()):
    print(f"  {year}: {len(year_data[year])} rows")

# Save files for each year
print(f"\nCreating output files...")
for year in sorted(year_data.keys()):
    output_file = os.path.join(data_dir, f"XAU_1m_data_{year}.csv")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(year_data[year])
    
    print(f"✓ Created: XAU_1m_data_{year}.csv ({len(year_data[year])} rows)")

print(f"\n✓ Done! Created {len(year_data)} files")
