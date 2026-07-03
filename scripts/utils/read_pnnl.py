import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel(r'PNNL_Prototype_Scorecards_Medium Office_HO CHI MINH CITY.xlsx', 
                   header=None, sheet_name='HO CHI MINH CITY')

print("=== Rows 80-112 ===")
for i in range(80, min(112, len(df))):
    row_data = []
    for j in range(min(8, df.shape[1])):
        val = df.iloc[i, j]
        if pd.notna(val):
            row_data.append(f"[{j}]{str(val)[:60]}")
    if row_data:
        print(f"  Row {i}: {' | '.join(row_data)}")
