"""
=====================================================================
 aggregate_lhs_results.py — Systematized jEPlus LHS result aggregator
 ICAE2026 pipeline | Framework step: SIMULATION & CALIBRATION -> PART 1
=====================================================================
 Walks all 9 climate-scenario folders under data/jEPlus-LHS/
 (1_Baseline = TMYx 2011-2025; 2..9 = CMIP6 ACCESS-CM2 / MRI-ESM2-0
  x SSP245/585 x 2050s/2080s), each holding 250 LHS-* job folders,
 parses every eplustbl.csv, joins with the jEPlus parameter table
 (AllCombinedResults.csv), and writes ONE tidy CSV consumed by the
 notebook (Section: Load LHS Sampling Data):

     data/aggregated_LHS_results.csv   (expected 9 x 250 = 2,250 rows)

 Output columns are backward-compatible with the notebook loader
 (@@P*@@ params, Scenario, EUI_MJ_m2) plus end-use breakdowns needed
 for the paper's figures.

 Usage (from the CODE/ directory):
     python scripts/data/aggregate_lhs_results.py
=====================================================================
"""

import sys
import os
import argparse
import pandas as pd

# --- Configuration ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LHS_ROOT = os.path.join(BASE_DIR, 'data', 'jEPlus-LHS')
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'aggregated_LHS_results.csv')

# All 9 climate scenarios (folder name -> mean warming dT vs baseline, degC).
# dT values follow the notebook's climate_delta_map (CMIP6 delta-morphing,
# Tran-Anh et al. 2023; registry row B4).
SCENARIOS = {
    '1_Baseline': 0.0,
    '2': 1.879,   # ACCESS-CM2  SSP245 2050s
    '3': 2.665,   # ACCESS-CM2  SSP245 2080s
    '4': 2.179,   # ACCESS-CM2  SSP585 2050s
    '5': 4.472,   # ACCESS-CM2  SSP585 2080s
    '6': 1.270,   # MRI-ESM2-0  SSP245 2050s
    '7': 1.875,   # MRI-ESM2-0  SSP245 2080s
    '8': 1.611,   # MRI-ESM2-0  SSP585 2050s
    '9': 3.144,   # MRI-ESM2-0  SSP585 2080s
}

# (table row label, electricity col index, output column name)
# eplustbl 'End Uses' rows: col2 = Electricity [GJ], col3 = Natural Gas [GJ]
END_USE_ROWS = {
    'Cooling':            'Cooling_Elec_GJ',
    'Interior Lighting':  'IntLighting_Elec_GJ',
    'Interior Equipment': 'IntEquip_Elec_GJ',
    'Fans':               'Fans_Elec_GJ',
    'Pumps':              'Pumps_Elec_GJ',
}


def parse_eplustbl(path):
    """Extract site energy, EUI and electricity end uses from one eplustbl.csv.

    Only the first 'End Uses' report table is read (the later
    'End Uses By Subcategory' table repeats the same row labels).
    """
    out = {}
    in_end_uses_done = False
    current = None
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                s = line.strip()
                if s == 'End Uses' and not in_end_uses_done:
                    current = 'EndUses'
                    continue
                if s.startswith('End Uses By Subcategory'):
                    current = None
                    in_end_uses_done = True
                    continue

                if s.startswith(',Total Site Energy,'):
                    p = s.split(',')
                    out['Total_Site_Energy_GJ'] = float(p[2])
                    out['EUI_MJ_m2'] = float(p[3])          # per total bldg area
                elif s.startswith(',Total Building Area,'):
                    out['Building_Area_m2'] = float(s.split(',')[2])
                elif current == 'EndUses':
                    p = s.split(',')
                    if len(p) > 3 and p[1] in END_USE_ROWS and END_USE_ROWS[p[1]] not in out:
                        out[END_USE_ROWS[p[1]]] = float(p[2])
                    elif len(p) > 3 and p[1] == 'Total End Uses':
                        out['Electricity_EndUse_GJ'] = float(p[2])
                        out['NaturalGas_EndUse_GJ'] = float(p[3])
    except (OSError, ValueError, IndexError) as e:
        print(f'    [FAIL] {path}: {e}')
        return None
    return out if 'EUI_MJ_m2' in out else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--external-test', action='store_true', help='Aggregate external test data')
    args = parser.parse_args()

    global LHS_ROOT, OUTPUT_CSV, SCENARIOS
    if args.external_test:
        LHS_ROOT = os.path.join(BASE_DIR, 'data', 'jEPlus-LHS', '0_Test', '0_LHS external test')
        OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'external_test_results.csv')
        SCENARIOS = {k: v for k, v in SCENARIOS.items() if k in ['1_Baseline', '2', '5']}

    all_rows, n_fail = [], 0

    for scen, delta_t in SCENARIOS.items():
        scen_dir = os.path.join(LHS_ROOT, scen)
        params_csv = os.path.join(scen_dir, 'AllCombinedResults.csv')
        if not os.path.exists(params_csv):
            print(f'[SKIP] {scen}: AllCombinedResults.csv not found')
            continue

        df = pd.read_csv(params_csv)
        n_ok = 0
        for _, row in df.iterrows():
            job_id = str(row['Job_ID']).strip()
            # NOTE: the jEPlus 'Errors' column counts E+ Severe errors, which for
            # this prototype are warmup-convergence notices only — the run still
            # completes and the annual tables are valid. Filter on Message instead;
            # 'Errors'/'Warnings' stay in the output as QA columns.
            if 'Completed Successfully' not in str(row.get('Message', '')):
                print(f'    [SKIP] {scen}/{job_id}: {row.get("Message", "no message")}')
                n_fail += 1
                continue
            tbl = os.path.join(scen_dir, job_id, 'eplustbl.csv')
            res = parse_eplustbl(tbl) if os.path.exists(tbl) else None
            if res is None:
                print(f'    [MISS] {scen}/{job_id}: eplustbl.csv missing/unreadable')
                n_fail += 1
                continue
            rec = row.to_dict()
            rec['Scenario'] = scen
            rec['Climate_DeltaT'] = delta_t
            rec.update(res)
            rec['EUI_kWh_m2'] = round(res['EUI_MJ_m2'] / 3.6, 3)
            all_rows.append(rec)
            n_ok += 1
        print(f'{scen}: {n_ok} jobs aggregated')

    if not all_rows:
        print('No data extracted — nothing written.')
        sys.exit(1)

    out = pd.DataFrame(all_rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f'\nWrote {len(out)} rows x {len(out.columns)} cols -> {OUTPUT_CSV}')
    print(f'Failures/skips: {n_fail}')
    print('\nEUI (kWh/m2/yr) by scenario:')
    print(out.groupby('Scenario')['EUI_kWh_m2'].agg(['count', 'min', 'mean', 'max']).round(1))


if __name__ == '__main__':
    main()
