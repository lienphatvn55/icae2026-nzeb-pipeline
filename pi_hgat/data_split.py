"""Single source of truth for the LHS dataset and the scenario split.

Both the notebook (S2, cell 10) and scripts/analysis/step3_robustness.py import
from here, so the split, feature definition and combo hashing can never diverge
between the interactive pipeline and the batch robustness studies.

SCENARIO_SPLIT design (fixed, deliberate — replaces the old random
GroupShuffleSplit(random_state=42) which happened to pick val={8}, test={2,6}):

  ΔT (°C, delta-morphing) per scenario:
    1_Baseline=0.000 · 6=1.270 · 8=1.611 · 7=1.875 · 2=1.879 · 4=2.179
    · 3=2.665 · 9=3.144 · 5=4.472

  train {1_Baseline, 6, 7, 4, 9, 5} — spans the FULL ΔT hull [0, 4.472]
        including both extremes, so the surrogate never extrapolates ΔT when
        the MOO queries it at 0 / +2.03 / +4.47 °C.
  val   {8 (1.611), 2 (1.879)}      — two interpolation points in different
        regions → early stopping / model selection far more stable than the
        previous single-scenario val.
  test  {3 (2.665)}                 — a genuine interpolation gap (±0.5 °C to
        the nearest train ΔT of 2.179 / 3.144), unlike the old test scenario 2
        which sat 0.004 °C from train scenario 7.

  Climate EXTRApolation evidence comes from the LOSO 9-fold study (S10), and
  parameter generalization from the combo-grouped split + the external LHS
  test set (seed 2810: scenarios {1_Baseline, 2, 5} × 150 unseen combos, S8).
"""
import os

import numpy as np
import pandas as pd

LHS_CSV = os.path.join('data', 'aggregated_LHS_results.csv')
EXTERNAL_CSV = os.path.join('data', 'external_test_results.csv')

FEATURE_NAMES = ['P1_Wall_U', 'P2_Roof_U', 'P3_Roof_Reflectance', 'P4_Win_U',
                 'P4_Win_SHGC', 'P5_COP', 'P6_Cool_SP', 'P7_LPD', 'Climate_DeltaT']

PARAM_COLS = ['@@P1_Wall_R@@', '@@P2_Roof_R@@', '@@P3_Roof_Abs@@', '@@P4_U@@',
              '@@P4_SHGC@@', '@@P5_COP@@', '@@P6_ClgSetp@@', '@@P7_LPD@@']

SCENARIO_SPLIT = {
    'train': ['1_Baseline', '6', '7', '4', '9', '5'],
    'val':   ['8', '2'],
    'test':  ['3'],
}


def row_to_params(row):
    """jEPlus CSV row -> physical parameter dict (the surrogate feature space)."""
    return {
        'P1_Wall_U': 1.0 / row['@@P1_Wall_R@@'],
        'P2_Roof_U': 1.0 / row['@@P2_Roof_R@@'],
        'P3_Roof_Reflectance': 1.0 - row['@@P3_Roof_Abs@@'],
        'P4_Win_U': row['@@P4_U@@'],
        'P4_Win_SHGC': row['@@P4_SHGC@@'],
        'P5_COP': row['@@P5_COP@@'],
        'P6_Cool_SP': row['@@P6_ClgSetp@@'],
        'P7_LPD': row['@@P7_LPD@@'],
        'Climate_DeltaT': row['Climate_DeltaT'],
    }


def build_combo_id(df):
    """Hash of the 7 E+ parameters. The same 249 unique combos repeat in all 9
    scenarios (250 LHS rows/scenario; one pair collides after round(4))."""
    return df[PARAM_COLS].round(4).astype(str).agg('|'.join, axis=1).values


def load_lhs_dataframe(csv_path=LHS_CSV):
    return pd.read_csv(csv_path)


def load_lhs_arrays(csv_path=LHS_CSV):
    """-> (df, samples, X, Y, groups, combo_id). Y = GROSS site EUI (1-D)."""
    df = load_lhs_dataframe(csv_path)
    samples = [row_to_params(row) for _, row in df.iterrows()]
    X = np.array([[s[k] for k in FEATURE_NAMES] for s in samples])
    Y = df['EUI_kWh_m2'].to_numpy()
    groups = df['Scenario'].astype(str).to_numpy()
    return df, samples, X, Y, groups, build_combo_id(df)


def split_indices(groups, split=None):
    """groups (array of scenario names) -> (idx_train, idx_val, idx_test).

    Raises if any scenario is missing or unassigned, so a data change can
    never silently produce an inconsistent split.
    """
    split = split or SCENARIO_SPLIT
    groups = np.asarray(groups).astype(str)
    assigned = [s for part in ('train', 'val', 'test') for s in split[part]]
    unique = set(np.unique(groups))
    if set(assigned) != unique:
        raise ValueError(f'SCENARIO_SPLIT does not cover the data: '
                         f'split={sorted(assigned)} vs data={sorted(unique)}')
    idx = {part: np.where(np.isin(groups, split[part]))[0]
           for part in ('train', 'val', 'test')}
    return idx['train'], idx['val'], idx['test']


def describe_split(groups, X=None, delta_col=FEATURE_NAMES.index('Climate_DeltaT')):
    """Pretty table of the split (scenario, ΔT, n rows, part) for the notebook."""
    groups = np.asarray(groups).astype(str)
    rows = []
    for part in ('train', 'val', 'test'):
        for s in SCENARIO_SPLIT[part]:
            mask = groups == s
            dt = float(X[mask][0, delta_col]) if X is not None else float('nan')
            rows.append({'part': part, 'scenario': s, 'delta_T': round(dt, 3),
                         'n_rows': int(mask.sum())})
    return pd.DataFrame(rows)
