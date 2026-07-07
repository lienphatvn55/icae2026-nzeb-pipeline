"""Single source of truth for the LHS dataset and the scenario split.

Both the notebook (S2, cell 10) and scripts/analysis/step3_robustness.py import
from here, so the split, feature definition and combo hashing can never diverge
between the interactive pipeline and the batch robustness studies.

SCENARIO_SPLIT design (fixed, deliberate):

  Canonical scenario IDs (2026-07-07 rename; ΔT °C, delta-morphing):
    Baseline=0.000 (TMYx 2011-2025) · S5=1.270 · S7=1.611 · S6=1.875
    · S1=1.879 · S3=2.179 · S2=2.665 · S8=3.144 · S4=4.472
    (full GCM/SSP/horizon metadata: SCENARIO_LABELS below)

  train {Baseline, S5, S6, S3, S8} — spans ΔT [0, 3.144].
  val   {S7 (1.611), S1 (1.879)}  — two interpolation points in different
        regions -> stable early stopping / model selection.
  test  {S2 (2.665)}             — a genuine interpolation gap (±0.5 °C to
        the nearest train ΔT of 2.179 / 3.144).
  extrapolation_test {S4 (4.472)} — SSP5-8.5 2080s, the most severe warming,
        held out ENTIRELY from train/val (its 250 MAIN rows are never used
        to fit or select the model). This is deliberate: an earlier version
        of this pipeline kept scenario 5 in train while ALSO evaluating it
        via a same-climate external LHS replicate (seed 2810) — that setup
        let the model see the exact climate signature during training, so a
        good "external" score there proved parameter generalization only,
        not climate extrapolation (a Q1 reviewer would call this data
        snooping on the pipeline's own marquee claim: "does the surrogate
        hold up under the worst 2080s scenario?"). Excluding scenario S4 from
        train/val turns it into a genuine held-out test on BOTH unseen
        retrofit combos AND unseen climate severity — see
        extrapolation_test_arrays() below, which merges S4's MAIN
        rows (250, never trained on) with its external LHS replicate (150,
        seed 2810) into one n=400 evaluation set.

  NOTE: S1 (ΔT=1.879) and S6 (ΔT=1.875, in train) are two
  distinct CMIP6 GCM/SSP combinations that happen to produce nearly
  identical mean warming (Δ=0.004 °C). Since the surrogate only sees ΔT as
  a scalar, it cannot distinguish them regardless of which side of the
  split S1 sits on — a structural limitation of the ΔT-scalar
  climate representation, not a split design choice.
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
    'train': ['Baseline', 'S5', 'S6', 'S3', 'S8'],
    'val':   ['S7', 'S1'],
    'test':  ['S2'],
    'extrapolation_test': ['S4'],
}

# Canonical scenario metadata (single source for every figure/table label).
# Old folder names (pre 2026-07-07): Baseline=1_Baseline, S1=2, S2=3, S3=4,
# S4=5, S5=6, S6=7, S7=8, S8=9.
SCENARIO_LABELS = {
    'Baseline': dict(gcm='TMYx 2011-2025', ssp='',        horizon='',      dt=0.0),
    'S1':       dict(gcm='ACCESS-CM2',     ssp='SSP2-4.5', horizon='2050s', dt=1.879),
    'S2':       dict(gcm='ACCESS-CM2',     ssp='SSP2-4.5', horizon='2080s', dt=2.665),
    'S3':       dict(gcm='ACCESS-CM2',     ssp='SSP5-8.5', horizon='2050s', dt=2.179),
    'S4':       dict(gcm='ACCESS-CM2',     ssp='SSP5-8.5', horizon='2080s', dt=4.472),
    'S5':       dict(gcm='MRI-ESM2-0',     ssp='SSP2-4.5', horizon='2050s', dt=1.270),
    'S6':       dict(gcm='MRI-ESM2-0',     ssp='SSP2-4.5', horizon='2080s', dt=1.875),
    'S7':       dict(gcm='MRI-ESM2-0',     ssp='SSP5-8.5', horizon='2050s', dt=1.611),
    'S8':       dict(gcm='MRI-ESM2-0',     ssp='SSP5-8.5', horizon='2080s', dt=3.144),
}


def scenario_label(sid, multiline=True, with_dt=True):
    """Figure-ready label, e.g. 'S1\\nACCESS-CM2\\nSSP2-4.5 2050s\\n(+1.9°C)'.

    GCM and SSP+horizon go on separate lines: a single combined detail line
    (~25 chars) is wider than one column slot in a 9-column figure and made
    neighbouring tick labels overlap (FigC1 review fix, 2026-07-07).
    """
    m = SCENARIO_LABELS[str(sid)]
    parts = [str(sid), m['gcm'], f"{m['ssp']} {m['horizon']}".strip()]
    if with_dt:
        parts.append(f"(+{m['dt']:.1f}°C)")
    parts = [p for p in parts if p]
    return ('\n' if multiline else ' — ').join(parts)

# Scenarios with an independent-seed external LHS replicate (seed 2810,
# 150 unseen combos each). Since 2026-07-07 ALL 9 scenarios have one, which
# enables the systematic loso_ext study (per-climate combo+climate
# generalization) instead of a hand-picked single extrapolation scenario.
EXTERNAL_SCENARIOS = list(SCENARIO_LABELS.keys())


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


def load_external_dataframe(csv_path=EXTERNAL_CSV):
    """External LHS replicate (seed 2810): scenarios in EXTERNAL_SCENARIOS,
    unseen retrofit combos. Filters to E+ runs that completed successfully."""
    ext = pd.read_csv(csv_path)
    return ext[ext['Message'].str.contains('Completed Successfully', na=False)].reset_index(drop=True)


def split_indices(groups, split=None):
    """groups (array of scenario names) -> (idx_train, idx_val, idx_test).

    The extrapolation_test scenario (5) is deliberately excluded from all
    three and handled separately by extrapolation_test_arrays(). Raises if
    any scenario is missing or unassigned across all four buckets, so a data
    change can never silently produce an inconsistent split.
    """
    split = split or SCENARIO_SPLIT
    groups = np.asarray(groups).astype(str)
    assigned = [s for part in ('train', 'val', 'test', 'extrapolation_test')
                for s in split[part]]
    unique = set(np.unique(groups))
    if set(assigned) != unique:
        raise ValueError(f'SCENARIO_SPLIT does not cover the data: '
                         f'split={sorted(assigned)} vs data={sorted(unique)}')
    idx = {part: np.where(np.isin(groups, split[part]))[0]
           for part in ('train', 'val', 'test')}
    return idx['train'], idx['val'], idx['test']


def extrapolation_test_arrays(df, X, Y, groups):
    """Combined climate+parameter extrapolation test for scenario 5
    (SSP585-2080s): its 250 MAIN rows (never trained on) + its 150-row
    external LHS replicate (seed 2810, unseen combos) = 400 rows, all at a
    ΔT the model never saw during training or model selection.

    Returns (X_main, Y_main, X_ext, Y_ext) — kept separate because MAIN rows
    can reuse the already-built PyG graphs / dataset index, while EXT rows
    need graphs built fresh from row_to_params. Concatenate predictions
    (not raw arrays) for the combined n=400 metric.
    """
    scenario = SCENARIO_SPLIT['extrapolation_test'][0]
    groups = np.asarray(groups).astype(str)
    idx_main = np.where(groups == scenario)[0]

    ext = load_external_dataframe()
    ext = ext[ext['Scenario'].astype(str) == scenario].reset_index(drop=True)
    ext_samples = [row_to_params(row) for _, row in ext.iterrows()]
    X_ext = np.array([[s[k] for k in FEATURE_NAMES] for s in ext_samples])
    Y_ext = ext['EUI_kWh_m2'].to_numpy()

    return idx_main, X[idx_main], Y[idx_main], ext, X_ext, Y_ext


def describe_split(groups, X=None, delta_col=FEATURE_NAMES.index('Climate_DeltaT')):
    """Pretty table of the split (scenario, ΔT, n rows, part) for the notebook."""
    groups = np.asarray(groups).astype(str)
    rows = []
    for part in ('train', 'val', 'test', 'extrapolation_test'):
        for s in SCENARIO_SPLIT[part]:
            mask = groups == s
            dt = float(X[mask][0, delta_col]) if X is not None else float('nan')
            rows.append({'part': part, 'scenario': s, 'delta_T': round(dt, 3),
                         'n_rows': int(mask.sum())})
    return pd.DataFrame(rows)
