# -*- coding: utf-8 -*-
"""Patch #3 for NZEB_PIPELINE_ICAE2026_v3.ipynb (user feedback 2026-07-06 evening).

Redesign: scenario 5 (SSP585-2080s, ΔT=4.472) fully excluded from train/val
(pi_hgat/data_split.py SCENARIO_SPLIT, already updated). This avoids the data
-snooping issue where the external LHS replicate (seed 2810) for scenario 5
was evaluated at a climate the model had already trained on — a good score
there proved parameter generalization only, not climate extrapolation.

  10  S2: update split description text (5 train, not 6) + mention
      extrapolation_test bucket
  32  S8 markdown: describe the two-tier external evaluation
  35  S8: full rewrite — split external eval into (a) combo generalization
      at seen climate [1_Baseline, 2] and (b) combo+climate extrapolation
      [scenario 5, MAIN 250 (never trained) + external 150 unseen combos]
  42  S10 markdown: "6/2/1" -> "5/2/1 + dedicated extrapolation test"
  44  S10: Fig7 x-axis label "6 scenarios" -> "5 scenarios"
"""
import json

NB = 'NZEB_PIPELINE_ICAE2026_v3.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']


def sub(i, old, new, count=1):
    cur = ''.join(cells[i]['source'])
    if old not in cur:
        raise SystemExit(f'cell {i}: substring not found:\n{old[:200]}')
    cells[i]['source'] = cur.replace(old, new, count).splitlines(keepends=True)


def set_source(i, text):
    cells[i]['source'] = text.splitlines(keepends=True)


# --- cell 10: S2 split description ------------------------------------------ #
sub(10, """# Fixed 6/2/1 scenario split (deliberate — see pi_hgat/data_split.py docstring):
#   train spans the FULL ΔT hull [0, 4.472] incl. both extremes -> the surrogate
#   never extrapolates ΔT when the MOO queries it at 0 / +2.03 / +4.47 °C;
#   val = 2 scenarios (stable early stopping); test = scenario 3 (ΔT 2.665, a
#   genuine ±0.5 °C interpolation gap). Climate EXTRApolation evidence = LOSO
#   9-fold (S10); parameter generalization = external LHS seed 2810 (S8).
idx_train, idx_val, idx_test = split_indices(groups)
print('\\nScenario split (fixed 6 train / 2 val / 1 test):')
print(describe_split(groups, X_flat).to_string(index=False))
print(f'Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test')""",
    """# Fixed scenario split (deliberate — see pi_hgat/data_split.py docstring):
#   train (5 scenarios) spans ΔT [0, 3.144]; val = 2 interpolation scenarios
#   (stable early stopping); test = scenario 3 (ΔT 2.665, a genuine ±0.5 °C
#   interpolation gap). Scenario 5 (SSP585-2080s, ΔT 4.472) is held out of
#   train/val ENTIRELY -> S8 evaluates it as a genuine combined climate +
#   parameter extrapolation test (never just "parameter generalization at a
#   climate the model already trained on"). Climate interpolation evidence =
#   LOSO 9-fold (S10); parameter generalization = combo-split (S10) + external
#   LHS seed 2810 at seen climate (S8a); climate+parameter extrapolation =
#   external LHS seed 2810 at scenario 5, merged with its held-out MAIN rows (S8b).
idx_train, idx_val, idx_test = split_indices(groups)
print('\\nScenario split (fixed 5 train / 2 val / 1 interpolation-test / '
      '1 extrapolation-test):')
print(describe_split(groups, X_flat).to_string(index=False))
print(f'Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test '
      f'(scenario 5, n=250, held out entirely -> S8 extrapolation test)')""")

# --- cell 32: S8 markdown ----------------------------------------------------- #
set_source(32, '''## PART 1 · Section 8 — Evaluation & Benchmark (R2, RMSE, MAE)
Produces the surrogate comparison table (paper Table: model benchmark), then two
external-LHS evaluations (seed 2810, unseen retrofit combos) that answer different
questions:
- **(a) Combo generalization at seen climate** (scenarios 1_Baseline, 2) — the model
  already trained/selected at these ΔT values, so this isolates pure parameter
  generalization. Expect near-ceiling scores; this is a sanity check, not a
  climate-extrapolation claim.
- **(b) Combo + climate extrapolation** (scenario 5, SSP585-2080s, ΔT=4.472) — this
  scenario is held out of train/val entirely (see S2), so its MAIN rows (250) and
  external replicate (150, unseen combos) together form a genuine n=400 test on
  BOTH unseen retrofit combos and unseen climate severity — the real evidence for
  "does the surrogate hold under the worst 2080s scenario?"
''')

# --- cell 35: S8 external test full rewrite ----------------------------------- #
set_source(35, '''# --- EXTERNAL / EXTRAPOLATION EVALUATION (seed 2810 replicate, unseen combos) ---
# Split into two tiers (see S2/S8 markdown): (a) combo generalization at a climate
# the model already trained on [1_Baseline, 2] vs (b) combo + climate extrapolation
# at scenario 5, which is excluded from train/val entirely (pi_hgat/data_split.py).
# An earlier version of this evaluation lumped all 3 external scenarios together;
# since scenario 5 was ALSO in train at the time, a good score there only proved
# parameter generalization, not climate extrapolation (Q1 review finding).
from pi_hgat.data_split import extrapolation_test_arrays, load_external_dataframe

ext_df = load_external_dataframe()
print(f'Loaded {len(ext_df)} external LHS samples (seed 2810, unseen combos), '
      f'{ext_df["Scenario"].nunique()} scenarios: {sorted(ext_df["Scenario"].unique())}')

def eval_external_subset(sub_df):
    """Build graphs + feature rows for an external subset; return per-model preds."""
    ds, xs = [], []
    for _, row in sub_df.iterrows():
        s = row_to_params(row)
        x_vec = [s[k] for k in FEATURE_NAMES]
        d = builder.create_sample_graph(s)
        d.y = torch.tensor([[row['EUI_kWh_m2']]], dtype=torch.float)
        d.global_params = torch.tensor([x_vec], dtype=torch.float)
        ds.append(d); xs.append(x_vec)
    loader = PyGDL(ds, batch_size=TRAIN_PARAMS['batch_size'])
    model.eval()
    p_hgat, t_hgat = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch.global_params)
            p_hgat.extend(out.cpu().numpy().flatten()); t_hgat.extend(batch.y.cpu().numpy().flatten())
    xs = np.array(xs)
    xs_s = scaler.transform(xs)
    ann.eval()
    with torch.no_grad():
        p_ann = ann(torch.tensor(xs_s, dtype=torch.float).to(device)).cpu().numpy().flatten()
    return np.array(t_hgat), np.array(p_hgat), xgb_model.predict(xs_s), p_ann, lr_model.predict(xs_s)

# ---- (a) Combo generalization at SEEN climate: scenarios 1_Baseline, 2 ----
seen_df = ext_df[ext_df.Scenario.astype(str).isin(['1_Baseline', '2'])].reset_index(drop=True)
t_a, hgat_a, xgb_a, ann_a, lr_a = eval_external_subset(seen_df)
df_ext_seen = pd.DataFrame({
    'PI-HGAT':    metrics(t_a, hgat_a),
    'XGBoost':    metrics(t_a, xgb_a),
    'ANN (MLP)':  metrics(t_a, ann_a),
    'Linear Reg': metrics(t_a, lr_a),
}).T
df_ext_seen.columns = ['R²', 'RMSE', 'MAE', 'MAPE (%)']
print(f'\\n===== (a) COMBO GENERALIZATION at seen climate (external LHS, n={len(seen_df)}) =====')
print('Unseen retrofit combos, ΔT already covered by train/val — sanity check, NOT a climate-extrapolation test.')
display(df_ext_seen.round(4))

# ---- (b) COMBO + CLIMATE EXTRAPOLATION: scenario 5 (SSP585-2080s) ----
# MAIN rows (250, never trained/validated on) + external replicate (150, unseen
# combos, seed 2810) -> n=400, all at a ΔT the model never saw before evaluation.
idx5_main, X5_main, Y5_main, ext5_df, X5_ext, Y5_ext = extrapolation_test_arrays(
    df_lhs, X_flat, Y_eui.ravel(), groups)

loader5_main = PyGDL([dataset[i] for i in idx5_main], batch_size=TRAIN_PARAMS['batch_size'])
model.eval()
p5m_hgat, t5m_hgat = [], []
with torch.no_grad():
    for batch in loader5_main:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch.global_params)
        p5m_hgat.extend(out.cpu().numpy().flatten()); t5m_hgat.extend(batch.y.cpu().numpy().flatten())
X5m_s = scaler.transform(X5_main)
ann.eval()
with torch.no_grad():
    p5m_ann = ann(torch.tensor(X5m_s, dtype=torch.float).to(device)).cpu().numpy().flatten()
p5m_xgb, p5m_lr = xgb_model.predict(X5m_s), lr_model.predict(X5m_s)

t5e, p5e_hgat, p5e_xgb, p5e_ann, p5e_lr = eval_external_subset(ext5_df)

t5_all    = np.concatenate([np.array(t5m_hgat), t5e])
hgat5_all = np.concatenate([np.array(p5m_hgat), p5e_hgat])
xgb5_all  = np.concatenate([p5m_xgb, p5e_xgb])
ann5_all  = np.concatenate([p5m_ann, p5e_ann])
lr5_all   = np.concatenate([p5m_lr, p5e_lr])

df_ext_extrap = pd.DataFrame({
    'PI-HGAT':    metrics(t5_all, hgat5_all),
    'XGBoost':    metrics(t5_all, xgb5_all),
    'ANN (MLP)':  metrics(t5_all, ann5_all),
    'Linear Reg': metrics(t5_all, lr5_all),
}).T
df_ext_extrap.columns = ['R²', 'RMSE', 'MAE', 'MAPE (%)']
print(f'\\n===== (b) COMBO + CLIMATE EXTRAPOLATION: scenario 5 / SSP585-2080s, ΔT=4.472 '
      f'(n={len(t5_all)} = {len(idx5_main)} held-out MAIN + {len(ext5_df)} unseen-combo external) =====')
print('Model saw ZERO rows at this ΔT during training or model selection — the genuine')
print('extrapolation evidence for "does the surrogate hold under the worst 2080s scenario?"')
display(df_ext_extrap.round(4))

df_results = pd.concat([
    df_results,
    df_ext_seen.rename(index=lambda m: f'{m} (ext: seen climate)'),
    df_ext_extrap.rename(index=lambda m: f'{m} (ext: climate extrapolation)'),
])

# Per-scenario external error (both tiers) for the paper's data statement
all_ext_scen = list(seen_df['Scenario'].astype(str)) + [SCENARIO_SPLIT['extrapolation_test'][0]] * len(ext5_df)
all_ext_err = np.abs(np.concatenate([t_a, t5e]) - np.concatenate([hgat_a, p5e_hgat]))
ext_err = pd.DataFrame({'Scenario': all_ext_scen, 'abs_err': all_ext_err})
print('\\nPI-HGAT external MAE per scenario (kWh/m2/yr) — scenario 5 row is external-combo-only (n=150):')
display(ext_err.groupby('Scenario')['abs_err'].agg(['mean', 'max']).round(3))''')

# --- cell 42: S10 markdown split description --------------------------------- #
sub(42, "**same** `pi_hgat.data_split` (fixed 6/2/1 scenario split) and `TRAIN_PARAMS` as this",
        "**same** `pi_hgat.data_split` (fixed 5/2/1 scenario split + dedicated scenario-5 "
        "extrapolation holdout) and `TRAIN_PARAMS` as this")

# --- cell 44: Fig7 x-axis label ----------------------------------------------- #
sub(44, "ax.set_xlabel('LHS combos per climate scenario (train, 6 scenarios)')",
        "ax.set_xlabel('LHS combos per climate scenario (train, 5 scenarios)')")

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched', NB)
