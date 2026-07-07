# -*- coding: utf-8 -*-
"""Patch #9: enable real physics-informed training (lambda_mono) + add PDP.

1. Cell 26 (S6 training): lambda_mono 0.0 -> 0.05 (validated safe/comparable
   R2 in the S10 ablation study, mono_on vs mono_off). Wires the same
   gradient-based monotonicity mechanism scripts/analysis/step3_robustness.py
   already uses for its ablation variant (features.requires_grad_(True) on
   global_params during training only), so "Physics-Informed" is literally
   true for the model driving every downstream section (S7-S16), not just an
   offline ablation variant nobody trains with by default.
2. New Section 15b — Partial Dependence (physics validation): sweeps
   Wall_U / COP / Climate_DeltaT across their physical range (all other
   params fixed at the TOPSIS-optimal package) for all 4 models, plots the
   curves + a Spearman-monotonicity table. This is the full continuous-curve
   version of what the S10 ablation only checked at one finite-difference
   point; ties the PDP figure directly to the physics-informed claim.
"""
import json
import uuid

NB = 'NZEB_PIPELINE_ICAE2026_v3.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']


def sub(i, old, new, count=1):
    cur = ''.join(cells[i]['source'])
    if old not in cur:
        raise SystemExit(f'cell {i}: substring not found:\n{old[:200]}')
    cells[i]['source'] = cur.replace(old, new, count).splitlines(keepends=True)


def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'id': uuid.uuid4().hex[:8],
            'metadata': {}, 'outputs': [], 'source': src.splitlines(keepends=True)}


def md_cell(src):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {},
            'source': src.splitlines(keepends=True)}


# --- cell 26: enable lambda_mono for the main model ------------------------- #
sub(26, """criterion = PhysicsLoss(lambda_bound=0.1, lambda_mono=0.0)

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total = 0.
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict,
                        batch.batch_dict, batch.global_params)
            loss, _, _, _ = criterion(out, batch.y)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)""",
    """# lambda_mono=0.05: validated in the S10 ablation (mono_on vs mono_off gave
# comparable R2, near-zero violation either way) -- enabling it here for the
# main model makes "Physics-Informed" literal (every downstream section uses
# this model), not just an offline ablation variant nobody actually trains with.
LAMBDA_MONO = 0.05
criterion = PhysicsLoss(lambda_bound=0.1, lambda_mono=LAMBDA_MONO)

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, total_mono = 0., 0.
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            gp = batch.global_params
            if is_train and LAMBDA_MONO > 0:
                gp = gp.detach().requires_grad_(True)
            out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, gp)
            feats = gp if (is_train and LAMBDA_MONO > 0) else None
            loss, _, _, loss_mono = criterion(out, batch.y, features=feats)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * batch.num_graphs
            total_mono += float(loss_mono) * batch.num_graphs
    return total / len(loader.dataset), total_mono / len(loader.dataset)""")

sub(26, """hist = {'train': [], 'val': []}
best_val, patience_cnt = 1e9, 0
print('Training PI-HGAT...')
t0 = time.time()

for ep in range(1, TRAIN_PARAMS['epochs']+1):
    tl = run_epoch(model, train_loader, optimizer)
    vl = run_epoch(model, val_loader)
    scheduler.step()   # review fix M5: CosineAnnealing steps by epoch, NOT by val loss
    hist['train'].append(tl); hist['val'].append(vl)""",
    """hist = {'train': [], 'val': [], 'train_mono': []}
best_val, patience_cnt = 1e9, 0
print(f'Training PI-HGAT (lambda_mono={LAMBDA_MONO})...')
t0 = time.time()

for ep in range(1, TRAIN_PARAMS['epochs']+1):
    tl, tm = run_epoch(model, train_loader, optimizer)
    vl, _ = run_epoch(model, val_loader)
    scheduler.step()   # review fix M5: CosineAnnealing steps by epoch, NOT by val loss
    hist['train'].append(tl); hist['val'].append(vl); hist['train_mono'].append(tm)""")

sub(26, """    if ep % 20 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f'  Ep {ep:3d}: train={tl:.2f}  val={vl:.2f}  lr={lr:.1e}')""",
    """    if ep % 20 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f'  Ep {ep:3d}: train={tl:.2f}  val={vl:.2f}  mono_loss={tm:.4f}  lr={lr:.1e}')""")

# --- New Section 15b: Partial Dependence (physics validation) --------------- #
md_src = '''## PART 3 · Section 15b — Partial Dependence (Physics Validation)
Complements the S10 finite-difference ablation with the full continuous curve: sweeps
Wall U-value, cooling COP and climate ΔT across their physical range (all other retrofit
parameters held at the TOPSIS-optimal package from S13) and predicts Gross EUI with all 4
models. A monotonic curve in the physically-expected direction is direct, visual evidence
that the model (not just a single perturbation check) has learned the right physical
relationship -- ties directly to the lambda_mono>0 physics-informed training in S6.
'''

code_src = '''# --- PDP: physics validation (all 4 models, 3 physically-grounded features) ---
from pi_hgat.config import P1_LEVELS, P5_LEVELS
from scripts.analysis.fig_style import apply_style, savefig, MODEL_COLORS
from scipy.stats import spearmanr
apply_style()

ORDER = ['PI-HGAT', 'XGBoost', 'ANN (MLP)', 'Linear Reg']
PDP_FEATURES = [
    ('P1_Wall_U', np.linspace(min(P1_LEVELS), max(P1_LEVELS), 25),
     'Wall U-value (W/m²K)', '+'),
    ('P5_COP', np.linspace(min(P5_LEVELS), max(P5_LEVELS), 25),
     'Cooling COP (gross)', '-'),
    ('Climate_DeltaT', np.linspace(0.0, 4.472, 25),
     'Climate warming ΔT (°C)', '+'),
]

def predict_all_models(params):
    """PI-HGAT + 3 baselines for one physical-parameter dict."""
    x_vec = [params[k] for k in FEATURE_NAMES]
    d = builder.create_sample_graph(params)
    d.global_params = torch.tensor([x_vec], dtype=torch.float)
    bd = {nt: torch.zeros(d[nt].x.size(0), dtype=torch.long, device=device)
          for nt in d.node_types}
    model.eval()
    with torch.no_grad():
        hgat_pred = model({nt: d[nt].x.to(device) for nt in d.node_types},
                          {et: d[et].edge_index.to(device) for et in d.edge_types},
                          bd, d.global_params.to(device)).item()
    xs = scaler.transform([x_vec])
    ann.eval()
    with torch.no_grad():
        ann_pred = float(ann(torch.tensor(xs, dtype=torch.float).to(device)).item())
    return {'PI-HGAT': hgat_pred, 'XGBoost': float(xgb_model.predict(xs)[0]),
            'ANN (MLP)': ann_pred, 'Linear Reg': float(lr_model.predict(xs)[0])}

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
mono_rows = []
for ax, (feat_key, grid, xlabel, expect) in zip(axes, PDP_FEATURES):
    curves = {m: [] for m in ORDER}
    for v in grid:
        p = dict(best_params); p[feat_key] = float(v)
        preds = predict_all_models(p)
        for m in ORDER:
            curves[m].append(preds[m])
    for m in ORDER:
        ax.plot(grid, curves[m], marker='o', ms=3, lw=1.6, color=MODEL_COLORS[m], label=m)
        rho, _ = spearmanr(grid, curves[m])
        mono_rows.append({'feature': feat_key, 'model': m, 'expected_sign': expect,
                          'spearman_rho': round(rho, 3)})
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Predicted Gross EUI (kWh/m²/yr)')
    arrow = '↑' if expect == '+' else '↓'
    ax.set_title(f'Expected: EUI {arrow} with {xlabel.split(" (")[0]}',
                fontsize=9, fontweight='bold')
axes[0].legend(prop={'size': 7}, loc='best')
plt.suptitle('Partial Dependence — all other params fixed at TOPSIS-optimal package',
             fontweight='bold', fontsize=10)
plt.tight_layout()
savefig(fig, 'Fig15_PartialDependence')
plt.show()

print('Monotonicity check (Spearman ρ between swept feature and prediction; '
      '|ρ|→1 = fully monotonic in the expected direction):')
display(pd.DataFrame(mono_rows).pivot(index='feature', columns='model', values='spearman_rho'))
'''

insert_at = 76  # right before cell 76 (S16 markdown header), after cell 75 (S15 timer)
cells[insert_at:insert_at] = [md_cell(md_src), code_cell(code_src)]

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'patched {NB}: cell 26 (lambda_mono) + inserted 2 cells at index {insert_at} '
      f'({len(cells)} total cells)')
