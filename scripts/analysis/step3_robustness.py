"""Step-3 robustness & validation studies (review plan, 2026-07-03).

Heavy-compute companion to the notebook: runs the studies and writes CSV
artifacts that notebook section S10b reads to draw Fig. 6 (2x2 robustness),
Fig. 7 (sample-size learning curve) and the validation tables.

Data + split come from pi_hgat.data_split (shared with the notebook: fixed
6/2/1 scenario split, train spans the full ΔT hull, test = scenario 3).
Training budget = pi_hgat.config.TRAIN_PARAMS (same as notebook S6).
Writes results/step3_meta.json (split + params + timestamp) so notebook S10
can verify the artifacts are not stale.

Studies
  multiseed   10 seeds x 4 models on the scenario split -> results/step3_multiseed.csv
  loso        leave-one-scenario-out (9 folds, climate generalization,
              SAME 249 combos as train) -> results/step3_loso.csv
  loso_ext    leave-one-scenario-out where each fold is ALSO evaluated on the
              scenario's independent seed-2810 external replicate (150 unseen
              combos) alongside its 250 held-out MAIN rows -> per-fold metrics
              for eval_set in {main, external, combined}. The main-vs-external
              gap isolates the parameter-generalization cost from the climate
              cost, systematically for all 9 climates (not just scenario 5)
              -> results/step3_loso_ext.csv
  combosplit  held-out parameter combos, all scenarios (parameter generalization)
              -> results/step3_combosplit.csv
  learncurve  all 4 models x 3 seeds vs combos/scenario (25..249)
              -> results/step3_learncurve.csv
  ablation    physics monotonicity loss on/off x 3 seeds + finite-difference
              violation rates -> results/step3_ablation.csv

Usage:  python scripts/analysis/step3_robustness.py --study all
Progress: results/logs/step3_progress.log (study-level lines are prefixed [S3]).
"""
import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as PyGDL
import xgboost as xgb

from pi_hgat.config import NEO4J_JSON_PATH, GNN_PARAMS, TRAIN_PARAMS
from pi_hgat.data_split import (FEATURE_NAMES, SCENARIO_SPLIT, load_lhs_arrays,
                                load_external_dataframe, row_to_params,
                                split_indices)
from pi_hgat.graph_builder import GraphBuilder
from pi_hgat.models import PI_HGAT, BaselineANN
from pi_hgat.physics_loss import PhysicsLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROGRESS = os.path.join('results', 'logs', 'step3_progress.log')
MODEL_ORDER = ['PI-HGAT', 'XGBoost', 'ANN (MLP)', 'Linear Reg']
# Same training budget as the notebook's main model (S6) — the previous
# hard-coded 200/150-epoch caps made the studies undertrain PI-HGAT and the
# learning curve answer "not enough data" when the truth was "not enough epochs".
EPOCHS_CAP = TRAIN_PARAMS['epochs']      # 300
PATIENCE = TRAIN_PARAMS['patience']      # 40


def log(msg, study_level=False):
    line = ('[S3] ' if study_level else '      ') + msg
    print(line, flush=True)
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M:%S ') + line + '\n')


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def metrics(yt, yp):
    return dict(r2=r2_score(yt, yp),
                rmse=mean_squared_error(yt, yp) ** 0.5,
                mae=mean_absolute_error(yt, yp),
                mape=mean_absolute_percentage_error(yt, yp) * 100)


# ------------------------------------------------------------------ data --- #
def load_data():
    """Delegates to pi_hgat.data_split — the same loader the notebook uses."""
    _, samples, X, Y, groups, combo_id = load_lhs_arrays()
    return X, Y, groups, combo_id, samples


def build_graphs(builder, samples, X, Y):
    dataset = []
    for i, s in enumerate(samples):
        d = builder.create_sample_graph(s)
        d.y = torch.tensor([[Y[i]]], dtype=torch.float)
        d.global_params = torch.from_numpy(X[i]).float().unsqueeze(0)
        dataset.append(d)
    return dataset


def scenario_split(X, Y, groups):
    """Fixed 6/2/1 scenario split from pi_hgat.data_split (same as notebook S2)."""
    return split_indices(groups)


def combo_split(X, Y, combo_id):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    idx_tr, idx_tmp = next(gss1.split(X, Y, groups=combo_id))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    iv, it = next(gss2.split(X[idx_tmp], Y[idx_tmp], groups=combo_id[idx_tmp]))
    return idx_tr, idx_tmp[iv], idx_tmp[it]


# -------------------------------------------------------------- training --- #
def train_hgat(dataset, idx_tr, idx_va, idx_te, seed, epochs_cap=EPOCHS_CAP,
               lambda_mono=0.0, patience=PATIENCE):
    seed_all(seed)
    model = PI_HGAT(metadata=dataset[0].metadata(),
                    hidden_channels=GNN_PARAMS['hidden_channels'], out_channels=1,
                    num_layers=GNN_PARAMS['num_layers'], heads=GNN_PARAMS['heads'],
                    dropout=GNN_PARAMS['dropout'], global_dim=len(FEATURE_NAMES)).to(DEVICE)
    tr = PyGDL([dataset[i] for i in idx_tr], batch_size=TRAIN_PARAMS['batch_size'], shuffle=True)
    va = PyGDL([dataset[i] for i in idx_va], batch_size=TRAIN_PARAMS['batch_size'])
    te = PyGDL([dataset[i] for i in idx_te], batch_size=TRAIN_PARAMS['batch_size'])

    dummy = next(iter(tr)).to(DEVICE)
    model(dummy.x_dict, dummy.edge_index_dict, dummy.batch_dict, dummy.global_params)

    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_PARAMS['lr'],
                           weight_decay=TRAIN_PARAMS['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_cap, eta_min=1e-6)
    crit = PhysicsLoss(lambda_bound=0.1, lambda_mono=lambda_mono)

    def run(loader, train):
        model.train() if train else model.eval()
        total = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for b in loader:
                b = b.to(DEVICE)
                gp = b.global_params
                if train and lambda_mono > 0:
                    gp = gp.detach().requires_grad_(True)
                out = model(b.x_dict, b.edge_index_dict, b.batch_dict, gp)
                feats = gp if (train and lambda_mono > 0) else None
                loss, _, _, _ = crit(out, b.y, features=feats)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * b.num_graphs
        return total / len(loader.dataset)

    t0 = time.time()
    best_val, best_state, bad = 1e9, None, 0
    for ep in range(1, epochs_cap + 1):
        run(tr, True)
        vl = run(va, False)
        sch.step()
        if vl < best_val:
            best_val, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    fit_s = time.time() - t0
    model.load_state_dict(best_state)

    def predict(loader):
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(DEVICE)
                out = model(b.x_dict, b.edge_index_dict, b.batch_dict, b.global_params)
                ps.extend(out.cpu().numpy().flatten())
                ts.extend(b.y.cpu().numpy().flatten())
        return np.array(ts), np.array(ps)

    yt_tr, yp_tr = predict(PyGDL([dataset[i] for i in idx_tr], batch_size=128))
    yt_te, yp_te = predict(te)
    return model, metrics(yt_tr, yp_tr), metrics(yt_te, yp_te), fit_s


def train_baselines(X, Y, idx_tr, idx_va, idx_te, seed, return_models=False):
    """Returns {name: (m_train, m_test, fit_seconds)} for XGB / ANN / LR.
    With return_models=True also returns the fitted estimators + scaler so a
    study can score additional evaluation sets (e.g. the external replicate)."""
    Xtr, Ytr = X[idx_tr], Y[idx_tr]
    Xv, Yv = X[idx_va], Y[idx_va]
    Xte, Yte = X[idx_te], Y[idx_te]
    sc = StandardScaler()
    Xtr_s, Xv_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xv), sc.transform(Xte)
    out = {}

    t0 = time.time()
    lr_m = LinearRegression().fit(Xtr_s, Ytr)
    out['Linear Reg'] = (metrics(Ytr, lr_m.predict(Xtr_s)),
                         metrics(Yte, lr_m.predict(Xte_s)), time.time() - t0)

    t0 = time.time()
    xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=seed, tree_method='exact')
    xgb_m.fit(Xtr_s, Ytr)
    out['XGBoost'] = (metrics(Ytr, xgb_m.predict(Xtr_s)),
                      metrics(Yte, xgb_m.predict(Xte_s)), time.time() - t0)

    seed_all(seed)
    ann = BaselineANN(Xtr_s.shape[1], 1).to(DEVICE)
    opt = torch.optim.Adam(ann.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr_s, dtype=torch.float),
                                        torch.tensor(Ytr, dtype=torch.float).unsqueeze(1))
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    xv_t = torch.tensor(Xv_s, dtype=torch.float).to(DEVICE)
    yv_t = torch.tensor(Yv, dtype=torch.float).unsqueeze(1).to(DEVICE)
    t0 = time.time()
    best, best_state, bad = 1e9, None, 0
    for ep in range(300):
        ann.train()
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad(); crit(ann(bx), by).backward(); opt.step()
        ann.eval()
        with torch.no_grad():
            vl = crit(ann(xv_t), yv_t).item()
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(ann.state_dict()), 0
        else:
            bad += 1
            if bad >= 20:
                break
    ann.load_state_dict(best_state)
    fit_s = time.time() - t0
    ann.eval()
    with torch.no_grad():
        p_tr = ann(torch.tensor(Xtr_s, dtype=torch.float).to(DEVICE)).cpu().numpy().flatten()
        p_te = ann(torch.tensor(Xte_s, dtype=torch.float).to(DEVICE)).cpu().numpy().flatten()
    out['ANN (MLP)'] = (metrics(Ytr, p_tr), metrics(Yte, p_te), fit_s)
    if return_models:
        return out, dict(scaler=sc, lr=lr_m, xgb=xgb_m, ann=ann)
    return out


def predict_single(model, builder, params):
    d = builder.create_sample_graph(params)
    d.global_params = torch.tensor([[params[k] for k in FEATURE_NAMES]], dtype=torch.float)
    bd = {nt: torch.zeros(d[nt].x.size(0), dtype=torch.long, device=DEVICE) for nt in d.node_types}
    with torch.no_grad():
        out = model({nt: d[nt].x.to(DEVICE) for nt in d.node_types},
                    {et: d[et].edge_index.to(DEVICE) for et in d.edge_types},
                    bd, d.global_params.to(DEVICE))
    return float(out.item())


# --------------------------------------------------------------- studies --- #
def study_multiseed(ctx, seeds=range(10)):
    X, Y, groups, dataset = ctx['X'], ctx['Y'], ctx['groups'], ctx['dataset']
    idx_tr, idx_va, idx_te = scenario_split(X, Y, groups)
    rows = []
    for seed in seeds:
        _, m_tr, m_te, fit_s = train_hgat(dataset, idx_tr, idx_va, idx_te, seed)
        rows.append(dict(model='PI-HGAT', seed=seed, fit_seconds=fit_s,
                         r2_train=m_tr['r2'], r2_test=m_te['r2'],
                         rmse=m_te['rmse'], mae=m_te['mae'], mape=m_te['mape']))
        for name, (b_tr, b_te, b_s) in train_baselines(X, Y, idx_tr, idx_va, idx_te, seed).items():
            rows.append(dict(model=name, seed=seed, fit_seconds=b_s,
                             r2_train=b_tr['r2'], r2_test=b_te['r2'],
                             rmse=b_te['rmse'], mae=b_te['mae'], mape=b_te['mape']))
        log(f'multiseed seed {seed}: PI-HGAT R2={m_te["r2"]:.4f} ({fit_s:.0f}s)')
    pd.DataFrame(rows).to_csv('results/step3_multiseed.csv', index=False)
    log(f'multiseed done ({len(list(seeds))} seeds x 4 models)', study_level=True)


def study_loso(ctx):
    X, Y, groups, combo_id, dataset = (ctx['X'], ctx['Y'], ctx['groups'],
                                       ctx['combo_id'], ctx['dataset'])
    delta_map = {g: X[groups == g][0, FEATURE_NAMES.index('Climate_DeltaT')]
                 for g in np.unique(groups)}
    rows = []
    for fold in sorted(np.unique(groups)):
        idx_te = np.where(groups == fold)[0]
        pool = np.where(groups != fold)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        i_tr, i_va = next(gss.split(X[pool], Y[pool], groups=combo_id[pool]))
        idx_tr, idx_va = pool[i_tr], pool[i_va]

        _, _, m_te, fit_s = train_hgat(dataset, idx_tr, idx_va, idx_te, seed=42)
        rows.append(dict(fold_scenario=fold, delta_t=delta_map[fold], model='PI-HGAT',
                         r2_test=m_te['r2'], rmse=m_te['rmse'], mae=m_te['mae'], mape=m_te['mape']))
        for name, (_, b_te, _) in train_baselines(X, Y, idx_tr, idx_va, idx_te, seed=42).items():
            rows.append(dict(fold_scenario=fold, delta_t=delta_map[fold], model=name,
                             r2_test=b_te['r2'], rmse=b_te['rmse'], mae=b_te['mae'], mape=b_te['mape']))
        log(f'LOSO fold {fold} (dT={delta_map[fold]:.2f}): PI-HGAT MAE={m_te["mae"]:.2f}')
    pd.DataFrame(rows).to_csv('results/step3_loso.csv', index=False)
    log('loso done (9 folds x 4 models)', study_level=True)


def study_loso_ext(ctx, folds=None):
    """LOSO with external replicate: for every scenario, hold it out of
    train/val ENTIRELY, then evaluate on three sets —
      main      its 250 MAIN rows (combos seen in training, climate unseen)
      external  its 150 seed-2810 rows (combos AND climate unseen)
      combined  both (n=400)
    The main-vs-external gap isolates the parameter-generalization cost from
    the climate cost, per climate. Generalizes the scenario-5-only S8
    extrapolation test to all 9 scenarios (no fixed scenario choice — the
    method section can state the protocol without hand-picking a climate)."""
    X, Y, groups, combo_id, dataset, builder = (
        ctx['X'], ctx['Y'], ctx['groups'], ctx['combo_id'],
        ctx['dataset'], ctx['builder'])
    delta_map = {g: X[groups == g][0, FEATURE_NAMES.index('Climate_DeltaT')]
                 for g in np.unique(groups)}

    ext_all = load_external_dataframe()
    ext_all['Scenario'] = ext_all['Scenario'].astype(str)

    def build_ext(sub_df):
        ds, xs, ys = [], [], []
        for _, row in sub_df.iterrows():
            s = row_to_params(row)
            d = builder.create_sample_graph(s)
            d.y = torch.tensor([[row['EUI_kWh_m2']]], dtype=torch.float)
            d.global_params = torch.tensor([[s[k] for k in FEATURE_NAMES]],
                                           dtype=torch.float)
            ds.append(d); xs.append([s[k] for k in FEATURE_NAMES])
            ys.append(row['EUI_kWh_m2'])
        return ds, np.array(xs), np.array(ys)

    def hgat_predict(model, ds):
        loader = PyGDL(ds, batch_size=TRAIN_PARAMS['batch_size'])
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(DEVICE)
                out = model(b.x_dict, b.edge_index_dict, b.batch_dict, b.global_params)
                ps.extend(out.cpu().numpy().flatten())
                ts.extend(b.y.cpu().numpy().flatten())
        return np.array(ts), np.array(ps)

    rows = []
    fold_list = folds or sorted(np.unique(groups))
    for fold in fold_list:
        idx_te = np.where(groups == fold)[0]
        pool = np.where(groups != fold)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        i_tr, i_va = next(gss.split(X[pool], Y[pool], groups=combo_id[pool]))
        idx_tr, idx_va = pool[i_tr], pool[i_va]

        ext_df = ext_all[ext_all['Scenario'] == fold]
        if not len(ext_df):
            log(f'loso_ext fold {fold}: no external replicate — skipped')
            continue
        ext_ds, ext_X, ext_Y = build_ext(ext_df)

        hgat, _, _, fit_s = train_hgat(dataset, idx_tr, idx_va, idx_te, seed=42)
        b_out, b_models = train_baselines(X, Y, idx_tr, idx_va, idx_te,
                                          seed=42, return_models=True)
        sc = b_models['scaler']
        Xm_s, Xe_s = sc.transform(X[idx_te]), sc.transform(ext_X)

        tm, pm_hgat = hgat_predict(hgat, [dataset[i] for i in idx_te])
        te_, pe_hgat = hgat_predict(hgat, ext_ds)
        b_models['ann'].eval()
        with torch.no_grad():
            pm_ann = b_models['ann'](torch.tensor(Xm_s, dtype=torch.float).to(DEVICE)).cpu().numpy().flatten()
            pe_ann = b_models['ann'](torch.tensor(Xe_s, dtype=torch.float).to(DEVICE)).cpu().numpy().flatten()
        preds = {
            'PI-HGAT':    (pm_hgat, pe_hgat),
            'XGBoost':    (b_models['xgb'].predict(Xm_s), b_models['xgb'].predict(Xe_s)),
            'ANN (MLP)':  (pm_ann, pe_ann),
            'Linear Reg': (b_models['lr'].predict(Xm_s), b_models['lr'].predict(Xe_s)),
        }
        y_main, y_ext = Y[idx_te], ext_Y
        for name, (p_main, p_ext) in preds.items():
            for eval_set, yt, yp in (('main', y_main, p_main),
                                     ('external', y_ext, p_ext),
                                     ('combined', np.concatenate([y_main, y_ext]),
                                                  np.concatenate([p_main, p_ext]))):
                m = metrics(yt, yp)
                rows.append(dict(fold_scenario=fold, delta_t=delta_map[fold],
                                 model=name, eval_set=eval_set, n=len(yt),
                                 r2=m['r2'], rmse=m['rmse'], mae=m['mae'],
                                 mape=m['mape']))
        r2c = [r for r in rows if r['fold_scenario'] == fold
               and r['model'] == 'PI-HGAT' and r['eval_set'] == 'combined'][0]['r2']
        log(f'loso_ext fold {fold} (dT={delta_map[fold]:.2f}): '
            f'PI-HGAT combined R2={r2c:.4f} ({fit_s:.0f}s)')
    pd.DataFrame(rows).to_csv('results/step3_loso_ext.csv', index=False)
    log(f'loso_ext done ({len(fold_list)} folds x 4 models x 3 eval sets)',
        study_level=True)


def study_combosplit(ctx):
    X, Y, combo_id, dataset = ctx['X'], ctx['Y'], ctx['combo_id'], ctx['dataset']
    idx_tr, idx_va, idx_te = combo_split(X, Y, combo_id)
    rows = []
    _, m_tr, m_te, fit_s = train_hgat(dataset, idx_tr, idx_va, idx_te, seed=42)
    rows.append(dict(model='PI-HGAT', r2_train=m_tr['r2'], r2_test=m_te['r2'],
                     rmse=m_te['rmse'], mae=m_te['mae'], mape=m_te['mape']))
    for name, (b_tr, b_te, _) in train_baselines(X, Y, idx_tr, idx_va, idx_te, seed=42).items():
        rows.append(dict(model=name, r2_train=b_tr['r2'], r2_test=b_te['r2'],
                         rmse=b_te['rmse'], mae=b_te['mae'], mape=b_te['mape']))
    pd.DataFrame(rows).to_csv('results/step3_combosplit.csv', index=False)
    log(f'combosplit done: PI-HGAT R2={m_te["r2"]:.4f} on unseen combos', study_level=True)


def study_learncurve(ctx, sizes=(25, 50, 100, 150, 200, 249), seeds=(42, 43, 44)):
    """Train on n combos/scenario (6 train scenarios), test on the held-out
    scenario — answers 'how many LHS runs per climate are enough?'.

    All 4 benchmark models (review fix: Linear Reg was silently missing from
    the old figure), multiple seeds for a mean ± range band, and the full
    TRAIN_PARAMS budget so small-n points measure data sufficiency rather
    than undertraining. The n subsets are nested (same shuffled combo order
    for every seed; seeds vary model init/minibatch order only)."""
    X, Y, groups, combo_id, dataset = (ctx['X'], ctx['Y'], ctx['groups'],
                                       ctx['combo_id'], ctx['dataset'])
    idx_tr_full, idx_va, idx_te = scenario_split(X, Y, groups)
    tr_combos = np.unique(combo_id[idx_tr_full])
    rng = np.random.RandomState(42)
    rng.shuffle(tr_combos)
    rows = []
    for n in sizes:
        keep = set(tr_combos[:n])
        idx_tr = idx_tr_full[np.isin(combo_id[idx_tr_full], list(keep))]
        for seed in seeds:
            _, _, m_te, fit_s = train_hgat(dataset, idx_tr, idx_va, idx_te, seed=seed)
            rows.append(dict(n_per_scenario=n, n_train=len(idx_tr), seed=seed,
                             model='PI-HGAT', r2_test=m_te['r2'],
                             rmse=m_te['rmse'], mae=m_te['mae']))
            b = train_baselines(X, Y, idx_tr, idx_va, idx_te, seed=seed)
            for name in ('XGBoost', 'ANN (MLP)', 'Linear Reg'):
                b_te = b[name][1]
                rows.append(dict(n_per_scenario=n, n_train=len(idx_tr), seed=seed,
                                 model=name, r2_test=b_te['r2'],
                                 rmse=b_te['rmse'], mae=b_te['mae']))
            log(f'learncurve n={n}/scenario seed {seed}: PI-HGAT R2={m_te["r2"]:.4f}')
    pd.DataFrame(rows).to_csv('results/step3_learncurve.csv', index=False)
    log('learncurve done', study_level=True)


def study_ablation(ctx, seeds=(0, 1, 2), lam=0.05):
    """Physics monotonicity loss on/off. Violation rate = share of test samples
    where a finite-difference perturbation moves EUI the physically wrong way."""
    X, Y, groups, dataset, samples, builder = (ctx['X'], ctx['Y'], ctx['groups'],
                                               ctx['dataset'], ctx['samples'], ctx['builder'])
    idx_tr, idx_va, idx_te = scenario_split(X, Y, groups)
    probe_idx = np.random.RandomState(0).choice(idx_te, size=100, replace=False)
    perturb = [('P1_Wall_U', +0.10, +1),    # U up   -> EUI must rise
               ('P5_COP',    +0.25, -1),    # COP up -> EUI must fall
               ('Climate_DeltaT', +0.50, +1)]  # dT up -> EUI must rise

    rows = []
    for variant, lam_v in (('mono_off', 0.0), ('mono_on', lam)):
        for seed in seeds:
            model, _, m_te, fit_s = train_hgat(dataset, idx_tr, idx_va, idx_te,
                                               seed=seed, lambda_mono=lam_v)
            viol = {}
            for key, d, sign in perturb:
                bad = 0
                for i in probe_idx:
                    p0 = dict(samples[i])
                    p1 = {**p0, key: p0[key] + d}
                    diff = predict_single(model, builder, p1) - predict_single(model, builder, p0)
                    if diff * sign < 0:
                        bad += 1
                viol[key] = bad / len(probe_idx)
            rows.append(dict(variant=variant, seed=seed, lambda_mono=lam_v,
                             r2_test=m_te['r2'], rmse=m_te['rmse'],
                             viol_wallU=viol['P1_Wall_U'], viol_cop=viol['P5_COP'],
                             viol_deltaT=viol['Climate_DeltaT']))
            log(f'ablation {variant} seed {seed}: R2={m_te["r2"]:.4f} '
                f'viol(U/COP/dT)={viol["P1_Wall_U"]:.2f}/{viol["P5_COP"]:.2f}/{viol["Climate_DeltaT"]:.2f}')
    pd.DataFrame(rows).to_csv('results/step3_ablation.csv', index=False)
    log('ablation done (mono on/off x 3 seeds)', study_level=True)


# ------------------------------------------------------------------ main --- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='all',
                    choices=['all', 'multiseed', 'loso', 'loso_ext', 'combosplit',
                             'learncurve', 'ablation'])
    ap.add_argument('--smoke', action='store_true',
                    help='tiny run (2 seeds, 2 LOSO folds, epochs_cap small) to validate code paths')
    args = ap.parse_args()

    log(f'START study={args.study} smoke={args.smoke} device={DEVICE}', study_level=True)
    builder = GraphBuilder(NEO4J_JSON_PATH)
    X, Y, groups, combo_id, samples = load_data()
    dataset = build_graphs(builder, samples, X, Y)
    ctx = dict(X=X, Y=Y, groups=groups, combo_id=combo_id,
               samples=samples, dataset=dataset, builder=builder)
    log(f'data ready: {X.shape[0]} rows, {len(np.unique(combo_id))} combos, '
        f'{len(np.unique(groups))} scenarios')

    # Provenance record so notebook S10 can assert artifacts match its config
    import json as _json
    meta = dict(scenario_split=SCENARIO_SPLIT, train_params=TRAIN_PARAMS,
                gnn_params=GNN_PARAMS, feature_names=FEATURE_NAMES,
                n_rows=int(X.shape[0]), n_combos=int(len(np.unique(combo_id))),
                smoke=bool(args.smoke), study=args.study,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'step3_meta.json'), 'w', encoding='utf-8') as f:
        _json.dump(meta, f, indent=2)

    if args.smoke:
        global train_hgat
        orig = train_hgat
        def fast(*a, **kw):
            kw['epochs_cap'] = 8; kw['patience'] = 8
            return orig(*a, **kw)
        train_hgat = fast

    todo = ([args.study] if args.study != 'all'
            else ['multiseed', 'combosplit', 'loso', 'loso_ext', 'learncurve', 'ablation'])
    failed = []
    for name in todo:
        try:
            if name == 'multiseed':
                study_multiseed(ctx, seeds=range(2) if args.smoke else range(10))
            elif name == 'loso':
                study_loso(ctx)
            elif name == 'loso_ext':
                study_loso_ext(ctx, folds=['Baseline', 'S4'] if args.smoke else None)
            elif name == 'combosplit':
                study_combosplit(ctx)
            elif name == 'learncurve':
                if args.smoke:
                    study_learncurve(ctx, sizes=(25, 249), seeds=(42,))
                else:
                    study_learncurve(ctx)
            elif name == 'ablation':
                study_ablation(ctx, seeds=(0,) if args.smoke else (0, 1, 2))
        except Exception as e:  # keep going; report at the end
            import traceback
            log(f'{name} FAILED: {e}', study_level=True)
            traceback.print_exc()
            failed.append(name)
    log('STEP3 FAILED: ' + ','.join(failed) if failed else 'STEP3 DONE', study_level=True)


if __name__ == '__main__':
    main()
