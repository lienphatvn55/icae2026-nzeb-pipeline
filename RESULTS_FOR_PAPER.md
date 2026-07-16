# RESULTS_FOR_PAPER — canonical numbers for the ICAE2026 manuscript

**Canonical pipeline: `NZEB_PIPELINE_ICAE2026_v3.ipynb`.**
Produced by a reproducible run: `TRAIN_SEED = 2`, deterministic mode on, every robustness study uses `lambda_mono = 0.05`.
The scientific results are bitwise-reproducible across independent runs (only wall-clock timing varies). Extracted 2026-07-09.

> This file supersedes any earlier number sheet built from the 2026-07-03 backup notebook, which described an older, different pipeline (Net-EUI target, a since-removed parameter ladder, inflated metrics).

---

## C1 · Surrogate benchmark (test = unseen climate scenarios)

**Test design:** the surrogate target is **gross site EUI** (features P1–P7 + climate; PV/BESS enter only at the MOO stage). Scenario-based split:
- Train (5 scenarios, ΔT 0–3.144): Baseline, S5, S6, S3, S8 → 1250 samples
- Val (2 scenarios): S7 (ΔT 1.611), S1 (ΔT 1.879) → 500 samples
- **Interpolation test: S2** (ACCESS-CM2 SSP2-4.5 2080s, ΔT 2.665) → 250 samples
- **Extrapolation test: S4** (ACCESS-CM2 SSP5-8.5 2080s, ΔT 4.472) — held out of train/val entirely

### Table 1a — Interpolation test S2 (n=250, single run, seed 2)
| Model | R² | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|
| **PI-HGAT** | 0.9741 | 0.9886 | 0.8008 | 0.66 |
| XGBoost | 0.6683 | 3.5394 | 3.5147 | 2.90 |
| ANN (MLP) | 0.9656 | 1.1398 | 0.9209 | 0.77 |
| Linear Reg | 0.9808 | 0.8525 | 0.6667 | 0.55 |

### Table 1b — Climate extrapolation S4 (n=400 = 250 held-out MAIN + 150 external seed-2810)
| Model | R² | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|
| **PI-HGAT** | **0.9017** | 2.2370 | 1.7997 | 1.34 |
| XGBoost | **−0.8716** | 9.7619 | 9.6639 | 7.19 |
| ANN (MLP) | 0.8934 | 2.3292 | 1.8451 | 1.37 |
| Linear Reg | 0.7696 | 3.4252 | 2.8402 | 2.07 |

> **Honest framing (required):** on interpolation, Linear Regression (0.981) matches/slightly exceeds PI-HGAT (0.974) — do NOT claim "baselines fail". PI-HGAT's advantage is in **(i) high-ΔT climate extrapolation** (XGBoost collapses to R²=−0.87 while PI-HGAT holds 0.90) and **(ii) spatial explainability (XAI)**. The abstract should say "R² = 0.97 interpolation, 0.90 on the unseen worst-case 2080s scenario", not "R² ≥ 0.99".

### Table 1c — Combo generalization (external, seen climate, n=300) — sanity check, NOT extrapolation
| Model | R² | RMSE | MAE |
|---|---|---|---|
| PI-HGAT | 0.9794 | 1.0114 | 0.7864 |
| XGBoost | 0.9919 | 0.6325 | 0.4981 |
| ANN (MLP) | 0.9385 | 1.7469 | 1.3433 |
| Linear Reg | 0.9799 | 0.9994 | 0.7957 |

---

## C2 · PI-HGAT training

| Quantity | Value |
|---|---|
| Model parameters | **43,329** |
| λ_mono (enabled) | **0.05** (`PhysicsLoss(lambda_bound=0.1, lambda_mono=0.05)`) |
| Optimizer | Adam (lr 5e-4, weight_decay 1e-5) + CosineAnnealingLR (T_max=300, eta_min 1e-6) + early stopping (patience 40) |
| Early-stopped at | **epoch 291** |
| Training time (GPU) | **~305 s** (multiseed mean 304.7 ± 118.6 s — the stable figure to report; single-run wall-clock varies with machine load, e.g. measured 311–397 s. Deterministic mode is ~1.6× slower than non-deterministic) |
| Best val loss (MSE) | **0.6066** |
| Batch size / epochs cap | 64 / 300 |
| Seed | TRAIN_SEED = 2 (representative; see §C2b) |

### C2b — Representative-seed selection (reproducibility)
Seeds alone are NOT enough to reproduce on GPU (GATConv scatter uses CUDA atomicAdd, whose float summation order varies run to run). Enabling `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` makes two independent runs bitwise-identical.
A 7-seed sweep against the 10-seed multiseed distribution (mean R²_test 0.9693 ± 0.0091) found **seed 2** closest to the mean → pinned. The notebook re-run gives R²=0.9741, matching the sweep prediction.

### C2c — Multiseed (10 seeds, λ_mono=0.05) → Fig P1.2
| Model | R²_test (mean ± σ) | RMSE (mean ± σ) | fit_s (mean ± σ) |
|---|---|---|---|
| **PI-HGAT** | 0.9693 ± 0.0091 | 1.067 ± 0.151 | 304.7 ± 118.6 |
| XGBoost | 0.6683 ± 0.0000 | 3.539 ± 0.000 | 0.3 ± 0.1 |
| ANN (MLP) | 0.9474 ± 0.0223 | 1.383 ± 0.288 | 11.7 ± 3.2 |
| Linear Reg | 0.9808 ± 0.0000 | 0.853 ± 0.000 | 0.0 ± 0.0 |

---

## C3 · Monotonicity validation (physics check) → Fig P3.5 (PDP, Section 14)

Section 14 (cell 77) sweeps each feature and reports the Spearman ρ between the feature and the prediction (|ρ|→1 = monotonic in the expected direction), for all 4 models.

| Feature (expected physical direction) | ANN | Linear Reg | PI-HGAT | XGBoost |
|---|---|---|---|---|
| Climate_ΔT (EUI ↑) | 1.00 | 1.00 | **1.00** | 0.960 |
| P1_Wall_U (EUI ↑) | 1.00 | 1.00 | **1.00** | 0.895 |
| P5_COP (EUI ↓) | −1.00 | −1.00 | **−1.00** | −0.982 |

PI-HGAT is perfectly monotonic in all 3 directions; XGBoost shows staircase/flat behavior (ρ departs from ±1).

### C3b — λ_mono ablation (3 seeds, including the S4 extrapolation set)
| variant | R²_test | RMSE | R²_extra (S4) | MAE_extra | viol (U/COP/ΔT) |
|---|---|---|---|---|---|
| mono_off (λ=0.0) | 0.9691 | 1.078 | 0.8855 | 1.953 | 0/0/0 |
| mono_on (λ=0.05) | 0.9633 | 1.162 | 0.8585 | 2.145 | 0/0/0 |

> **Required framing of "PI":** the monotonicity loss does NOT improve accuracy (both interpolation and S4 extrapolation differences are within seed noise), and the violation rate is 0 for BOTH variants. Describe "PI" as **a physics-consistency constraint verified at zero cost**, not as a source of performance gains.

---

## C3c · LOSO & additional robustness (λ_mono=0.05) → Fig P1.2, Fig P1.3, Fig P1.4

**LOSO test MAE (kWh/m²/yr) — each climate scenario held out in turn:**
| Fold (ΔT) | PI-HGAT | XGBoost | ANN | LR |
|---|---|---|---|---|
| Baseline (0.00) | 3.69 | 6.52 | 1.17 | 2.49 |
| S1 (1.88) | 1.02 | 0.35 | 0.62 | 0.81 |
| S2 (2.67) | 0.69 | 3.48 | 0.98 | 0.83 |
| S3 (2.18) | 0.80 | 1.90 | 1.10 | 0.80 |
| S4 (4.47) | 1.92 | 9.68 | 1.47 | 2.97 |
| S5 (1.27) | 0.64 | 1.87 | 1.28 | 0.68 |
| S6 (1.88) | 0.66 | 0.37 | 1.43 | 0.56 |
| S7 (1.61) | 0.67 | 1.63 | 1.11 | 0.64 |
| S8 (3.14) | 0.76 | 3.53 | 0.98 | 0.81 |

> Honest weak point: the **Baseline** fold (MAE 3.69) — PI-HGAT extrapolates DOWNWARD out of the ΔT range (to real TMYx weather) less well than LR/ANN. State this in Limitations.

**Combo-split (unseen combos, all scenarios):** PI-HGAT R²_test **0.9842** (train 0.9843 → no overfitting), MAE 0.962. XGBoost 0.9938, ANN 0.8938, LR 0.9811.

---

## C4 · NSGA-III

| Quantity | v3 value | (earlier prototype value — do not use) |
|---|---|---|
| Reference directions | Das–Dennis, p=16 → **153 directions** | "91 / 16 partitions" |
| Population size | **156** | 92 |
| Generations | **200** | 50 |
| Total evaluations | **31,200** | 4,600 |
| Inequality constraints | **0** (`n_ieq_constr=0`) — NO hard PMV constraint | — |
| Decision variables | 9 (integer level; P4 is one paired U+SHGC variable) | — |
| Runtime (baseline) | **499.3 s** (S12) | 29.4 s |
| Pareto solutions | **9** | 23 |
| Seed | 42 | 42 |
| Encoding | **integer level index** (`levels_to_params`); the surrogate is only ever queried at real jEPlus ladder values | — |

> PMV: describe per the code — the 26 °C setpoint ceiling of the design space is chosen to stay within the comfort band; do NOT call it a "hard constraint |PMV|≤0.5".

---

## C5 · Entropy-TOPSIS & compromise solution

| Quantity | v3 value | (earlier prototype — do not use) |
|---|---|---|
| Entropy weights (EUI/LCC/LCE) | **0.206 / 0.616 / 0.178** | 0.098/0.881/0.021 |
| Optimal package (P1…P9, integer level) | **L0 L0 L0 L0 L0 L4 L4 L5 L0** | L0…L2 L4 L5 L0 |
| Gross EUI | 114.59 kWh/m²/yr | — |
| **Net-import EUI** | **88.93 kWh/m²/yr** | 51.25 |
| Site-balance EUI (gross − PV_gen) | 71.83 kWh/m²/yr | — |
| Reduction vs baseline 122.1 | **27.2%** (net-import) | "58%" |
| NZE class | **Below target** (self-consumed RE 0.22) | "NZEB achieved" |
| LCC (20 yr, 8% real) | **$703,546** | $248,521 |
| LCE (WLC, 20 yr) | **6,085,330 kgCO₂eq** | 903,658 |
| Pareto solutions | 9 (all "Below target") | — |

Reading the optimal package: **keep the envelope unchanged (L0 wall/roof/glazing/cool-roof/HVAC)** + **raise the setpoint (L4=26 °C)** + **deepest LED (L4)** + **maximum roof PV (L5=150 kWp)** + **no BESS (L0)**. Message: operational measures (setpoint, LED) + roof PV are prioritized; envelope investment does not enter the compromise solution because LCC dominates (weight 0.616).

> **Required NZE framing:** with roof-limited PV (≤150 kWp, self-consumption ~0.22 under zero-export), **no solution reaches Net-Zero** — a valid finding. Title it "NZE feasibility assessment under roof-limited PV", do not promise "NZEB achieved".

---

## C6 · Derived numbers for the compromise solution (from `objectives.net_energy`)
- Floor area: 4,982 m² → gross demand = 114.59 × 4,982 = **570,911 kWh/yr**
- PV (P8=L5=150 kWp) × yield 1,420 kWh/kWp/yr = **213,000 kWh/yr**
- BESS (P9=L0=0) → self-consumption factor **sc = 0.60** (base, no BESS gain)
- Self-consumed = min(570911, 213000×0.60) = **127,800 kWh/yr**
- RE fraction (self-consumed / gross) = **0.224** ✓ (matches "0.22" in the output)
- Grid import = 570,911 − 127,800 = 443,111 kWh → net-import EUI = **88.94 kWh/m²/yr** ✓
- Site balance = (570,911 − 213,000) / 4,982 = **71.84 kWh/m²/yr** ✓

---

## C7 · GNNExplainer — top node features (mask score, on the compromise-solution graph) → Fig P3.1
| Node type | Top features (score) |
|---|---|
| Zone | height **0.687**, volume 0.686, area 0.648 |
| Envelope | tilt **0.742**, ShapeIndex 0.709, area 0.504 |
| System | COP **0.795**, Heat_SP 0.790, Cool_SP 0.779 |

> A "west-facing glazing dominant" narrative is NOT supported by this output — write per the table above.

### C7b — Explainer fidelity (deterministic, with matched-sparsity random controls) → Fig P3.4
| Metric | Deviation (kWh/m²/yr) | Random control |
|---|---|---|
| Fidelity+ (drop top-25% edges; necessity) | 0.37 | 0.28 ± 0.18 |
| Fidelity− (keep only top-25%; sufficiency) | 1.25 | 1.39 ± 0.65 |
| Drop ALL edges (total topology reliance) | 6.96 | — |

Necessity holds (Fid+ > control); sufficiency is limited (Fid− ≈ control) — reported honestly.

---

## C8 · SHAP (Fig P3.2)
The notebook saves `Fig_P3_2_SHAP.png/.pdf` but does not print numeric values. A numeric top-5 mean|SHAP| table would need a small extra `print` and a re-run; for now read the values from the figure when inserting it.

---

## C9 · Formulas & configuration from the source code

### B1 — Physics loss (`pi_hgat/physics_loss.py`)
Total: `L = L_MSE + λ_bound · L_bound + λ_mono · L_mono`, with λ_bound=0.1, λ_mono=0.05.
- **L_bound** (keep EUI within [10, 200] kWh/m²/yr): `mean(ReLU(10 − ŷ)) + mean(ReLU(ŷ − 200))`
- **L_mono** (gradient monotonicity ∂ŷ/∂x): for features that must make EUI **increase** (idx 0 Wall_U, 1 Roof_U, 3 Win_U, 4 SHGC, 8 ΔT): `Σ mean(ReLU(−∂ŷ/∂xᵢ))`; for features that must make EUI **decrease** (idx 5 COP): `Σ mean(ReLU(+∂ŷ/∂xᵢ))`. Gradients via `torch.autograd.grad(create_graph=True)`.

### B2 — PI-HGAT architecture (`pi_hgat/models.py`, `config.py`)
Heterogeneous GAT. Each node type has its own encoder `Linear(-1→32) → BatchNorm → ReLU`. Message passing: **2 `HeteroConv` layers**, each edge type uses **`GATConv`** (hidden 32, **2 heads**, `add_self_loops=False`, dropout 0.05), aggr='sum'; each layer is followed by `LayerNorm → ReLU → dropout → residual`. Pooling: **global mean pool** per node type, then concatenate (max-pool removed to reduce overfitting). Head MLP: `[pool_dim + 9 global_params] → 128 → 64 → 1` (ReLU, dropout 0.05). The 9 global_params are skip-connected into the head. GNN_PARAMS: hidden 32, layers 2, heads 2, dropout 0.05. TRAIN_PARAMS: lr 5e-4, wd 1e-5, epochs 300, patience 40, batch 64.

### B3 — LCC & LCE (`pi_hgat/objectives.py`)
- **LCC** (Kadric et al. 2026, Eq. 6–9): `LCC = IC + OC + MC`. IC = initial investment + discounted replacements (only components with service life < 20 yr; factor `((1+i)/(1+d))^year`). OC = present value of electricity cost on **grid imports** (imports only, zero-export): `annual_import_kwh × 0.137 $/kWh × pwf`, `pwf = (1−(1+r)^−n)/r`, r = (d−i)/(1+i). MC = 1%·IC_initial × pwf. d = 8% real, n = 20 yr, i = 0.
- **LCE** (Kadric Eq. 5, GLA 2022, Table 3): A1-A3 (registry) + A4-A5 (10% of A1-A3) + B2-B3 + B4 (registry) + **operational B6** (`import_kwh × 0.6592 kgCO₂e/kWh × 20`) + C1-C4. Excludes B1, B5, B7.
- **PV/BESS (the single place energy is touched):** zero-export, `sc = min(1, 0.6 + 0.4·min(1, BESS/daily_PV))`, self_consumed = min(load, PV·sc), import = gross − self_consumed. Yield 1,420 kWh/kWp/yr; EF_grid 0.6592.

### B4 — Node features (`pi_hgat/graph_builder.py`) — for the Nomenclature
- **Zone (4):** area (m²), volume (m³), height (m), LPD (W/m²)
- **Envelope (11):** area, tilt (°), azimuth (°), is_wall, is_roof, is_floor, is_window (one-hot), U-value (W/m²K), Reflectance, SHGC, ShapeIndex
- **Material (3):** conductance, U_mod, SHGC_mod
- **System (5):** cooling_cap, heating_cap, COP, Cool_SP (°C), Heat_SP (°C)
- **Climate (6):** dbt_mean, dbt_max, dbt_min, rh_mean, ghi_mean, Climate_ΔT (°C)

### B5 — Test scenarios
Interpolation test = **S2** (ACCESS-CM2 SSP2-4.5 2080s, ΔT 2.665). Extrapolation test = **S4** (ACCESS-CM2 SSP5-8.5 2080s, ΔT 4.472), held out entirely.

---

## C10 · Nomenclature (from notebook + code)

**Abbreviations:** NZEB (Net-Zero Energy Building), PI-HGAT (Physics-Informed Heterogeneous Graph Attention Network), HGAT, KG (Knowledge Graph), GNN, GAT (Graph Attention), MOO (Multi-Objective Optimization), NSGA-III, TOPSIS, XAI, SHAP, PDP (Partial Dependence Plot), LHS (Latin Hypercube Sampling), CMIP6, SSP, GCM, TMYx, EUI (Energy Use Intensity), LCC (Life-Cycle Cost), LCE (Life-Cycle Emissions), WLC (Whole-Life Carbon), NPV, EPD, EF (Emission Factor), COP, SHGC, LPD (Lighting Power Density), WWR, PV, BESS, LOSO (Leave-One-Scenario-Out).

**Symbols:** f1 (net-import EUI), f2 (LCC), f3 (LCE); λ_bound (=0.1), λ_mono (=0.05); ΔT (climate warming, °C); U (thermal transmittance, W/m²K); R² / RMSE / MAE / MAPE; d (discount rate 8%); n (study period 20 yr); sc (self-consumption fraction); ρ (Spearman).

> Keep only entries that actually appear. PMV appears only as setpoint rationale, not as a constraint — verify before including it in the Nomenclature.

---

## C11 · Figures in `results/figures/`
| File | Location | Suggested caption |
|---|---|---|
| Fig_P0_1_KGSchema | §3.4.1 | Heterogeneous KG meta-schema (5 node / 5 edge types) + case-study building |
| Fig_P2_1_Lifespan | §3.5 | Component service lives P1–P9 vs the 20-year study period |
| Fig_P2_2_LCEDistribution | §3.6 | Embodied LCE distribution by module |
| Fig_P1_1_PredictionPerf | §4.1 | Predicted vs actual, 4 models (test S2) |
| Fig_P1_2_BenchmarkRobustness | §4.1 | Multiseed R² (mean±σ) + boxplot + training time |
| Fig_P1_3_LearningCurve | §4.1 | Learning curve vs combos per scenario |
| Fig_P2_3_Pareto3D + Fig_P2_4_Convergence | §4.2 | 3D Pareto front + NSGA-III convergence |
| Fig_P2_5_Pairwise, Fig_P2_7_Heatmap | §4.2–4.3 | Pairwise Pareto colored by TOPSIS + heatmap |
| Fig_P2_6_LCEComparison | §4.3 | LCE by module: Baseline vs Optimal vs Max |
| Fig_P3_1_NodeImportance, Fig_P3_2_SHAP | §4.4 | Node importance + SHAP |
| Fig_P3_3_EdgeImportance, Fig_P3_4_SpatialExplanation | §4.4 | Edge importance + spatial explanation map + fidelity |
| Fig_P3_5_PartialDependence | §4.4 | PDP physics-monotonicity validation (4 models) |
| Fig_P0_2 – Fig_P0_4 | §3.x | Climate fingerprint (EUI shift, paired sensitivity, end-use) |
| Fig_P1_4_LOSOExternal | §4.1 | LOSO combined R² + MAE vs ΔT |
| Fig_P2_8_ClimateMOO | §4.x | Climate-aware MOO (median/worst 2080s) |
| FigS1–S3 | Appendix | Training loss, benchmark bar, compute cost |

---

### C11b · Manuscript figures (ICAE2026)
| Paper Fig | File | Generated by |
|---|---|---|
| Fig. 1 | `Paper_Fig1_Framework.png` | drawn externally |
| Fig. 2 | `Fig_P0_2_ClimateEUI` | Section 3 (violin of the 2,250-run gross-EUI corpus) |
| Fig. 3 | `Paper_Fig3_ExtrapolationBenchmark` | notebook Part 4 (R² bars + PI-HGAT scatter, withheld S4, n=400) |
| Fig. 4 | `Paper_Fig4_WorstPareto3D` | notebook Part 4 (worst-2080s front, entropy-TOPSIS star) |
| Fig. 5a | `Fig_P3_4_SpatialExplanation` | Section 15 |
| Fig. 5b | `Fig_P3_1_NodeImportance` | Section 13 (node importance, % of top feature) |
