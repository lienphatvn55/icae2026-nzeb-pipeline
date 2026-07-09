# PI-HGAT: Physics-Informed Heterogeneous Graph Attention Network for Climate-Aware NZEB Retrofit

Reproducibility artifact for the ICAE2026 paper *(title / authors — to be filled in on
acceptance)*. The pipeline couples a knowledge-graph surrogate of a building energy model
with multi-objective retrofit optimization and explainability, evaluated for climate
extrapolation under CMIP6 2050s/2080s scenarios.

**One notebook runs the whole study end to end:** [`NZEB_PIPELINE_ICAE2026_v3.ipynb`](NZEB_PIPELINE_ICAE2026_v3.ipynb).
All paper numbers are collected in [`RESULTS_FOR_PAPER.md`](RESULTS_FOR_PAPER.md).

---

## What the pipeline does

1. **Knowledge graph** (`pi_hgat/graph_builder.py`): BIM/Neo4j → PyG `HeteroData`
   (5 node types: Zone, Envelope, Material, System, Climate; 5 edge types; 69 nodes / 356 edges).
2. **PI-HGAT surrogate** (`pi_hgat/models.py`, `physics_loss.py`): heterogeneous GAT predicting
   **gross site EUI** from 9 design + climate features, with a physics-consistency
   (monotonicity + bound) penalty.
3. **Benchmark & robustness** (`scripts/analysis/step3_robustness.py`): XGBoost / ANN / Linear
   baselines; multiseed, leave-one-scenario-out, held-out-combo, and **climate extrapolation**
   (scenario S4, ΔT = 4.47 °C, held out entirely from training/selection).
4. **Multi-objective optimization** (`pi_hgat/objectives.py`): NSGA-III over 9 integer retrofit
   levels, three objectives — net-import EUI, life-cycle cost (LCC), life-cycle emissions (LCE);
   Entropy-TOPSIS compromise selection and NZE assessment.
5. **Explainability**: GNNExplainer (node/edge importance + fidelity), SHAP, and partial-dependence
   physics validation.

## Key results (see `RESULTS_FOR_PAPER.md` for the full set)

- Surrogate accuracy: R² = **0.974** on an unseen-climate interpolation test (S2), and **0.90**
  on the held-out worst-case 2080s extrapolation (S4) — where **XGBoost collapses to R² = −0.87**.
- Linear Regression is competitive on interpolation (R² = 0.981); PI-HGAT's advantage is in
  **climate extrapolation** and **spatial explainability**, reported honestly throughout.
- Optimization: all Pareto solutions remain **below the net-zero target** under roof-limited PV —
  reported as an NZE *feasibility* finding, not an NZEB claim.

## Reproducibility

- Training is made **bitwise-deterministic** (`torch.use_deterministic_algorithms(True)` +
  `CUBLAS_WORKSPACE_CONFIG`) with a fixed `TRAIN_SEED = 2` (chosen as the representative draw of
  a 7-seed sweep against the 10-seed multiseed distribution). Two independent full runs produced
  identical key outputs.
- Determinism trades ~1.6× training / ~2.5× inference speed for exact reproducibility; the reported
  compute costs reflect this deterministic configuration.

### Run it

```bash
pip install -r requirements.txt          # see the file header for torch/PyG + CUDA notes
# Full pipeline (~30 min on GPU): run NZEB_PIPELINE_ICAE2026_v3.ipynb top to bottom, or:
jupyter nbconvert --to notebook --execute --inplace NZEB_PIPELINE_ICAE2026_v3.ipynb

# Heavy robustness studies (multiseed / LOSO / learning curve, hours on GPU):
python scripts/analysis/step3_robustness.py --study all     # --smoke for a fast smoke test
```

The notebook reads the pre-computed `results/step3_*.csv`; re-run the script above only to
regenerate them.

## Repository layout

```
NZEB_PIPELINE_ICAE2026_v3.ipynb   the pipeline (single canonical notebook)
RESULTS_FOR_PAPER.md              all numbers/tables for the manuscript
requirements.txt                  verified package versions
pi_hgat/                          config, graph builder, model, physics loss, objectives, data split
scripts/analysis/                 step3 robustness studies + fig_style (figure system)
scripts/data/                     jEPlus/LHS aggregation
data/                             inputs: aggregated LHS results, KG (Neo4j JSON),
                                  sourced cost/carbon registry (xlsx), weather (EPW), baseline E+ output
results/figures/                  all paper figures (PNG + PDF)
results/*.csv                     benchmark / robustness / Pareto / compute-cost tables
best_hgat_v2.pt                   trained surrogate checkpoint
```

## Data & licensing notes

- `data/jEPlus-LHS/**` raw EnergyPlus run outputs (≈57k files) are git-ignored; the **aggregated
  results CSV** used by the surrogate is tracked.
- Weather files are CMIP6 morphed EPWs for Ho Chi Minh City (ACCESS-CM2 / MRI-ESM2-0, SSP2-4.5 /
  SSP5-8.5, 2050s / 2080s) plus a TMYx baseline.
- Cost/carbon factors come from the sourced data registry (`data/**/ICAE2026_DataRegistry_P1-P9.xlsx`).

## Limitations (stated in the paper)

Single building archetype (ASHRAE 90.1 Medium Office) and location (Ho Chi Minh City); two GCMs;
PV self-consumption uses a documented heuristic (no hourly dispatch); the surrogate extrapolates
less well *downward* to the real-TMYx baseline (LOSO Baseline fold).
