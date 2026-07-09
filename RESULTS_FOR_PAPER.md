# RESULTS_FOR_PAPER — nguồn số liệu chuẩn cho bản thảo ICAE2026

**Pipeline chuẩn: `NZEB_PIPELINE_ICAE2026_v3.ipynb`** (KHÔNG phải backup 03/07).
Sinh từ lần chạy tái lập được: `TRAIN_SEED = 2`, deterministic mode bật, mọi robustness study dùng `lambda_mono = 0.05`.
Commit gốc: `9ce9bb7` (artifacts) + notebook seed-2. Ngày trích: 2026-07-09.

> ⚠️ **QUAN TRỌNG — file này thay thế `MASTER_NUMBERS_TU_NOTEBOOK.md`.** MASTER dựa trên backup 03/07 và mô tả một pipeline CŨ, KHÁC (Net-EUI target, ladder đã bị xóa vì sai, số bị thổi phồng). Xem §C12 để biết từng điểm lệch. **Không dùng số trong MASTER nữa.**

---

## C1 · Benchmark surrogate (test = kịch bản khí hậu chưa thấy)

**Thiết kế test của v3 (khác MASTER):** biến mục tiêu là **GROSS site EUI** (P1–P7 + khí hậu; PV/BESS chỉ vào ở tầng MOO). Chia theo kịch bản:
- Train (5 kịch bản, ΔT 0–3.144): Baseline, S5, S6, S3, S8 → 1250 mẫu
- Val (2 kịch bản): S7 (ΔT 1.611), S1 (ΔT 1.879) → 500 mẫu
- **Test nội suy: S2** (ACCESS-CM2 SSP2-4.5 2080s, ΔT 2.665) → 250 mẫu
- **Test ngoại suy: S4** (ACCESS-CM2 SSP5-8.5 2080s, ΔT 4.472) — giữ riêng hoàn toàn, không train/val

### Bảng 1a — Test nội suy S2 (n=250, single run seed 2)
| Model | R² | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|
| **PI-HGAT** | 0.9741 | 0.9886 | 0.8008 | 0.66 |
| XGBoost | 0.6683 | 3.5394 | 3.5147 | 2.90 |
| ANN (MLP) | 0.9656 | 1.1398 | 0.9209 | 0.77 |
| Linear Reg | 0.9808 | 0.8525 | 0.6667 | 0.55 |

### Bảng 1b — Ngoại suy khí hậu S4 (n=400 = 250 MAIN giữ riêng + 150 external seed-2810)
| Model | R² | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|
| **PI-HGAT** | **0.9017** | 2.2370 | 1.7997 | 1.34 |
| XGBoost | **−0.8716** | 9.7619 | 9.6639 | 7.19 |
| ANN (MLP) | 0.8934 | 2.3292 | 1.8451 | 1.37 |
| Linear Reg | 0.7696 | 3.4252 | 2.8402 | 2.07 |

> **Cách kể chuyện trung thực (bắt buộc):** ở nội suy, Linear Regression (0.981) ngang/nhỉnh hơn PI-HGAT (0.974) — KHÔNG được viết "baselines thất bại". Lợi thế của PI-HGAT nằm ở **(i) ngoại suy khí hậu ΔT cao** (XGBoost sụp đổ về R²=−0.87 còn PI-HGAT giữ 0.90) và **(ii) khả năng giải thích không gian (XAI)**. Abstract nên viết "R² = 0.97 nội suy, 0.90 ngoại suy kịch bản 2080s chưa thấy", KHÔNG viết "R² ≥ 0.99".

### Bảng 1c — Combo generalization (external, khí hậu đã thấy, n=300) — sanity check, KHÔNG phải ngoại suy
| Model | R² | RMSE | MAE |
|---|---|---|---|
| PI-HGAT | 0.9794 | 1.0114 | 0.7864 |
| XGBoost | 0.9919 | 0.6325 | 0.4981 |
| ANN (MLP) | 0.9385 | 1.7469 | 1.3433 |
| Linear Reg | 0.9799 | 0.9994 | 0.7957 |

---

## C2 · Huấn luyện PI-HGAT

| Đại lượng | Giá trị |
|---|---|
| Số tham số model | **43,329** |
| λ_mono (đã bật) | **0.05** (`PhysicsLoss(lambda_bound=0.1, lambda_mono=0.05)`) |
| Optimizer | Adam (lr 5e-4, weight_decay 1e-5) + CosineAnnealingLR (T_max=300, eta_min 1e-6) + early stopping (patience 40) |
| Dừng sớm tại | **epoch 291** |
| Thời gian train (GPU) | **~305 s** (multiseed mean 304.7 ± 118.6 s — số ổn định để báo cáo; wall-clock 1 lần chạy dao động theo tải máy, vd. các lần đo được 311–397 s. Deterministic mode ~1.6× chậm hơn non-deterministic) |
| Best val loss (MSE) | **0.6066** |
| Batch size / epochs cap | 64 / 300 |
| Seed | TRAIN_SEED = 2 (chọn đại diện; xem §C2b) |

### C2b — Chọn seed đại diện (tái lập)
Seeds đơn lẻ KHÔNG đủ để tái lập trên GPU (GATConv scatter dùng CUDA atomicAdd, thứ tự cộng float đổi mỗi lần chạy). Đã bật `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` → hai lần chạy độc lập cho output giống bitwise.
Sweep 7 seed so với phân bố multiseed 10-seed (mean R²_test 0.9693 ± 0.0091) → **seed 2** gần mean nhất → chốt. Notebook chạy lại cho R²=0.9741, khớp dự đoán sweep.

### C2c — Multiseed (10 seeds, λ_mono=0.05) → Fig. 6
| Model | R²_test (mean ± σ) | RMSE (mean ± σ) | fit_s (mean ± σ) |
|---|---|---|---|
| **PI-HGAT** | 0.9693 ± 0.0091 | 1.067 ± 0.151 | 304.7 ± 118.6 |
| XGBoost | 0.6683 ± 0.0000 | 3.539 ± 0.000 | 0.3 ± 0.1 |
| ANN (MLP) | 0.9474 ± 0.0223 | 1.383 ± 0.288 | 11.7 ± 3.2 |
| Linear Reg | 0.9808 ± 0.0000 | 0.853 ± 0.000 | 0.0 ± 0.0 |

---

## C3 · Kiểm chứng monotonicity (physics validation) → Fig. 15 (PDP, Section 15b)

**A5 trong prompt đã có sẵn trong v3** dưới dạng PDP Section 15b (cell 77): sweep từng đặc trưng, báo Spearman ρ giữa đặc trưng và dự đoán (|ρ|→1 = đơn điệu đúng hướng), cho cả 4 model.

| Đặc trưng (hướng vật lý kỳ vọng) | ANN | Linear Reg | PI-HGAT | XGBoost |
|---|---|---|---|---|
| Climate_ΔT (EUI ↑) | 1.00 | 1.00 | **1.00** | 0.960 |
| P1_Wall_U (EUI ↑) | 1.00 | 1.00 | **1.00** | 0.895 |
| P5_COP (EUI ↓) | −1.00 | −1.00 | **−1.00** | −0.982 |

PI-HGAT đơn điệu hoàn hảo cả 3 hướng; XGBoost có bậc thang/phẳng (ρ chệch khỏi ±1).

### C3b — Ablation λ_mono (3 seeds, có cả test S4 ngoại suy)
| variant | R²_test | RMSE | R²_extra (S4) | MAE_extra | viol (U/COP/ΔT) |
|---|---|---|---|---|---|
| mono_off (λ=0.0) | 0.9691 | 1.078 | 0.8855 | 1.953 | 0/0/0 |
| mono_on (λ=0.05) | 0.9633 | 1.162 | 0.8585 | 2.145 | 0/0/0 |

> **Framing bắt buộc cho "PI":** monotonicity loss KHÔNG cải thiện độ chính xác (cả nội suy lẫn ngoại suy S4 đều trong biên độ nhiễu seed), và violation rate = 0 ở CẢ hai biến thể. Viết "PI" như **ràng buộc nhất quán vật lý được xác minh với chi phí bằng 0**, KHÔNG phải "physics loss là nguồn cải thiện hiệu năng".

---

## C3c · LOSO & robustness bổ sung (λ_mono=0.05) → Fig. 6, Fig. 7, FigC4

**LOSO test MAE (kWh/m²/yr) — giữ từng kịch bản khí hậu ra ngoài:**
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

> Điểm yếu trung thực: fold **Baseline** (MAE 3.69) — PI-HGAT ngoại suy XUỐNG DƯỚI dải ΔT (về thời tiết TMYx thực) yếu hơn LR/ANN. Nêu rõ trong Limitations.

**Combo-split (unseen combos, tất cả kịch bản):** PI-HGAT R²_test **0.9842** (train 0.9843 → không overfit), MAE 0.962. XGBoost 0.9938, ANN 0.8938, LR 0.9811.

---

## C4 · NSGA-III

| Đại lượng | Giá trị v3 | Số trong MASTER (SAI — bỏ) |
|---|---|---|
| Reference directions | Das–Dennis, p=16 → **153 hướng** | "91 hướng / 16 partitions" |
| Population size | **156** | 92 |
| Số thế hệ | **200** | 50 |
| Tổng đánh giá | **31,200** | 4,600 |
| Ràng buộc bất đẳng thức | **0** (`n_ieq_constr=0`) — KHÔNG có hard constraint PMV | — |
| Số biến quyết định | 9 (integer level; P4 là 1 biến cặp U+SHGC) | — |
| Thời gian chạy (baseline) | **499.3 s** (S12) | 29.4 s |
| Số nghiệm Pareto | **9** | 23 |
| Seed | 42 | 42 |
| Encoding | **integer level index** (`levels_to_params`), surrogate chỉ query tại giá trị ladder jEPlus thật | — |

> PMV: viết theo code — trần setpoint 26 °C của không gian thiết kế được chọn để nằm trong vùng tiện nghi, KHÔNG gọi là "hard constraint |PMV|≤0.5".

---

## C5 · Entropy-TOPSIS & nghiệm thỏa hiệp

| Đại lượng | Giá trị v3 | MASTER (SAI — bỏ) |
|---|---|---|
| Trọng số entropy (EUI/LCC/LCE) | **0.206 / 0.616 / 0.178** | 0.098/0.881/0.021 |
| Gói tối ưu (P1…P9, integer level) | **L0 L0 L0 L0 L0 L4 L4 L5 L0** | L0…L2 L4 L5 L0 |
| Gross EUI | 114.59 kWh/m²/yr | — |
| **Net-import EUI** | **88.93 kWh/m²/yr** | 51.25 |
| Site balance EUI (gross − PV_gen) | 71.83 kWh/m²/yr | — |
| Giảm so baseline 122.1 | **27.2%** (net-import) | "58%" |
| Phân loại NZE | **Below target** (RE tự tiêu thụ 0.22) | "đạt NZEB" |
| LCC (20 năm, 8% real) | **$703,546** | $248,521 |
| LCE (WLC 20 năm) | **6,085,330 kgCO₂eq** | 903,658 |
| Số nghiệm Pareto | 9 (tất cả "Below target") | — |

Gói tối ưu dịch nghĩa: **giữ nguyên vỏ bao che (L0 tường/mái/kính/cool-roof/HVAC)** + **nâng setpoint (L4=26 °C)** + **LED sâu nhất (L4)** + **PV mái kịch trần (L5=150 kWp)** + **không BESS (L0)**. Thông điệp: measures vận hành (setpoint, LED) + PV mái được ưu tiên; đầu tư vỏ bao che không lọt nghiệm thỏa hiệp do LCC chi phối (trọng số 0.616).

> **Framing NZE bắt buộc:** với PV mái giới hạn (≤150 kWp, self-consumption ~0.22 do zero-export), **không nghiệm nào đạt Net-Zero** — đây là finding hợp lệ. Đặt tiêu đề là "đánh giá tính khả thi NZE dưới ràng buộc PV mái", KHÔNG hứa "đạt NZEB".

---

## C6 · Số dẫn xuất của nghiệm thỏa hiệp (tự tính từ `objectives.net_energy`)
- Diện tích sàn: 4,982 m² → gross demand = 114.59 × 4,982 = **570,911 kWh/yr**
- PV (P8=L5=150 kWp) × yield 1,420 kWh/kWp/yr = **213,000 kWh/yr**
- BESS (P9=L0=0) → hệ số tự tiêu thụ **sc = 0.60** (base, không có BESS gain)
- Self-consumed = min(570911, 213000×0.60) = **127,800 kWh/yr**
- RE fraction (self-consumed / gross) = **0.224** ✓ (khớp "0.22" trong output)
- Grid import = 570,911 − 127,800 = 443,111 kWh → net-import EUI = **88.94 kWh/m²/yr** ✓
- Site balance = (570,911 − 213,000) / 4,982 = **71.84 kWh/m²/yr** ✓

> Câu "self-consumed renewable fraction 0.22" trong bản Word CŨ vẫn ĐÚNG với v3 (0.224) — giữ, KHÔNG xóa như MASTER đề nghị (MASTER dùng pipeline khác).

---

## C7 · GNNExplainer — top node features (mask score, trên đồ thị nghiệm thỏa hiệp) → Fig. 12
| Node type | Top features (score) |
|---|---|
| Zone | height **0.687**, volume 0.686, area 0.648 |
| Envelope | tilt **0.742**, ShapeIndex 0.709, area 0.504 |
| System | COP **0.795**, Heat_SP 0.790, Cool_SP 0.779 |

> Narrative "west-facing glazing dominant" KHÔNG có bằng chứng trong output này (giống cảnh báo của MASTER) — viết theo bảng trên. **Lưu ý thêm:** MASTER liệt kê "Zone PV_share 0.718" nhưng v3 KHÔNG có feature PV_share ở Zone (Zone chỉ có 4 feature: area, volume, height, LPD) — đó là dấu hiệu MASTER thuộc graph structure cũ.

### C7b — Explainer fidelity (deterministic, có đối chứng random) → Fig. 14
| Metric | Deviation (kWh/m²/yr) | Random control |
|---|---|---|
| Fidelity+ (bỏ top-25% cạnh; necessity) | 0.37 | 0.28 ± 0.18 |
| Fidelity− (giữ chỉ top-25%; sufficiency) | 1.25 | 1.39 ± 0.65 |
| Bỏ TẤT CẢ cạnh (tổng phụ thuộc topology) | 6.96 | — |

Necessity giữ vững (Fid+ > đối chứng); sufficiency hạn chế (Fid− ≈ đối chứng) — báo cáo trung thực.

---

## C8 · SHAP (Fig. 12b)
Notebook lưu `Fig12b_SHAP.png/.pdf` nhưng **KHÔNG in giá trị số** trong output. Muốn có bảng top-5 mean|SHAP| dạng số cần thêm một `print` nhỏ và chạy lại — **CHƯA làm** (tránh scope creep; báo để bạn quyết). Hiện lấy trực tiếp từ hình khi chèn.

---

## C9 · Công thức & cấu hình từ source code (Part B)

### B1 — Physics loss (`pi_hgat/physics_loss.py`)
Tổng: `L = L_MSE + λ_bound · L_bound + λ_mono · L_mono`, với λ_bound=0.1, λ_mono=0.05.
- **L_bound** (giữ EUI trong [10, 200] kWh/m²/yr): `mean(ReLU(10 − ŷ)) + mean(ReLU(ŷ − 200))`
- **L_mono** (đơn điệu theo gradient ∂ŷ/∂x): với đặc trưng phải làm EUI **tăng** (idx 0 Wall_U, 1 Roof_U, 3 Win_U, 4 SHGC, 8 ΔT): `Σ mean(ReLU(−∂ŷ/∂xᵢ))`; với đặc trưng phải làm EUI **giảm** (idx 5 COP): `Σ mean(ReLU(+∂ŷ/∂xᵢ))`. Gradient tính bằng `torch.autograd.grad(create_graph=True)`.

### B2 — Kiến trúc PI-HGAT (`pi_hgat/models.py`, `config.py`)
Heterogeneous GAT. Mỗi loại node có encoder riêng `Linear(-1→32) → BatchNorm → ReLU`. Message passing: **2 lớp `HeteroConv`**, mỗi edge-type dùng **`GATConv`** (hidden 32, **2 heads**, `add_self_loops=False`, dropout 0.05), aggr='sum'; sau mỗi lớp `LayerNorm → ReLU → dropout → residual`. Pooling: **global mean pool** theo từng loại node rồi concat (max-pool đã bỏ để giảm overfit). Head MLP: `[pool_dim + 9 global_params] → 128 → 64 → 1` (ReLU, dropout 0.05). global_params (9) nối skip vào head. GNN_PARAMS: hidden 32, layers 2, heads 2, dropout 0.05. TRAIN_PARAMS: lr 5e-4, wd 1e-5, epochs 300, patience 40, batch 64.

### B3 — LCC & LCE (`pi_hgat/objectives.py`)
- **LCC** (Kadric et al. 2026, Eq. 6–9): `LCC = IC + OC + MC`. IC = đầu tư ban đầu + thay thế chiết khấu (chỉ thành phần có tuổi thọ < 20 năm; hệ số `((1+i)/(1+d))^year`). OC = PV chi phí điện trên **grid imports** (chỉ import, zero-export): `annual_import_kwh × 0.137 $/kWh × pwf`, `pwf = (1−(1+r)^−n)/r`, r = (d−i)/(1+i). MC = 1%·IC_initial × pwf. d = 8% real, n = 20 năm, i = 0.
- **LCE** (Kadric Eq. 5, GLA 2022, Table 3): gồm A1-A3 (registry) + A4-A5 (10% A1-A3) + B2-B3 + B4 (registry) + **B6 vận hành** (`import_kwh × 0.6592 kgCO₂e/kWh × 20`) + C1-C4. Loại trừ B1, B5, B7.
- **PV/BESS (nguồn duy nhất chạm energy):** zero-export, `sc = min(1, 0.6 + 0.4·min(1, BESS/daily_PV))`, self_consumed = min(load, PV·sc), import = gross − self_consumed. Yield 1,420 kWh/kWp/yr; EF_grid 0.6592.

### B4 — Node features (`pi_hgat/graph_builder.py`) — cho Nomenclature
- **Zone (4):** area (m²), volume (m³), height (m), LPD (W/m²)
- **Envelope (11):** area, tilt (°), azimuth (°), is_wall, is_roof, is_floor, is_window (one-hot), U-value (W/m²K), Reflectance, SHGC, ShapeIndex
- **Material (3):** conductance, U_mod, SHGC_mod
- **System (5):** cooling_cap, heating_cap, COP, Cool_SP (°C), Heat_SP (°C)
- **Climate (6):** dbt_mean, dbt_max, dbt_min, rh_mean, ghi_mean, Climate_ΔT (°C)

### B5 — Kịch bản test (v3, KHÁC MASTER)
v3 KHÔNG dùng "2 kịch bản test ngẫu nhiên". Test nội suy = **S2** (ACCESS-CM2 SSP2-4.5 2080s, ΔT 2.665). Test ngoại suy = **S4** (ACCESS-CM2 SSP5-8.5 2080s, ΔT 4.472), giữ riêng hoàn toàn.

---

## C10 · Nomenclature (quét notebook + code)

**Abbreviations:** NZEB (Net-Zero Energy Building), PI-HGAT (Physics-Informed Heterogeneous Graph Attention Network), HGAT, KG (Knowledge Graph), GNN, GAT (Graph Attention), MOO (Multi-Objective Optimization), NSGA-III, TOPSIS, XAI, SHAP, PDP (Partial Dependence Plot), LHS (Latin Hypercube Sampling), CMIP6, SSP, GCM, TMYx, EUI (Energy Use Intensity), LCC (Life-Cycle Cost), LCE (Life-Cycle Emissions), WLC (Whole-Life Carbon), NPV, EPD, EF (Emission Factor), COP, SHGC, LPD (Lighting Power Density), WWR, PV, BESS, LOSO (Leave-One-Scenario-Out).

**Symbols:** f1 (net-import EUI), f2 (LCC), f3 (LCE); λ_bound (=0.1), λ_mono (=0.05); ΔT (climate warming, °C); U (thermal transmittance, W/m²K); R² / RMSE / MAE / MAPE; d (discount rate 8%); n (study period 20 yr); sc (self-consumption fraction); ρ (Spearman).

> Chỉ giữ mục thực sự xuất hiện. Các từ trong MASTER nhưng KHÔNG có trong v3: IFC, IDF, BIM, PMV (PMV chỉ xuất hiện như diễn giải setpoint, không phải constraint) — kiểm lại trước khi đưa vào.

---

## C11 · Danh sách hình trong `results/figures/` (bộ FIGURE_PLAN hiện hành)
| File | Vị trí | Caption gợi ý |
|---|---|---|
| Fig2_KGSchema | §3.4.1 | Heterogeneous KG meta-schema (5 node / 5 edge types) + case-study building |
| Fig3_Lifespan | §3.5 | Tuổi thọ linh kiện P1–P9 vs kỳ phân tích 20 năm |
| Fig4_LCEDistribution | §3.6 | Phân bố embodied LCE theo module |
| Fig5_PredictionPerf | §4.1 | Predicted vs actual, 4 model (test S2) |
| Fig6_BenchmarkRobustness | §4.1 | R² multiseed (mean±σ) + boxplot + thời gian train |
| Fig7_LearningCurve | §4.1 | Learning curve theo số combo/kịch bản |
| Fig8_Pareto3D + Fig8b_Convergence | §4.2 | Mặt Pareto 3D + hội tụ NSGA-III |
| Fig9_Pairwise, Fig11_Heatmap | §4.2–4.3 | Pareto cặp đôi tô màu TOPSIS + heatmap |
| Fig10_LCEComparison | §4.3 | LCE theo module: Baseline vs Optimal vs Max |
| Fig12a_NodeImportance, Fig12b_SHAP | §4.4 | Tầm quan trọng node + SHAP |
| Fig13_EdgeImportance, Fig14_SpatialExplanation | §4.4 | Tầm quan trọng cạnh + bản đồ giải thích không gian + fidelity |
| Fig15_PartialDependence | §4.4 | PDP kiểm chứng đơn điệu vật lý (4 model) |
| FigC1–C3 | §3.x | Climate fingerprint (EUI shift, paired sensitivity, end-use) |
| FigC4_LOSOExternal | §4.1 | LOSO combined R² + MAE theo ΔT |
| FigC5_ClimateMOO | §4.x | MOO nhận biết khí hậu (median/worst 2080s) |
| FigS1–S3 | Phụ lục | Training loss, benchmark bar, compute cost |

---

## C12 · Các điểm LỆCH giữa MASTER (backup 03/07) và v3 — đã kiểm chứng
| Hạng mục | MASTER (03/07) | v3 (chuẩn) | Lý do v3 đúng |
|---|---|---|---|
| Biến mục tiêu | Net EUI | **Gross EUI** | PV/BESS chỉ vào ở tầng objectives; tránh double-count |
| Chia dữ liệu | 6/1/2, 2 test ngẫu nhiên | **5/2/1, S4 giữ riêng** | Test ngoại suy khí hậu thật, không data-snooping |
| λ_mono lúc chạy | 0.0 (đề nghị bật) | **0.05 (đã bật + ablation)** | Đã kiểm chứng, có ablation trên S4 |
| NSGA encoding | continuous (đề nghị đổi) | **integer level (đã có)** | Surrogate chỉ query giá trị ladder thật |
| Ladder P1 Wall U | {1.07…0.29} | **{2.08…0.34}** (jEPlus thật) | config.py: ladder cũ là "stale design values" đã XÓA |
| Zone features | 5 (có PV_share) | **4** (không PV_share) | Graph structure khác — MASTER thuộc bản cũ |
| Model params | 43,617 | 43,329 | Do khác feature set |
| PI-HGAT R²_test | 0.992 | 0.974 | Số cũ thổi phồng |
| XGBoost R²_test | 0.932 | 0.668 (fail extrapolation −0.87) | Câu chuyện cảnh báo có giá trị của v3 |
| TOPSIS Net EUI | 51.25 (đạt NZEB) | 88.93 (Below target) | v3 không đạt NZE với PV mái — finding hợp lệ |
| LCC / LCE | $248k / 904k | $704k / 6.09M | Khác registry + target |

---

## ⚠️ Việc còn treo (cần chạy lại notebook để đồng bộ hình)
Các bảng/hình đọc từ step3 CSV trong notebook (cell 44: Fig6, FigC4, bảng LOSO/combo/ablation) và compute-cost (cell 83) hiện được RENDER từ lần chạy TRƯỚC khi sửa `lambda_mono` cho step3. **Số trong file này đã là số CSV mới (đúng), nhưng file PNG Fig6/FigC4 và `computational_cost.csv` cần một lần chạy lại notebook để khớp.** Đang tiến hành ở branch `paper-consistency-rerun`.
