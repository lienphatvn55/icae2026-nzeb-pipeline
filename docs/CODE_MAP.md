# CODE MAP — ICAE2026 PI-HGAT NZEB Pipeline

> Cập nhật 2026-07-03. Bản đồ toàn bộ file `.py` trong `CODE/`, phân loại theo Framework
> (`results/figures/0. FRAMEWORK DEMO.png`), kèm bản đồ nguồn gốc Fig3–Fig11.
> Điểm vào duy nhất của pipeline: **`NZEB_PIPELINE_ICAE2026.ipynb`** (14 sections = Part 0→3).

## 1. Trạng thái tổng quan

```
Framework block          Code thật sự chạy                     Trạng thái
──────────────────────   ───────────────────────────────────   ──────────────────────────────
PART 0 Data Input        pi_hgat/graph_builder.py (S1, S4)     ACTIVE
PART 0 Simulation        jEPlus (offline, ngoài repo) →        DONE (9×250 = 2,250 runs)
                         scripts/data/aggregate_lhs_results.py ACTIVE (mới, 2026-07-03)
PART 1 AI Prediction     pi_hgat/{models,physics_loss,train}   ACTIVE — CẦN CHẠY LẠI với 2,250
                                                                dòng data thật (kết quả cũ =
                                                                synthetic!)
PART 2 MOO               pi_hgat/{objectives,config} + S11-13  ACTIVE (20-yr/8% basis locked)
PART 3 XAI               scripts/analysis/run_explainer.py+S14 ACTIVE
Figures cho bài báo      scripts/analysis/publication_figures  ⚠ TOÀN BỘ MOCK — xem §3
```

## 2. Kiểm kê từng file

### 2.1 Package lõi `pi_hgat/` — ACTIVE (single source of truth)

| File | Vai trò | Ghi chú |
|---|---|---|
| `config.py` | MỌI hằng số: P-levels, kinh tế (20-yr, 8% real, $0.137/kWh), GRID_EMISSION_FACTOR 0.6592, PV 1420, PV_SELF_CONSUMPTION, lifespans | 2026-07-03: XÓA COST_FACTORS/LCA_FACTORS (thang lỗi thời — review B2); cost/carbon chỉ còn từ Excel registry |
| `graph_builder.py` | Neo4j JSON → PyG HeteroData (node/edge types, per-node features). Zone = 4 features (bỏ pv_share — PV không qua surrogate) | S1, S4 |
| `models.py` | PI-HGAT v2 + baseline models (global_dim=9) | S5, S8 |
| `physics_loss.py` | Physics-informed loss (indices theo 9 features gross-EUI) | S5 |
| `train.py` | Training loop (early stopping) | S6–7 |
| `objectives.py` | `levels_to_params()` (level index→giá trị vật lý, single source); `net_energy()` (MÔ HÌNH PV/BESS DUY NHẤT, zero-export self-consumption — review B1); f2 LCC (Kadrić Eq.6–9, OC theo import) + f3 LCE (EN 15978, B6 theo import); `assess_nze()` 4 lớp. LCC/LCA nhận GROSS EUI | S11–S13. PV/BESS vào đúng MỘT chỗ |
| `synthetic_data.py` | Sinh EUI giả lập theo vật lý | **DEPRECATED** — đã có data E+ thật. Chỉ giữ cho ablation/debug |

### 2.2 Data pipeline `scripts/data/` — ACTIVE

| File | Vai trò |
|---|---|
| `aggregate_lhs_results.py` | **(MỚI)** Quét `data/jEPlus-LHS/{1_Baseline,2..9}/LHS-*/eplustbl.csv` → `data/aggregated_LHS_results.csv` (2,250 dòng × 32 cột: tham số @@P*@@, Scenario, Climate_DeltaT, EUI, end-uses Cooling/Lighting/Equip/Fans/Pumps). Filter theo `Message` (cột `Errors` chỉ là warmup-convergence, không phải lỗi) |

### 2.3 Phân tích `scripts/analysis/`

| File | Trạng thái | Ghi chú |
|---|---|---|
| `publication_figures.py` | ⚠ **MOCK TOÀN BỘ** | Fig3–11 từ `generate_mock_pareto_data()` + số bịa. Fig3 còn vẽ mốc 60-yr (sai, nay là 20-yr). Xem §3 |
| `run_explainer.py` | ACTIVE | GNNExplainer standalone (bản script của S14) |
| `xai_explainer.py` | Bán trùng lặp | Phiên bản cũ hơn của run_explainer — hợp nhất về một file khi rà soát XAI |
| `analyze_graph.py` | One-off | Thống kê KG |
| `check_outputs.py`, `check_benchmark.py` | One-off | Đọc output notebook để kiểm tra nhanh |

### 2.4 Tiện ích notebook `scripts/utils/` — one-off, KHÔNG thuộc pipeline

`run_notebook.py`, `fix_nb.py`, `add_nsga3_cells.py`, `extract_nsga3.py`, `add_explain_cells.py`,
`extract_nb.py` — script chỉnh sửa/chạy notebook bằng code (lịch sử phát triển).
`extract_docx.py`, `extract_pdf.py`, `extract_pdf2.py`, `read_eplus_html.py`, `read_pnnl.py` —
trích xuất tài liệu tham khảo. Giữ nguyên, không cần bảo trì.

### 2.5 File rời ở root — CẦN DỌN

| File | Trạng thái | Hành động đề xuất |
|---|---|---|
| `extract_jeplus_results.py` | **SUPERSEDED** — chỉ aggregate 2/9 scenario, ít cột | Xóa hoặc chuyển vào `scripts/_archive/`; dùng `scripts/data/aggregate_lhs_results.py` |
| `calculate_lcc_lca.py` | **SUPERSEDED + NGUY HIỂM** — prototype của `objectives.py` nhưng hardcode hằng số lỗi thời (60-yr, d=2.5%, EF 0.72, $0.15) | Không chạy. Archive/xóa |
| `update_notebook.py` | One-off cũ | Archive |
| `test_obj.py` | Smoke test cho `objectives.py` | Giữ (nhẹ, hữu ích) |
| `data/jEPlus-LHS/0_Test/Midterm/**` | Coursework midterm (extractor cũ + surrogate 10k) | Archive, không thuộc paper |

## 3. Bản đồ Figures (results/figures/) — NGUỒN GỐC & KẾ HOẠCH THAY THẾ

**⚠ Fig3–Fig11 hiện tại đều do `scripts/analysis/publication_figures.py` sinh từ MOCK DATA
(random + hardcode) — KHÔNG được đưa vào bài báo.** Bố cục figure học theo
`docs/papers/2025_BEM with replaceable components using graph attention networks.pdf`.
Sau khi chạy end-to-end notebook với data thật, nguồn thay thế như sau:

| Figure | Hiện tại | Nguồn THẬT sau khi chạy end-to-end |
|---|---|---|
| Fig3_Lifespan | Mock + mốc 60-yr SAI | Vẽ từ `config.LIFESPANS_SHORT`, mốc 20-yr — có thể regenerate ngay |
| Fig4_LCEDistribution | Số bịa | `objectives.calculate_lca_breakdown()` @ max-level scenario |
| Fig5_PredictionPerf | Random scatter | **S10** notebook (actual vs predicted trên test set) |
| Fig6_NSGA3Evolution | Mock Pareto | **S12** (lưu history các generation) |
| Fig7_PairwisePareto | Mock | **S12/S13** Pareto set thật |
| Fig8_DecisionVars | Mock | **S13** phân bố biến quyết định trong Pareto set |
| Fig9_TOPSIS | Mock | **S13** front tô màu theo closeness coefficient |
| Fig10_OptimalLCE | Số bịa | `calculate_lca_breakdown()` @ nghiệm TOPSIS tốt nhất |
| Fig11_Heatmap | Mock | **S13** heatmap mức cải tạo của top solutions |
| `pareto_front_3d.png`, `gnn_explanation_subgraph.png`, `benchmark_results_v2.png` | Từ lần chạy synthetic cũ | Regenerate từ S12/S14 sau khi retrain |
| `page*_img*.png` | Ảnh trích từ PDF benchmark | Tham khảo, không phải kết quả |

## 4. Thứ tự chạy chuẩn (end-to-end validation)

1. *(đã xong)* jEPlus 9×250 → `data/jEPlus-LHS/{1_Baseline..9}/`
2. *(đã xong, 2026-07-03)* `python scripts/data/aggregate_lhs_results.py` → 2,250 dòng
3. Chạy notebook S1→S10: retrain PI-HGAT + baselines trên **data thật** (kết quả hiện lưu trong notebook là synthetic → mọi số R²/RMSE cũ vô hiệu)
4. S11→S13: NSGA-III + TOPSIS (f2/f3 dùng basis 20-yr đã chốt) — **lưu Pareto set ra `results/pareto_solutions.csv`** để figures tái lập được
5. S14: GNNExplainer trên nghiệm khuyến nghị
6. Viết lại `publication_figures.py` để đọc artifacts thật (bước 3–5) thay vì mock → xuất Fig3–11 vào `results/figures/`
