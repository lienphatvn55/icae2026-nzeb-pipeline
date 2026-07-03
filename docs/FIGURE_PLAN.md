# FIGURE PLAN — ICAE2026 (chuẩn Q1, hệ màu xanh dương–trắng thống nhất)

> Cập nhật 2026-07-03. Hệ figure hợp nhất từ 2 paper lõi:
> **Kadrić 2026** (LCA+MOO lineup Fig6–11) × **GAT-BEM 2025** (XAI/graph Fig5–10) + 6 hình minh họa (a–f) của user.
> Style module: `scripts/analysis/fig_style.py`. Vị trí sản xuất từng hình: tìm "📊 FIGURE SLOT" trong notebook.

## 1. Design system (đã machine-validate colorblind-safety)

| Vai trò màu | Giá trị | Validation |
|---|---|---|
| 4 model (categorical, thứ tự CỐ ĐỊNH) | PI-HGAT `#2a78d6` xanh dương · XGBoost `#1baf7a` aqua · ANN `#eda100` vàng · LR `#4a3aa7` tím | ALL PASS (CVD ΔE≥21.6); aqua/vàng <3:1 trên nền trắng → **luôn ghi value label trực tiếp** |
| 5 module embodied LCA (ordinal, sáng→đậm theo vòng đời) | A1-A3 `#86b6ef` → A4-A5 `#5598e7` → B2-B3 `#2a78d6` → B4 `#1c5cab` → C1-C4 `#0d366b` | ALL PASS (--ordinal) |
| B6 operational | `#1baf7a` aqua (tách khỏi ramp xanh embodied) | — |
| Sequential (TOPSIS closeness, heatmap) | ramp xanh `#cde2fb→#0d366b` (thay viridis) | 1 hue, monotone |
| Diverging (SHAP ±) | xanh `#0d366b` ↔ đỏ `#7a1f1f`, midpoint xám `#f0efec` | warm/cool chuẩn |
| Emphasis | focus `#2a78d6` / context xám `#c3c2b7` | — |
| Chrome | grid hairline LIỀN `#e1e0d9`, spine `#c3c2b7`, bỏ top/right spine, chữ Arial 8–10pt, 600 dpi PNG+PDF | — |

Quy tắc cứng: không dual-axis; không rainbow; không số trên MỌI điểm (chỉ label chọn lọc); màu theo model cố định — lọc bớt model không đổi màu model còn lại.

## 2. Lineup figure cuối (14 hình) — nguồn dữ liệu & vị trí

| Fig | Nội dung | Part | Slot trong NB | Gốc | Trạng thái |
|---|---|---|---|---|---|
| 1 | Framework end-to-end | — | (đã có, vẽ ngoài) | — | ✅ |
| 2 | KG schema + case building (kiểu BCGS) | 0 | sau S1 | GAT-BEM Fig6 | ⬜ MỚI |
| 3 | Lifespan P1–P9 vs mốc 20 năm | 2 | sau S11 | Kadrić Fig6 | ⬜ vẽ ngay được (config thật) |
| 4 | Embodied LCE theo module @ max level, **B6 loại** | 2 | sau S11 | Kadrić Fig7 | ⬜ từ `calculate_lca_breakdown` |
| 5 | Pred-vs-Actual 4 panel (b) | 1 | sau S10 | Kadrić Fig3 + minh họa (b) | ⬜ chờ retrain |
| 6 | Benchmark 2×2: bars R²/RMSE/MAE/MAPE (a) + seed boxplot + train-vs-test + runtime (e) | 1 | sau S10 | GAT-BEM Table6 + minh họa (a)(e) | ⬜ cần code multi-seed/timing |
| 7 | Learning curve — đủ mẫu 250/62.500 (f) | 1 | sau S10 | minh họa (f) | ⬜ cần code |
| 8 | Pareto 3D + hypervolume convergence | 2 | sau S12 | Kadrić Fig4 (nâng cấp) | ⬜ cần `save_history=True` |
| 9 | Pairwise Pareto màu TOPSIS | 2 | sau S13 | Kadrić Fig9 (gộp Fig5) | ⬜ |
| 10 | LCE module: Baseline vs Optimal vs Max (gộp Fig7+10 cũ) | 2 | sau S13 | Kadrić Fig10 | ⬜ |
| 11 | Heatmap level cải tạo × Pareto set (xếp TOPSIS) | 2 | sau S13 | Kadrić Fig11 | ⬜ |
| 12 | Feature importance per node type + SHAP beeswarm (c)(d) | 3 | sau S14 | GAT-BEM Fig5–8 + minh họa (c)(d) | ⬜ |
| 13 | Edge-type importance (quan hệ không gian) | 3 | sau S14 | GAT-BEM Fig9 | ⬜ MỚI — contribution |
| 14 | Spatial explanation map (floor plan tô importance) + centrality scatter | 3 | sau S14 | GAT-BEM Fig10 | ⬜ MỚI — contribution |

### Hình BỎ / GỘP (và lý do)
- **Fig4 cũ "NSGA3Evolution"** (scatter early-gen mock) → BỎ; thay bằng **hypervolume theo generation** trong Fig8 — bằng chứng hội tụ định lượng, Q1 reviewer tin hơn.
- **Fig5 cũ "PairwisePareto"** (pairwise trơn + histogram chéo) → BỎ; trùng hoàn toàn với Fig9 (pairwise + màu TOPSIS mang nhiều thông tin hơn trên cùng số panel).
- **Fig8 cũ "DecisionVars"** (9 histogram) → chuyển thành **violin/strip 1 hàng** nếu cần, hoặc BỎ vì Fig11 heatmap đã thể hiện phân bố level trong Pareto set; giữ nếu muốn nói "biến nào linh hoạt/biến nào cứng" (Kadrić dùng ý này). ĐỀ XUẤT: giữ dạng gọn 1×9 violin, ghép làm panel (b) của Fig11.
- **Fig7 & Fig10 cũ** → GỘP một phần: Fig7 giữ vai trò methods (embodied @ max), Fig10 thành so sánh 3 cấu hình — tránh 2 hình bar gần giống nhau.

## 3. Mapping 6 hình minh họa (a–f) của user

| Minh họa | Vào figure | Ghi chú chỉnh |
|---|---|---|
| (a) bars R²/MAE/MAPE | Fig 6a | thêm RMSE; màu MODEL_COLORS; value label đậm |
| (b) 4-panel scatter | Fig 3 | ★ đánh dấu PI-HGAT (hero); mỗi model đúng màu cố định |
| (c) feature importance | Fig 12a | đổi từ GBR-importance sang GNNExplainer mask per node-type; GBR/SHAP làm đối chứng |
| (d) SHAP beeswarm | Fig 12b | dùng `div_cmap` xanh↔đỏ (thay hồng-xanh mặc định của shap) |
| (e) robustness boxplot | Fig 6b–d | thêm train-vs-test (overfit) + runtime log-scale |
| (f) learning curve | Fig 5 | trục x = số mẫu/scenario (25→250) × 9 climate; đánh dấu n=250 điểm chọn |

## 4. Đề xuất XAI bổ sung (câu hỏi 3 của user — minh bạch & bảo toàn không gian)

**Đã đưa vào lineup:** Fig13 (edge-type importance) và Fig14 (spatial map + centrality) — đây là 2 cách GAT-BEM chứng minh "GNN học được cấu trúc nhiệt-không gian", và là chỗ PI-HGAT ăn điểm so với XGBoost (XGBoost không có khái niệm edge).

**Nên thêm dạng BẢNG (rẻ mà tăng uy tín Q1):**
1. **Bảng fidelity/sparsity của GNNExplainer:** Fidelity+ (xóa subgraph quan trọng → prediction đổi bao nhiêu), Fidelity− (giữ nguyên subgraph → đổi ít), Sparsity. Định lượng chất lượng giải thích thay vì chỉ hình đẹp.
2. **Bảng benchmark có cột train/test tách riêng + thời gian** (đúng format Table 6 GAT-BEM) — chính là "Overfitting Check" user muốn, dạng bảng cho phần text.
3. **Error phân theo 9 kịch bản khí hậu** (MAE per scenario): chứng minh surrogate không degrade ở SSP585-2080s — độc đáo so với cả 2 paper lõi.

**Cân nhắc thêm nếu còn chỗ:** attention-weight trung bình theo edge type của chính HGAT (so với mask GNNExplainer — 2 nguồn evidence hội tụ thì rất thuyết phục); uncertainty (MC-dropout) trên Pareto front.

## 5. Việc code còn thiếu để hiện thực plan

1. S12: `minimize(..., save_history=True)` để có hypervolume (Fig8b).
2. S13: lưu `results/pareto_solutions.csv` (X, F, closeness) — marker đã nhắc.
3. S10: vòng multi-seed (10–20 seeds) + đo train time từng model (Fig4).
4. S10: learning-curve loop (Fig5).
5. S14: trích `edge_mask_dict` theo edge type (Fig13) + tọa độ zone cho spatial map (Fig14; lấy từ Neo4j JSON geometry).
6. Viết lại `publication_figures.py` → đọc artifacts thật + `fig_style.py` (sau khi 1–5 xong).
