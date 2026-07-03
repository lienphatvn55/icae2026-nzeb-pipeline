# PROMPT — VIỆC CẦN LÀM TIẾP (handoff 2026-07-03, sau Step 1–5)

> Dán/đọc prompt này ở phiên làm việc tiếp theo. Ngữ cảnh đầy đủ:
> `docs/REVIEW_Q1_2026-07-03.md` (review + kế hoạch 6 bước), `docs/CODE_MAP.md`, `docs/FIGURE_PLAN.md`.
> Notebook chính: `NZEB_PIPELINE_ICAE2026.ipynb` (47 cells). Backup trước khi sửa:
> `NZEB_PIPELINE_ICAE2026.backup-2026-07-03.ipynb` (chứa bug B1–B4, chỉ để đối chiếu).

## 0. Xác nhận Step 3 đã chạy xong (BẮT BUỘC trước mọi việc)

1. Mở `results/logs/step3_progress.log`, kiểm tra dòng cuối là **`STEP3 DONE`**.
   - Nếu một study FAILED: chạy lại riêng study đó, ví dụ
     `python scripts/analysis/step3_robustness.py --study learncurve`
2. Kiểm tra 5 file artifacts có timestamp SAU 05:44 ngày 2026-07-03 (bản smoke cũ ghi lúc 05:23–05:29):
   `results/step3_multiseed.csv`, `step3_combosplit.csv`, `step3_loso.csv`,
   `step3_learncurve.csv`, `step3_ablation.csv`.

## 1. Chạy lại notebook từ đầu để kiểm chứng (việc anh định làm)

Restart kernel → **Run All** (~15 phút GPU). Checklist số liệu kỳ vọng:

| Chỗ kiểm tra | Giá trị kỳ vọng (±nhiễu seed nhỏ) |
|---|---|
| S2-3 | X (2250, 9); Gross EUI 95.1–150.4; 249 combos |
| S9 benchmark | PI-HGAT R² ≈ 0.976, RMSE ≈ 0.91; XGBoost thấp (~0.81) |
| S10b (sau khi Step 3 DONE) | combosplit PI-HGAT ≈ **0.98**; multiseed **0.9699 ± 0.0056**; LOSO MAE ~0.7–4.7 kWh/m²; bảng ablation KHÔNG còn R² âm |
| S12 | ~14 nghiệm Pareto (integer levels) |
| S13 TOPSIS | Gói: envelope L0 + P6=L2 + LED L4 + PV L5; Net-import ≈ 89; LCC ≈ $706k; cả front 'Below target' |
| S13b FigC5 | Net EUI 88.8 → ~101 → ~119 khi ΔT 0 → 2.03 → 4.47 |

⚠ Nếu bảng S10b vẫn hiện combosplit 0.315 / LOSO MAE 14 / ablation R² âm → đó là số SMOKE,
quay lại mục 0.

## 2. Sửa quirk FigC5 (front thưa) — ưu tiên cao, rẻ

Hiện gói tối ưu ở kịch bản nóng chọn P6 setpoint L0 (24°C) — phi logic vì setpoint cao hơn
dominate mọi mục tiêu khi chưa có ràng buộc tiện nghi. Nguyên nhân: front chỉ 14–15 nghiệm.
- Trong **S12** và **S13b**: đổi `termination=('n_gen', 50)` → `('n_gen', 100)`; `pop_size=92` → `124`
  (ref_dirs giữ das-dennis 12 partitions = 91 hướng; pop ≥ số hướng).
- Chạy lại S12→S13b, kiểm tra: P6 của gói TOPSIS không được GIẢM khi ΔT tăng.
- Nếu vẫn lạ: in cả front từng ΔT, kiểm tra dominance thủ công cột P6.

## 3. Đọc kết quả ablation và chốt claim "Physics-Informed"

Mở `results/step3_ablation.csv` (bản full):
- Nếu `mono_on` giảm violation rate rõ (viol_wallU/viol_cop ↓) mà R² không giảm đáng kể
  → giữ claim PI, ghi λ_mono=0.05 vào methods, cân nhắc bật trong training chính (S6-7).
- Nếu không có khác biệt → 2 lựa chọn trung thực: (a) tăng λ thử 0.1/0.2 rồi chạy lại
  `--study ablation`; (b) hạ claim thành "physics-guided loss (bound penalty)" trong bài.

## 4. Bảng cho bài báo (xuất từ artifacts, không gõ tay)

1. Benchmark chính: bảng S9 + cột train R², time từ `step3_multiseed.csv` (mean±σ).
2. MAE theo 9 kịch bản khí hậu: `step3_loso.csv` pivot (đã có sẵn trong S10b) — nhấn fold
   SSP585-2080s (ΔT 4.47).
3. Tổng quát hóa tham số: `step3_combosplit.csv`.
4. Sensitivity TOPSIS: bảng S13b (entropy/equal/cost/carbon).
5. Top-10 Pareto: bảng trong cell Fig10.
6. Fidelity GNNExplainer: bảng S14b.

## 5. (Tùy chọn — nếu còn thời gian trước deadline) Step 6: external test

1. jEPlus: tạo LHS mới **seed KHÁC** (~100–150 tổ hợp) × 3 EPW: `1_Baseline`, kịch bản ΔT≈2.0,
   kịch bản ΔT=4.47 (~300–450 runs).
2. Aggregate vào CSV riêng (sửa nhẹ `scripts/data/aggregate_lhs_results.py`, output
   `data/external_test_results.csv` — KHÔNG trộn vào file train).
3. Load surrogate đã train (`best_hgat_v2.pt`) → predict → thêm dòng "External test" vào
   bảng benchmark. Đây là bằng chứng mạnh nhất cho generalization.

## 6. Caveats PHẢI ghi trong bài (đã cài trong code, chỉ cần viết)

- PV/BESS self-consumption là heuristic `sc = 0.6 + 0.4·min(1, BESS/daily_PV)` (config
  `PV_SELF_CONSUMPTION`), zero-export theo bối cảnh VN — chưa mô phỏng dispatch theo giờ.
- PV-inverter replacement (~1 lần @ năm 12) chưa tính trong B4/IC (config LIFESPANS_SHORT note).
- Khí hậu mã hóa bằng ΔT scalar (delta-morphing nhiệt độ khô) — chưa xét biến đổi độ ẩm/bức xạ;
  hạn chế có chủ đích, nêu ở Limitations (quan trọng với khí hậu nóng-ẩm).
- Không có ràng buộc tiện nghi tường minh; thang setpoint bị chặn 24–26°C theo catalog mô phỏng.
- Kết quả chính sách: KHÔNG nghiệm nào đạt target giảm 50% demand (61 kWh/m²) với catalog
  hiện tại → NZE bất khả thi nếu chỉ dựa các biện pháp này + PV 150kWp → thảo luận NZEB-paradox
  (Fig10) + đề xuất mở rộng catalog (shading, daylighting, VRF...).

## 7. Dọn dẹp repo trước khi nộp (theo CODE_MAP §2.5)

- Archive: `calculate_lcc_lca.py` (hằng số lỗi thời NGUY HIỂM), `extract_jeplus_results.py`,
  `update_notebook.py`, các file fix_*.py/inject_*.py/patch_*.py ở root (script một lần).
- Giữ: notebook chính, backup, `pi_hgat/`, `scripts/`, `docs/`, `results/`, `test_obj.py`.

## 8. Figure còn thiếu (làm ngoài code)

- **Fig 1**: framework (đã có, vẽ ngoài).
- **Fig 2 panel (a)**: ảnh 3D DOE Medium Office + bản đồ HCMC (panel (b) KG schema đã
  programmatic — `Fig2_KGSchema.png`); ghép 2 panel trong PowerPoint/Illustrator.
