import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR

# File paths
NEO4J_JSON_PATH = os.path.join(DATA_DIR, 'data', 'registry', 'neo4j_query_table_data_2026-6-2.json')

# Baseline building parameters (DOE Medium Office, HCMC)
BASELINE = {
    'floor_area': 4982.0,
    'eui_site': 122.1,
    'cooling_cop': 2.96,
    'lpd': 6.66,
    'wwr': 0.33,
}

# Retrofit parameter ladders — the ACTUAL simulated jEPlus values, L0 (baseline)
# -> Lmax. Derived from data/aggregated_LHS_results.csv (9 climates x 250 LHS,
# 2026-07-03). These are the single source of truth for surrogate features, the
# objectives' nearest-level cost/carbon mapping, AND the NSGA-III bounds — the
# surrogate must never be queried outside these ranges (out-of-distribution).
P1_LEVELS = [2.0825, 0.9048, 0.5780, 0.4246, 0.3355]  # wall U = 1/@@P1_Wall_R@@ (W/m2K)
P2_LEVELS = [0.2315, 0.2023, 0.1796, 0.1614, 0.1467]  # roof U = 1/@@P2_Roof_R@@
P3_LEVELS = [0.55, 0.63, 0.70, 0.77]                  # roof reflectance = 1 - @@P3_Roof_Abs@@
P4_LEVELS = [(2.8618, 0.219, 0.2409), (1.60, 0.19, 0.27), (1.50, 0.17, 0.24),
             (1.50, 0.15, 0.20), (1.50, 0.13, 0.17)]  # glazing (U, SHGC, VT)
P5_LEVELS = [3.3993, 3.65, 3.90, 4.15, 4.40]          # GROSS COP (net baseline 2.96; gross/net = 1.149, registry B8)
P6_LEVELS = [24.0, 24.5, 25.0, 25.5, 26.0]            # cooling setpoint (degC)
P7_LEVELS = [6.6636, 6.0, 5.3, 4.6, 4.0]              # LPD (W/m2)
P8_LEVELS = [0, 30, 60, 90, 120, 150]                 # PV (kWp) — not in E+; post-proc only
P9_LEVELS = [0, 30, 60, 90, 120, 150]                 # BESS (kWh) — not in E+; post-proc only
# GNN hyperparameters (tuned)
GNN_PARAMS = {
    'hidden_channels': 32,
    'num_layers': 2,
    'heads': 2,
    'dropout': 0.05,
}

# Training hyperparameters (tuned)
TRAIN_PARAMS = {
    'batch_size': 64,
    'epochs': 300,
    'lr': 5e-4,
    'weight_decay': 1e-5,
    'patience': 40,
}

# -------------------------------------------------------------------- #
#  PHASE 2: MOO NSGA-III — economics / LCA basis                        #
# -------------------------------------------------------------------- #

# Economic and Lifecycle assumptions — 20-yr / 8% REAL basis (progress_doc f2; registry-consistent).
# The whole sourced Data Registry (EPDs, B4=0 justifications, PV/BESS 1-replacement schedule) is
# built on a 20-year study period; the Kadric method (Eq.6-9) is retained but re-parameterised to it.
# "8% real" = constant (real) cash flows discounted at 8% => inflation/escalation terms are 0
# (an explicit real energy escalation, if desired, goes in inflation_oc; kept 0 to avoid double-counting).
ECONOMIC_PARAMS = {
    'discount_rate': 0.08,        # d  = 8% real (VN social discount rate; progress_doc f2)
    'project_life_yr': 20,        # n  = 20 years study period (registry / progress_doc basis)
    'elec_price_kwh': 0.137,      # p_el ($/kWh) — EVN QD 1279/QD-BCT before-VAT blended office (registry S9)
    'inflation_ic': 0.0,          # i_IC = 0 (real basis; escalation folded into the 8% real rate)
    'inflation_mc': 0.0,          # i_MC = 0 (real basis)
    'inflation_oc': 0.0,          # i_OC = 0 (real basis; set >0 only for explicit real energy escalation)
    'maintenance_rate': 0.01,     # p = 1% of initial investment spent on maintenance annually (Kadric Eq.9)
}

# Rooftop PV specific yield in HCMC (kWh generated per installed kWp per year)
# Registry B6: ~1,420 kWh/kWp/yr (tilt 10 deg, az 180, 14% loss); World Bank GSA / VietnamSolar
# cross-check 3.91 kWh/kWp/day = 1,427/yr. FINALIZE via a PVWatts run if time allows.
PV_SPECIFIC_YIELD = 1420.0

# --- Single PV/BESS energy-balance model (objectives.net_energy — the ONLY place
# PV/BESS touch energy). Zero-export dispatch (registry B7: no commercial rooftop
# feed-in tariff in VN after Decision 13/2020 expired): only SELF-CONSUMED PV
# offsets grid imports. Self-consumption fraction:
#   sc = min(1, base + bess_gain * min(1, BESS_kWh / mean_daily_PV_kWh))
# base 0.6  = daytime-office load/PV coincidence (occupied weekdays);
# bess_gain 0.4 = evening/weekend share a full-day battery can shift to on-site use.
# Heuristic pending an hourly dispatch run — documented as such in the paper.
PV_SELF_CONSUMPTION = {'base': 0.6, 'bess_gain': 0.4}

# VN grid emission factor (kgCO2eq/kWh) — MONRE Cong van 1726/2024 (2023 data; registry S4)
GRID_EMISSION_FACTOR = 0.6592

# Same service lives keyed by the short P-code used in the registry/LCC engine (years).
# Drives the number of replacements over the 20-yr study period (Kadric Eq.7 / module B4).
# At n=20: P5 (HVAC) 1x@yr15, P7 (LED) 1x@yr12, P9 (BESS) 1x@yr15; P1-P4/P8 => 0 replacements
# so windows/coating B4=0 is correct by construction (registry, 20-yr scope). NOTE: PV (P8=30)
# yields 0 replacements here, so the PV-inverter swap (registry B10/EC6: 1x@yr12, ~10-15% of PV
# cost) is NOT captured by this whole-component replacement loop -- a documented simplification.
LIFESPANS_SHORT = {
    'P1': 75,   # external wall insulation
    'P2': 75,   # roof insulation
    'P3': 20,   # cool-roof coating
    'P4': 20,   # glazing
    'P5': 15,   # DX / HVAC
    'P6': 60,   # cooling setpoint (operational only, no replacement)
    'P7': 12,   # LED lighting (~11.5 yr)
    'P8': 30,   # rooftop PV
    'P9': 15,   # BESS
}

# NOTE (2026-07-03 review fix B2): the old COST_FACTORS / LCA_FACTORS ladders were
# removed — they used stale design values (P1 1.07..0.29 etc.) that did not match the
# simulated P*_LEVELS and had leaked into the notebook as NSGA-III bounds. Cost and
# carbon now come EXCLUSIVELY from the Excel registry (LevelInventory) keyed L0..Ln,
# addressed by integer level indices — see objectives.levels_to_params().

# LCE (life-cycle emissions) system-boundary proxies, aligned to Kadric et al. 2026 Table 3
# (GLA 2022 whole-life-carbon guidance). Modules B1, B5, B7 are excluded.
LCA_PROXY = {
    'A4A5_pct_a1a3': 0.10,   # A4-A5 construction ~10% of A1-A3 production (no VN logistics data)
    'B2_per_m2': 10.0,       # B2 maintenance = max(10 kgCO2e/m2 conditioned area, ...)
    'B2_pct_a1a5': 0.01,     #                       ... or 1% of A1-A5), whichever greater
    'B3_pct_b2': 0.25,       # B3 repair = 25% of B2
    'C1_per_m2': 3.4,        # C1 demolition = 3.4 kgCO2e/m2 GIA
    'C2C4_pct_a1a3': 0.10,   # C2-C4 transport + waste processing proxy (~7% embodied in Kadric)
}

