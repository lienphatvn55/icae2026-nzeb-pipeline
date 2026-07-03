import numpy as np
import pandas as pd
import math
import os
from .config import (ECONOMIC_PARAMS, LIFESPANS_SHORT, GRID_EMISSION_FACTOR,
                     LCA_PROXY, PV_SPECIFIC_YIELD, PV_SELF_CONSUMPTION, BASELINE,
                     P1_LEVELS, P2_LEVELS, P3_LEVELS, P4_LEVELS,
                     P5_LEVELS, P6_LEVELS, P7_LEVELS, P8_LEVELS, P9_LEVELS)


def levels_to_params(levels, climate_delta_t=0.0):
    """Map 9 integer level indices (P1..P9, 0 = baseline/none) to physical parameters.

    Single source of the level->value mapping used by the NSGA-III problem, the
    figures, and XAI — so the surrogate is only ever queried at simulated jEPlus
    ladder values (no out-of-distribution), and P4 glazing stays a physical
    catalog pair (U, SHGC, VT) instead of two free-floating variables.
    """
    i1, i2, i3, i4, i5, i6, i7, i8, i9 = [int(v) for v in levels]
    u4, shgc4, _vt4 = P4_LEVELS[i4]
    return {
        'P1_Wall_U': P1_LEVELS[i1],
        'P2_Roof_U': P2_LEVELS[i2],
        'P3_Roof_Reflectance': P3_LEVELS[i3],
        'P4_Win_U': u4,
        'P4_Win_SHGC': shgc4,
        'P5_COP': P5_LEVELS[i5],
        'P6_Cool_SP': P6_LEVELS[i6],
        'P7_LPD': P7_LEVELS[i7],
        'P8_PV_kW': float(P8_LEVELS[i8]),
        'P9_BESS_kWh': float(P9_LEVELS[i9]),
        'Climate_DeltaT': float(climate_delta_t),
    }


N_LEVELS = [len(P1_LEVELS), len(P2_LEVELS), len(P3_LEVELS), len(P4_LEVELS),
            len(P5_LEVELS), len(P6_LEVELS), len(P7_LEVELS),
            len(P8_LEVELS), len(P9_LEVELS)]


class ObjectiveCalculator:
    def __init__(self, graph_builder):
        self.builder = graph_builder
        self.zones = self.builder.get_zone_topology()
        self.total_floor_area = sum(z['area'] for z in self.zones)
        
        # Load Data Registry
        # The script is run from CODE directory, so path is relative to it
        excel_path = os.path.join('data', 'jEPlus-LHS', 'ICAE2026_DataRegistry_P1-P9.xlsx')
        try:
            self.inventory = pd.read_excel(excel_path, sheet_name='LevelInventory')
            self._parse_inventory()
            self.use_dummy = False
        except Exception as e:
            print(f"Warning: Could not load {excel_path}. Using dummy data. Error: {e}")
            self.use_dummy = True
            
        # Continuous -> discrete level mapping. Single-sourced from config's
        # TRUE simulated jEPlus ladders (fixes the old P1 ladder that used the
        # stale design values 1.07..0.29 instead of the actual U = 1/R values
        # 2.08..0.34, which misassigned wall cost/carbon levels).
        self.LEVELS = {
            'P1': list(P1_LEVELS),
            'P2': list(P2_LEVELS),
            'P3': list(P3_LEVELS),
            'P4': [t[1] for t in P4_LEVELS],  # mapped by SHGC
            'P5': list(P5_LEVELS),
            'P6': list(P6_LEVELS),
            'P7': list(P7_LEVELS),
            'P8': list(P8_LEVELS),
            'P9': list(P9_LEVELS),
        }

    def _parse_inventory(self):
        col_A1_A3 = self.inventory.columns[5]
        col_B4 = self.inventory.columns[6]
        col_IC = self.inventory.columns[7]

        self.inv_dict = {}
        for _, row in self.inventory.iterrows():
            param = str(row['Param']).strip()
            lvl = str(row['Level']).strip()
            if param not in self.inv_dict:
                self.inv_dict[param] = {}
                
            def parse_float(v):
                # Registry cost cells are ranges like "10–17" (en-dash); take the mid-point.
                if pd.isna(v): return 0.0
                s = str(v).strip()
                if s in ('', 'NaN', 'n/a'): return 0.0
                try:
                    return float(s)
                except ValueError:
                    import re
                    parts = [p for p in re.split(r'[–—\-]', s) if p.strip()]
                    vals = []
                    for p in parts:
                        try: vals.append(float(p.strip()))
                        except ValueError: pass
                    return sum(vals) / len(vals) if vals else 0.0

            self.inv_dict[param][lvl] = {
                'A1_A3': parse_float(row[col_A1_A3]),
                'B4': parse_float(row[col_B4]),
                'IC': parse_float(row[col_IC])
            }

    def _get_nearest_level(self, param_key, value):
        levels = self.LEVELS[param_key]
        idx = np.argmin(np.abs(np.array(levels) - value))
        return f"L{idx}"

    # ---------------- Single PV/BESS energy-balance model ---------------- #
    def net_energy(self, params, gross_eui_kwh_m2):
        """THE one place PV (P8) / BESS (P9) touch energy (review fix B1).

        The surrogate predicts GROSS site EUI (demand side, P1-P7 + climate only);
        this converts it to grid imports under a ZERO-EXPORT dispatch (registry B7:
        no commercial rooftop feed-in tariff in VN): only self-consumed PV offsets
        imports. Self-consumption fraction (config.PV_SELF_CONSUMPTION):
            sc = min(1, base + bess_gain * min(1, BESS_kWh / mean_daily_PV_kWh))
        Self-consumption is also capped by the building's own load.

        Returns dict:
          gross_kwh          annual site demand (kWh)
          pv_gen_kwh         annual PV generation (kWh)
          sc_factor          self-consumption fraction of PV generation
          self_consumed_kwh  min(load, PV*sc)
          import_kwh         grid imports = gross - self_consumed
          net_eui            import_kwh / area  (>= 0; drives f1, OC, B6)
          net_balance_eui    (gross - PV_gen) / area, may be < 0 (site annual
                             balance incl. exports; drives NZE classification)
          re_fraction        self_consumed / gross
        """
        area = self.total_floor_area
        gross_kwh = max(0.0, gross_eui_kwh_m2) * area
        pv_gen_kwh = params.get('P8_PV_kW', 0.0) * PV_SPECIFIC_YIELD
        daily_pv = pv_gen_kwh / 365.0
        bess_ratio = (params.get('P9_BESS_kWh', 0.0) / daily_pv) if daily_pv > 0 else 0.0
        sc = min(1.0, PV_SELF_CONSUMPTION['base']
                 + PV_SELF_CONSUMPTION['bess_gain'] * min(1.0, bess_ratio))
        self_consumed_kwh = min(gross_kwh, pv_gen_kwh * sc)
        import_kwh = gross_kwh - self_consumed_kwh
        return {
            'gross_kwh': gross_kwh,
            'pv_gen_kwh': pv_gen_kwh,
            'sc_factor': sc,
            'self_consumed_kwh': self_consumed_kwh,
            'import_kwh': import_kwh,
            'net_eui': import_kwh / area,
            'net_balance_eui': (gross_kwh - pv_gen_kwh) / area,
            're_fraction': (self_consumed_kwh / gross_kwh) if gross_kwh > 0 else 0.0,
        }

    def get_base_costs_and_carbon(self, params):
        if self.use_dummy:
            raise ValueError("Excel registry not loaded. Cannot compute accurate LCC/LCA.")
            
        mapped = {
            'P1': self._get_nearest_level('P1', params.get('P1_Wall_U', self.LEVELS['P1'][0])),
            'P2': self._get_nearest_level('P2', params.get('P2_Roof_U', self.LEVELS['P2'][0])),
            'P3': self._get_nearest_level('P3', params.get('P3_Roof_Reflectance', self.LEVELS['P3'][0])),
            'P4': self._get_nearest_level('P4', params.get('P4_Win_SHGC', self.LEVELS['P4'][0])),
            'P5': self._get_nearest_level('P5', params.get('P5_COP', self.LEVELS['P5'][0])),
            'P6': self._get_nearest_level('P6', params.get('P6_Cool_SP', self.LEVELS['P6'][0])),
            'P7': self._get_nearest_level('P7', params.get('P7_LPD', self.LEVELS['P7'][0])),
            'P8': self._get_nearest_level('P8', params.get('P8_PV_kW', 0.0)),
            'P9': self._get_nearest_level('P9', params.get('P9_BESS_kWh', 0.0)),
        }
        
        # Multiply by total_floor_area since Excel factors are normalized per m2 floor area
        base_costs = {}
        base_a1a3 = {}
        base_b4 = {}
        
        for p_key, lvl in mapped.items():
            data = self.inv_dict.get(p_key, {}).get(lvl, {'IC': 0.0, 'A1_A3': 0.0, 'B4': 0.0})
            base_costs[p_key] = data['IC'] * self.total_floor_area
            base_a1a3[p_key] = data['A1_A3'] * self.total_floor_area
            base_b4[p_key] = data['B4'] * self.total_floor_area
            
        return base_costs, base_a1a3, base_b4

    def calculate_lcc_breakdown(self, params, gross_eui_kwh_m2):
        """LCC = IC + OC + MC over the building lifetime (Kadric et al. 2026, Eq. 6-9).

        IC  = initial investment + discounted replacements (Eq. 7)
        OC  = present value of operational energy cost on GRID IMPORTS only
              (Eq. 8; imports from the single net_energy() model — review fix B1)
        MC  = present value of routine maintenance, % of initial investment (Eq. 9)
        Present values use the Fisher real rate r=(d-i)/(1+i), since (1+i)/(1+d)=1/(1+r),
        so the discounted geometric sum over t=1..n equals (1-(1+r)^-n)/r.

        `gross_eui_kwh_m2` MUST be the surrogate's demand-side (gross) EUI —
        never a PV-netted value, or PV would be double counted.
        """
        base_costs, _, _ = self.get_base_costs_and_carbon(params)
        ic_initial = sum(base_costs.values())          # Eq. 7, first period (t=0, undiscounted)

        d = ECONOMIC_PARAMS['discount_rate']
        i_ic = ECONOMIC_PARAMS['inflation_ic']
        i_oc = ECONOMIC_PARAMS['inflation_oc']
        i_mc = ECONOMIC_PARAMS['inflation_mc']
        n = ECONOMIC_PARAMS['project_life_yr']
        maint_rate = ECONOMIC_PARAMS['maintenance_rate']

        # --- OC: operational energy cost of grid imports, present value (Eq. 8) ---
        annual_energy_kwh = self.net_energy(params, gross_eui_kwh_m2)['import_kwh']
        annual_cost = annual_energy_kwh * ECONOMIC_PARAMS['elec_price_kwh']
        r_oc = (d - i_oc) / (1 + i_oc)
        pwf_oc = n if r_oc == 0 else (1 - (1 + r_oc)**-n) / r_oc
        oc = annual_cost * pwf_oc

        # --- IC replacements: discounted re-investment over lifetime (Eq. 7, replacement periods) ---
        replacement_cost = 0.0
        for p_key, cost in base_costs.items():
            lifespan = LIFESPANS_SHORT.get(p_key)
            if not lifespan or cost <= 0:
                continue
            n_repl = int((n - 1) // lifespan)          # replacements strictly before end-of-life
            for rep in range(1, n_repl + 1):
                year = rep * lifespan
                replacement_cost += cost * ((1 + i_ic) / (1 + d)) ** year
        ic = ic_initial + replacement_cost

        # --- MC: routine maintenance as % of INITIAL investment, present value (Eq. 9) ---
        annual_mc = ic_initial * maint_rate
        r_mc = (d - i_mc) / (1 + i_mc)
        pwf_mc = n if r_mc == 0 else (1 - (1 + r_mc)**-n) / r_mc
        mc = annual_mc * pwf_mc

        return {'IC': ic, 'IC_initial': ic_initial, 'IC_replacement': replacement_cost,
                'OC': oc, 'MC': mc, 'Total': ic + oc + mc}

    def calculate_lcc(self, params, gross_eui_kwh_m2):
        return self.calculate_lcc_breakdown(params, gross_eui_kwh_m2)['Total']

    def calculate_lca_breakdown(self, params, gross_eui_kwh_m2):
        """LCE by LCA module over the building lifetime (Kadric et al. 2026, Eq. 5, Table 3).

        Included: A1-A3, A4-A5, B2, B3, B4, B6, C1-C4.  Excluded: B1, B5, B7.
        Embodied (A1-A3, B4) come from the product registry; the remaining modules use
        the GLA 2022 whole-life-carbon proxies stored in config.LCA_PROXY.
        """
        _, base_a1a3, base_b4 = self.get_base_costs_and_carbon(params)
        n = ECONOMIC_PARAMS['project_life_yr']

        # A1-A3 production (registry) + A4-A5 construction proxy
        a1_a3 = sum(base_a1a3.values())
        a4_a5 = a1_a3 * LCA_PROXY['A4A5_pct_a1a3']
        a1_a5 = a1_a3 + a4_a5

        # B2 maintenance = max(10 kgCO2e/m2 conditioned area, 1% of A1-A5); B3 repair = 25% of B2
        b2 = max(LCA_PROXY['B2_per_m2'] * self.total_floor_area,
                 LCA_PROXY['B2_pct_a1a5'] * a1_a5)
        b3 = LCA_PROXY['B3_pct_b2'] * b2
        b2_b3 = b2 + b3

        # B4 replacement embodied (registry; per-component replacement carbon already resolved)
        b4 = sum(base_b4.values())

        # B6 operational carbon: GRID IMPORTS x grid emission factor x lifetime
        # (imports from the single net_energy() model — review fix B1; gross in, once)
        annual_energy_kwh = self.net_energy(params, gross_eui_kwh_m2)['import_kwh']
        b6 = annual_energy_kwh * GRID_EMISSION_FACTOR * n

        # C1-C4 end of life: demolition + transport/waste-processing proxy
        c1 = LCA_PROXY['C1_per_m2'] * self.total_floor_area
        c2_c4 = LCA_PROXY['C2C4_pct_a1a3'] * a1_a3
        c1_c4 = c1 + c2_c4

        total = a1_a3 + a4_a5 + b2_b3 + b4 + b6 + c1_c4
        return {
            'A1-A3': a1_a3,
            'A4-A5': a4_a5,
            'B2-B3': b2_b3,
            'B4': b4,
            'B6': b6,
            'C1-C4': c1_c4,
            'Total': total
        }

    def calculate_lca(self, params, gross_eui_kwh_m2):
        return self.calculate_lca_breakdown(params, gross_eui_kwh_m2)['Total']

    def assess_nze(self, params, gross_eui_kwh_m2):
        """NZEB attainment for one solution, on top of the single net_energy() model.

        Classification uses the ANNUAL SITE BALANCE (gross - total PV generation,
        exports counted, may be < 0) per the common net-ZEB site definition, while
        cost/carbon elsewhere use imports only (zero-export dispatch) — both come
        from the same net_energy() call, stated explicitly in the paper.

        Classes (demand target = 50% cut vs baseline EUI, i.e. <= 61 kWh/m2/yr):
          'Net-Zero (site balance)'  gross - PV_gen <= 0 over the year
          'Nearly-ZEB'               demand target met AND self-consumed RE >= 50%
          'NZEB-ready (demand)'      demand target met, RE fraction < 50%
          'Below target'             demand-side EUI still above the 50%-cut target
        """
        demand_target = BASELINE['eui_site'] * 0.5          # 61.05 kWh/m2/yr
        eb = self.net_energy(params, gross_eui_kwh_m2)

        if eb['net_balance_eui'] <= 0:
            nze_class = 'Net-Zero (site balance)'
        elif gross_eui_kwh_m2 <= demand_target and eb['re_fraction'] >= 0.5:
            nze_class = 'Nearly-ZEB'
        elif gross_eui_kwh_m2 <= demand_target:
            nze_class = 'NZEB-ready (demand)'
        else:
            nze_class = 'Below target'

        return {'gross_eui': gross_eui_kwh_m2, 'net_eui': eb['net_eui'],
                'net_balance_eui': eb['net_balance_eui'],
                'pv_gen_kwh': eb['pv_gen_kwh'], 're_fraction': eb['re_fraction'],
                'demand_target': demand_target, 'nze_class': nze_class}
