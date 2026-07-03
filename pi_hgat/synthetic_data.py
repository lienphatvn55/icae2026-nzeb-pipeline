"""
Synthetic Data Generator v2: Zone-aware physics-based EUI calculation.

Key fix: EUI formula uses ACTUAL zone-level envelope areas (wall/window/roof)
from the KG, creating genuine spatial correlations the GNN can learn.
Calibrated to E+ baseline = 122.1 kWh/m²/yr.
"""
import numpy as np
from scipy.stats import qmc
from .config import P1_LEVELS, P2_LEVELS, P3_LEVELS, P4_LEVELS
from .config import P5_LEVELS, P6_LEVELS, P7_LEVELS, P8_LEVELS, P9_LEVELS


class SyntheticDataGenerator:
    def __init__(self, graph_builder, num_samples=5000, seed=42):
        self.builder = graph_builder
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)

        # Extract zone topology from KG
        self.zones = graph_builder.get_zone_topology()
        self.total_floor_area = sum(z['area'] for z in self.zones)
        print(f"  Zones: {len(self.zones)}, Total floor area: {self.total_floor_area:.0f} m²")

        # Calibrate so baseline → 122.1 kWh/m²/yr
        baseline = dict(P1_Wall_U=1.07, P2_Roof_U=0.45, P3_Roof_Reflectance=0.30,
                        P4_Win_U=2.87, P4_Win_SHGC=0.22, P5_COP=2.96,
                        P6_Cool_SP=24., P7_LPD=6.66, P8_PV_kW=0., P9_BESS_kWh=0., Climate_DeltaT=0.)
        raw = self._raw_eui(baseline)
        self.cal = 122.1 / raw if raw > 0 else 1.0
        print(f"  Calibration factor: {self.cal:.4f}  (raw baseline={raw:.2f})")

    # ---------------------------------------------------------------- #
    #  LHS SAMPLING                                                     #
    # ---------------------------------------------------------------- #
    def _lhs_samples(self):
        sampler = qmc.LatinHypercube(d=10, seed=self.seed)
        raw = sampler.random(n=self.num_samples)

        def pick(col, levels):
            idx = np.clip(np.floor(col * len(levels)).astype(int), 0, len(levels)-1)
            return [levels[i] for i in idx]

        climate_deltas = [0., 1.879, 2.665, 2.179, 4.472, 1.27, 1.875, 1.611, 3.144]

        p4_idx = np.clip(np.floor(raw[:, 3] * len(P4_LEVELS)).astype(int), 0, len(P4_LEVELS)-1)
        cl_idx = np.clip(np.floor(raw[:, 9] * 9).astype(int), 0, 8)

        samples = []
        for i in range(self.num_samples):
            samples.append({
                'P1_Wall_U':          pick(raw[:, 0], P1_LEVELS)[i],
                'P2_Roof_U':          pick(raw[:, 1], P2_LEVELS)[i],
                'P3_Roof_Reflectance': pick(raw[:, 2], P3_LEVELS)[i],
                'P4_Win_U':           P4_LEVELS[p4_idx[i]][0],
                'P4_Win_SHGC':        P4_LEVELS[p4_idx[i]][1],
                'P5_COP':             pick(raw[:, 4], P5_LEVELS)[i],
                'P6_Cool_SP':         pick(raw[:, 5], P6_LEVELS)[i],
                'P7_LPD':             pick(raw[:, 6], P7_LEVELS)[i],
                'P8_PV_kW':           pick(raw[:, 7], P8_LEVELS)[i],
                'P9_BESS_kWh':        pick(raw[:, 8], P9_LEVELS)[i],
                'Climate_DeltaT':     climate_deltas[cl_idx[i]],
                'Climate_Idx':        int(cl_idx[i]),
            })
        return samples

    # ---------------------------------------------------------------- #
    #  ZONE-AWARE PHYSICS EUI                                           #
    # ---------------------------------------------------------------- #
    def _raw_eui(self, s):
        """Un-calibrated zone-aware EUI (kWh/m²/yr)."""
        if self.total_floor_area <= 0:
            return 100.

        solar_f = {'S': 1.3, 'E': 1.05, 'W': 1.1, 'N': 0.7, 'Core': 0.3}
        T_out = 28.3 + s['Climate_DeltaT']
        dT = max(0., T_out - s['P6_Cool_SP'])
        occ_hrs = 2600.  # ~10 h/d × 260 d/yr

        total = 0.
        for z in self.zones:
            sf = solar_f.get(z['orientation'], 0.5)
            # Conduction: walls
            q_wall = z['wall_area'] * s['P1_Wall_U'] * dT * 8.76       # kWh/yr
            # Conduction: roof (reflectance reduces solar gain on roof)
            ref_eff = (1. - s['P3_Roof_Reflectance']) / 0.7
            q_roof = z['roof_area'] * s['P2_Roof_U'] * dT * ref_eff * 8.76
            # Solar through windows
            q_solar = z['window_area'] * s['P4_Win_SHGC'] * sf * 1752.  # kWh/yr
            # Internal gains → cooling load (70 %)
            q_int = z['area'] * (s['P7_LPD'] + 11.38) * occ_hrs / 1000. * 0.7
            # Cooling energy
            cooling = (q_wall + q_roof + q_solar + q_int) / s['P5_COP']
            # Direct electricity
            lighting = z['area'] * s['P7_LPD'] * occ_hrs / 1000.
            plug = z['area'] * 11.38 * occ_hrs / 1000.
            total += cooling + lighting + plug

        pv = s['P8_PV_kW'] * 1500.   # kWh/yr in HCMC
        return (total - pv) / self.total_floor_area

    def calculate_physics_eui(self, s):
        eui = self._raw_eui(s) * self.cal
        eui += np.random.normal(0, 0.02 * abs(eui))
        return max(20., eui)

    # ---------------------------------------------------------------- #
    #  GENERATE FULL DATASET                                            #
    # ---------------------------------------------------------------- #
    def generate_dataset(self):
        samples = self._lhs_samples()
        X, Y = [], []
        for s in samples:
            X.append([s['P1_Wall_U'], s['P2_Roof_U'], s['P3_Roof_Reflectance'],
                      s['P4_Win_U'], s['P4_Win_SHGC'], s['P5_COP'],
                      s['P6_Cool_SP'], s['P7_LPD'], s['P8_PV_kW'], s['P9_BESS_kWh'],
                      s['Climate_DeltaT']])
            Y.append([self.calculate_physics_eui(s)])
        return samples, np.array(X), np.array(Y)
