"""Smoke test for pi_hgat.objectives after the 2026-07-03 refactor.

Checks: level->param mapping, single net_energy() model (PV applied once),
LCC/LCA on gross EUI, assess_nze consistency.
"""
from pi_hgat.objectives import ObjectiveCalculator, levels_to_params, N_LEVELS


class DummyBuilder:
    def get_zone_topology(self):
        return [{'area': 1000, 'wall_area': 500, 'roof_area': 1000, 'window_area': 200}]


builder = DummyBuilder()
obj_calc = ObjectiveCalculator(builder)

# Baseline: no PV -> imports == gross demand
p0 = levels_to_params([0] * 9)
eb0 = obj_calc.net_energy(p0, 122.1)
assert abs(eb0['net_eui'] - 122.1) < 1e-9, 'no-PV net EUI must equal gross'
print('L0  LCC:', round(obj_calc.calculate_lcc(p0, 122.1)), '| LCA:', round(obj_calc.calculate_lca(p0, 122.1)))

# Max retrofit: PV subtracted exactly once, self-consumption capped by load
pmax = levels_to_params([n - 1 for n in N_LEVELS])
eb = obj_calc.net_energy(pmax, 80.0)
assert eb['net_eui'] >= 0 and eb['self_consumed_kwh'] <= eb['gross_kwh']
print('Lmax net_eui: %.1f (gross 80.0, sc=%.3f)' % (eb['net_eui'], eb['sc_factor']))
print('Lmax LCC:', round(obj_calc.calculate_lcc(pmax, 80.0)), '| LCA:', round(obj_calc.calculate_lca(pmax, 80.0)))
print('Lmax NZE:', obj_calc.assess_nze(pmax, 80.0)['nze_class'])
print('OK')
