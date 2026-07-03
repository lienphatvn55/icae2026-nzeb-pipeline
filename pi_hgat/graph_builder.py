"""
Graph Builder v2: Neo4j JSON → PyG HeteroData with per-node differentiated features.

Key fix: Each envelope node gets TYPE-SPECIFIC parameters (wall→P1, roof→P2/P3, window→P4).
This creates genuine spatial variation that the GNN can learn through message passing.
"""
import json
import torch
from collections import Counter
from torch_geometric.data import HeteroData


class GraphBuilder:
    def __init__(self, json_path):
        self.json_path = json_path
        self._raw = self._load_json()
        self.nodes, self.edges = self._parse_graph()
        self._classify_envelopes()
        self._precompute_edge_indices()

    def _load_json(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _parse_graph(self):
        nodes = {t: {} for t in ('Zone', 'Envelope', 'Material', 'System', 'Climate')}
        edges = []
        for rec in self._raw:
            fl, tl = rec['from_labels'][0], rec['to_labels'][0]
            fi, ti = rec['from_id'], rec['to_id']
            nodes[fl].setdefault(fi, rec['from_props'])
            nodes[tl].setdefault(ti, rec['to_props'])
            edges.append(dict(src=(fl, fi), tgt=(tl, ti),
                              type=rec['rel_type'],
                              props=rec.get('rel_props', {})))
        return nodes, edges

    # ---------- Envelope classification ---------- #
    def _classify_envelopes(self):
        """wall / roof / floor / window via MADE_OF material + tilt."""
        env_mat = {}
        for e in self.edges:
            if e['type'] == 'MADE_OF':
                st, si = e['src']
                tt, ti = e['tgt']
                if st == 'Envelope' and tt == 'Material':
                    env_mat[si] = self.nodes['Material'][ti].get('name', '').upper()

        self.envelope_types = {}
        for eid in self.nodes['Envelope']:
            mat = env_mat.get(eid, '')
            tilt = self.nodes['Envelope'][eid].get('tilt_deg', 90.0)
            if 'WINDOW' in mat:
                self.envelope_types[eid] = 'window'
            elif 'SLAB' in mat or 'FLOOR' in mat:
                self.envelope_types[eid] = 'roof' if tilt < 45 else 'floor'
            else:
                self.envelope_types[eid] = 'wall'
        print(f"  Envelope types: {dict(Counter(self.envelope_types.values()))}")

    # ---------- Edge indices (shared topology) ---------- #
    def _precompute_edge_indices(self):
        self.id_maps = {nt: {} for nt in self.nodes}
        for nt, nd in self.nodes.items():
            for i, oid in enumerate(nd):
                self.id_maps[nt][oid] = i

        raw = {}
        for e in self.edges:
            st, si = e['src']
            tt, ti = e['tgt']
            key = (st, e['type'], tt)
            raw.setdefault(key, {'s': [], 't': []})
            raw[key]['s'].append(self.id_maps[st][si])
            raw[key]['t'].append(self.id_maps[tt][ti])

        self._edge_tensors = {}
        for key, idx in raw.items():
            self._edge_tensors[key] = torch.stack([
                torch.tensor(idx['s'], dtype=torch.long),
                torch.tensor(idx['t'], dtype=torch.long)])

    # ---------- Zone topology for physics EUI ---------- #
    def get_zone_topology(self):
        """Return list of dicts with zone areas + connected envelope areas by type."""
        solar_map = {'SOUTH': 'S', 'NORTH': 'N', 'EAST': 'E', 'WEST': 'W'}
        zones = []
        for zid, props in self.nodes['Zone'].items():
            name = props.get('name', '').upper()
            orient = 'Core'
            for key, val in solar_map.items():
                if key in name:
                    orient = val
                    break
            z = dict(id=zid, area=props.get('area_m2', 0.),
                     volume=props.get('volume_m3', 0.),
                     height=props.get('height_m', 0.),
                     orientation=orient,
                     floor=props.get('floor', ''),
                     wall_area=0., window_area=0., roof_area=0., floor_area=0.)
            # sum connected envelope areas by type
            for e in self.edges:
                if e['type'] == 'hasEnvelope' and e['src'] == ('Zone', zid):
                    eid = e['tgt'][1]
                    et = self.envelope_types.get(eid, 'wall')
                    ea = self.nodes['Envelope'][eid].get('area_m2', 0.)
                    z[f'{et}_area'] += ea
            zones.append(z)
        return zones

    # ---------- Per-sample HeteroData ---------- #
    def create_sample_graph(self, params):
        """
        Build HeteroData with TYPE-SPECIFIC parameter injection.

        PV/BESS (P8/P9) are deliberately NOT graph features: the surrogate models
        demand-side (gross) EUI only; supply enters via objectives.net_energy().

        Node feature dims:
          Zone:     4  [area, vol, height, lpd]
          Envelope: 11 [area, tilt, az, 4×onehot, u, refl, shgc, shape_index]
          Material: 3  [cond, u_mod, shgc_mod]
          System:   5  [cool_cap, heat_cap, cop, cool_sp, heat_sp]
          Climate:  6  [dbt_mean, dbt_max, dbt_min, rh, ghi, delta_t]
        """
        import math
        data = HeteroData()

        # --- Zone ---
        zf = []
        for zid, p in self.nodes['Zone'].items():
            zf.append([p.get('area_m2', 0.), p.get('volume_m3', 0.),
                       p.get('height_m', 0.), params['P7_LPD']])
        data['Zone'].x = torch.tensor(zf, dtype=torch.float)

        # --- Envelope (type-specific injection) ---
        ef = []
        for eid, p in self.nodes['Envelope'].items():
            et = self.envelope_types.get(eid, 'wall')
            ow = float(et == 'wall')
            orr = float(et == 'roof')
            of = float(et == 'floor')
            og = float(et == 'window')
            u = {'wall': params['P1_Wall_U'], 'roof': params['P2_Roof_U'],
                 'window': params['P4_Win_U'], 'floor': 0.5}[et]
            refl = params['P3_Roof_Reflectance'] if et == 'roof' else 0.
            shgc = params['P4_Win_SHGC'] if et == 'window' else 0.
            
            area = p.get('area_m2', 0.)
            perimeter = p.get('perimeter_m', None)
            if perimeter is None:
                if et in ['wall', 'window']:
                    h = 3.0
                    w = area / h if h > 0 else 0
                    perimeter = 2 * (h + w)
                else:
                    w = math.sqrt(area) if area > 0 else 0
                    perimeter = 4 * w
            
            # Shape Index (SI): Ratio of perimeter to the circumference of a circle with same area
            si = perimeter / (2 * math.sqrt(math.pi * area)) if area > 0 else 1.0

            ef.append([area, p.get('tilt_deg', 0.),
                       p.get('azimuth_deg', 0.), ow, orr, of, og, u, refl, shgc, si])
        data['Envelope'].x = torch.tensor(ef, dtype=torch.float)

        # --- Material ---
        mf = []
        for mid, p in self.nodes['Material'].items():
            nm = p.get('name', '').upper()
            if 'WINDOW' in nm:
                mf.append([params['P4_Win_U'], params['P4_Win_SHGC'], 0.])
            elif 'SLAB' in nm:
                mf.append([p.get('conductance', 3.84), params['P2_Roof_U'],
                           params['P3_Roof_Reflectance']])
            else:
                mf.append([p.get('conductance', 6.3), params['P1_Wall_U'], 0.])
        data['Material'].x = torch.tensor(mf, dtype=torch.float)

        # --- System ---
        sf = []
        for sid, p in self.nodes['System'].items():
            sf.append([p.get('cooling_cap_W', 50000.) / 1e5,
                       p.get('heating_cap_W', 10000.) / 1e5,
                       params['P5_COP'], params['P6_Cool_SP'],
                       p.get('heating_sp', 21.)])
        data['System'].x = torch.tensor(sf, dtype=torch.float)

        # --- Climate ---
        cf = []
        for cid, p in self.nodes['Climate'].items():
            cf.append([p.get('dbt_mean', 28.), p.get('dbt_max', 35.),
                       p.get('dbt_min', 22.), p.get('rh_mean', 75.),
                       p.get('ghi_mean', 200.), params['Climate_DeltaT']])
        data['Climate'].x = torch.tensor(cf, dtype=torch.float)

        # --- Edges (shared topology) ---
        for key, ei in self._edge_tensors.items():
            data[key].edge_index = ei.clone()

        return data

    # backward compat — baseline (L0) values from the TRUE simulated ladders
    def create_heterodata(self, modifications=None):
        return self.create_sample_graph(
            dict(P1_Wall_U=2.0825, P2_Roof_U=0.2315, P3_Roof_Reflectance=0.55,
                 P4_Win_U=2.8618, P4_Win_SHGC=0.219, P5_COP=3.3993,
                 P6_Cool_SP=24., P7_LPD=6.6636, Climate_DeltaT=0.))
