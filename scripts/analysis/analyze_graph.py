import json, sys
sys.stdout.reconfigure(encoding='utf-8')

data = json.load(open('neo4j_query_table_data_2026-6-2.json', 'r', encoding='utf-8'))

# Extract all unique nodes
zones = {}
envelopes = {}
materials = {}
systems = {}
climates = {}

for r in data:
    # From nodes
    if 'Zone' in r['from_labels']:
        zones[r['from_id']] = r['from_props']
    if 'Envelope' in r['from_labels']:
        envelopes[r['from_id']] = r['from_props']
    # To nodes
    if 'Zone' in r['to_labels']:
        zones[r['to_id']] = r['to_props']
    if 'Envelope' in r['to_labels']:
        envelopes[r['to_id']] = r['to_props']
    if 'Material' in r['to_labels']:
        materials[r['to_id']] = r['to_props']
    if 'System' in r['to_labels']:
        systems[r['to_id']] = r['to_props']
    if 'Climate' in r['to_labels']:
        climates[r['to_id']] = r['to_props']

print(f"=== ZONES ({len(zones)}) ===")
for k, v in sorted(zones.items()):
    print(f"  id={k}: {v['name']}, floor={v['floor']}, area={v.get('area_m2','?')}, orient={v.get('orientation','?')}")

print(f"\n=== ENVELOPES ({len(envelopes)}) ===")
for k, v in sorted(envelopes.items()):
    print(f"  id={k}: {v['name']}, type={v['type']}, area={v.get('area_m2','?')}, tilt={v.get('tilt_deg','?')}, azimuth={v.get('azimuth_deg','?')}")

print(f"\n=== MATERIALS ({len(materials)}) ===")
for k, v in sorted(materials.items()):
    print(f"  id={k}: {v}")

print(f"\n=== SYSTEMS ({len(systems)}) ===")
for k, v in sorted(systems.items()):
    print(f"  id={k}: {v['name']}, type={v['type']}, cop={v.get('cop_cooling','?')}, floor={v.get('floor','?')}")

print(f"\n=== CLIMATES ({len(climates)}) ===")
for k, v in sorted(climates.items()):
    print(f"  id={k}: {v['scenario']}, dbt_mean={v.get('dbt_mean','?')}, delta_t={v.get('delta_t','?')}")

print(f"\n=== SUMMARY ===")
print(f"Total nodes: {len(zones) + len(envelopes) + len(materials) + len(systems) + len(climates)}")
print(f"  Zones: {len(zones)}, Envelopes: {len(envelopes)}, Materials: {len(materials)}, Systems: {len(systems)}, Climates: {len(climates)}")
print(f"Total edges: {len(data)}")
