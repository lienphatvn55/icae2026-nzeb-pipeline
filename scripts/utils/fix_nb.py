import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 10: PI-HGAT Definition
src10 = nb['cells'][10]['source']
nb['cells'][10]['source'] = [s for s in src10 if 'total_params' not in s and 'Total trainable parameters' not in s]
# remove last \n from print(model)
if len(nb['cells'][10]['source']) > 0 and nb['cells'][10]['source'][-1] == 'print(model)\n':
    nb['cells'][10]['source'][-1] = 'print(model)'

# Cell 12: Training loop
src12 = nb['cells'][12]['source']
# insert parameter printing after dummy batch
insert_idx = src12.index('model(dummy_batch.x_dict, dummy_batch.edge_index_dict, dummy_batch.batch_dict)\n') + 1

src12.insert(insert_idx, '\n')
src12.insert(insert_idx+1, 'total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n')
src12.insert(insert_idx+2, 'print(f"\\nTotal trainable parameters: {total_params:,}")\n')

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
