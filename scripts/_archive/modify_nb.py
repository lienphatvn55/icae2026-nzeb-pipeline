import json

filename = "NZEB_PIPELINE_ICAE2026.ipynb"
with open(filename, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        for i, line in enumerate(cell["source"]):
            if "termination=('n_gen', 50)" in line:
                cell["source"][i] = line.replace("termination=('n_gen', 50)", "termination=('n_gen', 100)")
            if "pop_size=92" in line:
                cell["source"][i] = line.replace("pop_size=92", "pop_size=124")

with open(filename, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
