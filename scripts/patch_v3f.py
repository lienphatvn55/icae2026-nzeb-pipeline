# -*- coding: utf-8 -*-
"""Patch #6: FigC3 Pumps note collided with the legend (both at upper-left).
Move the note below the legend and reorder so legend is set first."""
import json

NB = 'NZEB_PIPELINE_ICAE2026_v3.ipynb'
nb = json.load(open(NB, encoding='utf-8'))

old = ("""ax.text(0.02, 0.98, f"Pumps ≈ {values5['Pumps'].mean():.3f} kWh/m²/yr in all 3 "\n"""
       """        "columns (negligible, not visible at this scale)",\n"""
       """        transform=ax.transAxes, ha='left', va='top', fontsize=6.5,\n"""
       """        color='#6b6a63', style='italic')\n"""
       """ax.set_ylabel('Mean end-use intensity (kWh/m²/yr)')\n"""
       """ax.set_title('End-use shift: cooling drives the climate-change penalty\\n(3 representative of 9 scenarios)')\n"""
       """ax.legend(prop={'size': 7}, loc='upper left')""")

new = ("""ax.set_ylabel('Mean end-use intensity (kWh/m²/yr)')\n"""
       """ax.set_title('End-use shift: cooling drives the climate-change penalty\\n(3 representative of 9 scenarios)')\n"""
       """ax.legend(prop={'size': 7}, loc='upper left')\n"""
       """# Note placed below the legend (upper-left alone collided with it)\n"""
       """ax.text(0.02, 0.60, f"Pumps ≈ {values5['Pumps'].mean():.3f} kWh/m²/yr in all 3 "\n"""
       """        "columns\\n(negligible, not visible at this scale)",\n"""
       """        transform=ax.transAxes, ha='left', va='top', fontsize=6.5,\n"""
       """        color='#6b6a63', style='italic')""")

src = ''.join(nb['cells'][14]['source'])
assert old in src, 'marker not found'
src = src.replace(old, new)
nb['cells'][14]['source'] = src.splitlines(keepends=True)
json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

import ast
ast.parse(src)
print('patched cell 14, syntax OK')
