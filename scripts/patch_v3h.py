# -*- coding: utf-8 -*-
"""Patch #8: FigC3 Pumps note still collided with bar segment value labels
(axes-fraction y=0.60 landed inside the Equipment segment). Move it out of
the plot area entirely as a figure-level footnote below the x-axis."""
import json

NB = 'NZEB_PIPELINE_ICAE2026_v3.ipynb'
nb = json.load(open(NB, encoding='utf-8'))

old = ("""ax.set_ylabel('Mean end-use intensity (kWh/m²/yr)')\n"""
       """ax.set_title('End-use shift: cooling drives the climate-change penalty\\n(3 representative of 9 scenarios)')\n"""
       """ax.legend(prop={'size': 7}, loc='upper left')\n"""
       """# Note placed below the legend (upper-left alone collided with it)\n"""
       """ax.text(0.02, 0.60, f"Pumps ≈ {values5['Pumps'].mean():.3f} kWh/m²/yr in all 3 "\n"""
       """        "columns\\n(negligible, not visible at this scale)",\n"""
       """        transform=ax.transAxes, ha='left', va='top', fontsize=6.5,\n"""
       """        color='#6b6a63', style='italic')""")

new = ("""ax.set_ylabel('Mean end-use intensity (kWh/m²/yr)')\n"""
       """ax.set_title('End-use shift: cooling drives the climate-change penalty\\n(3 representative of 9 scenarios)')\n"""
       """ax.legend(prop={'size': 7}, loc='upper left')\n"""
       """plt.subplots_adjust(bottom=0.22)\n"""
       """# Figure-level footnote (outside the axes, below the x-axis) -- any position\n"""
       """# inside the plot area collides with a bar segment somewhere across the 3 columns.\n"""
       """fig.text(0.5, 0.02, f"Pumps \\u2248 {values5['Pumps'].mean():.3f} kWh/m\\u00b2/yr in all 3 columns "\n"""
       """        "(negligible, not visible at this scale)",\n"""
       """        ha='center', va='bottom', fontsize=6.5, color='#6b6a63', style='italic')""")

src = ''.join(nb['cells'][14]['source'])
assert old in src, 'marker not found'
src = src.replace(old, new)
nb['cells'][14]['source'] = src.splitlines(keepends=True)
json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

import ast
ast.parse(src)
print('patched cell 14, syntax OK')
