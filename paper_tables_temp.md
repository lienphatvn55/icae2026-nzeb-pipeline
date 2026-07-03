`	ext\n   100 |    12400 |     11 |  0.000000E+00 |             f
Optimization finished in 88.4s
Found 11 Pareto optimal solutions.
\n`\n\n`	ext\nObjective Weights (Entropy): EUI=0.192, LCC=0.644, LCE=0.163

===== OPTIMAL COMPROMISE SOLUTION (TOPSIS) =====
Levels: P1_Wall=L0, P2_Roof=L0, P3_CoolRoof=L0, P4_Glazing=L0, P5_HVAC=L0, P6_SetPt=L2, P7_LED=L4, P8_PV=L5, P9_BESS=L0
Net-import EUI: 88.63 kWh/m2/yr (gross 114.29, site balance 71.53)
LCC: $701,546 | LCE: 6,065,719 kgCO2eq
NZE class: Below target (self-consumed RE fraction 0.22)

Saved 11 solutions to results/pareto_solutions.csv
NZE_class
Below target    11
Name: count, dtype: int64
\n`\n\n`	ext\nSaved: results/figures/Fig8_Pareto3D.png|.pdf
\n`\n\n`	ext\nTop-10 Pareto solutions (sorted by TOPSIS closeness):
\n`\n\n`	ext\nTOPSIS weight sensitivity (baseline climate):
\n`\n\n`	ext\nMedian future +2.03°C: 11 Pareto solutions
\n`\n\n`	ext\nWorst 2080s +4.47°C: 12 Pareto solutions
\n`\n\n`	ext\nExplaining TOPSIS best solution: P1_Wall=L0, P2_Roof=L0, P3_CoolRoof=L0, P4_Glazing=L0, P5_HVAC=L0, P6_SetPt=L2, P7_LED=L4, P8_PV=L5, P9_BESS=L0
Running GNNExplainer (learning masks for mutual information)...
\n`\n