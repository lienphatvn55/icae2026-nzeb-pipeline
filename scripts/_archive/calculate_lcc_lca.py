import pandas as pd
import numpy as np
import random
import math

random.seed(42)

def calculate_lcc_lca():
    # Base params
    d = 0.025
    project_life = 60
    elec_price = 0.15
    grid_emission_factor = 0.72
    i_IC = 0.035
    i_MC = 0.035
    i_OC = 0.040
    p_MC = 0.01  # 1% annual maintenance
    total_floor_area = 4982.0

    LIFESPANS = {
        'P1': 75, 'P2': 75, 'P3': 20, 'P4': 20, 'P5': 15,
        'P6': 60, 'P7': 11.5, 'P8': 30, 'P9': 15
    }

    # As per Table 3.5
    P1_LEVELS = [1.07, 0.64, 0.46, 0.36, 0.29]
    P2_LEVELS = [0.22, 0.19, 0.17, 0.16, 0.14] 
    P3_LEVELS = [0.55, 0.63, 0.70, 0.77]       
    P4_LEVELS = [0.219, 0.19, 0.17, 0.15, 0.13] # using SHGC
    P5_LEVELS = [3.40, 3.65, 3.90, 4.15, 4.40] 
    P6_LEVELS = [24.0, 24.5, 25.0, 25.5, 26.0]
    P7_LEVELS = [6.66, 6.0, 5.3, 4.6, 4.0]
    P8_LEVELS = [0, 30, 60, 90, 120, 150]
    P9_LEVELS = [0, 30, 60, 90, 120, 150]

    def closest_level(val, levels):
        idx = np.argmin(np.abs(np.array(levels) - val))
        return f"L{idx}"

    print("Loading Data Registry...")
    excel_path = 'data/jEPlus-LHS/ICAE2026_DataRegistry_P1-P9.xlsx'
    inventory = pd.read_excel(excel_path, sheet_name='LevelInventory')
    
    col_A1_A3 = inventory.columns[5]
    col_B4 = inventory.columns[6]
    col_IC = inventory.columns[7]

    inventory_dict = {}
    for _, row in inventory.iterrows():
        param = str(row['Param']).strip()
        lvl = str(row['Level']).strip()
        if param not in inventory_dict:
            inventory_dict[param] = {}
            
        a1a3 = row[col_A1_A3]
        b4 = row[col_B4]
        ic = row[col_IC]
        
        # safely convert to float
        def parse_float(v):
            if pd.isna(v): return 0.0
            s = str(v).strip()
            if s == 'NaN' or s == '': return 0.0
            try:
                return float(s)
            except:
                return 0.0

        inventory_dict[param][lvl] = {
            'A1_A3': parse_float(a1a3),
            'B4': parse_float(b4),
            'IC': parse_float(ic)
        }

    print("Loading LHS Results...")
    df = pd.read_csv('data/aggregated_LHS_results.csv')
    
    net_eui_list = []
    lce_list = []
    lcc_list = []
    p8_list = []
    p9_list = []

    for _, row in df.iterrows():
        p1_val = 1.0 / row['@@P1_Wall_R@@']
        p2_val = 1.0 / row['@@P2_Roof_R@@']
        p3_val = 1.0 - row['@@P3_Roof_Abs@@']
        p4_val = row['@@P4_SHGC@@']
        p5_val = row['@@P5_COP@@']
        p6_val = row['@@P6_ClgSetp@@']
        p7_val = row['@@P7_LPD@@']
        
        p8_val = random.choice(P8_LEVELS)
        p9_val = random.choice(P9_LEVELS)
        
        p8_list.append(p8_val)
        p9_list.append(p9_val)

        levels_mapped = {
            'P1': closest_level(p1_val, P1_LEVELS),
            'P2': closest_level(p2_val, P2_LEVELS),
            'P3': closest_level(p3_val, P3_LEVELS),
            'P4': closest_level(p4_val, P4_LEVELS),
            'P5': closest_level(p5_val, P5_LEVELS),
            'P6': closest_level(p6_val, P6_LEVELS),
            'P7': closest_level(p7_val, P7_LEVELS),
            'P8': closest_level(p8_val, P8_LEVELS),
            'P9': closest_level(p9_val, P9_LEVELS)
        }

        # Calculate Embodied LCE
        embodied_lce = 0.0
        for p, lvl in levels_mapped.items():
            if lvl in inventory_dict.get(p, {}):
                embodied_lce += inventory_dict[p][lvl]['A1_A3']
                embodied_lce += inventory_dict[p][lvl]['B4']
        
        # Calculate Net EUI
        gross_eui_kwh_m2 = row['EUI_MJ_m2'] / 3.6
        pv_generation_kwh = p8_val * 1500.0
        net_eui_kwh_m2 = max(10.0, gross_eui_kwh_m2 - (pv_generation_kwh / total_floor_area))
        net_eui_list.append(net_eui_kwh_m2)

        # Operational LCE
        operational_lce = net_eui_kwh_m2 * grid_emission_factor * project_life
        total_lce = (embodied_lce + operational_lce) / 1000.0  # tCO2eq/m2
        lce_list.append(total_lce)

        # LCC Calculation
        initial_ic = 0.0
        replacement_ic = 0.0
        for p, lvl in levels_mapped.items():
            if lvl in inventory_dict.get(p, {}):
                base_cost = inventory_dict[p][lvl]['IC']
                initial_ic += base_cost
                
                lifespan = LIFESPANS[p]
                reps = math.floor(project_life / lifespan)
                if project_life % lifespan == 0:
                    reps -= 1
                for r in range(1, int(reps) + 1):
                    yr = r * lifespan
                    replacement_ic += base_cost * ((1 + i_IC) / (1 + d))**yr

        total_ic = initial_ic + replacement_ic

        mc_total = 0.0
        for yr in range(1, project_life + 1):
            mc_total += p_MC * initial_ic * ((1 + i_MC) / (1 + d))**yr

        oc_total = 0.0
        annual_energy_cost = net_eui_kwh_m2 * elec_price
        for yr in range(1, project_life + 1):
            oc_total += annual_energy_cost * ((1 + i_OC) / (1 + d))**yr

        total_lcc = total_ic + mc_total + oc_total
        lcc_list.append(total_lcc)

    df['P8_PV_kW'] = p8_list
    df['P9_BESS_kWh'] = p9_list
    df['Net_EUI_kWh_m2'] = net_eui_list
    df['LCE_tCO2_m2'] = lce_list
    df['LCC_USD_m2'] = lcc_list

    df.to_csv('data/aggregated_LHS_LCC_LCA_results.csv', index=False)
    print("Successfully generated data/aggregated_LHS_LCC_LCA_results.csv")

if __name__ == '__main__':
    calculate_lcc_lca()
