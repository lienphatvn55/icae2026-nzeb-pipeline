import os
import pandas as pd

def main():
    base_dir = r"d:\1. Research\0. CONFERENCE PAPER\2026.09_ICAE2026\3. DATA_CODE\CODE\data\jEPlus-LHS"
    output_path = r"d:\1. Research\0. CONFERENCE PAPER\2026.09_ICAE2026\3. DATA_CODE\CODE\data\aggregated_LHS_results.csv"
    
    # Define scenarios to process. Add '3', '4', ..., '9' when ready.
    scenarios = ['1_Baseline', '2']
    
    all_extracted_data = []

    for scenario in scenarios:
        scenario_path = os.path.join(base_dir, scenario)
        combined_results_file = os.path.join(scenario_path, "AllCombinedResults.csv")
        
        if not os.path.exists(combined_results_file):
            print(f"Skipping {scenario}: AllCombinedResults.csv not found.")
            continue
            
        print(f"Processing scenario: {scenario}")
        
        # Read the jEPlus job summary
        df_params = pd.read_csv(combined_results_file)
        
        # Ensure we have rows to process
        if df_params.empty:
            print(f"No rows in {combined_results_file}.")
            continue
            
        count = 0
        for index, row in df_params.iterrows():
            job_id = row['Job_ID'] # e.g., 'LHS-000000'
            
            lhs_folder_path = os.path.join(scenario_path, job_id)
            eplustbl_path = os.path.join(lhs_folder_path, "eplustbl.csv")
            
            if not os.path.exists(eplustbl_path):
                print(f"  Warning: eplustbl.csv not found in {lhs_folder_path}")
                continue
                
            # Variables to extract
            total_site_energy_gj = None
            eui_mj_m2 = None
            electricity_gj = None
            natural_gas_gj = None
            
            # Parse eplustbl.csv
            with open(eplustbl_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line_clean = line.strip()
                    # eplustbl format places these in specific comma-separated positions
                    if line_clean.startswith(",Total Site Energy,"):
                        parts = line_clean.split(',')
                        try:
                            if len(parts) > 2: total_site_energy_gj = float(parts[2])
                            if len(parts) > 3: eui_mj_m2 = float(parts[3])
                        except ValueError:
                            pass
                    elif line_clean.startswith(",Total End Uses,"):
                        parts = line_clean.split(',')
                        try:
                            if len(parts) > 2: electricity_gj = float(parts[2])
                            if len(parts) > 3: natural_gas_gj = float(parts[3])
                        except ValueError:
                            pass
                            
                    # Optional: Break early if all vars are found to speed up
                    if (total_site_energy_gj is not None and 
                        electricity_gj is not None):
                        pass # keep parsing in case there are multiple matches, but usually only one
                        
            # Create a dictionary for this record
            record = row.to_dict()
            record['Scenario'] = scenario
            record['Total_Site_Energy_GJ'] = total_site_energy_gj
            record['EUI_MJ_m2'] = eui_mj_m2
            record['Electricity_EndUse_GJ'] = electricity_gj
            record['NaturalGas_EndUse_GJ'] = natural_gas_gj
            
            all_extracted_data.append(record)
            count += 1
            
        print(f"  Extracted {count} records from {scenario}")

    if all_extracted_data:
        # Convert to DataFrame
        final_df = pd.DataFrame(all_extracted_data)
        
        # Save to CSV
        final_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved {len(final_df)} aggregated records to:\n{output_path}")
    else:
        print("\nNo data was extracted.")

if __name__ == "__main__":
    main()
