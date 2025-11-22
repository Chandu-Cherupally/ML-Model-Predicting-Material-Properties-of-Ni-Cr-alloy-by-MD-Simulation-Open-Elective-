import numpy as np
import pandas as pd
from pathlib import Path
import re

class LAMMPSOutputParser:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
    
    def parse_stress_strain(self, filename):
        """Parse stress-strain data from LAMMPS output"""
        try:
            data = np.loadtxt(filename, skiprows=1)
            if len(data) == 0:
                return None
            
            # Assuming format: step lx pxx
            steps = data[:, 0]
            lengths = data[:, 1]
            stresses = data[:, 2]  # pxx in pressure units
            
            # Calculate engineering strain and stress
            initial_length = lengths[0]
            strains = (lengths - initial_length) / initial_length
            stresses_eng = -stresses * 1e-4  # Convert to GPa (from bar)
            
            return {
                'strains': strains,
                'stresses': stresses_eng,
                'steps': steps
            }
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return None
    
    def extract_mechanical_properties(self, stress_strain_data):
        """Extract mechanical properties from stress-strain curve"""
        if stress_strain_data is None:
            return None
        
        strains = stress_strain_data['strains']
        stresses = stress_strain_data['stresses']
        
        # Young's modulus from linear region (first 2% strain)
        linear_mask = strains < 0.02
        if np.sum(linear_mask) > 3:
            youngs_modulus = np.polyfit(strains[linear_mask], 
                                      stresses[linear_mask], 1)[0]
        else:
            youngs_modulus = np.nan
        
        # Yield strength (0.2% offset)
        offset_strain = strains - 0.002
        yield_idx = np.where(stresses >= youngs_modulus * offset_strain)[0]
        yield_strength = stresses[yield_idx[0]] if len(yield_idx) > 0 else np.nan
        
        # Ultimate tensile strength
        uts = np.max(stresses) if len(stresses) > 0 else np.nan
        
        # Strain at UTS
        uts_idx = np.argmax(stresses)
        strain_at_uts = strains[uts_idx] if len(strains) > uts_idx else np.nan
        
        return {
            'youngs_modulus': youngs_modulus,
            'yield_strength': yield_strength,
            'uts': uts,
            'strain_at_uts': strain_at_uts
        }
    
    def parse_filename_parameters(self, filename):
        """Extract parameters from filename"""
        filename = Path(filename).name
        pattern = r'stress_strain_comp([\d.]+)_temp(\d+)_rate([\d.e+-]+)\.out'
        match = re.search(pattern, filename)
        
        if match:
            return {
                'composition': float(match.group(1)),
                'temperature': int(match.group(2)),
                'strain_rate': float(match.group(3))
            }
        return None
    
    def process_all_simulations(self):
        """Process all simulation outputs and create dataset"""
        output_dir = self.base_dir / "lammps/outputs"
        results = []
        
        for file_path in output_dir.glob("stress_strain_*.out"):
            print(f"Processing {file_path.name}")
            
            # Extract parameters from filename
            params = self.parse_filename_parameters(file_path)
            if params is None:
                continue
            
            # Parse stress-strain data
            stress_strain_data = self.parse_stress_strain(file_path)
            if stress_strain_data is None:
                continue
            
            # Extract mechanical properties
            properties = self.extract_mechanical_properties(stress_strain_data)
            if properties is None:
                continue
            
            # Combine all data
            result = {**params, **properties}
            results.append(result)
            
            # Save individual stress-strain curves
            curve_df = pd.DataFrame({
                'strain': stress_strain_data['strains'],
                'stress': stress_strain_data['stresses']
            })
            curve_filename = f"stress_strain_comp{params['composition']}_temp{params['temperature']}_rate{params['strain_rate']:.0e}.csv"
            curve_df.to_csv(self.base_dir / "data/raw" / curve_filename, index=False)
        
        # Create main dataset
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.base_dir / "data/processed/mechanical_properties.csv", index=False)
            print(f"Processed {len(results)} simulations")
            return df
        else:
            print("No valid simulations found")
            return None

if __name__ == "__main__":
    parser = LAMMPSOutputParser()
    df = parser.process_all_simulations()