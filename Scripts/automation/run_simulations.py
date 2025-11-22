import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import sys

class LAMMPSRunner:
    def __init__(self, lammps_executable=None, base_dir="."):
        self.base_dir = Path(base_dir)
        
        # Auto-detect LAMMPS executable
        if lammps_executable is None:
            self.lammps_executable = self.find_lammps_executable()
        else:
            self.lammps_executable = lammps_executable
            
        print(f"Using LAMMPS executable: {self.lammps_executable}")
        self.setup_directories()
    
    def find_lammps_executable(self):
        """Find LAMMPS executable automatically"""
        # Try common executable names
        possible_names = [
            "lmp_mpi", "lmp_serial", "lmp", "lammps",
            "lmp_mpi.exe", "lmp_serial.exe", "lmp.exe", "lammps.exe"
        ]
        
        # Check if lammps is in PATH
        for name in possible_names:
            try:
                result = subprocess.run([name, "-help"], capture_output=True, text=True)
                if result.returncode == 0 or "LAMMPS" in result.stderr or "LAMMPS" in result.stdout:
                    return name
            except:
                continue
        
        # Check common installation paths on Windows
        common_paths = [
            "C:\\Program Files\\LAMMPS\\bin\\lmp_serial.exe",
            "C:\\Program Files\\LAMMPS\\bin\\lmp_mpi.exe",
            "C:\\LAMMPS\\bin\\lmp_serial.exe",
            os.path.expanduser("~\\AppData\\Local\\LAMMPS\\bin\\lmp_serial.exe"),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # If using conda, try to find in conda environment
        try:
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix:
                conda_path = os.path.join(conda_prefix, 'Library', 'bin', 'lmp_serial.exe')
                if os.path.exists(conda_path):
                    return conda_path
        except:
            pass
        
        print("LAMMPS executable not found. Please install LAMMPS or specify the path manually.")
        print("You can install it using: conda install -c conda-forge lammps")
        return None
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'lammps/outputs', 'lammps/inputs', 'lammps/potentials',
            'data/raw', 'data/processed', 'visualization', 'ml_models'
        ]
        for dir_path in dirs:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def generate_input_file(self, composition=0.8, temperature=300, 
                          strain_rate=1e9, lattice_param=3.52, size=10):
        """Generate LAMMPS input file with given parameters"""
        
        template = f"""# NiCr Alloy Tensile Test - MEAM Potential
# Composition: Ni{composition*100:.0f}Cr{(1-composition)*100:.0f}

units metal
atom_style atomic
dimension 3
boundary p p p

# Create lattice
lattice fcc {lattice_param}
region box block 0 {size} 0 {size} 0 {size}
create_box 2 box
create_atoms 1 box

# Set composition
set type 1 type/fraction 2 {1-composition} 12345

# MEAM Potential
pair_style meam
pair_coeff * * ../potentials/library.meam Ni Cr ../potentials/CrNi.meam Ni Cr

# Minimization
min_style cg
minimize 1e-10 1e-10 1000 1000

# Equilibration
velocity all create {temperature} 54321 rot yes dist gaussian
timestep 0.001
fix 1 all npt temp {temperature} {temperature} 0.1 iso 0.0 0.0 1.0
run 1000  # Reduced for testing

# Tensile test
unfix 1
fix 1 all nvt temp {temperature} {temperature} 0.1
fix 2 all deform 1 x erate {strain_rate} units box remap x

# Stress calculation
compute p all stress/atom NULL
compute pxx all reduce sum c_p[1]

# Output
thermo 50
thermo_style custom step lx ly lz press pxx pyy pzz pe temp

fix stress_out all print 50 "$(step) $(lx) $(pxx)" file ../outputs/stress_strain_comp{composition}_temp{temperature}_rate{strain_rate:.0e}.out screen no

run 500  # Reduced for testing
"""
        
        input_file = self.base_dir / f"lammps/inputs/in_tensile_comp{composition}_temp{temperature}_rate{strain_rate:.0e}.lmp"
        with open(input_file, 'w') as f:
            f.write(template)
        
        return input_file
    
    def run_simulation(self, composition=0.8, temperature=300, strain_rate=1e9):
        """Run a single simulation with given parameters"""
        if self.lammps_executable is None:
            print("ERROR: LAMMPS executable not found!")
            return False
            
        print(f"Running simulation: Ni{composition*100:.0f}Cr{(1-composition)*100:.0f}, "
              f"T={temperature}K, rate={strain_rate:.1e}")
        
        # Generate input file
        input_file = self.generate_input_file(composition, temperature, strain_rate)
        
        # Run LAMMPS
        cmd = [self.lammps_executable, "-in", str(input_file.name)]
        
        try:
            # Change to inputs directory
            original_dir = os.getcwd()
            os.chdir(self.base_dir / "lammps/inputs")
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Return to original directory
            os.chdir(original_dir)
            
            if result.returncode != 0:
                print(f"LAMMPS Error (return code: {result.returncode})")
                if result.stderr:
                    print(f"Stderr: {result.stderr[:500]}...")  # First 500 chars
                return False
            else:
                print("Simulation completed successfully")
                # Check if output file was created
                output_file = self.base_dir / f"lammps/outputs/stress_strain_comp{composition}_temp{temperature}_rate{strain_rate:.0e}.out"
                if output_file.exists():
                    print(f"Output file created: {output_file}")
                else:
                    print("Warning: Output file not created")
                return True
                
        except subprocess.TimeoutExpired:
            print("Simulation timed out after 5 minutes")
            os.chdir(original_dir)
            return False
        except Exception as e:
            print(f"Failed to run simulation: {e}")
            os.chdir(original_dir)
            return False
    
    def run_test_simulation(self):
        """Run a single test simulation to verify setup"""
        print("Running test simulation...")
        return self.run_simulation(composition=0.8, temperature=300, strain_rate=1e9)
    
    def run_parameter_sweep(self, compositions=None, temperatures=None, strain_rates=None):
        """Run simulations for multiple parameter combinations"""
        if self.lammps_executable is None:
            print("Cannot run simulations: LAMMPS executable not found")
            return pd.DataFrame()
            
        if compositions is None:
            compositions = [0.8]  # Start with just one composition for testing
        if temperatures is None:
            temperatures = [300]  # Start with one temperature
        if strain_rates is None:
            strain_rates = [1e9]  # Start with one strain rate
        
        results = []
        for comp in compositions:
            for temp in temperatures:
                for rate in strain_rates:
                    success = self.run_simulation(comp, temp, rate)
                    results.append({
                        'composition': comp,
                        'temperature': temp,
                        'strain_rate': rate,
                        'success': success
                    })
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test the LAMMPS setup
    runner = LAMMPSRunner()
    
    if runner.lammps_executable:
        # Run a single test simulation first
        print("Running test simulation...")
        success = runner.run_test_simulation()
        
        if success:
            print("Test simulation successful! Running parameter sweep...")
            # Run full parameter sweep
            df_results = runner.run_parameter_sweep(
                compositions=[0.7, 0.8, 0.9],
                temperatures=[300, 600],
                strain_rates=[1e8, 1e9]
            )
        else:
            print("Test simulation failed. Please check LAMMPS installation.")
    else:
        print("LAMMPS not found. Please install LAMMPS first.")
    
    # Save simulation log
    if 'df_results' in locals():
        df_results.to_csv('data/raw/simulation_log.csv', index=False)