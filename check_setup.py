import os
import subprocess
import sys
from pathlib import Path

def main():
    print("NiCr ML Project - Setup Verification")
    print("=" * 50)
    
    # Create directories first
    directories = [
        'lammps/inputs', 'lammps/outputs', 'lammps/potentials',
        'data/raw', 'data/processed', 'visualization', 'ml_models'
    ]
    
    print("Creating directories...")
    for dir_path in directories:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ {dir_path}")
        except Exception as e:
            print(f"✗ {dir_path}: {e}")
    
    print("\nChecking LAMMPS installation...")
    # Try to find LAMMPS
    possible_names = ["lmp_mpi", "lmp_serial", "lmp", "lmp_mpi.exe", "lmp_serial.exe", "lmp.exe"]
    
    lammps_found = False
    for name in possible_names:
        try:
            result = subprocess.run([name, "-help"], capture_output=True, text=True, timeout=5)
            if "LAMMPS" in result.stderr or "LAMMPS" in result.stdout:
                print(f"✓ LAMMPS found: {name}")
                lammps_found = True
                break
        except:
            continue
    
    if not lammps_found:
        print("✗ LAMMPS not found")
        print("\nTo install LAMMPS:")
        print("  conda install -c conda-forge lammps")
    
    print("\nChecking potential files...")
    potential_files = {
        'library.meam': 'lammps/potentials/library.meam',
        'CrNi.meam': 'lammps/potentials/CrNi.meam'
    }
    
    all_potentials = True
    for name, path in potential_files.items():
        if Path(path).exists():
            print(f"✓ {name}")
        else:
            print(f"✗ {name} - missing")
            all_potentials = False
    
    if not all_potentials:
        print("\nDownload potential files from:")
        print("https://www.ctcms.nist.gov/potentials/")
        print("Search for: 2025--Sharifi-H--Cr-Ni--LAMMPS--ipr1")
    
    print("\n" + "=" * 50)
    if lammps_found and all_potentials:
        print("✓ Setup complete! Ready to run simulations.")
        print("\nNext steps:")
        print("1. python run_project.py --test-only")
        print("2. python run_project.py --all")
    else:
        print("✗ Setup incomplete. Please fix the issues above.")

if __name__ == "__main__":
    main()