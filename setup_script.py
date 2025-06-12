#!/usr/bin/env python3
"""
Setup script to install dependencies and verify environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"📦 {description}...")
    try:
        # Split command into parts for proper Windows handling
        if sys.platform == 'win32':
            command_parts = command.split()
            result = subprocess.run(command_parts, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
        if result.returncode != 0:
            print(f"❌ Error in {description}:")
            print(result.stderr)
            return False
        else:
            print(f"✅ {description} completed successfully")
            return True
    except Exception as e:
        print(f"❌ Error in {description}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def install_requirements():
    """Install required packages from requirements.txt"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt file not found")
        return False
    
    command = f"{sys.executable} -m pip install -r {requirements_file}"
    return run_command(command, "Installing dependencies")

def verify_model_files():
    """Check if required model files exist"""
    current_dir = Path(__file__).parent
    
    required_files = [
        "lead_pipeline.joblib",
        "sac_lead_prioritization_model_1.zip"  # SAC models are typically saved as zip
    ]
    
    missing_files = []
    for file in required_files:
        if not (current_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n   Please ensure these files are in the same directory as the scripts.")
        return False
    else:
        print("✅ All required model files found")
        return True

def create_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = Path(__file__).parent / "outputs"
    if not output_dir.exists():
        output_dir.mkdir()
        print("✅ Created outputs directory")
    else:
        print("✅ Outputs directory already exists")
    return True

def verify_imports():
    """Test import of key modules"""
    print("🔍 Verifying package imports...")
    
    required_packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", None),
        ("torch", None),
        ("sentence_transformers", None),
        ("stable_baselines3", None),
        ("gymnasium", "gym"),
        ("joblib", None)
    ]
    
    failed_imports = []
    
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All required packages imported successfully")
        return True

def main():
    """Main setup function"""
    print("🚀 Lead Prioritization System Setup")
    print("=" * 40)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing requirements", install_requirements),
        ("Verifying package imports", verify_imports),
        ("Checking model files", verify_model_files),
        ("Creating output directory", create_output_directory)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n🔧 {step_name}...")
        if not step_func():
            failed_steps.append(step_name)
    
    print("\n" + "=" * 40)
    
    if failed_steps:
        print("❌ Setup completed with errors:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease resolve the above issues before running predictions.")
        return False
    else:
        print("🎉 Setup completed successfully!")
        print("\nYou can now run predictions using:")
        print("   python predict.py --test_data_path your_test_data.csv")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)