#!/usr/bin/env python3
"""
Startup script for Swiss TLM3D Route Planner
This script checks dependencies and starts the web server
"""

import sys
import subprocess
import importlib

def check_dependency(module_name, package_name=None):
    """Check if a Python module is available"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is available")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT available")
        if package_name:
            print(f"   Install with: pip install {package_name}")
        return False

def main():
    print("🔍 Checking dependencies for Swiss TLM3D Route Planner...")
    print("=" * 60)
    
    # Check required dependencies
    dependencies = [
        ("flask", "flask"),
        ("geopandas", "geopandas"),
        ("networkx", "networkx"),
        ("shapely", "shapely"),
        ("numpy", "numpy"),
    ]
    
    missing_deps = []
    for module, package in dependencies:
        if not check_dependency(module, package):
            missing_deps.append(package)
    
    print("=" * 60)
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_deps)}")
        print("\n   Or install all requirements with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    print("\n🚀 Starting the web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🗺️  Click on the map to set start and end points")
    print("⚡ Routes are calculated using the optimized clipping approach!")
    print("\n" + "=" * 60)
    
    # Start the Flask server
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
