# Testing Directory

This directory contains all testing scripts, debugging tools, and intermediate files used during the development of the robust routing system.

## Test Scripts
- `test_routing_validation.py` - Main validation script for testing routing with specific coordinate pairs
- `test_specific_route.py` - Original test script for specific route testing
- `debug_robust_routing.py` - Comparison script between simple and robust routing approaches
- `fix_network_connectivity.py` - Script to analyze and fix network connectivity issues
- `plot_test_points.py` - Visualization script for test points on the network
- `routing_debug.py` - Additional debugging utilities

## Visualizations
- `routing_validation_results.png` - Final successful routing results showing detailed geometry
- `routing_comparison.png` - Comparison between simple and robust routing approaches  
- `test_points_on_network.png` - Visualization of test points on the ski touring network

## Intermediate Files
- `intermediate_steps.gpkg` - Intermediate processing steps during network cleaning
- `routing_network.gpkg` - Network file used during routing development

## Purpose
These files were crucial for:
1. Debugging the robust routing geometry preservation issues
2. Validating that routes follow actual ski touring paths instead of straight lines
3. Testing segment splitting with detailed coordinate preservation
4. Ensuring routing works for all segment lengths and configurations

The main routing system is now production-ready and these files are kept for reference and future debugging if needed.
