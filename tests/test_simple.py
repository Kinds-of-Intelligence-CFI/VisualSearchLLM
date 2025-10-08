#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import os
import sys
from rds6_config import ensure_rds6_dirs, get_experiment_path

def test_basic_functionality():
    """Test basic RDS6 functionality."""
    print("🧪 Testing basic functionality...")
    
    # Test RDS6 directory creation
    ensure_rds6_dirs()
    print("✓ RDS6 directories created")
    
    # Test experiment path creation
    test_dir = "test_experiment"
    rds6_path = get_experiment_path(test_dir, "results")
    print(f"✓ Experiment path created: {rds6_path}")
    
    # Test file operations
    test_file = os.path.join(rds6_path, "test.txt")
    with open(test_file, 'w') as f:
        f.write("Test content")
    
    if os.path.exists(test_file):
        print("✓ File creation test passed")
        os.remove(test_file)
        print("✓ File cleanup test passed")
    else:
        print("✗ File creation test failed")
    
    print("🎉 Basic functionality test completed!")

if __name__ == "__main__":
    test_basic_functionality()
