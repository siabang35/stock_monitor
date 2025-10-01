#!/usr/bin/env python3
"""
Test script for the improved area picker
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.area_picker import AreaPicker
import cv2
import numpy as np

def create_test_image():
    """Create a test image for picker testing"""
    # Create a simple warehouse-like image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Draw some warehouse elements
    # Floor lines
    for i in range(0, 800, 100):
        cv2.line(img, (i, 0), (i, 600), (150, 150, 150), 1)
    for i in range(0, 600, 100):
        cv2.line(img, (0, i), (800, i), (150, 150, 150), 1)
    
    # Some pallet-like rectangles
    pallets = [
        (100, 100, 80, 60),
        (300, 150, 80, 60),
        (500, 200, 80, 60),
        (200, 350, 80, 60),
        (400, 400, 80, 60)
    ]
    
    for x, y, w, h in pallets:
        cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 100), -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 50), 2)
    
    # Add title
    cv2.putText(img, "Test Warehouse Image", (250, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

def main():
    print("=== Area Picker Test ===")
    
    # Create test image
    test_img = create_test_image()
    test_image_path = "data/test_warehouse.png"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save test image
    cv2.imwrite(test_image_path, test_img)
    print(f"Created test image: {test_image_path}")
    
    # Create picker instance
    picker = AreaPicker(save_file="data/test_areas.pkl")
    
    print("Starting interactive picker...")
    print("Instructions:")
    print("- Left Click: Add point (4 points = complete area)")
    print("- Right Click: Remove area")
    print("- Middle Click: Clear current points")
    print("- S Key: Save manually")
    print("- C Key: Clear all areas")
    print("- ESC: Exit")
    
    # Run picker
    success = picker.run_interactive(test_image_path)
    
    if success:
        areas = picker.get_areas()
        print(f"\n=== Picker Results ===")
        print(f"Total areas defined: {len(areas)}")
        
        for i, area in enumerate(areas):
            print(f"Area {i+1}: {area}")
        
        # Export as JSON
        json_data = picker.export_areas_json()
        print(f"\nJSON Export: {json_data}")
        
    else:
        print("Picker failed or was cancelled")

if __name__ == "__main__":
    main()
