import pickle
import os
from typing import List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path

class AreaPicker:
    def __init__(self, save_file: str = "data/object_positions.pkl"):
        self.save_file = save_file
        self.points: List[Tuple[int, int]] = []  # Temporary points for polygon being constructed
        self.object_positions: List[List[Tuple[int, int]]] = self.load_positions()
        self.image = None
        self.original_image = None
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    def load_positions(self) -> List[List[Tuple[int, int]]]:
        """Load saved object positions from a file."""
        try:
            with open(self.save_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error loading positions: {e}")
            return []
    
    def save_positions(self) -> None:
        """Save object positions to a file."""
        try:
            with open(self.save_file, "wb") as f:
                pickle.dump(self.object_positions, f)
            print(f"Positions saved to {self.save_file}")
        except Exception as e:
            print(f"Error saving positions: {e}")
    
    def mouse_events(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events for adding or removing polygons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add points for the polygon
            self.points.append((x, y))
            print(f"Point added: ({x}, {y}). Total points: {len(self.points)}")
            
            # If 4 points are selected, save the polygon
            if len(self.points) == 4:
                self.object_positions.append(self.points.copy())
                print(f"Polygon completed! Total areas: {len(self.object_positions)}")
                self.points = []  # Clear the temporary points
                self.save_positions()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove polygons on right mouse click
            for i, polygon in enumerate(self.object_positions):
                if (cv2.pointPolygonTest(
                    np.array(polygon, dtype=np.int32), (x, y), False
                ) >= 0):
                    removed_polygon = self.object_positions.pop(i)
                    print(f"Polygon {i+1} removed: {removed_polygon}")
                    self.save_positions()
                    break
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Clear current points on middle mouse click
            if self.points:
                print(f"Cleared {len(self.points)} temporary points")
                self.points = []
    
    def draw_polygons(self, img: np.ndarray) -> None:
        """Draw saved polygons on the image."""
        for i, polygon in enumerate(self.object_positions):
            pts = np.array(polygon, np.int32)
            # Draw polygon outline
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
            
            # Draw polygon number
            if polygon:
                center_x = int(sum(p[0] for p in polygon) / len(polygon))
                center_y = int(sum(p[1] for p in polygon) / len(polygon))
                
                # Draw background circle for number
                cv2.circle(img, (center_x, center_y), 20, (255, 255, 255), -1)
                cv2.circle(img, (center_x, center_y), 20, (255, 0, 255), 2)
                
                # Draw area number
                cv2.putText(img, str(i+1), (center_x-8, center_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    def draw_temp_polygon(self, img: np.ndarray) -> None:
        """Draw the polygon currently being constructed."""
        if len(self.points) > 1:
            pts = np.array(self.points, np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
        
        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(img, point, 8, (0, 255, 255), -1)
            cv2.circle(img, point, 8, (0, 0, 0), 2)
            # Draw point number
            cv2.putText(img, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def draw_instructions(self, img: np.ndarray) -> None:
        """Draw instructions on the image."""
        instructions = [
            "Left Click: Add point (4 points = complete area)",
            "Right Click: Remove area",
            "Middle Click: Clear current points",
            "ESC: Exit",
            f"Current points: {len(self.points)}/4",
            f"Total areas: {len(self.object_positions)}"
        ]
        
        # Draw background for instructions
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (500, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 20
            cv2.putText(img, instruction, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def load_image(self, image_path: str) -> bool:
        """Load image from path."""
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                print(f"Error: Could not load image from {image_path}")
                return False
            print(f"Image loaded: {image_path}")
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def load_image_from_array(self, image_array: np.ndarray) -> None:
        """Load image from numpy array."""
        self.original_image = image_array.copy()
    
    def run_interactive(self, image_path: str) -> bool:
        """Run the interactive area picker."""
        if not self.load_image(image_path):
            return False
        
        print("=== Area Picker Started ===")
        print("Instructions:")
        print("- Left Click: Add point (4 points complete an area)")
        print("- Right Click: Remove area")
        print("- Middle Click: Clear current points")
        print("- ESC: Exit")
        print(f"Loaded {len(self.object_positions)} existing areas")
        
        window_name = "Area Picker - Warehouse Stock Counting"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_events)
        
        while True:
            # Create a copy of the original image
            img = self.original_image.copy()
            
            # Draw all elements
            self.draw_polygons(img)
            self.draw_temp_polygon(img)
            self.draw_instructions(img)
            
            # Display the image
            cv2.imshow(window_name, img)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC key to exit
                break
            elif key == ord('s'):  # Save manually
                self.save_positions()
                print("Positions saved manually")
            elif key == ord('c'):  # Clear all
                self.object_positions = []
                self.points = []
                self.save_positions()
                print("All areas cleared")
        
        cv2.destroyAllWindows()
        print(f"=== Area Picker Finished ===")
        print(f"Total areas defined: {len(self.object_positions)}")
        return True
    
    def get_areas(self) -> List[List[Tuple[int, int]]]:
        """Get all defined areas."""
        return self.object_positions.copy()
    
    def set_areas(self, areas: List[List[Tuple[int, int]]]) -> None:
        """Set areas programmatically."""
        self.object_positions = areas
        self.save_positions()
    
    def clear_areas(self) -> None:
        """Clear all areas."""
        self.object_positions = []
        self.points = []
        self.save_positions()
    
    def add_area(self, points: List[Tuple[int, int]]) -> bool:
        """Add an area programmatically."""
        if len(points) == 4:
            self.object_positions.append(points)
            self.save_positions()
            return True
        return False
    
    def remove_area(self, index: int) -> bool:
        """Remove area by index."""
        if 0 <= index < len(self.object_positions):
            removed = self.object_positions.pop(index)
            self.save_positions()
            print(f"Area {index+1} removed: {removed}")
            return True
        return False
    
    def export_areas_json(self) -> dict:
        """Export areas as JSON-serializable dict."""
        return {
            "areas": self.object_positions,
            "total_count": len(self.object_positions),
            "save_file": self.save_file
        }
    
    def import_areas_json(self, data: dict) -> bool:
        """Import areas from JSON data."""
        try:
            self.object_positions = data.get("areas", [])
            self.save_positions()
            print(f"Imported {len(self.object_positions)} areas")
            return True
        except Exception as e:
            print(f"Error importing areas: {e}")
            return False


def create_picker_from_image_path(image_path: str, save_file: str = "data/object_positions.pkl") -> AreaPicker:
    """Create and return an AreaPicker instance."""
    return AreaPicker(save_file=save_file)

def run_picker_interactive(image_path: str, save_file: str = "data/object_positions.pkl") -> List[List[Tuple[int, int]]]:
    """Run picker interactively and return defined areas."""
    picker = AreaPicker(save_file=save_file)
    if picker.run_interactive(image_path):
        return picker.get_areas()
    return []
