import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import pickle
import os

class VideoProcessor:
    def __init__(self, empty_threshold: float = 0.15):
        self.empty_threshold = empty_threshold
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        
    def load_areas(self, file_path: str) -> List[List[Tuple[int, int]]]:
        """Load object positions from pickle file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            return []
        except Exception as e:
            print(f"Error loading areas: {e}")
            return []
    
    def process_frame(self, frame: np.ndarray, areas: List[List[Tuple[int, int]]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame for stock counting"""
        if not areas:
            return frame, {"empty_count": 0, "total_areas": 0, "details": []}
        
        overlay = frame.copy()
        
        # Frame processing
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
        img_thresh = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
        )
        
        empty_count = 0
        area_details = []
        
        for i, polygon in enumerate(areas):
            # Convert polygon to numpy array
            pts = np.array(polygon, dtype=np.int32)
            
            # Create mask for polygon
            mask = np.zeros(img_thresh.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            
            # Calculate non-zero pixel count
            non_zero_count = cv2.countNonZero(cv2.bitwise_and(img_thresh, mask))
            full_area = cv2.countNonZero(mask)
            
            if full_area == 0:
                continue
            
            # Calculate ratio
            ratio = non_zero_count / full_area
            
            # Determine status and color
            if ratio < self.empty_threshold:
                color = (0, 255, 0)  # Green for empty
                status = "empty"
                empty_count += 1
            else:
                color = (0, 0, 255)  # Red for occupied
                status = "occupied"
            
            # Draw polygon
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=2)
            cv2.fillPoly(overlay, [pts], color)
            
            # Add text
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(overlay, f"{ratio:.2f}", (centroid[0], centroid[1]), 
                       self.font, 0.7, (255, 255, 255), 1)
            
            area_details.append({
                "area_id": i,
                "status": status,
                "ratio": ratio,
                "centroid": [int(centroid[0]), int(centroid[1])]
            })
        
        # Blend overlay with frame
        alpha = 0.48
        result_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add analytics
        self.add_analytics(result_frame, len(areas), empty_count)
        
        count_data = {
            "empty_count": empty_count,
            "occupied_count": len(areas) - empty_count,
            "total_areas": len(areas),
            "details": area_details
        }
        
        return result_frame, count_data
    
    def add_analytics(self, frame: np.ndarray, total_areas: int, empty_count: int):
        """Add analytics overlay to frame"""
        occupied_count = total_areas - empty_count
        
        # Calculate estimated stock
        PALLET_HEIGHT = 4
        PALLET_CAPACITY = 20
        SACK_WEIGHT = 50  # kg
        
        total_pallets = occupied_count * PALLET_HEIGHT
        total_sacks = total_pallets * PALLET_CAPACITY
        total_weight = total_sacks * SACK_WEIGHT / 1000  # tons
        
        # Draw analytics box
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        
        # Add text
        texts = [
            f"Total Areas: {total_areas}",
            f"Occupied: {occupied_count}",
            f"Empty: {empty_count}",
            f"Est. Pallets: {total_pallets}",
            f"Est. Sacks: {total_sacks}",
            f"Est. Weight: {total_weight:.1f} tons"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 30 + (i * 20)
            cv2.putText(frame, text, (20, y_pos), self.font, 1, (255, 255, 255), 1)
