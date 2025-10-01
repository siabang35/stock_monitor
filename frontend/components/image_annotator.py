import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
from typing import List, Tuple, Optional

class ImageAnnotator:
    def __init__(self):
        self.areas = []
        self.current_polygon = []
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    def reset(self):
        """Reset all annotations"""
        self.areas = []
        self.current_polygon = []
    
    def add_point(self, x: int, y: int) -> bool:
        """Add a point to current polygon"""
        self.current_polygon.append((x, y))
        
        # Complete polygon with 4 points
        if len(self.current_polygon) == 4:
            self.areas.append(self.current_polygon.copy())
            self.current_polygon = []
            return True
        return False
    
    def remove_area(self, area_index: int):
        """Remove an area by index"""
        if 0 <= area_index < len(self.areas):
            self.areas.pop(area_index)
    
    def get_areas(self) -> List[List[Tuple[int, int]]]:
        """Get all defined areas"""
        return self.areas
    
    def load_areas(self, areas: List[List[Tuple[int, int]]]):
        """Load areas from external source"""
        self.areas = areas
    
    def draw_annotations(self, image: Image.Image) -> Image.Image:
        """Draw annotations on image"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Draw completed areas
        for i, area in enumerate(self.areas):
            if len(area) >= 3:
                color = self.colors[i % len(self.colors)]
                
                # Draw filled polygon with transparency
                draw.polygon(area, outline=color, width=3)
                
                # Draw area number
                if area:
                    center_x = sum(p[0] for p in area) // len(area)
                    center_y = sum(p[1] for p in area) // len(area)
                    
                    # Draw background circle for number
                    draw.ellipse(
                        [center_x-15, center_y-15, center_x+15, center_y+15],
                        fill='white',
                        outline=color,
                        width=2
                    )
                    
                    # Draw area number
                    draw.text((center_x-5, center_y-8), str(i+1), fill=color)
        
        # Draw current polygon being created
        if len(self.current_polygon) > 0:
            # Draw points
            for point in self.current_polygon:
                draw.ellipse(
                    [point[0]-5, point[1]-5, point[0]+5, point[1]+5],
                    fill='yellow',
                    outline='black',
                    width=2
                )
            
            # Draw lines between points
            if len(self.current_polygon) > 1:
                for i in range(len(self.current_polygon) - 1):
                    draw.line(
                        [self.current_polygon[i], self.current_polygon[i+1]],
                        fill='yellow',
                        width=2
                    )
        
        return img_copy
    
    def create_interactive_interface(self, image: Image.Image):
        """Create interactive annotation interface"""
        st.markdown("### Interactive Area Definition")
        
        # Display annotated image
        annotated_image = self.draw_annotations(image)
        
        # Image click handling (simplified for demo)
        st.image(annotated_image, caption="Click to define areas", use_column_width=True)
        
        # Manual point input
        with st.expander("Manual Point Input"):
            col1, col2 = st.columns(2)
            
            with col1:
                x_coord = st.number_input("X Coordinate", min_value=0, max_value=image.width)
            
            with col2:
                y_coord = st.number_input("Y Coordinate", min_value=0, max_value=image.height)
            
            if st.button("Add Point"):
                completed = self.add_point(int(x_coord), int(y_coord))
                if completed:
                    st.success(f"Area {len(self.areas)} completed!")
                else:
                    st.info(f"Point {len(self.current_polygon)} added. Need {4 - len(self.current_polygon)} more points.")
                st.rerun()
        
        # Area management
        if self.areas:
            st.markdown("### Defined Areas")
            
            for i, area in enumerate(self.areas):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**Area {i+1}:** {len(area)} points")
                
                with col2:
                    if st.button("Edit", key=f"edit_{i}"):
                        st.info("Edit functionality coming soon")
                
                with col3:
                    if st.button("Delete", key=f"delete_{i}"):
                        self.remove_area(i)
                        st.rerun()
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Current"):
                self.current_polygon = []
                st.rerun()
        
        with col2:
            if st.button("Clear All"):
                self.reset()
                st.rerun()
        
        with col3:
            if st.button("Add Rectangle"):
                self.add_rectangle_area(image)
                st.rerun()
    
    def add_rectangle_area(self, image: Image.Image):
        """Add a rectangular area"""
        width, height = image.size
        
        # Create rectangle in center
        center_x, center_y = width // 2, height // 2
        rect_width, rect_height = min(width, height) // 6, min(width, height) // 8
        
        rectangle = [
            (center_x - rect_width//2, center_y - rect_height//2),
            (center_x + rect_width//2, center_y - rect_height//2),
            (center_x + rect_width//2, center_y + rect_height//2),
            (center_x - rect_width//2, center_y + rect_height//2)
        ]
        
        self.areas.append(rectangle)
    
    def export_areas(self) -> dict:
        """Export areas as JSON"""
        return {
            "areas": self.areas,
            "total_count": len(self.areas),
            "timestamp": st.session_state.get("timestamp", "")
        }
    
    def import_areas(self, data: dict):
        """Import areas from JSON"""
        self.areas = data.get("areas", [])
