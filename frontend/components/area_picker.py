import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import tempfile
import os
from typing import List, Tuple, Optional
from backend.utils.area_picker import AreaPicker

class StreamlitAreaPicker:
    def __init__(self):
        # Initialize session state
        if 'picker_areas' not in st.session_state:
            st.session_state.picker_areas = []
        if 'picker_current_points' not in st.session_state:
            st.session_state.picker_current_points = []
        if 'picker_instance' not in st.session_state:
            st.session_state.picker_instance = None
    
    def create_interface(self, image: Optional[Image.Image] = None, image_path: Optional[str] = None):
        """Create the area picker interface."""
        st.markdown("### 📍 Area Definition - Pallet Picker")
        st.markdown("Define areas where pallets will be counted in the warehouse")
        
        if image is None and image_path is None:
            st.error("No image provided")
            return
        
        # Method selection
        method = st.radio(
            "Choose definition method:",
            ["Interactive OpenCV Picker", "Web Interface Picker"],
            help="OpenCV provides more precise control, Web Interface is easier to use"
        )
        
        if method == "Interactive OpenCV Picker":
            self._opencv_picker_interface(image, image_path)
        else:
            self._web_picker_interface(image, image_path)
    
    def _opencv_picker_interface(self, image: Optional[Image.Image], image_path: Optional[str]):
        """OpenCV-based picker interface."""
        st.markdown("#### 🖱️ OpenCV Interactive Picker")
        
        # Save image temporarily if PIL Image provided
        temp_path = None
        if image is not None and image_path is None:
            temp_path = tempfile.mktemp(suffix='.png')
            image.save(temp_path)
            image_path = temp_path
        
        if image_path and os.path.exists(image_path):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("Click 'Start Picker' to open OpenCV window for area definition")
                
                if st.button("🚀 Start Interactive Picker", type="primary"):
                    try:
                        # Create picker instance
                        picker = AreaPicker(save_file="data/warehouse_areas.pkl")
                        
                        # Run interactive picker
                        with st.spinner("Opening OpenCV picker window..."):
                            success = picker.run_interactive(image_path)
                        
                        if success:
                            # Update session state with new areas
                            st.session_state.picker_areas = picker.get_areas()
                            st.session_state.picker_instance = picker
                            st.success(f"✅ Picker completed! {len(st.session_state.picker_areas)} areas defined")
                            st.rerun()
                        else:
                            st.error("Failed to start picker")
                    
                    except Exception as e:
                        st.error(f"Error running picker: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
            
            with col2:
                st.markdown("**Instructions:**")
                st.markdown("""
                - **Left Click**: Add point (4 points = complete area)
                - **Right Click**: Remove area
                - **Middle Click**: Clear current points  
                - **S Key**: Save manually
                - **C Key**: Clear all areas
                - **ESC**: Exit picker
                """)
        
        # Display current areas
        self._display_current_areas(image)
    
    def _web_picker_interface(self, image: Optional[Image.Image], image_path: Optional[str]):
        """Web-based picker interface."""
        st.markdown("#### 🌐 Web Interface Picker")
        
        if image is None:
            if image_path:
                image = Image.open(image_path)
            else:
                st.error("No image available")
                return
        
        # Manual coordinate input
        with st.expander("➕ Add Area Manually", expanded=True):
            st.markdown("Enter 4 corner points to define a rectangular area:")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Point 1 (Top-Left)**")
                x1 = st.number_input("X1", min_value=0, max_value=image.width, key="x1")
                y1 = st.number_input("Y1", min_value=0, max_value=image.height, key="y1")
            
            with col2:
                st.markdown("**Point 2 (Top-Right)**")
                x2 = st.number_input("X2", min_value=0, max_value=image.width, key="x2")
                y2 = st.number_input("Y2", min_value=0, max_value=image.height, key="y2")
            
            with col3:
                st.markdown("**Point 3 (Bottom-Right)**")
                x3 = st.number_input("X3", min_value=0, max_value=image.width, key="x3")
                y3 = st.number_input("Y3", min_value=0, max_value=image.height, key="y3")
            
            with col4:
                st.markdown("**Point 4 (Bottom-Left)**")
                x4 = st.number_input("X4", min_value=0, max_value=image.width, key="x4")
                y4 = st.number_input("Y4", min_value=0, max_value=image.height, key="y4")
            
            col_add, col_preset = st.columns(2)
            
            with col_add:
                if st.button("➕ Add Area", type="primary"):
                    new_area = [(int(x1), int(y1)), (int(x2), int(y2)), 
                               (int(x3), int(y3)), (int(x4), int(y4))]
                    st.session_state.picker_areas.append(new_area)
                    st.success(f"Area {len(st.session_state.picker_areas)} added!")
                    st.rerun()
            
            with col_preset:
                if st.button("📐 Add Center Rectangle"):
                    # Add a rectangle in the center
                    center_x, center_y = image.width // 2, image.height // 2
                    rect_w, rect_h = min(image.width, image.height) // 6, min(image.width, image.height) // 8
                    
                    center_rect = [
                        (center_x - rect_w//2, center_y - rect_h//2),
                        (center_x + rect_w//2, center_y - rect_h//2),
                        (center_x + rect_w//2, center_y + rect_h//2),
                        (center_x - rect_w//2, center_y + rect_h//2)
                    ]
                    st.session_state.picker_areas.append(center_rect)
                    st.success("Center rectangle added!")
                    st.rerun()
        
        # Display current areas
        self._display_current_areas(image)
    
    def _display_current_areas(self, image: Optional[Image.Image]):
        """Display current defined areas."""
        if st.session_state.picker_areas:
            st.markdown("#### 📋 Defined Areas")
            
            # Draw areas on image
            if image:
                annotated_image = self._draw_areas_on_image(image, st.session_state.picker_areas)
                st.image(annotated_image, caption=f"Defined Areas: {len(st.session_state.picker_areas)}", 
                        use_container_width=False)

            # Area management
            for i, area in enumerate(st.session_state.picker_areas):
                with st.expander(f"🏷️ Area {i+1} - {len(area)} points"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.json(area)
                    
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_area_{i}"):
                            st.session_state.picker_areas.pop(i)
                            st.success(f"Area {i+1} deleted!")
                            st.rerun()
            
            # Control buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Areas"):
                    self._save_areas()
                    st.success("Areas saved!")
            
            with col2:
                if st.button("📤 Export JSON"):
                    areas_json = json.dumps(st.session_state.picker_areas, indent=2)
                    st.download_button(
                        label="Download Areas JSON",
                        data=areas_json,
                        file_name="warehouse_areas.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("🗑️ Clear All", type="secondary"):
                    st.session_state.picker_areas = []
                    st.success("All areas cleared!")
                    st.rerun()
        
        else:
            st.info("No areas defined yet. Use the methods above to define pallet areas.")
    
    def _draw_areas_on_image(self, image: Image.Image, areas: List[List[Tuple[int, int]]]) -> Image.Image:
        """Draw areas on image using OpenCV."""
        # Convert PIL to OpenCV
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Draw areas
        colors = [(255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255), 
                 (255, 0, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255)]
        
        for i, area in enumerate(areas):
            if len(area) >= 3:
                pts = np.array(area, dtype=np.int32)
                color = colors[i % len(colors)]
                
                # Draw polygon
                cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=3)
                
                # Draw area number
                if area:
                    center_x = int(sum(p[0] for p in area) / len(area))
                    center_y = int(sum(p[1] for p in area) / len(area))
                    
                    # Background circle
                    cv2.circle(img_bgr, (center_x, center_y), 25, (255, 255, 255), -1)
                    cv2.circle(img_bgr, (center_x, center_y), 25, color, 3)
                    
                    # Area number
                    cv2.putText(img_bgr, str(i+1), (center_x-10, center_y+8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Convert back to PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    def _save_areas(self):
        """Save areas to file."""
        try:
            # Create AreaPicker instance and save
            picker = AreaPicker(save_file="data/warehouse_areas.pkl")
            picker.set_areas(st.session_state.picker_areas)
            st.session_state.picker_instance = picker
        except Exception as e:
            st.error(f"Error saving areas: {str(e)}")
    
    def get_areas(self) -> List[List[Tuple[int, int]]]:
        """Get current areas."""
        return st.session_state.picker_areas.copy()
    
    def load_areas(self, areas: List[List[Tuple[int, int]]]):
        """Load areas into session state."""
        st.session_state.picker_areas = areas
    
    def clear_areas(self):
        """Clear all areas."""
        st.session_state.picker_areas = []
        st.session_state.picker_current_points = []
