import pickle
from typing import List, Tuple

import cv2
import numpy as np


class Picker:
    def __init__(
        self, image_path: str, save_file: str = "tmp/object_positions"
    ) -> None:
        self.image_path: str = image_path
        self.save_file: str = save_file
        self.points: List[Tuple[int, int]] = (
            []
        )  # Temporary points for the polygon being constructed
        self.object_positions: List[List[Tuple[int, int]]] = self.load_positions()

    def load_positions(self) -> List[List[Tuple[int, int]]]:
        """Load saved object positions from a file."""
        try:
            with open(self.save_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return []

    def save_positions(self) -> None:
        """Save object positions to a file."""
        with open(self.save_file, "wb") as f:
            pickle.dump(self.object_positions, f)

    def mouse_events(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events for adding or removing polygons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add points for the polygon
            self.points.append((x, y))

            # If 4 points are selected, save the polygon
            if len(self.points) == 4:
                self.object_positions.append(self.points.copy())
                self.points = []  # Clear the temporary points
                self.save_positions()

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Remove polygons on middle mouse click
            for i, polygon in enumerate(self.object_positions):
                if (
                    cv2.pointPolygonTest(
                        np.array(polygon, dtype=np.int32), (x, y), False
                    )
                    >= 0
                ):
                    self.object_positions.pop(i)
                    break
            self.save_positions()

    def draw_polygons(self, img: np.ndarray) -> None:
        """Draw saved polygons on the image."""
        for polygon in self.object_positions:
            pts = np.array(polygon, np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

    def draw_temp_polygon(self, img: np.ndarray) -> None:
        """Draw the polygon currently being constructed."""
        if len(self.points) > 1:
            pts = np.array(self.points, np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=1)
        for point in self.points:
            cv2.circle(img, point, 5, (0, 255, 255), -1)

    def run(self) -> None:
        """Run the main loop."""
        while True:
            img = cv2.imread(self.image_path)

            # Draw polygons and the temporary polygon
            self.draw_polygons(img)
            self.draw_temp_polygon(img)

            # Display the image
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.imshow("image", img)
            cv2.setMouseCallback("image", self.mouse_events)

            key = cv2.waitKey(30)
            if key == 27:  # ESC key to exit
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    picker = Picker("data/gudang.png")
    picker.run()
