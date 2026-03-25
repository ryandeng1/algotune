import numpy as np
from scipy.ndimage import map_coordinates

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        angle = problem["angle"]
        n = image.shape[0]
        
        theta_rad = np.radians(angle)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        
        center = (n / 2.0, n / 2.0)
        
        x = np.arange(n)
        y = np.arange(n)
        x, y = np.meshgrid(x, y)
        
        x_in = (x - center[0]) * cos_theta - (y - center[1]) * sin_theta + center[0]
        y_in = (x - center[0]) * sin_theta + (y - center[1]) * cos_theta + center[1]
        
        coords = np.dstack((x_in, y_in))
        
        rotated_image = map_coordinates(
            image, 
            coords, 
            order=3, 
            mode='constant', 
            fill_value=0.0
        )
        
        return {"rotated_image": rotated_image}
