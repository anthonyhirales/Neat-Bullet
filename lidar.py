import pybullet as p
import pybullet_data
import time
import numpy as np
import math

"""
How would the class work?
We created several Lidar objects, setting up independently for each vehicle, and update functions as fundamental
- setup: stablish the start and end positions and orientations per ray.
- update: we used the rayTestBatch function to update the lidars, re-calculating each postion, orientation and hit fraction.
- get data: this function retrieves the data for each step and fowards to its assigned NN in the car.py file. 
"""

class Lidar():
    def __init__(self, car_id, hokuyo_joint=4):
        self.car = car_id       
        self.hokuyo_joint = hokuyo_joint  
        self.num_rays = 10
        self.ray_ids = []
        self.ray_hit_color = [1, 0, 0]  
        self.ray_miss_color = [0, 1, 0] 
        self.ray_length = 2
        self.ray_start_length = 0.25
        self.last_lidar_time = time.time()

    def setup(self):
        angles = [2 * math.pi * 0.5 * float(i) / self.num_rays for i in range(self.num_rays)]
        self.ray_from = []
        self.ray_to = []

        for angle in angles:
            start = [self.ray_start_length * math.sin(angle), self.ray_start_length * math.cos(angle), 0]
            end = [self.ray_length * math.sin(angle), self.ray_length * math.cos(angle), 0]

            self.ray_from.append(start)
            self.ray_to.append(end)
            self.ray_ids.append(p.addUserDebugLine(start, end, self.ray_miss_color, parentObjectUniqueId=self.car,
                                                parentLinkIndex=self.hokuyo_joint))
            
            # Debugging print
            print(f"LiDAR rayo {len(self.ray_from)} - Inicio: {start}, Fin: {end}, Angulo: {math.degrees(angle)}°")

    
    def update_lidar(self):
        now_time = time.time()
        if now_time - self.last_lidar_time > 0.1:
            lidar_position, orientation = p.getBasePositionAndOrientation(self.car)[:2]
            yaw = p.getEulerFromQuaternion(orientation)[2]  # Get car's rotation

            for i in range(self.num_rays):
                angle = i * 2 * 0.5 * math.pi / self.num_rays - yaw  # Adjusted yaw correction

                self.ray_from[i] = [
                    lidar_position[0],
                    lidar_position[1],
                    lidar_position[2] + 0.2
                ]

                self.ray_to[i] = [
                    lidar_position[0] + self.ray_length * math.sin(angle),
                    lidar_position[1] + self.ray_length * math.cos(angle),
                    lidar_position[2] + 0.2
                ]

            results = p.rayTestBatch(self.ray_from, self.ray_to)
            
            for i, result in enumerate(results):
                hit_fraction = result[2]
                hit_position = result[3]

                color = self.ray_hit_color if hit_fraction < 1.0 else self.ray_miss_color
                hit_position = self.ray_from[i] if hit_fraction == 1.0 else [
                    self.ray_from[i][0] + hit_fraction * (self.ray_to[i][0] - self.ray_from[i][0]),
                    self.ray_from[i][1] + hit_fraction * (self.ray_to[i][1] - self.ray_from[i][1]),
                    self.ray_from[i][2] + hit_fraction * (self.ray_to[i][2] - self.ray_from[i][2])
                ]

                self.ray_ids[i] = p.addUserDebugLine(
                    self.ray_from[i], hit_position, color, replaceItemUniqueId=self.ray_ids[i]
                )

                print(f"Rayo {i}: Fraccion: {hit_fraction}, Posicion: {hit_position}")
            
            self.last_lidar_time = now_time




    # Function to pass data to the NN. 
    def get_lidar_data(self):
        lidar_readings = []
        results = p.rayTestBatch(self.ray_from, self.ray_to)

        for result in results:
            hit_fraction = result[2] 
            lidar_readings.append(hit_fraction)

        # Debug
        print(f"LiDAR readings for car {self.car}: {lidar_readings}")

        return np.array(lidar_readings) if lidar_readings else None  
    




