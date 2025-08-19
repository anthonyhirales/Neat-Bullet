
import pybullet as p
import pybullet_data
import time
import numpy as np
from lidar import Lidar
import math
from Neaat import Genome, InnovationTracker


class Simulation():
    def __init__(self, genomas,maximo, real_time_simulation=False):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(1.0 / 120.0)
        p.setRealTimeSimulation(real_time_simulation)

        # Load the track and cars
        self.track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)

        # Define joints
        self.genomapcarro = {}
        self.genomes = genomas
        self.Maximo = maximo
        self.wheels = [8, 15]
        self.steering = [0, 2]
        self.hokuyo_joint = 4 
        self.eval_car = 0

        self.last_time = time.time()
        self.lt_choque = time.time()
        self.last_atasco_time = time.time()

        self.last_control_time = time.time()
        self.contador = time.time()
        self.last_position = {}
        self.ultima_pos = {}
        self.inicio_medido = False

        self.Set_Cars()
        self.choques = {car_name: 0 for car_name in self.carros.keys()}
        self.Puntuaciones = {}
        self.fitness_final = {}  
        self.last_movement_time = {}
        self.tcf = time.time()
        self.pos_ant = {}


    """
    1.- load objects
    2.- restrict them
    3.- set up lidar
    """
    def Set_Cars(self):
        self.carros = {}
        self.lidars = {}
        self.angulos = {}
        if not hasattr(self, "tracker"):
            self.tracker = InnovationTracker()  

        for i in range(3):
            car_name = f"car{i}"
            if i == 0:
                position = [0,i ,.3]
            elif i ==1:
                position = [1, 1.5, 0.3]
            elif i ==2:
                position = [2.4, 3.5, 0.3]
            elif i==4:
                position = [-.5, -1, 0.3]
            
            orientation = p.getQuaternionFromEuler([0, 0, math.pi/3])
            self.carros[car_name] = self.load_car(position, orientation)
            self.constraints(self.carros[car_name])
            
            self.lidars[car_name] = Lidar(self.carros[car_name], self.hokuyo_joint)
            self.lidars[car_name].setup()

            self.genomapcarro[car_name] = self.genomes[car_name]
        

    def load_car(self, position, orientation):
        car_id = p.loadURDF("f10_racecar/racecar_differential.urdf", position, orientation)
        for wheel in range(p.getNumJoints(car_id)):
            p.setJointMotorControl2(car_id, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        return car_id
    
    def constraints (self, car):
        #restrict each axis
        constraints = [
            (9, 11, 1), (10, 13, -1), (9, 13, -1),
            (16, 18, 1), (16, 19, -1), (17, 19, -1),
            (1, 18, -1), (3, 19, -1)
        ]
        for parent, child, ratio in constraints:
            c = p.createConstraint(car, parent, car, child, jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=ratio, maxForce=10000)
    
    
    def choque(self):
        #Detect a collision 
        tci = time.time()
        if tci - self.tcf >= 2:
            for car_name, car in self.carros.items():
                if self.choques.get(car_name, 0) == 0:  
                    car_pos, _ = p.getBasePositionAndOrientation(car)
                    posicion = np.array(car_pos[:2])

                    if car_name not in self.pos_ant:
                        print("Position: ", posicion)
                        self.pos_ant[car_name] = posicion 
                        continue


                    distancia = np.linalg.norm(posicion - self.pos_ant[car_name])
                    print("First point ", self.pos_ant[car_name] ,"Distance: ", distancia )
                    # Validate if the vehicle is stuck and not moving 

                    if distancia < 0.05:
                        print(f"Vehicle stuck: {car_name}")
                        self.choques[car_name] = 1

                    # Verify collisions based on force perception 
                    contact_points = p.getContactPoints(car)
                    for contact in contact_points:
                        other_body_id = contact[2]
                        normal_force = contact[9]

                        if other_body_id != 0 and normal_force > 50:
                            print(f"Crashed {car_name} (normal_force={normal_force})")
                            self.choques[car_name] = 1

                    self.pos_ant[car_name] = posicion  # Update data

            self.tcf = tci  


   
    def Actualizar(self):

        now_time = time.time()
        if now_time - self.last_control_time > 0.01:
            for car_name, car in self.carros.items():
                if self.choques.get(car_name, 0) == 1:
                    if car_name not in self.fitness_final:
                        car_pos, _ = p.getBasePositionAndOrientation(car)
                        current_position = np.array(car_pos[:2])
                        if car_name in self.last_position:
                            distance_traveled = np.linalg.norm(current_position - self.last_position[car_name])
                            reward = distance_traveled * 10
                        else:
                            reward = 0

                        self.fitness_final[car_name] = reward
                        for wheel in self.wheels:
                            p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, 
                                                    targetVelocity=0, force=0)
                        for steer in self.steering:
                            p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, 
                                                    targetPosition=0)

                        continue


                if self.choques.get(car_name, 0) == 0:  #The vehicle is still moving 
                    car_pos, _ = p.getBasePositionAndOrientation(car)
                    current_position = np.array(car_pos[:2])

                    
                    if car_name in self.last_position:
                        distance_traveled = np.linalg.norm(current_position - self.last_position[car_name])
                        reward = distance_traveled * 10
                    else:
                        reward = 0  # If not initilized
                    self.last_position[car_name] = current_position

                        
                    if car_name in self.fitness_final:
                        self.fitness_final[car_name] += reward
                    else:
                        self.fitness_final[car_name] = reward

                    inputs = self.get_nn_inputs(car_name)
                    if inputs is None:
                        continue

                    # Here we obtianed the output of the nerual networks, a simple list with an element.
                    output = self.genomapcarro[car_name].forward(inputs)[0]

                    # Interpret output as steering angle in range [-0.5, 0.5]
                    steering_angle = float(np.clip(output, -1.0, 1.0)) * 0.5
                    target_velocity = 15

                    if not hasattr(self, 'choques'):
                        self.angulos = {car_name: 0 for car_name in self.carros.keys()}


                    # Implementation of the steering.
                    for wheel in self.wheels:
                        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, 
                                                targetVelocity=target_velocity,
                                                force=50)
                    for steer in self.steering:
                        p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, 
                                                targetPosition=steering_angle)
                    
                    if self.fitness_final[car_name] >= self.Maximo:
                        self.choques[car_name] = 1

            self.last_control_time = now_time 



    def Fitness(self):
        current_time = time.time()
        if current_time - self.last_time >= .1:  # evenry .5 seconds 
            for car_name, car in self.carros.items():
                if self.choques.get(car_name, 0) == 1:
                    continue  # Dont calculate the fitness score if they already crashed 
                
                car_pos, _ = p.getBasePositionAndOrientation(car)
                current_position = np.array(car_pos[:2])  # Solo (x, y)

                # Initilize if its the first cycle.
                if car_name not in self.last_position:
                    self.last_position[car_name] = current_position
                
                # Fitness score calculation
                distance_traveled = np.linalg.norm(current_position - self.last_position[car_name])
                reward = distance_traveled * 10  

                
                # Store real-time fitness
                self.Puntuaciones[car_name] = reward  
                self.last_position[car_name] = current_position
            self.last_time = current_time 

    def get_nn_inputs(self, car_name):
        if car_name not in self.carros or car_name not in self.lidars:
            print(f" Error: {car_name} No corresponding lidar detections")
            return None  

        self.lidars[car_name].update_lidar() 
        lidar_data = self.lidars[car_name].get_lidar_data() 

        if lidar_data is None:
            print(f"Invalid data, from:  {car_name}, receiving none ")

        return lidar_data

    def sortear(self, d):
        arr = []
        for x, y in d.items():
            arr.append(y)
            arr.sort()

        di = {}
        for i in arr:
                for x,y in d.items():
                    if i == y:
                        di.update({x:i})

        print(di)
        return di
    
    
    def Actualizaciones(self):
        current_time = time.time()

        if current_time - self.contador > 3:
            self.choque()

        self.Actualizar()
        self.Fitness()

        if all(value == 1 for value in self.choques.values()):
            print("End simulation")
            self.fitness_final = self.sortear(self.fitness_final)
            print("Final fitness score :", self.fitness_final)
            return False
        else:
            return True
    


    def run(self):
        si = True
        while si:
            p.stepSimulation()
            si = self.Actualizaciones()
            time.sleep(1 / 240)
        p.disconnect()
        return self.genomes, self.fitness_final
            

