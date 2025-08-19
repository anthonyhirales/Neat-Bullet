#neat
import random
import numpy as np
import copy

class Node:
    def __init__(self, node_id, node_type):  
        self.id = node_id
        self.type = node_type
        self.value = 0.0

class Connection:
    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

class Genome:
    def __init__(self, input_size, output_size, innovation_tracker):
        self.nodes = []
        self.connections = []
        self.input_size = input_size
        self.output_size = output_size
        self.innovation_tracker = innovation_tracker

        self.inicializar()

    def inicializar(self):
        for i in range(self.input_size):
            self.nodes.append(Node(i, 'input'))
        for i in range(self.output_size):
            self.nodes.append(Node(self.input_size + i, 'output'))

        for i in range(self.input_size):
            for j in range(self.output_size):
                in_node = i
                out_node = self.input_size + j
                innovation = self.innovation_tracker.get_innovation(in_node, out_node)
                self.connections.append(Connection(in_node, out_node, np.random.randn(), True, innovation))

    def forward(self, inputs):
        # Initialize  
        node_values = {node.id: 0.0 for node in self.nodes}

        # pass the lidars information to the neurons.
        for i in range(self.input_size):
            node_values[i] = inputs[i]

        # Sum the values
        for conn in self.connections:
            if conn.enabled:
                in_node = conn.in_node
                out_node = conn.out_node
                # Propagates
                node_values[out_node] += node_values[in_node] * conn.weight

        # Activate with Tahn
        for node in self.nodes:
            if node.type != 'input':
                node_values[node.id] = np.tanh(node_values[node.id])

        # Obtain results
        outputs = [node_values[node.id] for node in self.nodes if node.type == 'output']
        return outputs

    
    def NodoID(self, id):
        for node in self.nodes:
            if node.id == id:
                return node
        raise ValueError(f"No se encontro el nodo con ID {id} ") # Debugear.

    
    def Apuntar(self, in_id, out_id):
        tipo_in = self.NodoID(in_id).type
        tipo_out = self.NodoID(out_id).type

        if tipo_in == 'output':
            return False  
        if tipo_in == 'hidden' and tipo_out == 'input':
            return False
        if tipo_in == 'hidden' and tipo_out == 'hidden' and in_id > out_id:
            return False 
        return True

    
    def AN(self):
        # Choose a random connection
        posibles = [c for c in self.connections if c.enabled]
        if not posibles:
            print("No available connections to spli.")
            return

         # Disable original coonection
        conexion = random.choice(posibles)
        conexion.enabled = False 

        # New hidden node.
        nuevo_id = max(n.id for n in self.nodes) + 1
        nuevo_nodo = Node(nuevo_id, 'hidden')
        self.nodes.append(nuevo_nodo)

        # Wweadd 2 more  connections.
        innov1 = self.innovation_tracker.get_innovation(conexion.in_node, nuevo_id)
        innov2 = self.innovation_tracker.get_innovation(nuevo_id, conexion.out_node)

        self.connections.append(Connection(
            conexion.in_node, nuevo_id, 1.0, True, innov1
        ))

        self.connections.append(Connection(
            nuevo_id, conexion.out_node, conexion.weight, True, innov2
        ))

        print(f"NODO OCULTO {nuevo_id} ENTRE  {conexion.in_node} → {conexion.out_node}")




    def get_node_type(self, node_id):
        if node_id < 10:
            return "input"
        elif node_id == 10:
            return "output"
        else:
            return "hidden"
        
    def AC(self):
        max_int = 10 
        for _ in range(max_int):
            entrada = random.choice(self.nodes)
            salida = random.choice(self.nodes)

            if entrada.id == salida.id:
                continue 

            tipo_in = self.get_node_type(entrada.id)
            tipo_out = self. get_node_type(salida.id)

            if tipo_out == "input":
                continue
            if tipo_in == "output":
                continue
            if tipo_in == "input" and tipo_out == "input":
                continue

            existe = any(c.in_node == entrada.id and c.out_node == salida.id for c in self.connections)
            if existe:
                continue 

            peso = np.random.randn()
            innov = self.innovation_tracker.get_innovation(entrada.id, salida.id)
            self.connections.append(Connection(entrada.id, salida.id, peso, True, innov))

            print(f"Conexion agregada {entrada.id} → {salida.id}")
            return  

        print("No connection was added after multiple tries. ")


    def mutar(self):
        for conn in self.connections:
            if random.random() < 0.8:
                conn.weight += np.random.normal(0, 0.1) 
            else:
                conn.weight = np.random.randn()


        b = random.random()
        c = random.random()

        if len(self.connections) < 50:
            if b < 0.3:
                print("Adding node")
                self.AN()
            if c < 0.15:
                print("Adding connection")
                self.AC()
        print( "B ", b, "C ", c)
    
   


class InnovationTracker:
    def __init__(self):
        self.num_inovacion = 0
        self.inovacion = {}

    def get_innovation(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self.inovacion:
            self.inovacion[key] = self.num_inovacion
            self.num_inovacion += 1
        return self.inovacion[key]


tracker = InnovationTracker()
Genoma1 = Genome(10, 1, tracker)
