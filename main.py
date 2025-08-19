from car import Simulation
from Neaat import Connection, Genome, InnovationTracker, Node
import time
import copy
import random
from FuncionesMain import *

if __name__ == "__main__":
    Generacion = 0
    ncarros = 3
    maximo = 2000
    limite = 20
    prev_fitness_prom = 0
    estancamiento = 0
    max_estancamiento = 2

    tracker = InnovationTracker()
    Genes = Genomas(ncarros, tracker)
    for genoma in Genes.values():
        genoma.mutar()

    while Generacion < limite:
        sim = Simulation(Genes, maximo)
        genomes_out, ff = sim.run()

        print(f"Fitness gen {Generacion}:")
        for k, v in sorted(ff.items(), key=lambda x: x[1], reverse=True):
            print(f"{k}: {v:.3f}")

        fitness_prom = sum(ff.values()) / len(ff)
        print(f"Promedio fitness: {fitness_prom:.3f} | Estancamiento: {estancamiento}/{max_estancamiento}")

        if abs(fitness_prom - prev_fitness_prom) < 0.01:
            estancamiento += 1
        else:
            estancamiento = 0
        prev_fitness_prom = fitness_prom

        if estancamiento >= max_estancamiento:
            print("Stagnation  detected. Mutating every genome.")
            for gen in Genes.values():
                gen.mutar()
            estancamiento = 0

        Genes = Reproducir(
            genomes_out,
            ff,
            tracker,
            tam_poblacion=ncarros,
            elitismo=1  
        )


        print("Generation ends ", Generacion)
        Generacion += 1

        if Generacion == limite:
            print("Final Results:", ff)

        time.sleep(3)
