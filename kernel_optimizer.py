import random
import numpy as np
import datetime

from train import *

# GA constants
KERNEL_DIM = 3
POPULATION_SIZE = 5
GENERATIONS = 10
MUTATION_RATE = 0.25
ELITISM = 2  # Keep top 2 kernels

# Convolution constants
DATA_COUNT = 0
BATCH_SIZE = 64
NUM_EPOCHS = 30
SEED = None
LAYER_COUNT = 2


def conv(kernel):
    if kernel is None:
        kernel = random_kernel((LAYER_COUNT, 3, 3), seed=SEED)

    run(nn_model=FullyConnectedModel,
        DATA_COUNT=DATA_COUNT,
        BATCH_SIZE=BATCH_SIZE,
        NUM_EPOCHS=NUM_EPOCHS,
        LAYER_COUNT=LAYER_COUNT,
        SEED=SEED,
        kernel=kernel)


def mutate_kernel(kernel):
    kernel = kernel.copy()
    kernel = kernel.flatten()
    idx = random.randint(0, kernel.shape[0] - 1)
    kernel[idx] += random.choice([-1, 1])

    return kernel


def crossover_kernel(kernel1, kernel2):
    k1 = kernel1.flatten()
    k2 = kernel2.flatten()
    crossover_point = random.randint(1, k1.shape[0] - 1)
    new_kernel = np.append(
        k1[:crossover_point], k2[crossover_point:])
    new_kernel = new_kernel.reshape(3, 3)
    return new_kernel


def evaluate_kernel(kernel, population_number):
    print(f'Population {population_number}\n')
    result = conv(kernel)
    return result
    # try:
    # except e:
    #     print(f"Error during ML evaluation")
    #     return 0.0


all_kernels = []

population = [random_kernel((LAYER_COUNT, 3, 3), seed=SEED)
              for _ in range(POPULATION_SIZE)]

for generation in range(GENERATIONS):
    print(f'\nGeneration {generation}\n')
    fitness = [(kernel, evaluate_kernel(kernel, idx))
               for idx, kernel in enumerate(population)]
    fitness.sort(key=lambda x: x[1], reverse=True)

    next_population = [f[0] for f in fitness[:ELITISM]]

    while len(next_population) < POPULATION_SIZE:
        if random.random() < MUTATION_RATE:
            parent = random.choice(fitness[:5])[0]  # Mutate a top-5 kernel
            next_population.append(mutate_kernel(parent))
        else:
            parent1 = random.choice(fitness[:5])[0]
            parent2 = random.choice(fitness[:5])[0]
            next_population.append(crossover_kernel(parent1, parent2))

    population = next_population
    print(f"Best kernel: {fitness[0][0]}, Accuracy: {fitness[0][1]}")
    all_kernels.append(fitness[0])

best_kernel, best_accuracy = fitness[0]
print(f"Optimal Kernel: {best_kernel}, Best Accuracy: {best_accuracy}")

print(f'All kernels:\n{all_kernels}')

with open(f'GA_Kernels/{datetime.date.today().strftime(("%d-%m-%Y"))}.txt', 'w+') as f:
    f.write(str(all_kernels))
