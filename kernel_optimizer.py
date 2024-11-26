import random
import numpy as np
import datetime

from train import *

KERNEL_DIM = 9


def conv(kernel):
    image_dir = 'images'
    get_class_counts(image_dir)
    data_count = 0
    batch_size = 64
    num_epochs = 30
    seed = 5

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    dataset = EntropyImageDataset(image_dir=image_dir,
                                  data_count=data_count,
                                  kernel_override=kernel,
                                  # do_entropy=True,
                                  do_var=True,
                                  # do_convolution=True,
                                  resize=512,
                                  seed=seed,
                                  transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    print(f'Train size: {train_size} \nTest size: {test_size}')

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(dataset.seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              #   num_workers=4, persistent_workers=True,
                              pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_size,
                             #  num_workers=4, persistent_workers=True,
                             pin_memory=True, shuffle=False)

    model = torch.nn.DataParallel(
        FullyConnectedModel(num_classes=4).to('cuda'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# Exec
    torch.cuda.empty_cache()
    t = train_model(model, train_loader, criterion,
                    optimizer, num_epochs=num_epochs)
    result = test_model(model, test_loader)

    dataset[0]
    last_save_location = dataset.get_last_save_location()
    result_logger(result, t, num_epochs, batch_size, last_save_location)
    return result


def mutate_kernel(kernel):
    # Ensure mutation occurs within valid indices
    kernel = kernel.copy()  # To avoid modifying the original kernel
    row = random.randint(0, kernel.shape[0] - 1)
    col = random.randint(0, kernel.shape[1] - 1)

    kernel[row, col] += random.choice([-1, 1])

    return kernel


def crossover_kernel(kernel1, kernel2):
    crossover_point = random.randint(
        1, min(kernel1.shape[0], kernel2.shape[0]) - 1)
    return np.vstack((kernel1[:crossover_point], kernel2[crossover_point:]))


def evaluate_kernel(kernel):
    """Evaluates the kernel by passing it to the ML code and reading the accuracy."""
    kernel_str = ','.join(map(str, kernel))
    try:
        result = conv(kernel)
        return result
    except:
        print(f"Error during ML evaluation: {e.stderr}")
        return 0.0


POPULATION_SIZE = 5
GENERATIONS = 10
MUTATION_RATE = 0.2
ELITISM = 2  # Keep top 2 kernels
all_kernels = []

population = [random_kernel() for _ in range(POPULATION_SIZE)]

for generation in range(GENERATIONS):
    print(f"Generation {generation}")
    fitness = [(kernel, evaluate_kernel(kernel)) for kernel in population]
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
with open(f'{datetime.date.today().strftime(("%d-%m-%Y"))}.txt') as f:
    f.write(all_kernels)