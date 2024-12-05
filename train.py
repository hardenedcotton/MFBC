import datetime
import os
import random
import time
import json
import numpy as np
import scipy.signal
import skimage.measure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from numpy import cos, exp, sin, sqrt, tanh
from PIL import Image as im
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils


# Functions
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = divmod(self.end-self.start, 60)
        self.mins = int(self.elapsed[0])
        self.secs = int(self.elapsed[1])


functions = [
    lambda x: x,
    lambda x: sin(x),
    lambda x: cos(x),
    lambda x: sin(3 * x),
    lambda x: cos(3 * x),
    lambda x: sqrt(max(0, x)),
    lambda x: exp(-x),
    lambda x: sqrt(0.5 * x),
    lambda x: x ** 2,
    lambda x: tanh(x),
    lambda x: exp(-2 * x),
    lambda x: 1 / (1 + exp(-x)),
    lambda x: 0,
]

torch_functions = [
    lambda x: x,
    lambda x: torch.sin(x),
    lambda x: torch.cos(x),
    lambda x: torch.sin(3 * x),
    lambda x: torch.cos(3 * x),
    lambda x: torch.sqrt(torch.clamp(x, min=0)),
    lambda x: torch.exp(-x),
    lambda x: torch.sqrt(0.5 * x),
    lambda x: x ** 2,
    lambda x: torch.tanh(x),
    lambda x: torch.exp(-2 * x),
    lambda x: 1 / (1 + torch.exp(-x)),
    lambda x: torch.zeros_like(x),
]


def random_kernel(kernel_size=(3, 3), seed=None):
    assert kernel_size[-1] % 2 == 1, 'Kernel side length must be an odd number.'

    np.random.seed(seed)
    kernel = np.random.randint(1, len(functions), size=kernel_size)
    return kernel


def elementwise_convolution(img, kernel, resize=0):
    if resize:
        img.thumbnail((resize, resize), im.BILINEAR)

    img = np.array(img)
    rgb = np.transpose(img, (2, 0, 1))/255

    im_new = np.zeros(img.shape, 'uint8')

    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2

    kernel_flat = kernel.flatten().astype(int)
    kernel_results = np.zeros_like(kernel_flat)

    func_list = [functions[i] for i in kernel_flat]

    for idx, channel in enumerate(rgb):
        padded = np.pad(channel, pad_size, mode='edge')
        padded_h, padded_w = padded.shape
        for h in range(padded_h-kernel_size+1):
            for w in range(padded_w-kernel_size+1):

                window = padded[h:h+kernel_size, w:w+kernel_size].flatten()
                kernel_results = []

                for i in range(len(kernel_flat)):
                    kernel_results.append(func_list[i](window[i]))
                kernel_results = np.array(kernel_results)

                result = (kernel_results.mean() *
                          255).clip(0, 255).astype('uint8')
                im_new[h, w, idx] = result
    final_image = im.fromarray(im_new)
    return final_image


def quick_convolution(img, kernel, resize=0):
    if resize:
        img.thumbnail((resize, resize), im.BILINEAR)

    img = np.array(img)
    rgb = np.transpose(img, (2, 0, 1)) / 255.0

    im_new = np.zeros(img.shape, 'uint8')

    for idx, channel in enumerate(rgb):
        result = scipy.signal.convolve2d(
            channel, kernel, mode='same', boundary='symm')
        im_new[:, :, idx] = result

    im_new = np.clip(im_new * 255, 0, 255).astype('uint8')

    final_image = im.fromarray(im_new)
    return final_image


def convolution_var(img, kernel, resize=0):
    if resize:
        img = F.interpolate(img, size=(resize, resize),
                            mode='bilinear', align_corners=False)

    # Calculate variance and generate kernel
    variance = img.mean() / (img.std() + 1e-8)  # Avoid division by zero
    # Assume this generates a 2D kernel
    kernel_var = kernel_from_constant(kernel, variance)
    kernel_var = torch.tensor(
        kernel_var, dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    # kernel_var now has shape [1, 1, kernel_height, kernel_width]

    # Normalize input and permute to (batch_size, channels, height, width)
    # From (batch_size, height, width, channels)
    rgb = img.permute(0, 3, 1, 2).float() / 255.0

    # Prepare output tensor
    im_new = torch.zeros_like(rgb)

    # Apply convolution for each channel
    for idx in range(rgb.size(1)):  # Iterate over channels
        # Extract single channel, keep batch dimension
        channel = rgb[:, idx:idx + 1, :, :]
        # Apply 2D convolution
        result = F.conv2d(channel, kernel_var, padding='same')
        im_new[:, idx:idx + 1, :, :] = result

    # Scale back to Float range [0, 1]
    im_new = im_new.clamp(0, 1)

    return im_new  # Keep as Float for downstream operations


def kernel_from_constant(kernel, constant):
    kernel_flat = kernel.flatten().astype(int)
    cons_kernel = np.zeros_like(kernel_flat, float)
    for i in range(kernel_flat.shape[0]):
        cons_kernel[i] = torch_functions[kernel_flat[i]](constant)
    cons_kernel = cons_kernel.reshape(kernel.shape)
    return cons_kernel


def get_class_counts(path):
    class_names = {}
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                class_name = f.read()[0]
                if class_name in class_names:
                    class_names[class_name] += 1
                else:
                    class_names[class_name] = 1
    return class_names


def str_to_kernel(input_str):
    input_list = input_str[2:-1].split('_')
    return np.array(input_list, int).reshape((3, 3))


def result_logger(result, time_values, num_epochs, batch_size, kernel, path, seed):
    date = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    kernel = list(a.flatten()).__str__().replace(" ", "").replace(",", "_")
    log = f'{date}\n{kernel}\n{seed}\nEpochs: {num_epochs}\tBatch Size: {batch_size}\nProcess took {time_values[0]}m {time_values[1]}s\nAccuracy: {result}%'
    log_path = f'{path}/log.txt'
    with open(log_path, 'a') as file:
        file.write(log + '\n\n')

    log_json = {
        'date': date,
        'kernel': kernel,
        'epochs': num_epochs,
        'batch_size': batch_size,
        'process_time': {
            'mins': time_values[0],
            'secs': time_values[1]},
        'accuracy': result
    }

    json_log_path = f'{path}/log.json'
    if os.path.exists(json_log_path):
        with open(json_log_path, 'r') as json_file:
            try:
                logs = json.load(json_file)
                if not isinstance(logs, dict):
                    logs = {}
            except json.JSONDecodeError:
                logs = {}
    else:
        logs = {}

    logs[seed] = log_json

    with open(json_log_path, 'w') as json_file:
        json.dump(logs, json_file, indent=2)

    print(f'\nLogged:\n\n{log}\n')
    print(f'{log_path}\n{json_log_path}\n')


# Dataset Creator
class EntropyImageDataset(Dataset):
    def __init__(self, image_dir, data_count, convolution_type='', resize=None, kernel_size=3, seed=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.convolution_type = convolution_type
        self.resize = resize
        self.kernel_size = kernel_size
        self.seed = seed
        self.image_filenames = [f for f in os.listdir(image_dir) if (
            f.startswith('IMG') and f.endswith('.JPEG'))]
        self.save_location = f'/unlisted/{image_dir}'

        self.image_filenames = [f for f in self.image_filenames if os.path.exists(
            os.path.join(image_dir, f.replace('.JPEG', '.txt')))]

        if not self.seed:
            self.seed = np.random.randint(0, 2**31)
        np.random.seed(self.seed)
        print(f'Current seed: {np.random.get_state()[1][0]}')
        self.image_counter = 0
        if self.__len__() == 0:
            print(
                f"No images found in the directory: {image_dir}. Please check the directory path and file extensions.")
        else:
            print(f"Found {self.__len__()} images in /{image_dir} folder")

        if data_count and data_count <= self.__len__():
            self.image_filenames = random.sample(
                self.image_filenames, data_count)

        print(f"Selected {self.__len__()} images to be evaluated.")

    def __len__(self):
        return len(self.image_filenames)

    def get_last_save_location(self):
        return self.save_location

    def save_get_images(self, filename):
        directory_path = os.path.dirname(filename)
        image = im.open(filename)
        final_image = image
        self.save_location = f'{directory_path}/s{self.seed}'
        return final_image

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        # image = self.save_get_images(img_name)

        image = im.open(img_name)
        self.save_location = f'{self.image_dir}/s{self.seed}'
        if self.transform:
            image = self.transform(image)

        # adjust extension if necessary
        label_name = img_name.replace('.JPEG', '.txt')

        try:
            with open(label_name, 'r') as f:
                label = int(f.read()[0])
            return image, label
        except FileNotFoundError:
            print(f'No label for {img_name}, assuming 0')
            return image, 0

# Models


class FullyConnectedModel(nn.Module):
    def __init__(self, num_classes):
        super(FullyConnectedModel, self).__init__()
        self.fc1 = nn.Linear(3 * 512 * 512, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)  # Batch size
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Conv4LayerModel(nn.Module):
    def __init__(self, num_classes, kernel):
        super(Conv4LayerModel, self).__init__()
        self.kernel = kernel
        self.conv = convolution_var
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(3 * 512 * 512, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv(x, self.kernel[0]))
        x = F.relu(self.conv(x, self.kernel[1]))
        x = F.relu(self.conv(x, self.kernel[2]))
        x = F.relu(self.conv(x, self.kernel[3]))

        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# old model
class oldCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(oldCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(-1, 256 * 32 * 32)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class singleCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(singleCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 256 * 256, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Train / Test Model

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    with Timer() as t:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device is: {device}')
        model.to(device)
        torch.cuda.empty_cache()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    print('Finished Training')

    t_string = f'Process took {t.mins}m {t.secs}s'
    print(t_string)
    return t.mins, t.secs


def test_model(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    result = 100 * correct / total
    result_string = f'Accuracy: {result}%'
    print(result_string)
    return result


def run(nn_model, DATA_COUNT, BATCH_SIZE, NUM_EPOCHS, SEED, kernel):
    np.set_printoptions(linewidth=np.inf)
    if not SEED:
        SEED = np.random.randint(0, 2**31)
        np.random.seed(SEED)
        print(f'Current seed: {np.random.get_state()[1][0]}')

    image_dir = 'images/source_data'
    get_class_counts(image_dir)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    dataset = EntropyImageDataset(image_dir=image_dir,
                                  data_count=DATA_COUNT,
                                  # do_entropy=True,
                                  #   do_var=True,
                                  # do_convolution=True,
                                  resize=512,
                                  seed=SEED,
                                  transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    print(f'Train size: {train_size} \nTest size: {test_size}')

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(dataset.seed))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              #   num_workers=4, persistent_workers=True,
                              pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_size,
                             #  num_workers=4, persistent_workers=True,
                             pin_memory=True, shuffle=False)

    model = torch.nn.DataParallel(
        nn_model(num_classes=4, kernel=kernel).to('cuda'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Exec
    torch.cuda.empty_cache()
    t = train_model(model, train_loader, criterion,
                    optimizer, num_epochs=NUM_EPOCHS)
    result = test_model(model, test_loader)

    log_path = 'images\\exports\\variance4'
    result_logger(result, t, NUM_EPOCHS, BATCH_SIZE, kernel, log_path, SEED)


if __name__ == '__main__':
    DATA_COUNT = 0
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    SEED = None
    kernel = random_kernel((4, 3, 3), seed=SEED)
    run(nn_model=Conv4LayerModel,
        DATA_COUNT=DATA_COUNT,
        BATCH_SIZE=BATCH_SIZE,
        NUM_EPOCHS=NUM_EPOCHS,
        SEED=SEED,
        kernel=kernel,
        )
