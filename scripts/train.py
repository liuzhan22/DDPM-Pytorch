# ./scripts/train.py

import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from data.dataset import MnistDataset
from torch.utils.data import DataLoader
from models.diffusion import DiffusionProcess
from models.UNet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):

    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Create the diffusion process
    diffusion = DiffusionProcess(time_steps=diffusion_config['num_timesteps'], beta_1=diffusion_config['beta_1'], beta_T=diffusion_config['beta_T'])

    # Create the dataset
    mnist_train = MnistDataset(split='train', root_dir=dataset_config['root_dir'])
    mnist_test = MnistDataset(split='test', root_dir=dataset_config['root_dir'])

    # Create the DataLoader
    train_loader = DataLoader(mnist_train, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=train_config['batch_size'], shuffle=False, num_workers=4)

    # Instantiate the model
    model = UNet(model_config).to(device)
    model.train()

    # Create output directories
    experiment_dir = os.path.join('experiment', train_config['task_name'])
    if not os.path.exists('experiment'):
        os.makedirs('experiment')
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # Run training
    for epoch_idx in range(num_epochs):
        # Each epoch, the model is trained on the entire dataset
        losses = []
        for image, _ in tqdm(train_loader):
            # Each iteration, a batch of images is processed
            optimizer.zero_grad()
            image = image.float().to(device)

            noise = torch.randn_like(image).to(device)
            timesteps = torch.randint(0, diffusion_config['num_timesteps'], (image.shape[0], )).to(device)

            noisy_image = diffusion.forward(image, noise, timesteps)
            noise_pred = model(noisy_image, timesteps) 

            loss = criterion(noise, noise_pred) # here, we estimate the noise that has been added to the image, not a new one
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch {} with loss (mean of one batch) {:.4f}'.format(epoch_idx+1, np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(experiment_dir, 'model_{}.pt'.format(epoch_idx+1)))

    print('Training completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config_path', type=str, default='config/default.yaml', help='path to config file')
    args = parser.parse_args()
    train(args)
