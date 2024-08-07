# ./scripts/inference.py

import torch
import yaml
import argparse
import os
import torchvision
from tqdm import tqdm
from models.UNet import UNet
from models.diffusion import DiffusionProcess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, diffusion, train_config, model_config, diffusion_config):
    x_t = torch.randn((train_config['num_samples'],
                       model_config['im_channels'],
                       model_config['im_size'],
                       model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        noise_pred = model(x_t, torch.as_tensor(i).unsqueeze(0).to(device))
        x_t, x_0_pred = diffusion.backward(x_t, noise_pred, torch.as_tensor(i).to(device))

        ims = torch.clamp(x_t, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = torchvision.utils.make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        experiment_dir = os.path.join('experiment', train_config['task_name'], 'samples')
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        img.save(os.path.join(experiment_dir, 'x0_{}.png'.format(i+1)))
        img.close()

def inference(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    model = UNet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join('experiment', train_config['task_name'], train_config['ckpt_name']), map_location=device))
    model.eval()

    diffusion = DiffusionProcess(time_steps=diffusion_config['num_timesteps'], beta_1=diffusion_config['beta_1'], beta_T=diffusion_config['beta_T'])

    with torch.no_grad():
        sample(model, diffusion, train_config, model_config, diffusion_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script arguments. Need to modify ckpt_name first in the config file')
    parser.add_argument('--config_path', type=str, help='Path to the config file', default='config/default.yaml')
    args = parser.parse_args()
    inference(args)