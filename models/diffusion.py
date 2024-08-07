# ./models/diffusion.py
import torch

class DiffusionProcess:

    r"""
    The forward process of the diffusion model described in the paper DDPM
    """

    def __init__(self, time_steps = 1000, beta_1 = 1e-4, beta_T = 0.02):
        self.betas = torch.linspace(beta_1, beta_T, time_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def forward(self, x_0, noise, t):
        r"""
        add noise to the input image, and get x_t
        The forward process is calculated directly by x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * noise
        """
        sqrt_alpha_t_bar = self.sqrt_alphas_cumprod.to(x_0.device)[t] # shape: (B, )
        sqrt_1_minus_alpha_t_bar = self.sqrt_1_minus_alphas_cumprod.to(x_0.device)[t]
        x_t = sqrt_alpha_t_bar[:, None, None, None] * x_0 + sqrt_1_minus_alpha_t_bar[:, None, None, None] * noise # broadcast explicitly
        return x_t
    
    def backward(self, x_t, noise_pred, t):
        r"""
        Given x_t, and the predicted noise, we can get x_{t-1} and x_0
        """
        # calculate x_0
        sqrt_alpha_t_bar = self.sqrt_alphas_cumprod.to(x_t.device)[t]
        sqrt_1_minus_alpha_t_bar = self.sqrt_1_minus_alphas_cumprod.to(x_t.device)[t]
        x_0 = (x_t - sqrt_1_minus_alpha_t_bar * noise_pred) / sqrt_alpha_t_bar

        # calculate mean of x_{t-1}
        alpha_t = self.alphas.to(x_t.device)[t]
        beta_t = self.betas.to(x_t.device)[t]
        mean = (x_t - (beta_t * noise_pred)/sqrt_1_minus_alpha_t_bar) / torch.sqrt(alpha_t)

        if t == 1:
            return mean, x_0
        else:
            var = beta_t
            sigma = var ** 0.5
            z = torch.randn(x_t.size(), device=x_t.device)

            return mean + sigma * z, x_0
