# ./util/utils.py
import torch
import torch.nn as nn

def get_time_bedding(timesteps, embedding_dim: int):
    '''
    Time embedding fuction, refer to code from DDPM and https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch
    :param timesteps: the number of time steps -> (B, )
    :param embedding_dim: the dimension of the embedding -> int

    :return: the time embedding -> (B, embedding_dim)
    '''
    assert embedding_dim % 2 == 0, 'The dimension of the embedding must be even'

    denominator = 10000 ** (2*torch.arange(0, embedding_dim//2).float() / embedding_dim)
    time_embedding = timesteps[:, None] / denominator[None, :] # (B, embedding_dim//2)
    time_embedding = torch.cat([torch.sin(time_embedding), torch.cos(time_embedding)], dim=1) # (B, embedding_dim)

    return time_embedding

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, GroupNorm: bool = True, num_of_groups: int = 8, Activation: bool = True):
        super(ConvBlock, self).__init__()

        # GroupNorm
        self.GroupNorm = nn.GroupNorm(num_groups=num_of_groups, num_channels=in_channels) if GroupNorm else nn.Identity()

        # Activation
        self.Activation = nn.SiLU() if Activation else nn.Identity()

        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        x = self.Conv(self.Activation(self.GroupNorm(x)))
        return x
    
class TimeEmbedding(nn.Module):
    '''
    Maps the Time Embedding to the required output dimension
    '''
    def __init__(self, n_out: int, time_embedding_dim: int = 128):
        super(TimeEmbedding, self).__init__()
        self.TimeEmbeddingBlock = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, n_out))

    def forward(self, x):
        return self.TimeEmbeddingBlock(x)
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 8, num_head: int = 4, norm: bool = True):
        super(SelfAttentionBlock, self).__init__()

        # GroupNorm
        self.GroupNorm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels) if norm else nn.Identity()
        # Self-Attention
        self.Attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_head, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W)
        x = self.GroupNorm(x)
        x = x.transpose(1, 2)
        x, _ = self.Attention(x, x, x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
    
