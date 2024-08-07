# ./models/UNet.py
import torch
import torch.nn as nn


def get_time_bedding(timesteps, embedding_dim: int, device = 'cuda'):
    '''
    Time embedding fuction, refer to code from DDPM and https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch
    :param timesteps: the number of time steps -> (B, )
    :param embedding_dim: the dimension of the embedding -> int

    :return: the time embedding -> (B, embedding_dim)
    '''
    assert embedding_dim % 2 == 0, 'The dimension of the embedding must be even'

    denominator = 10000 ** (2*torch.arange(0, embedding_dim//2).float() / embedding_dim)
    time_embedding = timesteps[:, None].to(device) / denominator[None, :].to(device) # (B, embedding_dim//2)
    time_embedding = torch.cat([torch.sin(time_embedding), torch.cos(time_embedding)], dim=1) # (B, embedding_dim)

    return time_embedding


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, down_sample: bool = True, num_heads: int = 4, num_layers: int = 1):
        super(DownBlock, self).__init__()

        self.num_layers = num_layers
        self.down_sample = down_sample

        # List of modules belonging to the same category
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers)
            ]
        )

        self.time_embedding_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_embedding_dim, out_channels)
                )    
                for _ in range(num_layers)
            ]   
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3, stride=1, padding=1)
                )    
                for _ in range(num_layers)
            ]    
        )

        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads=num_heads, batch_first=True)
                for _ in range(num_layers)
            ]    
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)    
            ]    
        )

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x, time_embedding):
        output = x
        for i in range(self.num_layers):

            # Resnet block of each layer
            # A typical Unet downblock includes two convolution layers and one pooling layer
            resnet_input = output
            output = self.resnet_conv_first[i](output)
            output += self.time_embedding_layers[i](time_embedding)[:, :, None, None]
            output = self.resnet_conv_second[i](output)
            output += self.residual_input_conv[i](resnet_input)

            # Attention block of each layer
            B, C, H, W = output.shape
            attn_in = output.reshape(B, C, H*W)
            attn_in = self.attention_norms[i](attn_in)
            attn_in = attn_in.transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_in, attn_in, attn_in)
            attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
            output = output + attn_out

        output = self.down_sample_conv(output)
        return output
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, num_heads=4, num_layers=1):
        super(MidBlock, self).__init__()

        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers + 1)
            ]   
        )
        self.time_embedding_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_embedding_dim, out_channels)
                )    
                for _ in range(num_layers + 1)
            ]   
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for _ in range(num_layers + 1)
            ]    
        )
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)    
            ]
        )
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads=num_heads, batch_first=True)
                for _ in range(num_layers)
            ]    
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]    
        )

    def forward(self, x, time_embedding):
        output = x

        # first resnet block, and that's why for (num_layers + 1)
        resnet_input = output
        output = self.resnet_conv_first[0](output)
        output += self.time_embedding_layers[0](time_embedding)[:, :, None, None]
        output = self.resnet_conv_second[0](output)
        output += self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):

            # Attention Block
            B, C, H, W = output.shape
            attn_in = output.reshape(B, C, H*W)
            attn_in = self.attention_norms[i](attn_in)
            attn_in = attn_in.transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_in, attn_in, attn_in)
            attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
            output = output + attn_out

            # Resnet Block
            resnet_input = output
            output = self.resnet_conv_first[i+1](output)
            output += self.time_embedding_layers[i+1](time_embedding)[:, :, None, None]
            output = self.resnet_conv_second[i+1](output)
            output += self.residual_input_conv[i+1](resnet_input)

        return output
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, up_sample: bool = True, num_heads: int = 4, num_layers: int = 1):
        super(UpBlock, self).__init__()

        self.num_layers = num_layers
        self.up_sample = up_sample
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers)
            ]   
        )
        self.time_embedding_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_embedding_dim, out_channels)
                )    
                for _ in range(num_layers)
            ]   
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for _ in range(num_layers)
            ]    
        )
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)    
            ]
        )
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads=num_heads, batch_first=True)
                for _ in range(num_layers)
            ]    
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]    
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1) if self.up_sample else nn.Identity()


    def forward(self, x, crop_and_cat_x, time_embedding):
        x = self.up_sample_conv(x)
        x = torch.cat([x, crop_and_cat_x], dim=1)

        output = x
        for i in range(self.num_layers):
            resnet_input = output
            output = self.resnet_conv_first[i](output)
            output += self.time_embedding_layers[i](time_embedding)[:, :, None, None]
            output = self.resnet_conv_second[i](output)
            output += self.residual_input_conv[i](resnet_input)

            # Attention Block
            B, C, H, W = output.shape
            attn_in = output.reshape(B, C, H*W)
            attn_in = self.attention_norms[i](attn_in)
            attn_in = attn_in.transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_in, attn_in, attn_in)
            attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
            output = output + attn_out

        return output

class UNet(nn.Module):
    def __init__(self, model_config):
        super(UNet, self).__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.time_embedding_dim = model_config['time_embedding_dim']
        self.down_samples = model_config['down_samples']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        assert self.mid_channels[0] == self.down_channels[-1], 'The first channel of mid_channels must be equal to the last channel of down_channels'
        assert self.mid_channels[-1] == self.down_channels[-2], 'The last channel of mid_channels must be equal to the second last channel of down_channels'
        assert len(self.down_samples) == len(self.down_channels) - 1, 'The length of down_samples must be equal to the length of down_channels - 1'

        self.time_projection = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim)    
        )

        self.up_sample = list(reversed(self.down_samples))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.down_blocks.append(
                DownBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i+1],
                    time_embedding_dim=self.time_embedding_dim,
                    down_sample=self.down_samples[i],
                    num_layers=self.num_down_layers
                )
            )

        self.mid_blocks = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mid_blocks.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i+1],
                    time_embedding_dim=self.time_embedding_dim,
                    num_layers=self.num_mid_layers
                )
            )

        self.up_blocks = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.up_blocks.append(
                UpBlock(
                    in_channels=self.down_channels[i]*2,
                    out_channels=self.down_channels[i-1] if i != 0 else 16,
                    time_embedding_dim=self.time_embedding_dim,
                    up_sample=self.down_samples[i],
                    num_layers=self.num_up_layers
                )
            )

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, time_embedding):
        '''
        Shapes of downblocks should be [C1, C2, C3, C4]
        Shapes of midblocks should be [C4, C4, C3]
        Shapes of down_samples should be [True, True, False]
        '''
        # x -> (B, C, H, W)
        output = self.conv_in(x) # output -> (B, C1, H, W)

        # time_embedding -> (B, time_embedding_dim)
        time_embedding = get_time_bedding(torch.as_tensor(time_embedding).long(), self.time_embedding_dim, device=x.device)
        time_embedding = self.time_projection(time_embedding)

        down_crop_and_cat = []

        for down_block in self.down_blocks:
            down_crop_and_cat.append(output)
            output = down_block(output, time_embedding)

        # down_crop_and_cat -> [B, C1, H, W], [B, C2, H/2, W/2], [B, C3, H/4, W/4]
        # output -> (B, C4, H/4, W/4)

        for mid_block in self.mid_blocks:
            output = mid_block(output, time_embedding)
        # output -> (B, C3, H/4, W/4)

        for up_block in self.up_blocks:
            down_crop_and_cat_x = down_crop_and_cat.pop()
            output = up_block(output, down_crop_and_cat_x, time_embedding)
            # output -> (B, C2, H/4, W/4), (B, C1, H/2, W/2), (B, 16, H, W)

        output = self.norm_out(output)
        output = nn.SiLU()(output)
        output = self.conv_out(output)
        # output -> (B, C, H, W)

        return output