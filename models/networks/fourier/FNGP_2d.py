import torch
from torch import nn
from models.networks.FFB_encoder import FrequencyModulatedHashEncoder
from models.networks.Sine import Sine, sine_init, first_layer_sine_init

import math

def positional_encoding(x, num_frequencies=10, include_input=True):
    frequencies = 2.0 ** torch.arange(num_frequencies, dtype=x.dtype, device=x.device) * math.pi
    x_expanded = x[..., None] * frequencies  # (N, D, F)
    sin = torch.sin(x_expanded)
    cos = torch.cos(x_expanded)
    pe = torch.cat([sin, cos], dim=-1)  # (N, D, 2F)
    pe = pe.view(x.shape[0], -1)  # Flatten

    if include_input:
        return torch.cat([x, pe], dim=-1)
    else:
        return pe

class FourierNGP(nn.Module):
    def __init__(self, config, out_dims=3):
        super().__init__()
        print("\n======= Initializing FourierNGP =======")
        print(f"Config: {config}")
        
        # 初始化编码器
        print("\nInitializing xyz_encoder...")
        self.xyz_encoder = FrequencyModulatedHashEncoder(
            n_input_dims=2, 
            encoding_config=config["encoding"],
            network_config=config["SIREN"], 
            has_out=False
        )
        print(f"Encoder output dim: {self.xyz_encoder.out_dim}")
        
        self.num_freq = config["encoding"]["num_freq"]
        self.include_input = config["encoding"]["include_input"]
        if self.include_input:
            pe_dim = 2 *  (2 * self.num_freq + 1)
        else:
            pe_dim = 2 * 2* self.num_freq
            
        # 构建backbone网络
        backbone_dims = config["Backbone"]["dims"]
        grid_feat_len = self.xyz_encoder.out_dim
        backbone_dims = [grid_feat_len + pe_dim] + backbone_dims + [out_dims]
        self.num_backbone_layers = len(backbone_dims)
        
        print("\nBuilding backbone layers:")
        for layer in range(0, self.num_backbone_layers - 1):
            in_dim = backbone_dims[layer]
            out_dim = backbone_dims[layer + 1]
            setattr(self, "backbone_lin" + str(layer), nn.Linear(in_dim, out_dim))
            print(f"Layer {layer}: Linear({in_dim}, {out_dim})")

        # 初始化激活函数
        self.activation = Sine(w0=config["SIREN"]["w0"])
        print(f"\nUsing Sine activation with w0={config['SIREN']['w0']}")

        # 初始化权重
        self.init_siren()
        print("="*50 + "\n")
        
        

    def init_siren(self):
        for layer in range(self.num_backbone_layers - 1):
            lin = getattr(self, f"backbone_lin{layer}")
            if layer == 0:
                first_layer_sine_init(lin)
                print(f"First layer {layer} initialized with first_layer_sine_init")
            else:
                sine_init(lin, w0=self.activation.w0)
                print(f"Layer {layer} initialized with sine_init (w0={self.activation.w0})")
        
    def forward(self, in_pos):
        """
        Inputs:
            x: (N, 2) xy in [-scale, scale]
        Outputs:
            out: (N, 1 or 3), the RGB values
        """
        x = (in_pos - 0.5) * 2.0
        if self.num_freq > 0:
            in_pos = positional_encoding(in_pos, num_frequencies=self.num_freq, include_input=self.include_input)
        else:
            in_pos = in_pos.view(in_pos.shape[0], -1)
            
        grid_feature = self.xyz_encoder(x)
        grid_feature = torch.cat([grid_feature, in_pos], dim=-1)  # Concatenate input position
        
        
        for layer in range(0, self.num_backbone_layers - 1):
            lin = getattr(self, "backbone_lin" + str(layer))
            grid_feature = lin(grid_feature)
            if layer < self.num_backbone_layers - 2:
                grid_feature = self.activation(grid_feature)

        out_feat = grid_feature.clamp(-1.0, 1.0)

        return out_feat

    @torch.no_grad()
    # optimizer utils
    def get_params(self, LR_schedulers):
        params = [
            {'params': self.parameters(), 'lr': LR_schedulers[0]["initial"]}
        ]

        return params        