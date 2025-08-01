import torch
from torch import nn

import tinycudann as tcnn
import math

class NGP_2d(nn.Module):
    def __init__(self, config, out_dims=2):
        super().__init__()

        n_input_dims = 2
        grid_level = 19
        feat_dim = 2
        base_resolution = 16
        per_level_scale = 1.5

        self.grid_encoder = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": grid_level,
                "n_features_per_level": feat_dim,
                "log2_hashmap_size": 24,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        
        self.grid_encoder.out_dim = grid_level * feat_dim
        self.decoder = tcnn.Network(
            n_input_dims=self.grid_encoder.out_dim,
            n_output_dims=out_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )


    @torch.no_grad()
    def get_params(self, LR_schedulers):
        params = [
            {'params': self.parameters(), 'lr': LR_schedulers[0]["initial"]}
        ]
        return params

    def forward(self, in_pos):
        """
        Inputs:
            in_pos: (N, 2) xy in [-scale, scale]
        Outputs:
            out: (N, out_dims), the output values
        """
        x = (in_pos - 0.5) * 2.0
        
        grid_x = self.grid_encoder(x)
        out_feat = torch.cat(grid_x, dim=1)
        out_feat = self.decoder(out_feat)

        out_pixel = out_feat.clamp(-1.0, 1.0)
        return out_pixel