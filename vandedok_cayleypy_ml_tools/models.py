from pydantic import BaseModel
import torch
from torch import nn


class PermMLParams(BaseModel):
        hidden_dims: list[int] = [4096]
        dropout: float | None = None
        layer_norm: bool = False

class PermMLP(nn.Module):
    def __init__(
            self, 
            state_size: int,
            state_vocab_size: int,
            y_norm: float,
            hidden_dims: list[int] = [4096],
            dropout: int| None = None,
            layer_norm: bool = False,
           
        ):
        super(PermMLP, self).__init__()
        

        layers_list = []
        self.state_size = state_size
        self.state_vocab_size = state_vocab_size
        
        input_dim = self.state_size * self.state_vocab_size
        if layer_norm:
            layers_list.append(nn.LayerNorm(input_dim))
        
      
        for hidden_dim in hidden_dims:
            layers_list.append(nn.Linear(input_dim, hidden_dim))
            layers_list.append(nn.ReLU()),
            if dropout is not None:
                layers_list.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers_list.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.Sequential(*layers_list)
        self.y_norm=y_norm

        self.set_cayleypy_inference(False)

    def set_cayleypy_inference(self, which: bool):
        if which:
            self.final_squeeze = True
        else:
            self.final_squeeze = False

    def forward(self, x):
        x = nn.functional.one_hot(x.long(), num_classes= self.state_vocab_size).float().reshape(x.shape[0], -1)
        if self.training:
            y = self.layers(x)
        else:
            y = self.layers(x) * self.y_norm
        
        if self.final_squeeze:
            return y.squeeze(-1)
        else:
            return y
        

class StateDistanceTransformerParams(BaseModel):
        embed_dim: int =128
        num_heads: int = 4
        num_layers: int = 4
        ff_dim: int = 256
        dropout: float =0.1

class StateDistanceTransformer(nn.Module):
    def __init__(
        self, 
        state_size: int, 
        state_vocab_size: int,
        y_norm: float,
        embed_dim: int =128, 
        num_heads: int = 4, 
        num_layers: int = 4, 
        ff_dim: int = 256, 
        dropout: float =0.1,
        # vocab_size: int =88
        
    ):
    
        super().__init__()

        self.state_size = state_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(state_vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, state_size, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling + output
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # регрессия на расстояние
        )
        self.y_norm = y_norm

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding  # (batch, state_size, embed_dim)
        x = self.transformer(x)                     # (batch, state_size, embed_dim)
        x = x.mean(dim=1)                           # mean pooling по позициям
        out = self.fc_out(x).squeeze(-1)           # (batch,)
        if self.training:
            return out
        else:
            return out * self.y_norm