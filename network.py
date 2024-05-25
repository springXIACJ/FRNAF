import torch
import torch.nn as nn


class DensityNetwork(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, embedding_dim_pos=16,T=1000,hidden_dim=256, skips=[4], out_dim=1, last_activation="sigmoid"):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        self.embedding_dim_pos=embedding_dim_pos
        self.T = T
        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips 
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers-1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(0, num_layers-1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")
    def positional_encoding(self, x, L, step, is_pos=False):
        out = x
        # for j in range(L):
        #     out.append(torch.sin(2 ** j * x))
        #     out.append(torch.cos(2 ** j * x))
        # out = torch.cat(out, dim=1)

        Lmax = self.in_dim
        if is_pos:
            # out[:, int(step / self.T * Lmax)+2:int(step / self.T * Lmax) + 4] = torch.multiply(out[:, int(step / self.T * Lmax)+2:int(step / self.T * Lmax) + 4] , step / self.T * Lmax-int(step / self.T * Lmax))
            out[:, int(step / self.T * Lmax) + 2:] = 0.
        return out
    def forward(self, x,step):
        hash_x = self.encoder(x, self.bound)
        x= self.positional_encoding(hash_x, self.embedding_dim_pos, step, is_pos=True)
        #T=0 
        #x= self.positional_encoding(hash_x, self.embedding_dim_pos, step, is_pos=False)
        # print("emb_x",emb_x.shape)
        # print("hash_x",hash_x.shape)
        #x=torch.cat([emb_x, hash_x], dim=-1)
        #x= self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)
        
        return x
    