import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.output, bias=True)
        self.prelu = torch.nn.PReLU(args.output)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        z = self.prelu(z)
        return z

class Model(torch.nn.Module):
    def __init__(self, encoder: GCN, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: GCN = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, centers: torch.Tensor, 
             mean: bool = True):
        h1 = self.projection(z1[centers])
        h2 = self.projection(z2[centers])

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret