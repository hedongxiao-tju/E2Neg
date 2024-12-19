import torch
import argparse
import random
import numpy as np
import scipy.sparse as sp
from networks import Model, GCN
from utils import load_dataset, augment, construct_graph, spec_clustering
from eval import label_classification

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PubMed', help='dataset')
parser.add_argument('--log_dir', type=str, default='./log/', help='./log/')
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--output', type=int, default=1024, help='output size')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--cluster', type=int, default=50, help='numbers of cluster')
parser.add_argument('--num_neighbors', type=int, default=100, help='numbers of neighbors in every cluster')
args = parser.parse_args()

with open(args.log_dir + args.dataset + ".txt", 'a') as f:
    f.write('****'*20+'\n')
    f.write('\n\n'+'##'*20+'\n')
    f.write(str(args) + '\n')

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = load_dataset(args.dataset, '../../datasets/')
data = datasets[0]
args.num_features = data.num_features
edge_index = data.edge_index.numpy()
adj_csr = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(data.num_nodes, data.num_nodes)).tocsr()

centers = spec_clustering(adj_csr, args.cluster, batch_size = 2000)
new_edge_index, new_centers, node_indices = construct_graph(data, centers, args.num_neighbors) 

data = data.to(device)
sub_x = data.x[node_indices]
new_edge_index = new_edge_index.to(device)
new_centers = new_centers.to(device)
encoder = GCN(args).to(device)
model = Model(encoder, args.output, args.output, 0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()

print(f'New number of nodes: {sub_x.shape[0]}')
print(f'New number of edges: {new_edge_index.shape[1]}')

for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    
    sub_aug_x = augment(sub_x, new_centers)
    
    z1 = model(sub_x, new_edge_index) # original view
    z2 = model(sub_aug_x, new_edge_index) # augmented view
    
    loss = model.loss(z1, z2, new_centers)
    loss.backward()
    optimizer.step()
    
with torch.no_grad():
    emb = model(data.x, data.edge_index)
result = label_classification(emb, data.y, ratio=0.1)

with open(args.log_dir + args.dataset + ".txt", 'a') as f:
    f.write('epoch: ' + str(epoch) + '\n')
    f.write(str(result) + '\n')
print('-----------------')

