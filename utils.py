import torch
import numpy as np
import scipy.sparse as sp
import torch.utils.data
import torch_geometric.transforms as T
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork, ppi
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.transforms import NormalizeFeatures 
 
def augment(x, centers):
    center_feat = x[centers]
    perm = torch.randperm(len(centers)) 
    shuffled_feat = center_feat[perm]
    x[centers] = shuffled_feat
    return x

def cal_centrality(X, labels):
    centrality_scores = [
        torch.norm(X[labels == label], dim=1) if (labels == label).any() else 0
        for label in torch.unique(labels)
    ]
    return centrality_scores

def spec_clustering(adj_csr, k, batch_size=100):
    degree_matrix = sp.diags(np.array(adj_csr.sum(axis=1)).flatten())  
    norm_laplacian = sp.csgraph.laplacian(adj_csr, normed=True)  
    
    u, s, vt = svds(norm_laplacian, k=k)  
    eigvals = s[::-1] 
    eigvecs = u[:, ::-1]  
    # eigvals and eigvecs should be returned
    
    X = torch.tensor(eigvecs[:, :k], dtype=torch.float32)
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X.numpy())

    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42, n_init=10)
    labels = minibatch_kmeans.fit_predict(X_scaled)
    labels = torch.tensor(labels, dtype=torch.int64)
    # labels should be returned

    centrality_scores = cal_centrality(X, labels)

    centers = []
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        cluster_indices = torch.where(labels == label)[0]
        if len(cluster_indices) > 0:
            cluster_centrality_scores = centrality_scores[label.item()]
            max_centrality_index = cluster_indices[torch.argmax(cluster_centrality_scores)]
            centers.append(max_centrality_index.item())
        
    return torch.tensor(centers, dtype=torch.int64)

def get_neighbors(edge_index, nodes, order = 1, device = 'cpu'):
    if isinstance(nodes, torch.Tensor) and nodes.dim() == 0:
        nodes = torch.tensor([nodes.item()], device=device)

    neighbors_set = set(nodes.tolist())
    current_nodes = nodes

    edge_index = edge_index.to(device)
    
    for _ in range(order):
        mask = torch.isin(edge_index[0], current_nodes)
        neighbors = edge_index[1, mask].unique()
        neighbors_set.update(neighbors)
        current_nodes = neighbors

    return neighbors_set

def construct_graph(data, centers, num_neighbors = 100, min_neighbors = 50):
    edge_index = data.edge_index.clone()
    center_to_neighbors = {center.item(): set() for center in centers}
    node_to_center = {}

    for center in centers:
        order = 2  
        while order <= 10: 
            neighbors = set(get_neighbors(edge_index, torch.tensor([center]), order=order))
            neighbors.discard(center.item())  

            if len(neighbors) >= min_neighbors:
                break  
            order += 1
        
        if len(neighbors) > num_neighbors:
            selected_neighbors = np.random.choice(list(neighbors), num_neighbors, replace=False)
        else:
            selected_neighbors = list(neighbors)
        
        for neighbor in selected_neighbors:
            if neighbor not in node_to_center:
                center_to_neighbors[center.item()].add(neighbor)
                node_to_center[neighbor] = center.item()

    # Construct the new edge index as a tensor
    selected_edge_index = torch.tensor(
        [[neighbor, center] for center, neighbors in center_to_neighbors.items() for neighbor in neighbors],
        dtype=torch.long
    ).t().contiguous()

    # Reindex the graph
    return reindex_graph(selected_edge_index, centers)

def reindex_graph(edge_index, centers):
    nodes = torch.cat((edge_index[0], edge_index[1])).unique()
    node_map = {node.item(): idx for idx, node in enumerate(nodes)}
    new_edge_index = torch.tensor(
        [[node_map[src.item()], node_map[dst.item()]] for src, dst in edge_index.T], 
        dtype=torch.long
    ).T

    new_centers = torch.tensor(
        [node_map.get(center.item(), -1) for center in centers],  
        dtype=torch.long
    )

    new_centers = new_centers[new_centers != -1]
    new_edge_index, _ = add_self_loops(new_edge_index)
    return new_edge_index, new_centers, nodes

def get_wiki_cs(dataset_dir, transform=NormalizeFeatures()):
    dataset = WikiCS(dataset_dir, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data]

def load_dataset(dataset_name, dataset_dir):

    print('Dataloader: Loading Dataset', dataset_name)
    assert dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'dblp', 'Photo','Computers', 'CS','Physics', 
                'ogbn-products', 'ogbn-arxiv', 'Wiki-CS','ppi',
                'Cornell', 'Texas', 'Wisconsin',
                'chameleon', 'crocodile', 'squirrel']
    
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name, 
                transform=T.NormalizeFeatures())
        
    elif dataset_name == 'dblp':
        dataset = CitationFull(dataset_dir, name=dataset_name, 
                transform=T.NormalizeFeatures())
        
    elif dataset_name in ['Photo','Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name, 
                transform=T.NormalizeFeatures())
        
    elif dataset_name in ['CS','Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name, 
                transform=T.NormalizeFeatures())
    elif dataset_name in ['Wiki-CS']:
        dataset = get_wiki_cs(dataset_dir + "/Wiki")
        
    elif dataset_name in ['ppi']:
        train = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'train')
        val = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'val')
        test = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'test')
        dataset = [train, val, test]   
        
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
            return WebKB(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'crocodile', 'squirrel']:
            return WikipediaNetwork(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    
    print('Dataloader: Loading success.')
    print(dataset[0])
    
    return dataset

