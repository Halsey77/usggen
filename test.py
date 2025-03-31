import numpy as np
import torch


num_nodes = 3

# Create random edges_from and edges_to tensors
edges_from = torch.randint(0, 10, size=(num_nodes, num_nodes), dtype=torch.long)
edges_from = torch.triu(edges_from, diagonal=1)  # Keep only upper triangular part
edges_to = torch.randint(0, 10, size=(num_nodes, num_nodes), dtype=torch.long)
edges_to = torch.triu(edges_to, diagonal=1)  # Keep only upper triangular part

print("edges_from:")
print(edges_from)

print("edges_to:")
print(edges_to)

adj_matrix = edges_to + np.transpose(edges_from)
print("adj_matrix:")
print(adj_matrix)

new_e_from = torch.triu(np.transpose(adj_matrix), diagonal=1)
print("new_e_from:")
print(new_e_from)

new_e_to = torch.triu(adj_matrix, diagonal=1)
print("new_e_to:")
print(new_e_to)