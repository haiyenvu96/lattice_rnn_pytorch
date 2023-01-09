import torch
import torch.nn as nn

class LatticeRNN(nn.Module):
    def __init__(self, input_dim=128, output_dim=1, hidden_dim=128, n_dictionary=9455):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_dictionary, input_dim)
        self.RNNCell = nn.RNNCell(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_lattice, device='cpu'):
        tag_indices = torch.tensor(input_lattice.tags_idx).to(device)
        embedded_input = self.embedding(tag_indices)

        h_hidden_nodes = []
        for i, node in enumerate(input_lattice.nodes):
            if i == 0:
                z_hidden = torch.zeros(1, self.hidden_dim).to(device)
            else:
                in_h_hiddens = torch.cat([
                                    h_hidden_nodes[input_lattice.node2index[parent_node]]
                                        for parent_node in input_lattice.child2parent[node]
                                    ], dim=0).to(device)
                z_hidden = torch.mean(in_h_hiddens, 0, keepdim=True)
            h_hidden = self.RNNCell(embedded_input[i].view(1, -1), z_hidden)
            h_hidden_nodes.append(h_hidden)
        h_hidden_nodes = torch.cat(h_hidden_nodes, 0)

        output_nodes = self.output_proj(h_hidden_nodes)
        output_nodes = self.sigmoid(output_nodes)

        return output_nodes