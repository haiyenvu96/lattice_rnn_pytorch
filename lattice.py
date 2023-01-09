import numpy as np


class Lattice:
    def __init__(self, nodes=None, edges=None, tags = None, target_nodes = None):
        self.nodes = nodes
        self.node2index = {node: index for index, node in enumerate(self.nodes)}
        self.edges = edges
        self.tags = tags
        self.parent2child = {}
        self.child2parent = {}
        if self.edges is not None:
            for i, edge in enumerate(self.edges):
                node_from, node_to = edge
                if node_from not in self.parent2child:
                    self.parent2child[node_from] = {}
                self.parent2child[node_from][node_to] = i
                if node_to not in self.child2parent:
                    self.child2parent[node_to] = {}
                self.child2parent[node_to][node_from] = i
        self.target_nodes = target_nodes

def load_lattices(path):
    data = np.load(path, allow_pickle=True)
    nodes = data['nodes']
    edges = data['edges']
    tags = data['tags']
    target_nodes = data['target_nodes']
    list_of_lattices = []
    for i in range(len(nodes)):
        lattice = Lattice(nodes=nodes[i], edges=edges[i], tags=tags[i], target_nodes=target_nodes[i])
        list_of_lattices.append(lattice)
    return list_of_lattices

def drawLattice(lattice, target_nodes = None):
    from graphviz import Digraph
    if target_nodes is None:
        target_nodes = lattice.target_nodes
    # Create Digraph object
    dot = Digraph()
    # Add nodes
    node2index = {}
    for i, node in enumerate(lattice.nodes):
        node2index[node] = i
        if target_nodes[i] == 1:
            dot.node(str(node), label = lattice.tags[i], color = 'red')
        else:
            dot.node(str(node), label = lattice.tags[i])

    # Add edges
    for edge in lattice.edges:
        if target_nodes[node2index[edge[0]]] == 1:
            if edge[1] == 't':
                dot.edge(str(edge[0]), str(edge[1]), color = 'red')
            else:
                if target_nodes[node2index[edge[1]]] == 1:
                    dot.edge(str(edge[0]), str(edge[1]), color = 'red')
                else:
                    dot.edge(str(edge[0]), str(edge[1]))
        else:
            dot.edge(str(edge[0]), str(edge[1]))

    # Visualize the graph
    return dot

def load_dictionary(path):
    tag2idx = {}
    with open(path) as f:
        idx = 0
        for line in f:
            tag = line.split("\t")[0]
            tag2idx[tag] = int(idx)
            idx += 1
    print("Dictionary length:", len(tag2idx))
    return tag2idx