import pdb
import random
import warnings

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
from sklearn.neighbors import NearestNeighbors
from tnetwork import ComScenario
from tnetwork.experiments.experiments import *
from tqdm.notebook import tqdm

from scipy.sparse.linalg import eigsh
from collections import defaultdict
from itertools import product

warnings.simplefilter(action='ignore', category=FutureWarning)

colors_dict = {'blue': 0,
               'red': 1,
               'gray': 2}

dict_colors = {0: 'blue', 1: 'red', 2: 'gray'}


def get_efficient_eigenvectors(G, number_of_eigenvectors=20):
    laplacian_matrix = nx.laplacian_matrix(G).astype(np.float32)
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=number_of_eigenvectors, which='SM')
    return eigenvectors, eigenvalues


def create_edge_df(graphs):
    edge_list = []
    for i, graph in enumerate(graphs):
        for edge in graph.edges:
            a, b = edge
            state_a, state_b = graph.nodes[a]['state'], graph.nodes[b]['state']
            edge_list.append((a, b, i, state_a, state_b))

    df = pd.DataFrame(edge_list, columns=['a', 'b', 'i', 'state_a', 'state_b'])
    return df


def create_graphs_from_df(df):
    graphs = []
    for i in tqdm(df['i'].unique()):
        graph = nx.Graph()
        for _, row in tqdm(df[df['i'] == i].iterrows(), leave=False):
            a, b, state_a, state_b = row[['a', 'b', 'state_a', 'state_b']]
            if a not in graph.nodes:
                graph.add_node(a, state=state_a)
            if b not in graph.nodes:
                graph.add_node(b, state=state_b)
            graph.add_edge(a, b)
        graphs.append(graph)
    return graphs


# Create traces for nodes and edges
def draw_cloud(G, points, color=None, fig=None, draw_edges=True, sz=1, width=1000, height=500, zoom_out=2, title=None):
    x = np.zeros(len(points))
    y, z = x.copy(), x.copy()
    color_was_none = color is None
    if color_was_none:
        color = ['gray'] * len(points)
    for i, node in enumerate(G.nodes()):
        x[node] = points[node][0]
        y[node] = points[node][1]
        z[node] = points[node][2]
        if color_was_none:
            color[node] = dict_colors[G.nodes[node]['community']]
    try:
        name = 'Points: ' + ', '.join([f'{str(color.count(c)).zfill(5)} {c}' for c in set(color.values())])
        if 'gray' not in name:
            name += ', 00000 gray'
    except:
        name = ''

    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=sz, color=color),
        name=name
    )

    if draw_edges:
        edge_x = []
        edge_y = []
        edge_z = []

        # Extract edge coordinates
        for edge in G.edges():
            x0, y0, z0 = points[edge[0]]
            x1, y1, z1 = points[edge[1]]
            edge_x.extend([x0, x1, None])  # None creates a break in the line
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        # Create trace for edges
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=1)
        )

    if fig is None:
        fig = go.Figure()

    # Add traces to the figure
    fig.add_trace(node_trace)
    if draw_edges:
        fig.add_trace(edge_trace)

    if title:
        fig.update_layout(
            title=title,
            titlefont=dict(size=18),  # Adjust the font size of the title
            title_y=0.92  # Adjust the y-position of the title (value between 0 and 1)
        )

    fig.update_layout(scene=dict(
        aspectmode='data',
        camera=dict(
            eye=dict(x=0, y=0, z=zoom_out),  # Zoom out by adjusting 'x' value
            up=dict(x=0, y=1, z=0),  # Change orientation by adjusting 'up' vector
            center=dict(x=0, y=0, z=0)  # Set the center of rotation
        )
    ), width=width, height=height, margin=dict(
        l=5, r=5, b=10, t=10, pad=4
    ))
    return fig


class TorusDistance:
    def __init__(self, points):
        """
        Initialize the TorusDistance object.

        Args:
            points (numpy.ndarray): An array of 2D points on the torus grid.

        Assumptions:
            - The grid is defined on the xy-plane.
            - The points are equally spaced along each axis.
            - The grid has the same number of points along the x and y axes.
        """
        points = np.asarray(points)
        self.N = abs(points[:, 0].max() - points[:, 0].min())
        self.M = abs(points[:, 1].max() - points[:, 1].min())
        self.delta = self.M / np.sqrt(len(points))

    def __call__(self, point1, point2):
        """
        Calculate the torus distance between two points.

        Args:
            point1 (numpy.ndarray or list): A 1D array or list representing the first point.
            point2 (numpy.ndarray or list): A 1D array or list representing the second point.

        Returns:
            float: The torus distance between the two points.
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)

        regular_x_distance = np.abs(point1[0] - point2[0])
        regular_y_distance = np.abs(point1[1] - point2[1])
        dx = np.minimum(round(regular_x_distance, 15), round(self.N - regular_x_distance + self.delta, 15))
        dy = np.minimum(round(regular_y_distance, 15), round(self.M - regular_y_distance + self.delta, 15))

        return np.sqrt(dx ** 2 + dy ** 2)


# Generate random points (replace with your actual data)

def create_graph_from_clouds(indices, points, k=5, majority=False):
    if isinstance(k, int):
        k_values = [k] * len(points)
    else:
        k_values = k

    G = nx.Graph()
    for i in tqdm(range(len(points))):
        G.add_node(i, xyz=points[i])  # Add the point as an attribute to the node
        current_node_degree = k_values[i]
        for j in indices[i][:current_node_degree]:
            if i != j:  # Avoid self-loops
                G.add_edge(i, j)

    if majority:
        node_color = [np.random.choice(['blue', 'red', 'gray'], p=[0.1, 0.1, 0.8]) for _ in G.nodes()]
    else:
        node_color = [random.choice(['gray']) for _ in G.nodes()]

    for i, node in enumerate(G.nodes()):
        G.nodes[node]['community'] = colors_dict[node_color[i]]

    for i, node in enumerate(G.nodes()):
        G.nodes[node]['state'] = 0

    return G


def run_knn_once(points, max_k):
    torus_metric = TorusDistance(points)
    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='ball_tree', metric=torus_metric).fit(points)
    distances, indices = nbrs.kneighbors(points)
    return indices


# Function to update node communities based on neighbors
def update_communities(G, a, b, past_communities=None, history=3):
    d = {}
    if past_communities is None:
        past_communities = defaultdict(list)
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            community_counts = {0: 0, 1: 0}
            for neighbor in neighbors:
                community = G.nodes[neighbor]['community']
                if community in community_counts.keys():
                    community_counts[community] += 1
            past_communities[node].append(community_counts)
            past_communities[node] = past_communities[node][-history:]
            zero_count, one_count = 0, 0
            for i in range(len(past_communities[node])):
                zero_count += past_communities[node][i][0]
                one_count += past_communities[node][i][1]
            if zero_count == one_count:
                most_common_community = G.nodes[node]['community']
            else:
                most_common_community = int(one_count > zero_count)

            r = np.random.rand()
            if r < a:
                new_community = G.nodes[node]['community']
            elif r < a + b:
                new_community = 1 - G.nodes[node]['community']
            else:
                new_community = most_common_community

            d.update({node: new_community})
    for node, new_community in d.items():
        G.nodes[node]['community'] = new_community

    return past_communities


# Create a grid graph using NetworkX
def create_grid_graph(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    return G


# Update the state of each cell based on Conway's Game of Life rules
def update_grid(G, temporal, resource_stock, resource, max_age=100, critical_survival_period=2, ts=None):
    """Update the state of each cell in the grid G according to the rules of Conway's Game of Life.

    Args:
        G (networkx.Graph): The grid graph representing the current state of the game.
        temporal (bool): Whether to apply temporal rules for cell survival.
        resource_stock (int): The current stock of resources.
        resource (bool): Whether to apply resource dynamics.

    Returns:
        int: The updated resource stock after applying the rules.
    """
    next_state = {}
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for node in nodes:
        live_neighbors = 0
        neighbors = list(G.neighbors(node))
        k = len(neighbors)
        for neighbor in neighbors:
            if G.nodes[neighbor]['state'] == 1:
                live_neighbors += 1
        if 'memory' not in G.nodes[node]:
            G.nodes[node]['memory'] = 0

        lower_bound = max(2 * k / 8, 2)
        upper_bound = max(3 * k / 8, 3)
        next_state[node] = 1 if lower_bound <= live_neighbors <= upper_bound else 0
        # print('ts', ts, 'node', node, 'lower_bound', lower_bound, 'upper_bound', upper_bound, 'live_neighbors',
        #       live_neighbors, 'number_of_neighbors', k)

        time_alive = G.nodes[node]['memory']
        if temporal:
            if time_alive > max_age:
                next_state[node] = 0
            elif time_alive > critical_survival_period:
                next_state[node] = 1

        # Resource dynamics
        if resource:
            if resource_stock > 0:
                if next_state[node] == 1:
                    resource_stock -= time_alive
            else:
                next_state[node] = 0

        G.nodes[node]['memory'] = G.nodes[node]['memory'] + 1 if next_state[node] == 1 else 0

    for node, state in next_state.items():
        G.nodes[node]['state'] = state
    return resource_stock


def set_graph_colors_under_name(G, name, colors):
    for i, node in enumerate(G.nodes()):
        G.nodes[node][name] = colors[i]


def transfer_att_from_graph(from_g, to_g):
    for node in from_g.nodes:
        for key, value in from_g.nodes[node].items():
            to_g.nodes[node][key] = value
    return to_g


def create_unified_graph(graphs, circular=False, enumaraete=False):
    unified_graph = nx.Graph()

    # Iterate through each graph
    for i, graph in enumerate(graphs):
        # Add nodes with community information
        for node in graph.nodes():
            if 'state' not in graph.nodes[node].keys():
                unified_graph.add_node((i, node))
            else:
                unified_graph.add_node((i, node), state=graph.nodes[node]['state'])

        # Add edges
        for edge in graph.edges():
            unified_graph.add_edge((i, edge[0]), (i, edge[1]))

    for i, graph in enumerate(graphs):
        # Connect each node to its previous and next counterparts
        for node in graph.nodes():
            if i > 0:
                unified_graph.add_edge((i, node), (i - 1, node))
            # Connect to next counterpart
            if i < len(graphs) - 1:
                unified_graph.add_edge((i, node), (i + 1, node))

            if circular:
                last_layer = len(graphs) - 1
                first_layer = 0
                if i == first_layer:
                    unified_graph.add_edge((first_layer, node), (last_layer, node))
                if i == last_layer:
                    unified_graph.add_edge((last_layer, node), (first_layer, node))

    if enumaraete:
        num_nodes = len(graphs[0].nodes)

        def custom_mapping(node_label):
            i, t = node_label
            new_label = i * num_nodes + t
            return new_label

        # Generate the new labels using the custom mapping
        mapping = {node: custom_mapping(node) for node in unified_graph.nodes()}

        # Relabel the nodes in the graph G using the generated mapping
        unified_graph = nx.relabel_nodes(unified_graph, mapping, copy=True)

    return unified_graph


def expand_points(points, layers, shift=10):
    expanded_points = []
    for i in range(layers):
        for point in points:
            base_x, base_y, base_z = point
            expanded_points.append((base_x, base_y, base_z + shift * (i + 1)))
    return expanded_points


def get_V_F_of_obj(obj_path: str):
    V, F = [], []
    with open(obj_path) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                F.append([int(x) for x in values[1:4]])
    full_V, full_F = np.array(V), np.array(F) - 1
    return full_V, full_F


def linear_interpolation_np(set_a, set_b, t):
    if set_a.shape != set_b.shape:
        raise ValueError("Input sets must be of the same shape")

    set_c = set_a + (set_b - set_a) * t
    return set_c


def create_grid(x_range, y_range, resolution, z_const):
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_values = np.arange(x_min, round(x_max + resolution, 15), resolution)
    y_values = np.arange(y_min, round(y_max + resolution, 15), resolution)
    points = [(x, y, z_const) for x, y in product(x_values, y_values)]

    return points


def generate_toy_random_network(**kwargs):
    """
    Generate a small, toy dynamic graph

    Generate a toy dynamic graph with evolving communities, following scenario described in XXX
    Optional parameters are the same as those passed to the ComScenario class to generate custom scenarios

    :return: pair, (dynamic graph, dynamic reference partition) (as snapshots)
    """
    my_scenario = ComScenario(**kwargs)

    # Initialization with 4 communities of different sizes
    [A, B, C, T] = my_scenario.INITIALIZE([5, 8, 20, 8],
                                          ["A", "B", "C", "T"])
    # Create a theseus ship after 20 steps
    (T, U) = my_scenario.THESEUS(T, delay=20)

    # Merge two of the original communities after 30 steps
    B = my_scenario.MERGE([A, B], B.label(), delay=30)

    # Split a community of size 20 in 2 communities of size 15 and 5
    (C, C1) = my_scenario.SPLIT(C, ["C", "C1"], [15, 5], delay=75)

    # Split again the largest one, 40 steps after the end of the first split
    (C1, C2) = my_scenario.SPLIT(C, ["C", "C2"], [10, 5], delay=40)

    # Merge the smallest community created by the split, and the one created by the first merge
    my_scenario.MERGE([C2, B], B.label(), delay=20)

    # Make a new community appear with 5 nodes, disappear and reappear twice, grow by 5 nodes and disappear
    R = my_scenario.BIRTH(5, t=25, label="R")
    R = my_scenario.RESURGENCE(R, delay=10)
    R = my_scenario.RESURGENCE(R, delay=10)
    R = my_scenario.RESURGENCE(R, delay=10)

    # Make the resurgent community grow by 5 nodes 4 timesteps after being ready
    R = my_scenario.GROW_ITERATIVE(R, 5, delay=4)

    # Kill the community grown above, 10 steps after the end of the addition of the last node
    my_scenario.DEATH(R, delay=10)

    (dyn_graph, dyn_com) = my_scenario.run()
    dyn_graph_sn = dyn_graph.to_DynGraphSN(slices=1)
    GT_as_sn = dyn_com.to_DynCommunitiesSN(slices=1)
    return dyn_graph_sn, GT_as_sn


# Function to map sets to colors
def map_sets_to_colors(sets):
    color_map = {}
    assigned_colors = []

    for s in sets:
        # Convert the set to a tuple (hashable)
        s_tuple = tuple(s)

        # Check if the set is already in the color_map
        if s_tuple in color_map:
            assigned_colors.append(color_map[s_tuple])
        else:
            # Generate a new random RGB color
            r = random.randint(0, 255)
            color = r

            color_map[s_tuple] = color
            assigned_colors.append(color)

    return assigned_colors
