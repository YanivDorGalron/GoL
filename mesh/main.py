import random

import numpy as np
from plotly_gif import GIF
from tqdm import tqdm

from mesh.utils import create_graph_from_clouds, draw_cloud, update_communities, update_grid, dict_colors, \
    transfer_att_from_graph, create_edge_df, get_V_F_of_obj, create_grid


def consensus_dynamics(points_type='grid',
                       obj_path=None,
                       sample_size=20,
                       seed=42,
                       initial_k=9,
                       majority=True,
                       sz=2,
                       steps=50,
                       history=10,
                       a=0, b=0, length=500, prefix=''):
    name = f'{prefix}_consensus_dynamics_seed_{seed}_initial_k_{initial_k}_steps_{steps}_history_{history}_a_{a}_b_{b}'
    partial_V = get_points(points_type, obj_path, sample_size)
    np.random.seed(seed)
    G = create_graph_from_clouds(partial_V, k=initial_k, majority=majority)  # same as grid
    fig = draw_cloud(G, partial_V, sz=sz)

    gif = GIF(verbose=False)
    gif.create_image(fig)
    past_communities = None
    for _ in tqdm(range(steps)):
        past_communities = update_communities(G, a=a, b=b, past_communities=past_communities, history=history)
        fig = draw_cloud(G, partial_V, sz=sz)
        gif.create_image(fig)
    gif.create_gif(gif_path=f'saved/{name}.gif', length=length * steps)


def space_invariant_GoL(points_type='obj', obj_path=None, seed=42, sz=2, steps=100, k_delta=20, kmin=4,
                        w=0.1, temporal=True,
                        sample_size=20, oscilations=True, create_gif=True, critical_survival_period=2,
                        max_age=10, save_df=False, use_resource=False, length=500, prefix='', red_nodes=None):
    valid_k = [9, 13, 21, 25, 5]
    initial_k = valid_k[0]
    name = f'{prefix}_space_invariant_GoL_seed_{seed}_initial_k_{initial_k}_steps_{steps}_k_delta_{k_delta}_kmin_{kmin}_w_{w}_oscilations_{oscilations}'
    np.random.seed(seed)
    partial_V = get_points(points_type, obj_path, sample_size)
    G, color, k, partial_V, sorted_nodes = create_and_color_graph(partial_V, initial_k, red_nodes)

    gif = GIF(verbose=False)

    fig = draw_cloud(G, partial_V, color, sz=sz, title='Start')
    gif.create_image(fig)  # create_gif image for gif
    resource_stock = len(G.nodes) // 2  # 5000
    # Run the game for a few steps and save each step as an image
    Graphs = [G.copy()]
    for i in tqdm(range(steps)):
        resource_stock = update_grid(G, temporal=temporal, resource=use_resource, resource_stock=resource_stock,
                                     max_age=max_age, critical_survival_period=critical_survival_period)
        resource_stock = resource_stock + round(len(G.nodes) // 5)

        color = [dict_colors[G.nodes[node]['state']] for node in sorted_nodes]
        fig = draw_cloud(G, partial_V, color, sz=sz, title=f'index: {i}')
        Graphs.append(G.copy())

        if oscilations:
            k = valid_k[(i + 1) % len(valid_k)]
            # k = round(k_delta * (np.sin(i * w) + 1) / 2 + kmin) - round(k_delta / 2 + kmin - initial_k)
            G1 = create_graph_from_clouds(partial_V, k)
            G = transfer_att_from_graph(G, G1)

        if create_gif:
            gif.create_image(fig)

    if create_gif:
        gif.create_gif(gif_path=f'saved/{name}.gif', length=length * steps)

    if save_df:
        df = create_edge_df(Graphs)
        df.to_csv(f'saved/{name}.csv')


def create_and_color_graph(partial_V, initial_k, red_nodes):
    k = initial_k
    G = create_graph_from_clouds(partial_V, k=k)
    color, sorted_nodes = color_graph(G, partial_V, red_nodes)
    return G, color, k, partial_V, sorted_nodes


def color_graph(G, partial_V, red_nodes):
    color = ['gray'] * len(partial_V)
    sorted_nodes = list(G.nodes())
    sorted_nodes.sort()
    for node in sorted_nodes:
        G.nodes[node]['state'] = 0
        if red_nodes:
            if node in red_nodes:
                G.nodes[node]['state'] = 1
        else:
            G.nodes[node]['state'] = np.random.choice([0, 1], p=[0.95, 0.05])

        color[node] = dict_colors[G.nodes[node]['state']]
    return color, sorted_nodes


def sample_torus(R, r, n_points=100):
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, 2 * np.pi, n_points)
    u, v = np.meshgrid(u, v)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return points


def get_points(points_type, obj_path=None, sample_size=None):
    if points_type == 'obj':
        assert obj_path is not None
        assert sample_size is not None
        V, F = get_V_F_of_obj(obj_path=obj_path)
        partial_v = V[::sample_size]
    elif points_type == 'grid':
        partial_v = create_grid((-0.2, 0.2), (-0.2, 0.2), 1e-2, 2)
    elif points_type == 'torus':
        partial_v = sample_torus(R=5, r=2, n_points=200)
    else:
        raise ValueError('Points type could be either obj, grid or torus')
    return partial_v


def past_dependent_GoL(points_type='grid', obj_path=None, seed=42, sz=2, steps=100, kmax=24, kmin=4, w=1,
                       sample_size=20, create_gif=True, save_df=False, use_resource=False, prefix='', red_nodes=None):
    # kmax = 25
    # kmin = 5
    name = f'{prefix}_past_dependent_GoL_seed_{seed}_steps_{steps}_kmax_{kmax}_kmin_{kmin}_w_{w}_use_resource_{use_resource}'
    np.random.seed(seed)
    partial_V = get_points(points_type, obj_path, sample_size)

    # valid_k = [9, 13, 21, 25, 5]
    # k = np.random.choice(valid_k, size=len(partial_V))
    k = np.array([random.randint(kmin, kmax) for _ in range(len(partial_V))])
    G = create_graph_from_clouds(partial_V, k=k)

    color, sorted_nodes = color_graph(G, partial_V, red_nodes)

    gif = GIF(verbose=False)

    fig = draw_cloud(G, partial_V, color, sz=sz)
    gif.create_image(fig)  # create_gif image for gif
    resource_stock = len(G.nodes) // 2  # 5000
    # Run the game for a few steps and save each step as an image
    Graphs = [G.copy()]
    k_history = [k]
    for i in tqdm(range(steps)):
        resource_stock = update_grid(G, temporal=True, resource=use_resource, resource_stock=resource_stock)
        resource_stock = resource_stock + round(len(G.nodes) // 5)
        color = [dict_colors[G.nodes[node]['state']] for node in sorted_nodes]
        fig = draw_cloud(G, partial_V, color, sz=sz)
        Graphs.append(G.copy())

        if len(k_history) < 2:
            # k = np.random.choice(valid_k, size=len(partial_V))
            k = np.array([random.randint(kmin, kmax) for _ in range(len(partial_V))])
        else:
            k_history = k_history[-2:]
            k = (kmin + (kmax - kmin) * (0.5 + np.sin(w * k_history[0] * k_history[1]) / 2)).astype(int)
        G1 = create_graph_from_clouds(partial_V, k)
        G = transfer_att_from_graph(G, G1)
        k_history.append(k)

        if create_gif:
            gif.create_image(fig)

    if create_gif:
        gif.create_gif(gif_path=f'saved/{name}.gif', length=500 * steps)

    if save_df:
        df = create_edge_df(Graphs)
        df.to_csv(f'saved/{name}.csv')
