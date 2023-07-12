import torch
import torchvision
import matplotlib.pyplot as plt 
import seaborn as sns
import networkx as nx
import numpy as np


@torch.no_grad()
def visualize_ae_reconstruction(model, images, actions=None):
    """ Plots reconstructions of an autoencoder """
    reconst = model(images, actions=actions)
    if not isinstance(reconst, torch.Tensor):
        reconst = reconst.mean()
    images = torch.stack([images, reconst], dim=1).flatten(0, 1)
    if images.shape[1] == 12:
        images = flatten_causal_world_images(images)
    img_grid = torchvision.utils.make_grid(images, nrow=2, normalize=True, pad_value=0.25, value_range=(-1, 1))
    img_grid = img_grid.permute(1, 2, 0)
    img_grid = img_grid.cpu().numpy()
    return img_grid

def flatten_causal_world_images(images):
    images = images.unflatten(1, (3, 4))
    diffs = images[:,:,-1]
    diffs = torch.stack([diffs * (diffs > 0).float(),
                         -diffs * (diffs < 0).float(),
                         diffs * 0.], dim=2) * 2. - 1.
    obs = images[:,:,:3]
    images = torch.cat([obs, diffs], dim=1)
    images = images.permute(0, 2, 3, 1, 4)
    images = images.flatten(-2, -1)
    return images

@torch.no_grad()
def visualize_reconstruction(model, image, label, dataset):
    """ Plots the reconstructions of a VAE """
    fig_width = 10
    reconst, *_ = model(image[None])
    if reconst.shape[1] == 12:
        fig_width = 20
        reconst = flatten_causal_world_images(reconst)
        image = flatten_causal_world_images(image[None])[0]
        label = flatten_causal_world_images(label[None])[0]
    reconst = reconst.squeeze(dim=0)

    if dataset.num_labels() > 1:
        soft_img = dataset.label_to_img(torch.softmax(reconst, dim=0))
        hard_img = dataset.label_to_img(torch.argmax(reconst, dim=0))
        if label.dtype == torch.long:
            true_img = dataset.label_to_img(label)
            diff_img = (hard_img != true_img).any(dim=-1, keepdims=True).long() * 255
        else:
            true_img = label
            soft_reconst = soft_img.float() / 255.0 * 2.0 - 1.0
            diff_img = (label - soft_reconst).clamp(min=-1, max=1)
    else:
        soft_img = reconst
        hard_img = reconst
        true_img = label
        diff_img = (label - reconst).clamp(min=-1, max=1)

    imgs = [image, true_img, soft_img, hard_img, diff_img]
    titles = ['Original image', 'GT Labels', 'Soft prediction', 'Hard prediction', 'Difference']
    imgs = [t.permute(1, 2, 0) if (t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in imgs]
    imgs = [t.detach().cpu().numpy() for t in imgs]
    imgs = [((t + 1.0) * 255.0 / 2.0).astype(np.int32) if t.dtype == np.float32 else t for t in imgs]
    imgs = [t.astype(np.uint8) for t in imgs]

    fig, axes = plt.subplots(1, len(imgs), figsize=(fig_width, 3))
    for np_img, title, ax in zip(imgs, titles, axes):
        ax.imshow(np_img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig


@torch.no_grad()
def plot_graph_probabilities(graph_probs):
    graph_probs = graph_probs.detach().cpu().numpy().transpose(1, 0)
    fig = plt.figure(figsize=(graph_probs.shape[1]/1.5, graph_probs.shape[0]/1.5))
    sns.heatmap(graph_probs, annot=True,
                yticklabels=[f'Input Dim {i+1}' for i in range(graph_probs.shape[1])] + 
                            [f'Action Dim {i+1}' for i in range(graph_probs.shape[0]-graph_probs.shape[1])],
                xticklabels=[f'Output Dim {i+1}' for i in range(graph_probs.shape[1])])
    plt.title('Graph probabilities (input -> output)')
    plt.tight_layout()
    return fig


@torch.no_grad()
def plot_intervention_similarity(prior):
    similarities = prior.robotic_intervention_similarity.detach().cpu().numpy()
    fig = plt.figure(figsize=(similarities.shape[1]/1.25, similarities.shape[0]/1.25))
    sns.heatmap(similarities, annot=True,
                fmt='.1%',
                xticklabels=[f'Dim {i+1}' for i in range(similarities.shape[1]-1)] + ['Constant'],
                yticklabels=[f'Dim {i+1}' for i in range(similarities.shape[0]-1)] + ['Constant'])
    plt.title('Intervention Similarities')
    plt.tight_layout()
    return fig


@torch.no_grad()
def visualize_triplet_reconstruction(model, img_triplet, labels, sources, dataset=None, *args, **kwargs):
    """ Plots the triplet predictions against the ground truth for a VAE/Flow """
    sources = sources[0].to(model.device)
    labels = labels[-1]
    triplet_rec = model.triplet_prediction(img_triplet[None], sources[None])
    triplet_rec = triplet_rec.squeeze(dim=0)
    if labels.dtype == torch.long:
        triplet_rec = triplet_rec.argmax(dim=0)
        diff_img = (triplet_rec != labels).long() * 255
    else:
        diff_img = ((triplet_rec - labels).clamp(min=-1, max=1) + 1) / 2.0
    triplet_rec = dataset.label_to_img(triplet_rec)
    labels = dataset.label_to_img(labels)
    vs = [img_triplet, labels, sources, triplet_rec, diff_img]
    vs = [e.squeeze(dim=0) for e in vs]
    vs = [t.permute(0, 2, 3, 1) if (len(t.shape) == 4 and t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in vs]
    vs = [t.permute(1, 2, 0) if (len(t.shape) == 3 and t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in vs]
    vs = [e.detach().cpu().numpy() for e in vs]
    img_triplet, labels, sources, triplet_rec, diff_img = vs
    img_triplet = (img_triplet + 1.0) / 2.0
    s1 = np.where(sources == 0)[0]
    s2 = np.where(sources == 1)[0]

    fig, axes = plt.subplots(1, 5, figsize=(8, 3))
    for i, (img, title) in enumerate(zip([img_triplet[0], img_triplet[1], triplet_rec, labels, diff_img], 
                                         ['Image 1', 'Image 2', 'Reconstruction', 'GT Label', 'Difference'])):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    targets = dataset.target_names()
    fig.suptitle(f'Image 1: {[targets[i] for i in s1]}, Image 2: {[targets[i] for i in s2]}')
    plt.tight_layout()
    return fig


def visualize_graph(nodes, adj_matrix):
    if nodes is None:
        nodes = [f'c{i}' for i in range(adj_matrix.shape[0])]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    edges = np.where(adj_matrix == 1)
    edges = [(nodes[edges[0][i]], nodes[edges[1][i]]) for i in range(edges[0].shape[0])]
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)

    figsize = max(3, len(nodes))
    fig = plt.figure(figsize=(figsize, figsize))
    nx.draw(G, 
            pos=pos, 
            arrows=True,
            with_labels=True,
            font_weight='bold',
            node_color='lightgrey',
            edgecolors='black',
            node_size=600,
            arrowstyle='-|>',
            arrowsize=16)
    return fig