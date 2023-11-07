"""Network models."""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
import seaborn
import scipy
import sklearn
import sklearn.decomposition
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
from tqdm import tqdm


class GraphLoader:
  def __init__(self, graph_file):
    self.g = nx.read_gpickle(graph_file)
    print(f'num_nodes:{len(self.g.nodes())} num_edges:{len(self.g.edges())}')
    self.num_of_nodes = self.g.number_of_nodes()
    self.num_of_edges = self.g.number_of_edges()
    self.edges_raw = self.g.edges(data=True)
    self.nodes_raw = self.g.nodes(data=True)

    self.node_index = {}
    self.node_index_reversed = {}
    for index, (node, _) in enumerate(self.nodes_raw):
      self.node_index[node] = index
      self.node_index_reversed[index] = node

    if nx.is_directed(self.g):
      self.edges = [(self.node_index[u], self.node_index[v])
                    for u, v, _ in self.edges_raw]
    else:
      self.edges = ([(self.node_index[u], self.node_index[v])
                     for u, v, _ in self.edges_raw] +
                    [(self.node_index[v], self.node_index[u])
                     for u, v, _ in self.edges_raw])

  def fetch_signed(self):
    u_i = []
    u_j = []
    label = []

    for node_i,_ in self.nodes_raw:
      for node_j,_ in self.nodes_raw:
        if node_i == node_j:
          continue
        # They are node indies.
        node_id_i = self.node_index[node_i]
        node_id_j = self.node_index[node_j]
        u_i.append(node_id_i)
        u_j.append(node_id_j)
        if (node_id_i, node_id_j) in self.edges:
          label.append(1)
        else:
          label.append(-1)

    return u_i, u_j, label


  def fetch_binary(self):
    u_i = []
    u_j = []
    label = []

    for node_i,_ in self.nodes_raw:
      for node_j,_ in self.nodes_raw:
        if node_i == node_j:
          continue
        # They are node indies.
        node_id_i = self.node_index[node_i]
        node_id_j = self.node_index[node_j]
        u_i.append(node_id_i)
        u_j.append(node_id_j)
        if (node_id_i, node_id_j) in self.edges:
          label.append(1)
        else:
          label.append(0)

    return u_i, u_j, label


  def fetch_categorical(self):
    u_i = []
    u_j = []
    label = []
    weight_mat = nx.to_numpy_matrix(self.g)
    num_cat = np.max(weight_mat).astype(int) + 1

    for node_i,_ in self.nodes_raw:
      for node_j,_ in self.nodes_raw:
        if node_i == node_j:
          continue
        # They are node indies.
        node_id_i = self.node_index[node_i]
        node_id_j = self.node_index[node_j]
        u_i.append(node_id_i)
        u_j.append(node_id_j)
        one_hot = np.zeros(num_cat)
        if (node_id_i, node_id_j) in self.edges:
          edge_cat = self.g[node_i][node_j]['weight']
          one_hot[int(edge_cat)] = 1
        else:
          one_hot[0] = 1
        label.append(one_hot)

    label = np.stack(label, axis=0)
    return u_i, u_j, label


  def fetch_prob_mat(self):
    self_loop_weight = 0
    weight_mat = nx.to_numpy_matrix(self.g)
    np.fill_diagonal(weight_mat, self_loop_weight)
    degree = np.sum(weight_mat, axis=1)
    prob_mat = weight_mat / degree

    return prob_mat


  def fetch_cat_tensor(self):
    self_loop_weight = 0
    weight_mat = nx.to_numpy_matrix(self.g).astype(int)
    num_cat = np.max(weight_mat) + 1
    np.fill_diagonal(weight_mat, self_loop_weight)
    cat_tensor = np.zeros([num_cat, self.num_of_nodes, self.num_of_nodes])

    for i in range(self.num_of_nodes):
      for j in range(self.num_of_nodes):
        cat = weight_mat[i,j]
        cat_tensor[cat,i,j] = 1

    return cat_tensor


  def embedding_mapping(self, embedding):
    return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}


class MultiGraphLoader:
  def __init__(self, graph_file_list):
    """We assume the nodes are in the same order."""
    self.num_graphs = len(graph_file_list)
    self.g = [0] * self.num_graphs
    self.num_of_edges = np.zeros(self.num_graphs)
    # Use the first one as template.
    self.g[0] = nx.read_gpickle(graph_file_list[0])
    self.num_of_nodes = self.g[0].number_of_nodes()
    self.nodes_raw = self.g[0].nodes(data=True)

    print(f'num_nodes:{self.num_of_nodes}')
    print(f'graph:{0} num_edges:{self.g[0].number_of_edges()}')
    for f in range(1, self.num_graphs):
      self.g[f] = nx.read_gpickle(graph_file_list[f])
      if self.g[f].number_of_nodes() != self.num_of_nodes:
        raise ValueError('num of nodes do not match.')
      print(f'graph:{f} num_edges:{self.g[f].number_of_edges()}')
      self.num_of_edges[f] = self.g[f].number_of_edges()
      # Verify nodes order.
      nodes_tmp = list(self.nodes_raw)
      nodes = list(self.g[f].nodes(data=True))
      node_compare = [node == nodes[n][0] for n,(node,_) in enumerate(nodes_tmp)]
      if not all(node_compare):
        raise ValueError('Node list do not match.')

    self.node_index = {}
    self.node_index_reversed = {}
    for index, (node, _) in enumerate(self.nodes_raw):
      self.node_index[node] = index
      self.node_index_reversed[index] = node


  def fetch_binary_single(self, edges):
    u_i = []
    u_j = []
    label = []

    for node_i,_ in self.nodes_raw:
      for node_j,_ in self.nodes_raw:
        if node_i == node_j:
          continue
        # They are node indies.
        node_id_i = self.node_index[node_i]
        node_id_j = self.node_index[node_j]
        u_i.append(node_id_i)
        u_j.append(node_id_j)
        if (node_id_i, node_id_j) in edges:
          label.append(1)
        else:
          label.append(0)
    return u_i, u_j, label


  def fetch_binary(self):
    label = [0] * self.num_graphs
    for f in range(self.num_graphs):
      edges_raw = self.g[f].edges(data=True)
      if nx.is_directed(self.g[f]):
        edges = [(self.node_index[u], self.node_index[v])
                      for u, v, _ in edges_raw]
      else:
        edges = ([(self.node_index[u], self.node_index[v])
                       for u, v, _ in edges_raw] +
                      [(self.node_index[v], self.node_index[u])
                       for u, v, _ in edges_raw])
      u_i, u_j, label[f] = self.fetch_binary_single(edges)
    return u_i, u_j, np.array(label)


class NetworkModel(object):
  """Parent class of all network models."""

  def __init__(self):
    pass

  def plot_adj_hat(self):
    """
    Plot adj matrix.

    It's important to note that the diagonal of the true matrix is 0, but in the
    estimated matrix it is 1. As the product of the <x,x> has to be the largest
    value, we can't force the diagonal to be zero, in another word, the
    embedding of a node is closest to itself.
    """
    adj_true = nx.to_numpy_matrix(self.data_loader.g)
    np.fill_diagonal(adj_true, 1)
    adj_hat = self.embedding_trained @ self.embedding_trained.T
    adj_hat = 1/(1 + np.exp(-adj_hat))

    adj_init = self.embedding_init @ self.embedding_init.T
    adj_init = 1/(1 + np.exp(-adj_init))

    err = adj_true-adj_hat
    np.fill_diagonal(err, 0)
    err_sum = np.sum(np.abs(err))
    print(f'sum |adj_hat - adj_true|: {np.round(err_sum, 3)}')
    print('err ratio:', err_sum / adj_true.sum())
    print(f'err_max:{np.max(err)} err_min:{np.min(err)}')

    gs_kw = dict(width_ratios=[1] * 3)
    fig, axs = plt.subplots(figsize=(14, 3),
        gridspec_kw=gs_kw, nrows=1, ncols=3)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    ax = fig.add_subplot(axs[0])
    seaborn.heatmap(adj_true)
    ax = fig.add_subplot(axs[1])
    seaborn.heatmap(adj_hat)
    ax = fig.add_subplot(axs[2])
    seaborn.heatmap(err / adj_true, cmap='PiYG')
    # ax = fig.add_subplot(axs[2])
    # seaborn.heatmap(adj_init)
    plt.show()


  def plot_prob_adj_hat(self):
    """This is for NetworkSoftmaxModel."""
    adj_true = self.data_loader.fetch_prob_mat()
    adj_hat = self.embedding_trained @ self.embedding_trained.T
    offdiag_idx = ~np.eye(adj_hat.shape[0], dtype=bool)
    adj_hat_offdiag = adj_hat[offdiag_idx].reshape(adj_hat.shape[0], -1)
    adj_hat_offdiag = scipy.special.softmax(adj_hat_offdiag, axis=1)
    adj_hat[offdiag_idx] = adj_hat_offdiag.reshape(-1)
    np.fill_diagonal(adj_hat, 0)

    adj_init = self.embedding_init @ self.embedding_init.T
    adj_init = scipy.special.softmax(adj_init, axis=1)

    err = adj_true-adj_hat
    np.fill_diagonal(err, 0)
    err_sum = np.sum(np.abs(err))
    print(f'sum |adj_hat - adj_true|: {err_sum:.3e}')
    print('err ratio:', err_sum / adj_true.sum())
    print(f'err_max:{np.max(err)} err_min:{np.min(err)}')

    plt.figure(figsize=(14, 3))
    plt.subplot(131)
    seaborn.heatmap(adj_true)
    plt.subplot(132)
    seaborn.heatmap(adj_hat)
    plt.subplot(133)
    seaborn.heatmap(err / adj_true, cmap='PiYG')
    # plt.subplot(144)
    # seaborn.heatmap(adj_init)
    plt.show()


  def plot_query_adj_hat(self):
    """
    Plot adj matrix.

    It's important to note that the diagonal of the true matrix is 0, but in the
    estimated matrix it is 1. As the product of the <x,x> has to be the largest
    value, we can't force the diagonal to be zero, in another word, the
    embedding of a node is closest to itself.
    """
    adj_true = nx.to_numpy_matrix(self.data_loader.g)
    np.fill_diagonal(adj_true, 1)
    adj_hat = self.embedding_trained @ self.query_trained @ self.embedding_trained.T
    adj_hat = 1/(1 + np.exp(-adj_hat))

    err = adj_true-adj_hat
    np.fill_diagonal(err, 0)
    err_sum = np.sum(np.abs(err))
    print(f'sum |adj_hat - adj_true|: {np.round(err_sum, 3)}')
    print('err ratio:', err_sum / adj_true.sum())
    print(f'err_max:{np.max(err)} err_min:{np.min(err)}')

    gs_kw = dict(width_ratios=[1] * 4)
    fig, axs = plt.subplots(figsize=(18, 3),
        gridspec_kw=gs_kw, nrows=1, ncols=4)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    ax = fig.add_subplot(axs[0])
    seaborn.heatmap(adj_true)
    ax = fig.add_subplot(axs[1])
    seaborn.heatmap(adj_hat)
    ax = fig.add_subplot(axs[2])
    seaborn.heatmap(err / adj_true)
    ax = fig.add_subplot(axs[3])
    seaborn.heatmap(self.query_trained, cmap='PiYG')
    plt.show()


  def plot_multi_graph_adj(self):
    """Plot multiple graph adj."""
    gs_kw = dict(width_ratios=[1] * self.num_graphs)
    fig, axs = plt.subplots(figsize=(self.num_graphs*3.5, 2.5),
        gridspec_kw=gs_kw, nrows=1, ncols=self.num_graphs)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.05)
    for f in range(self.num_graphs):
      adj_true = nx.to_numpy_matrix(self.data_loader.g[f])
      np.fill_diagonal(adj_true, 1)
      ax = fig.add_subplot(axs[f])
      if f != 0:
        ax.axis('off')
      cbar = True if f == self.num_graphs-1 else False
      seaborn.heatmap(adj_true, cbar=cbar, square=True)
    plt.show()


  def plot_multi_graph_adj_contrast(self):
    """Pairwise comparison between query mat."""
    gs_kw = dict(width_ratios=[1] * self.num_graphs,
                 height_ratios=[1] * (self.num_graphs-1))
    fig, axs = plt.subplots(figsize=(self.num_graphs*2, (self.num_graphs-1)*2),
        gridspec_kw=gs_kw, nrows=self.num_graphs-1, ncols=self.num_graphs)
    plt.subplots_adjust(left=None, right=None, hspace=0.02, wspace=0.02)
    for i in range(self.num_graphs-1):
      adj_i = nx.to_numpy_matrix(self.data_loader.g[i])
      for j in range(self.num_graphs):
        if i >= j:
          ax = fig.add_subplot(axs[i,j])
          ax.axis('off')
          continue
        ax = fig.add_subplot(axs[i,j])
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        adj_j = nx.to_numpy_matrix(self.data_loader.g[j])
        seaborn.heatmap(adj_i - adj_j, cbar=False, vmin=-1, vmax=1)
    plt.show()


  def plot_multi_graph_adj_hat(self):
    """Plot multiple graph query."""
    adj_hat = self.embedding_trained @ self.query_trained @ self.embedding_trained.T
    adj_hat = 1/(1 + np.exp(-adj_hat))

    gs_kw = dict(width_ratios=[1] * self.num_graphs)
    fig, axs = plt.subplots(figsize=(self.num_graphs*3.5, 2.5),
        gridspec_kw=gs_kw, nrows=1, ncols=self.num_graphs)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.05)
    for f in range(self.num_graphs):
      np.fill_diagonal(adj_hat[f], 1)
      ax = fig.add_subplot(axs[f])
      if f != 0:
        ax.axis('off')
      cbar = True if f == self.num_graphs-1 else False
      seaborn.heatmap(adj_hat[f], cbar=cbar, square=True)
    plt.show()


  def plot_multi_graph_query_adj_hat(
      self,
      f):
    """Plot multiple graph query."""
    adj_true = nx.to_numpy_matrix(self.data_loader.g[f])
    np.fill_diagonal(adj_true, 1)
    adj_hat = self.embedding_trained @ self.query_trained[f] @ self.embedding_trained.T
    adj_hat = 1/(1 + np.exp(-adj_hat))

    err = adj_true-adj_hat
    np.fill_diagonal(err, 0)
    err_sum = np.sum(np.abs(err))
    print(f'sum |adj_hat - adj_true|: {np.round(err_sum, 3)}')
    print('err ratio:', err_sum / (self.num_of_nodes-1) / self.num_of_nodes)
    print(f'err_max:{np.max(err)} err_min:{np.min(err)}')

    gs_kw = dict(width_ratios=[1] * 4)
    fig, axs = plt.subplots(figsize=(18, 3),
        gridspec_kw=gs_kw, nrows=1, ncols=4)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    ax = fig.add_subplot(axs[0])
    seaborn.heatmap(adj_true)
    ax = fig.add_subplot(axs[1])
    seaborn.heatmap(adj_hat)
    ax = fig.add_subplot(axs[2])
    seaborn.heatmap(err)
    ax = fig.add_subplot(axs[3])
    seaborn.heatmap(self.query_trained[f], cmap='PiYG')
    plt.show()


  def plot_multi_graph_pca_query_adj_hat_dims(
      self,
      f,
      select_dim_left,
      select_dim_right):
    """Plot multiple graph query."""
    pca = sklearn.decomposition.PCA(n_components=self.embedding_dim)
    pca = pca.fit(self.embedding_trained)
    embedding_pca = pca.transform(self.embedding_trained)
    pca_components = pca.components_.T
    pca_sigma = pca.singular_values_
    pca_mean = pca.mean_

    offset = pca_mean.reshape(1,-1) @ pca_components
    embedding_sub_left = embedding_pca[:,select_dim_left] + offset[0,select_dim_left]
    embedding_sub_right = embedding_pca[:,select_dim_right] + offset[0,select_dim_right]
    query_pca = pca_components.T @ self.query_trained[f] @ pca_components
    query_sub = query_pca[np.ix_(select_dim_left, select_dim_right)]
    adj_hat = embedding_sub_left @ query_sub @ embedding_sub_right.T
    adj_hat = 1/(1 + np.exp(-adj_hat))

    adj_true = nx.to_numpy_matrix(self.data_loader.g[f])
    err = adj_true-adj_hat
    np.fill_diagonal(err, 0)
    err_sum = np.sum(np.abs(err))
    print(f'sum |adj_hat - adj_true|: {np.round(err_sum, 3)}')
    print('err ratio:', err_sum / adj_true.sum())
    print(f'err_max:{np.max(err)} err_min:{np.min(err)}')

    gs_kw = dict(width_ratios=[1] * 4)
    fig, axs = plt.subplots(figsize=(18, 3),
        gridspec_kw=gs_kw, nrows=1, ncols=4)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    ax = fig.add_subplot(axs[0])
    seaborn.heatmap(adj_true)
    ax = fig.add_subplot(axs[1])
    seaborn.heatmap(adj_hat)
    ax = fig.add_subplot(axs[2])
    seaborn.heatmap(err)
    ax = fig.add_subplot(axs[3])
    seaborn.heatmap(self.query_trained[f], cmap='PiYG')
    plt.show()


  def plot_multi_graph_pca_query_adj_hat_mask(
      self,
      f,
      query_mask):
    """Plot multiple graph query."""
    pca = sklearn.decomposition.PCA(n_components=self.embedding_dim)
    pca = pca.fit(self.embedding_trained)
    embedding_pca = pca.transform(self.embedding_trained)
    pca_components = pca.components_.T
    pca_sigma = pca.singular_values_
    pca_mean = pca.mean_

    mask_binary = ~np.isnan(query_mask)
    select_dim_left = np.where(mask_binary.sum(axis=1))[0]
    select_dim_right = np.where(mask_binary.sum(axis=0))[0]
    # query_mask[~mask_binary] = 0
    query_pca = pca_components.T @ self.query_trained[f] @ pca_components
    query_pca[mask_binary] = 0

    offset = pca_mean.reshape(1,-1) @ pca_components
    embedding_sub_left = embedding_pca[:,select_dim_left] + offset[0,select_dim_left]
    embedding_sub_right = embedding_pca[:,select_dim_right] + offset[0,select_dim_right]
    query_sub = query_pca[np.ix_(select_dim_left, select_dim_right)]
    adj_hat = embedding_sub_left @ query_sub @ embedding_sub_right.T
    adj_hat = 1/(1 + np.exp(-adj_hat))

    adj_true = nx.to_numpy_matrix(self.data_loader.g[f])
    err = adj_true-adj_hat
    np.fill_diagonal(err, 0)
    err_sum = np.sum(np.abs(err))
    print(f'sum |adj_hat - adj_true|: {np.round(err_sum, 3)}')
    print('err ratio:', err_sum / adj_true.sum())
    print(f'err_max:{np.max(err)} err_min:{np.min(err)}')

    gs_kw = dict(width_ratios=[1] * 4)
    fig, axs = plt.subplots(figsize=(18, 3),
        gridspec_kw=gs_kw, nrows=1, ncols=4)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    ax = fig.add_subplot(axs[0])
    seaborn.heatmap(adj_true)
    ax = fig.add_subplot(axs[1])
    seaborn.heatmap(adj_hat)
    ax = fig.add_subplot(axs[2])
    seaborn.heatmap(err)
    ax = fig.add_subplot(axs[3])
    seaborn.heatmap(self.query_trained[f], cmap='PiYG')
    plt.show()


  def plot_multi_graph_pca_query_spectrum(
      self,
      threshold):
    """Plot multiple graph query."""
    pca = sklearn.decomposition.PCA(n_components=self.embedding_dim)
    pca = pca.fit(self.embedding_trained)
    embedding_pca = pca.transform(self.embedding_trained)
    pca_components = pca.components_.T
    query_pca = pca_components.T @ self.query_trained @ pca_components
    vmin, vmax = np.min(query_pca), np.max(query_pca)
    query_diff = query_pca-query_pca[0]
    vmin_diff, vmax_diff = np.min(query_diff), np.max(query_diff)
    query_diff[(query_diff>threshold[0]) & (query_diff<threshold[1])] = np.nan

    gs_kw = dict(width_ratios=[1] * self.num_graphs, height_ratios=[1])
    fig, axs = plt.subplots(figsize=(self.num_graphs*2.5, 2),
        gridspec_kw=gs_kw, nrows=1, ncols=self.num_graphs)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.05)
    for f in range(self.num_graphs):
      ax = fig.add_subplot(axs[f])
      if f != 0:
        ax.axis('off')
      cbar = True if f == self.num_graphs-1 else False
      seaborn.heatmap(query_pca[f], vmin=vmin, vmax=vmax,
          cmap='PiYG', cbar=cbar, square=True)
    plt.show()

    gs_kw = dict(width_ratios=[1] * (self.num_graphs-1), height_ratios=[1])
    fig, axs = plt.subplots(figsize=((self.num_graphs-1)*2.5, 2),
        gridspec_kw=gs_kw, nrows=1, ncols=self.num_graphs-1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    for f in range(1, self.num_graphs):
      ax = fig.add_subplot(axs[f-1])
      if f != 1:
        ax.axis('off')
      cbar = True if f == self.num_graphs-1 else False
      seaborn.heatmap(query_diff[f], vmin=-20, vmax=20,
          cmap='PiYG', cbar=cbar, square=True)
    plt.show()

    query_diff_mask = query_diff[1:]
    return query_diff_mask


  def compare_multi_graph_query_mat(self):
    """Pairwise comparison between query mat."""
    pca = sklearn.decomposition.PCA(n_components=self.embedding_dim)
    pca = pca.fit(self.embedding_trained)
    pca_features = pca.transform(self.embedding_trained)
    pca_components = pca.components_.T
    pca_components = pca.components_.T
    query_pca = pca_components.T @ self.query_trained @ pca_components

    for i in range(self.num_graphs-1):
      # query_i = self.query_trained[i]
      query_i = query_pca[i]
      qi = query_i.reshape(-1)
      U_i, D_i, Vh_i = np.linalg.svd(query_i)

      for j in range(self.num_graphs):
        if i >= j:
          continue
        # query_j = self.query_trained[j]
        query_j = query_pca[j]
        qj = query_j.reshape(-1)
        U_j, D_j, Vh_j = np.linalg.svd(query_j)

        l2 = scipy.spatial.distance.euclidean(qi, qj)
        cos_dist = 1 - scipy.spatial.distance.cosine(qi, qj)
        print(f'({i},{j}) F:{np.round(l2,2)}\tcos:{np.round(cos_dist,3)}'+
            f'\t')

        U_cos = np.abs(np.diagonal(U_i.T @ U_j))
        V_cos = np.abs(np.diagonal(Vh_i @ Vh_j.T))
        print('U cos:', np.round(U_cos, 3))
        print('V cos:', np.round(V_cos, 3))
        print('D i:', np.round(D_i, 3))
        print('D j:', np.round(D_j, 3))
        print()


  def plot_multi_graph_query(self):
    """Compare different query matrix in the tensor."""
    gs_kw = dict(width_ratios=[1] * self.num_graphs, height_ratios=[1,1])
    fig, axs = plt.subplots(figsize=(self.num_graphs*3, 4),
        gridspec_kw=gs_kw, nrows=2, ncols=self.num_graphs)
    # plt.subplots_adjust(left=None, right=None, hspace=0.02, wspace=0.02)
    for c in range(self.num_graphs):
      query = self.query_trained[c]
      U, sigma, _ = np.linalg.svd(query)
      ax = fig.add_subplot(axs[0,c])
      ax.tick_params(bottom=False, labelbottom=False)
      if c > 0:
        ax.tick_params(left=True, labelleft=False, bottom=False, labelbottom=False)
      plt.stem(sigma)
      if c == 0:
        ymax = np.max(sigma)
      plt.ylim(0, ymax*1.5)
      ax = fig.add_subplot(axs[1,c])
      ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
      seaborn.heatmap(U, cbar=False)
    plt.show()


  def plot_prob_query_adj_hat(self):
    """This is for NetworkSoftmaxModel."""
    adj_true = self.data_loader.fetch_prob_mat()
    adj_hat = self.embedding_trained @ self.query_trained @ self.embedding_trained.T
    offdiag_idx = ~np.eye(adj_hat.shape[0], dtype=bool)
    adj_hat_offdiag = adj_hat[offdiag_idx].reshape(adj_hat.shape[0], -1)
    adj_hat_offdiag = scipy.special.softmax(adj_hat_offdiag, axis=1)
    adj_hat[offdiag_idx] = adj_hat_offdiag.reshape(-1)
    np.fill_diagonal(adj_hat, 0)
    adj_init = self.embedding_init @ self.embedding_init.T
    adj_init = scipy.special.softmax(adj_init, axis=1)

    err = adj_true-adj_hat
    np.fill_diagonal(err, 0)
    err_sum = np.sum(np.abs(err))
    print(f'sum |adj_hat - adj_true|: {err_sum:.3e}')
    print('err ratio:', err_sum / adj_true.sum())
    print(f'err_max:{np.max(err)} err_min:{np.min(err)}')

    plt.figure(figsize=(20,3.5))
    plt.subplot(141)
    seaborn.heatmap(adj_true)
    plt.subplot(142)
    seaborn.heatmap(adj_hat)
    plt.subplot(143)
    seaborn.heatmap(err / adj_true, cmap='PiYG')
    plt.subplot(144)
    seaborn.heatmap(self.query_trained)
    plt.show()


  def plot_cat_query_adj_hat(self):
    """This is for NetworkSoftmaxModel."""
    adj_true = self.data_loader.fetch_cat_tensor()
    adj_hat = self.embedding_trained @ self.query_trained @ self.embedding_trained.T
    adj_hat = scipy.special.softmax(adj_hat, axis=0)

    err = adj_true-adj_hat
    diag_idx = np.arange(self.num_of_nodes)
    err[:,diag_idx,diag_idx] = 0

    err_sum = np.sum(np.abs(err))
    print(f'sum |adj_hat - adj_true|: {err_sum}')
    print('err ratio:', err_sum / adj_true.sum())

    gs_kw = dict(width_ratios=[1] * self.num_cat, height_ratios=[1,1])
    fig, axs = plt.subplots(figsize=(self.num_cat*2, 4),
        gridspec_kw=gs_kw, nrows=2, ncols=self.num_cat)
    plt.subplots_adjust(left=None, right=None, hspace=0.02, wspace=0.02)
    for c in range(self.num_cat):
      ax = fig.add_subplot(axs[0,c])
      ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
      seaborn.heatmap(adj_true[c], cbar=False)
      ax = fig.add_subplot(axs[1,c])
      ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
      seaborn.heatmap(adj_hat[c], cbar=False)
    plt.show()

    vmin = np.min(self.query_trained) * 0.8
    vamx = np.max(self.query_trained) * 0.8
    gs_kw = dict(width_ratios=[1] * self.num_cat)
    fig, axs = plt.subplots(figsize=(self.num_cat*2, 2),
        gridspec_kw=gs_kw, nrows=1, ncols=self.num_cat)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.02)
    for c in range(self.num_cat):
      ax = fig.add_subplot(axs[c])
      ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
      seaborn.heatmap(self.query_trained[c], vmin=vmin, vmax=vamx, cbar=False)


  def plot_cat_query(self):
    """Compare different query matrix in the tensor."""
    gs_kw = dict(width_ratios=[1] * self.num_cat, height_ratios=[1,1])
    fig, axs = plt.subplots(figsize=(self.num_cat*3, 4),
        gridspec_kw=gs_kw, nrows=2, ncols=self.num_cat)
    # plt.subplots_adjust(left=None, right=None, hspace=0.02, wspace=0.02)
    for c in range(self.num_cat):
      query = self.query_trained[c]
      U, sigma, _ = np.linalg.svd(query)
      ax = fig.add_subplot(axs[0,c])
      ax.tick_params(bottom=False, labelbottom=False)
      if c > 0:
        ax.tick_params(left=True, labelleft=False, bottom=False, labelbottom=False)
      plt.stem(sigma)
      if c == 0:
        ymax = np.max(sigma)
      plt.ylim(0, ymax*1.5)
      ax = fig.add_subplot(axs[1,c])
      ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
      seaborn.heatmap(U, cbar=False)
    plt.show()


  def compare_cat_query_mat(self):
    """Pairwise comparison between query mat."""
    gs_kw = dict(width_ratios=[1] * self.num_cat,
                 height_ratios=[1] * (self.num_cat-1))
    fig, axs = plt.subplots(figsize=(self.num_cat*2, (self.num_cat-1)*2),
        gridspec_kw=gs_kw, nrows=self.num_cat-1, ncols=self.num_cat)
    plt.subplots_adjust(left=None, right=None, hspace=0.02, wspace=0.02)
    for i in range(self.num_cat-1):
      query_i = self.query_trained[i]
      U_i, sigma_i, _ = np.linalg.svd(query_i)
      for j in range(self.num_cat):
        if i >= j:
          ax = fig.add_subplot(axs[i,j])
          ax.axis('off')
          continue
        ax = fig.add_subplot(axs[i,j])
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        query_j = self.query_trained[j]
        U_j, sigma_j, _ = np.linalg.svd(query_j)
        cos = np.diagonal(U_i.T @ U_j)
        # seaborn.heatmap(U_i.T @ U_j, cbar=False)
        plt.stem(np.abs(cos))
        plt.ylim(0, 1)
    plt.show()


  def decompose_adj(self):
    """Decompose adj matrix."""
    adj_true = nx.to_numpy_matrix(self.data_loader.g)
    np.fill_diagonal(adj_true, 0)
    _, sigmas, _ = np.linalg.svd(adj_true)
    plt.figure(figsize=(4,2))
    plt.stem(sigmas)
    plt.show()


  def multi_graph_query_svd(self):
    """Decompose adj matrix."""
    pca = sklearn.decomposition.PCA(n_components=self.embedding_dim)
    pca = pca.fit(self.embedding_trained)
    pca_components = pca.components_.T

    gs_kw = dict(width_ratios=[1] * self.num_graphs)
    fig, axs = plt.subplots(figsize=(self.num_graphs*4, 2),
        gridspec_kw=gs_kw, nrows=1, ncols=self.num_graphs)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    for f in range(self.num_graphs):
      ax = fig.add_subplot(axs[f])
      query = self.query_trained[f]
      _, sigmas, _ = np.linalg.svd(query)
      plt.stem(sigmas)
    plt.show()

    gs_kw = dict(width_ratios=[1] * self.num_graphs)
    fig, axs = plt.subplots(figsize=(self.num_graphs*4, 2),
        gridspec_kw=gs_kw, nrows=1, ncols=self.num_graphs)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    for f in range(self.num_graphs):
      ax = fig.add_subplot(axs[f])
      query = self.query_trained[f]

      query_pca_proj = pca_components.T @ query @ pca_components
      sigmas = np.diag(np.abs(query_pca_proj))
      plt.stem(sigmas)
    plt.show()


  def graph_laplacian(self):
    """Decompose adj matrix."""
    adj_true = nx.to_numpy_matrix(self.data_loader.g)
    np.fill_diagonal(adj_true, 0)
    # num_conn = nx.number_connected_components(self.data_loader.g)
    # print('num connected components', num_conn)

    D = adj_true.sum(axis=1)
    D = np.diag(D)
    L = D - adj_true
    eigvals = np.linalg.eigvals(L)

    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    seaborn.heatmap(L)
    plt.subplot(122)
    plt.stem(np.abs(eigvals[1:]))
    plt.show()


  def multi_graph_laplacian(self):
    """Decompose adj matrix."""
    gs_kw = dict(width_ratios=[1] * self.num_graphs)
    fig, axs = plt.subplots(figsize=(self.num_graphs*4, 2),
        gridspec_kw=gs_kw, nrows=1, ncols=self.num_graphs)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    for f in range(self.num_graphs):
      ax = fig.add_subplot(axs[f])
      adj_true = np.array(nx.to_numpy_matrix(self.data_loader.g[f]))
      np.fill_diagonal(adj_true, 0)
      D_diag = np.sum(adj_true, axis=1)
      D = np.diag(D_diag)
      L = D - adj_true
      D_norm = np.diag(1 / np.sqrt(D_diag))
      L = D_norm @ L @ D_norm
      _, sigmas, _ = np.linalg.svd(L)
      plt.stem(sigmas)
    plt.show()


  def plot_raw_embedding(
      self):
    """Plot raw embeddings."""
    embedding = self.embedding_trained
    gs_kw = dict(width_ratios=[1] * 3)
    fig, axs = plt.subplots(figsize=(10, 2),
        gridspec_kw=gs_kw, nrows=1, ncols=3)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    ax = fig.add_subplot(axs[0])
    plt.plot(embedding[:,0], embedding[:,1], '.')
    ax.set_aspect('equal', adjustable='box')

    ax = fig.add_subplot(axs[1])
    plt.plot(embedding[:,0], embedding[:,2], '.')
    ax.set_aspect('equal', adjustable='box')

    ax = fig.add_subplot(axs[2])
    plt.plot(embedding[:,1], embedding[:,2], '.')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


  def plot_raw_query_embedding(
      self):
    """Plot raw embeddings."""
    U, sigma, Vh = np.linalg.svd(self.query_trained, full_matrices=True)
    embedding = self.embedding_trained @ U
    gs_kw = dict(width_ratios=[1] * 3)
    fig, axs = plt.subplots(figsize=(10, 2),
        gridspec_kw=gs_kw, nrows=1, ncols=3)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    ax = fig.add_subplot(axs[0])
    plt.plot(embedding[:,0], embedding[:,1], '.')
    ax.set_aspect('equal', adjustable='box')

    ax = fig.add_subplot(axs[1])
    plt.plot(embedding[:,0], embedding[:,2], '.')
    ax.set_aspect('equal', adjustable='box')

    ax = fig.add_subplot(axs[2])
    plt.plot(embedding[:,1], embedding[:,2], '.')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


  def plot_embedding_pca(
      self,
      pc_ids,
      highligh_nodes=None):
    """PCA analysis on embeddings."""
    if 'block' in list(self.data_loader.nodes_raw)[0][1]:
      node_group = [val['block'] for node,val in self.data_loader.nodes_raw]
    if 'probe' in list(self.data_loader.nodes_raw)[0][1]:
      map_probe_to_val = {'probeA':0, 'probeB':1, 'probeC':2, 'probeD':3,
          'probeE':4, 'probeF':5}
      node_group = [map_probe_to_val[val['probe']] 
                    for node,val in self.data_loader.nodes_raw]
    else:
      node_group = None

    pca = sklearn.decomposition.PCA(n_components=self.embedding_dim)
    pca = pca.fit(self.embedding_trained)
    features = pca.transform(self.embedding_trained)

    num_pcs = len(pc_ids)
    num_plots = int(num_pcs * (num_pcs-1) / 2)
    gs_kw = dict(width_ratios=[1] * num_plots)
    fig, axs = plt.subplots(figsize=(num_plots * 4, 3.5),
        gridspec_kw=gs_kw, nrows=1, ncols=num_plots)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.2)
    plt_cnt = 0
    for i, pc_i in enumerate(pc_ids):
      for j, pc_j in enumerate(pc_ids):
        if i >= j:
          continue
        ax = fig.add_subplot(axs[plt_cnt])
        sc = plt.scatter(features[:,pc_i], features[:,pc_j],
            s=8, c=node_group, cmap='jet')
        if highligh_nodes is not None:
          plt.scatter(features[highligh_nodes,pc_i],
                      features[highligh_nodes,pc_j],
                      s=35, facecolors='none', edgecolors='r')

        plt.axhline(y=0, c='lightgrey')
        plt.axvline(x=0, c='lightgrey')
        ax.axis('equal')
        # plt.grid()
        # if i == 0 and j == 1:
        #   circle = plt.Circle((0, 0), 0.6, color='lightgrey', lw=0.5, fill=False)
        #   ax.add_patch(circle)
        if plt_cnt == num_pcs*(num_pcs-1)/2-1:
          # plt.legend()
          plt.colorbar(sc)
        plt.title(f'PC{pc_i} -- PC{pc_j}')
        plt_cnt += 1
    plt.show()

    main_features = features[:,[0,1]]
    norm = np.linalg.norm(main_features, axis=1)
    neuron_idx = np.where(norm < 0.6)[0]
    print(neuron_idx)


  def embedding_group_analysis(
      self,
      selected_nodes=[0]):
    """Relations between groups."""
    group_idxs = [val['block'] for node,val in self.data_loader.nodes_raw]
    group_ids = set(group_idxs)
    group_idxs = np.array(group_idxs)
    num_groups = len(group_ids)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=self.embedding_dim)
    features = pca.fit(self.embedding_trained).transform(self.embedding_trained)
    features_mean = pca.mean_

    for i, g_i in enumerate(group_ids):
      for j, g_j in enumerate(group_ids):
        if i >= j:
          continue
        group_idx_i = group_idxs == g_i
        group_idx_j = group_idxs == g_j
        features_i = self.embedding_trained[group_idx_i].mean(axis=0)
        features_j = self.embedding_trained[group_idx_j].mean(axis=0)
        l2 = scipy.spatial.distance.euclidean(features_i, features_j)
        print(g_i, g_j, l2)

    pairwise_dist = np.zeros([self.num_of_nodes, self.num_of_nodes])
    for i in range(self.num_of_nodes):
      for j in range(self.num_of_nodes):
        if i >= j:
          continue
        # pairwise_dist[i,j] = scipy.spatial.distance.euclidean(
        #     self.embedding_trained[i,2:4], self.embedding_trained[j,2:4])
        pairwise_dist[i,j] = scipy.spatial.distance.euclidean(
            features[i,[0,1,7]], features[j,[0,1,7]])
    mask = np.zeros_like(pairwise_dist)
    mask[np.tril_indices_from(mask)] = True
    plt.figure(figsize=[6.5, 5])
    seaborn.heatmap(pairwise_dist, cmap='PiYG', mask=mask)
    plt.show()


  def embedding_mapping(self):
    """Map from node to its embedding."""
    map_node_embedding = {}
    for node, _ in self.nodes_raw:
      map_node_embedding[node] = self.embedding_trained[self.node_index[node]]
    return map_node_embedding


class NetworkSigmoidModel(NetworkModel):
  def __init__(self, args):
    super().__init__()
    self.data_loader = GraphLoader(graph_file=args.graph_file)
    self.num_of_nodes = self.data_loader.num_of_nodes
    args.num_of_nodes = self.num_of_nodes
    self.embedding_dim = args.embedding_dim

    tf.reset_default_graph()
    batch_size = args.num_of_nodes * (args.num_of_nodes-1)
    self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[batch_size])
    self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[batch_size])
    self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[batch_size])

    self.embedding = tf.get_variable('target_embedding',
        [self.num_of_nodes, args.embedding_dim],
        initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
    self.u_i_embedding = tf.matmul(
        tf.one_hot(self.u_i, depth=self.num_of_nodes), self.embedding)
    self.u_j_embedding = tf.matmul(
        tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
  #   self.context_embedding = tf.get_variable('context_embedding',
  #       [self.num_of_nodes, args.embedding_dim],
  #       initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
      # self.u_j_embedding = tf.matmul(
      #     tf.one_hot(self.u_j, depth=self.num_of_nodes), self.context_embedding)
    self.inner_product = tf.reduce_sum(
        self.u_i_embedding * self.u_j_embedding, axis=1)

    # This is for signed data. (no need now)
    # self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))
    # Other implementation for binary loss.
    # self.loss = -tf.reduce_mean(
    #     self.label * tf.log_sigmoid(self.inner_product) +
    #     (1-self.label) * tf.log_sigmoid(-self.inner_product))
    # Working.
    # self.loss = -tf.reduce_mean(self.label * self.inner_product +
    #                             tf.log_sigmoid(-self.inner_product))
    self.loss = tf.losses.sigmoid_cross_entropy(self.label, self.inner_product)
    self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
    # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)


  def train(
      self,
      args,
      print_step=None):
    """Train the model."""
    if print_step is None:
      print_step = int(args.num_epochs / 10)
    with tf.Session() as sess:
      print('itrs\tloss')
      tf.global_variables_initializer().run()
      self.embedding_init = sess.run(self.embedding)
      learning_rate = args.learning_rate
      u_i, u_j, label = self.data_loader.fetch_binary()

      for itr in range(args.num_epochs):
        feed_dict = {self.u_i: u_i, self.u_j: u_j, self.label: label,
                     self.learning_rate: learning_rate}
        sess.run(self.train_op, feed_dict=feed_dict)
        learning_rate = max(args.learning_rate * (1 - itr / args.num_epochs),
                            args.learning_rate * 0.0001)
        if itr % print_step == 0 or itr == args.num_epochs-1:
          loss = sess.run(self.loss, feed_dict=feed_dict)
          print(f'{itr}\t{loss:.3e}')
          if loss < 1e-5:  # Good enough to converge.
            break

      self.embedding_trained = sess.run(self.embedding)


class NetworkSoftmaxModel(NetworkModel):
  def __init__(self, args):
    super().__init__()
    self.data_loader = GraphLoader(graph_file=args.graph_file)
    self.num_of_nodes = self.data_loader.num_of_nodes
    args.num_of_nodes = self.num_of_nodes
    self.embedding_dim = args.embedding_dim

    tf.reset_default_graph()
    self.label = tf.placeholder(name='label', dtype=tf.float32,
        shape=[args.num_of_nodes, args.num_of_nodes])
    self.embedding = tf.get_variable('target_embedding',
        [args.num_of_nodes, args.embedding_dim],
        initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))

    rows, cols = np.where(~np.eye(args.num_of_nodes, dtype=bool))
    rows = rows.reshape(args.num_of_nodes, args.num_of_nodes-1)
    cols = cols.reshape(args.num_of_nodes, args.num_of_nodes-1)
    off_diag_mask = tf.stack((rows, cols), -1)

    self.inner_product = tf.matmul(self.embedding, tf.transpose(self.embedding))
    self.inner_product = tf.gather_nd(self.inner_product, off_diag_mask)
    self.weights = tf.gather_nd(self.label, off_diag_mask)
    self.log_p = tf.math.log_softmax(self.inner_product, axis=1)
    self.loss = -tf.reduce_sum(self.weights * self.log_p) / args.num_of_nodes

    self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
    # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)


  def train(
      self,
      args,
      print_step=None):
    """Train the model."""
    if print_step is None:
      print_step = int(args.num_epochs / 10)
    with tf.Session() as sess:
      print('itrs\tloss')
      tf.global_variables_initializer().run()
      self.embedding_init = sess.run(self.embedding)
      learning_rate = args.learning_rate
      label = self.data_loader.fetch_prob_mat()

      for itr in range(args.num_epochs):
        feed_dict = {self.label: label, self.learning_rate: learning_rate}
        sess.run(self.train_op, feed_dict=feed_dict)
        learning_rate = max(args.learning_rate * (1 - itr / args.num_epochs),
                            args.learning_rate * 0.0001)
        if itr % print_step == 0 or itr == args.num_epochs-1:
          loss = sess.run(self.loss, feed_dict=feed_dict)
          print(f'{itr}\t{loss:.3e}')
          if loss < 1e-5:  # Good enough to converge.
            break

      self.embedding_trained = sess.run(self.embedding)


class NetworkSigmoidQueryModel(NetworkModel):
  def __init__(self, args):
    super().__init__()
    self.data_loader = GraphLoader(graph_file=args.graph_file)
    self.num_of_nodes = self.data_loader.num_of_nodes
    args.num_of_nodes = self.num_of_nodes
    self.embedding_dim = args.embedding_dim

    tf.reset_default_graph()
    batch_size = args.num_of_nodes * (args.num_of_nodes-1)
    self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[batch_size])
    self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[batch_size])
    self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[batch_size])

    self.embedding = tf.get_variable('target_embedding',
        [self.num_of_nodes, args.embedding_dim],
        initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))

    query_init = tf.constant(np.eye(args.embedding_dim), dtype=tf.float32)
    # self.query = query_init
    self.query = tf.get_variable('query', initializer=query_init, dtype=tf.float32)

    self.u_i_embedding = tf.matmul(
        tf.one_hot(self.u_i, depth=self.num_of_nodes), self.embedding)
    self.u_j_embedding = tf.matmul(
        tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)

    # embedding_i_query = self.u_i_embedding
    embedding_i_query = tf.matmul(self.u_i_embedding, self.query)
    self.inner_product = tf.reduce_sum(
        embedding_i_query * self.u_j_embedding, axis=1)

    # For signed data. (no need now)
    # self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))
    # Other implementation of binary loss.
    # self.loss = -tf.reduce_mean(
    #     self.label * tf.log_sigmoid(self.inner_product) +
    #     (1-self.label) * tf.log_sigmoid(-self.inner_product))
    # Working.
    # self.loss = -tf.reduce_mean(self.label * self.inner_product +
    #                             tf.log_sigmoid(-self.inner_product))
    self.loss = tf.losses.sigmoid_cross_entropy(self.label, self.inner_product)
    self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
    # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)


  def train(
      self,
      args,
      epsilon=1e-5,
      print_step=None):
    """Train the model."""
    if print_step is None:
      print_step = int(args.num_epochs / 10)
    with tf.Session() as sess:
      print('itrs\tloss')
      tf.global_variables_initializer().run()
      self.embedding_init = sess.run(self.embedding)
      learning_rate = args.learning_rate
      u_i, u_j, label = self.data_loader.fetch_binary()

      for itr in range(args.num_epochs):
        feed_dict = {self.u_i: u_i, self.u_j: u_j, self.label: label,
                     self.learning_rate: learning_rate}
        sess.run(self.train_op, feed_dict=feed_dict)
        learning_rate = max(args.learning_rate * (1 - itr / args.num_epochs),
                            args.learning_rate * 0.0001)
        if itr % print_step == 0 or itr == args.num_epochs-1:
          loss = sess.run(self.loss, feed_dict=feed_dict)
          print(f'{itr}\t{loss:.3e}')
          if loss < epsilon:  # Good enough to converge.
            break

      self.embedding_trained = sess.run(self.embedding)
      self.query_trained = sess.run(self.query)


class NetworkSoftmaxQueryModel(NetworkModel):
  def __init__(self, args):
    super().__init__()
    self.data_loader = GraphLoader(graph_file=args.graph_file)
    self.num_of_nodes = self.data_loader.num_of_nodes
    args.num_of_nodes = self.num_of_nodes
    self.embedding_dim = args.embedding_dim

    tf.reset_default_graph()
    self.label = tf.placeholder(name='label', dtype=tf.float32,
        shape=[args.num_of_nodes, args.num_of_nodes])
    query_init = tf.constant(np.eye(args.embedding_dim), dtype=tf.float32)
    # self.query = query_init
    self.query = tf.get_variable('query', initializer=query_init, dtype=tf.float32)
    self.embedding = tf.get_variable('target_embedding',
        [args.num_of_nodes, args.embedding_dim],
        initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))

    rows, cols = np.where(~np.eye(args.num_of_nodes, dtype=bool))
    rows = rows.reshape(args.num_of_nodes, args.num_of_nodes-1)
    cols = cols.reshape(args.num_of_nodes, args.num_of_nodes-1)
    off_diag_mask = tf.stack((rows, cols), -1)

    # embedding_query = self.embedding
    embedding_query = tf.matmul(self.embedding, self.query)
    self.inner_product = tf.matmul(embedding_query, tf.transpose(self.embedding))
    self.inner_product = tf.gather_nd(self.inner_product, off_diag_mask)
    self.weights = tf.gather_nd(self.label, off_diag_mask)
    self.log_p = tf.math.log_softmax(self.inner_product, axis=1)
    self.loss = -tf.reduce_sum(self.weights * self.log_p) / args.num_of_nodes

    self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
    # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)


  def train(
      self,
      args,
      print_step=None):
    """Train the model."""
    if print_step is None:
      print_step = int(args.num_epochs / 10)
    with tf.Session() as sess:
      print('itrs\tloss')
      tf.global_variables_initializer().run()
      self.embedding_init = sess.run(self.embedding)
      learning_rate = args.learning_rate
      label = self.data_loader.fetch_prob_mat()

      for itr in range(args.num_epochs):
        feed_dict = {self.label: label, self.learning_rate: learning_rate}
        sess.run(self.train_op, feed_dict=feed_dict)
        learning_rate = max(args.learning_rate * (1 - itr / args.num_epochs),
                            args.learning_rate * 0.0001)
        if itr % print_step == 0 or itr == args.num_epochs-1:
          loss = sess.run(self.loss, feed_dict=feed_dict)
          print(f'{itr}\t{loss:.3e}')
          if loss < 1e-5:  # Good enough to converge.
            break

      self.embedding_trained = sess.run(self.embedding)
      self.query_trained = sess.run(self.query)


class NetworkCategoricalQueryModel(NetworkModel):
  def __init__(self, args):
    super().__init__()
    self.data_loader = GraphLoader(graph_file=args.graph_file)
    self.num_of_nodes = self.data_loader.num_of_nodes
    args.num_of_nodes = self.num_of_nodes
    self.embedding_dim = args.embedding_dim
    weight_mat = nx.to_numpy_matrix(self.data_loader.g)
    self.num_cat = np.max(weight_mat).astype(int) + 1

    tf.reset_default_graph()
    batch_size = args.num_of_nodes * (args.num_of_nodes-1)
    self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[batch_size])
    self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[batch_size])
    self.label = tf.placeholder(name='label', dtype=tf.float32,
        shape=[batch_size, self.num_cat])

    self.embedding = tf.get_variable('target_embedding',
        [self.num_of_nodes, args.embedding_dim],
        initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
    # query shape: num_cat x embedding_dim x embedding_dim.
    cat_axis = np.ones([self.num_cat,1,1])
    query_init = cat_axis * np.expand_dims(np.eye(args.embedding_dim), axis=0)
    query_init = tf.constant(query_init, dtype=tf.float32)
    # self.query = query_init
    self.query = tf.get_variable('query', initializer=query_init, dtype=tf.float32)

    self.u_i_embedding = tf.matmul(
        tf.one_hot(self.u_i, depth=self.num_of_nodes), self.embedding)
    self.u_j_embedding = tf.matmul(
        tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)

    # embedding_i_query: num_cat x batch_size x embedding_dim.
    embedding_i_query = tf.matmul(self.u_i_embedding, self.query)
    inner_product = embedding_i_query * self.u_j_embedding
    inner_product = tf.reduce_sum(inner_product, axis=2)
    self.inner_product = tf.transpose(inner_product)
    # print(batch_size)
    # print(embedding_i_query.shape)
    # print(self.inner_product.shape)
    # return

    self.loss = tf.losses.softmax_cross_entropy(
        onehot_labels=self.label, logits=self.inner_product)
    self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
    # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)


  def train(
      self,
      args,
      print_step=None):
    """Train the model."""
    if print_step is None:
      print_step = int(args.num_epochs / 10)
    with tf.Session() as sess:
      print('itrs\tloss')
      tf.global_variables_initializer().run()
      self.embedding_init = sess.run(self.embedding)
      learning_rate = args.learning_rate
      u_i, u_j, label = self.data_loader.fetch_categorical()

      for itr in range(args.num_epochs):
        feed_dict = {self.u_i: u_i, self.u_j: u_j, self.label: label,
            self.learning_rate: learning_rate}
        sess.run(self.train_op, feed_dict=feed_dict)
        learning_rate = max(args.learning_rate * (1 - itr / args.num_epochs),
                            args.learning_rate * 0.0001)
        if itr % print_step == 0 or itr == args.num_epochs-1:
          loss = sess.run(self.loss, feed_dict=feed_dict)
          print(f'{itr}\t{loss:.3e}')
          if loss < 1e-5:  # Good enough to converge.
            break
      self.embedding_trained = sess.run(self.embedding)
      self.query_trained = sess.run(self.query)


class NetworkMultiGraphSigmoidQueryModel(NetworkModel):
  def __init__(self, args):
    super().__init__()
    self.data_loader = MultiGraphLoader(graph_file_list=args.graph_file)
    self.num_of_nodes = self.data_loader.num_of_nodes
    self.num_graphs = self.data_loader.num_graphs
    args.num_of_nodes = self.num_of_nodes
    self.embedding_dim = args.embedding_dim

    tf.reset_default_graph()
    batch_size = args.num_of_nodes * (args.num_of_nodes-1)
    self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[batch_size])
    self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[batch_size])
    self.label = tf.placeholder(name='label', dtype=tf.float32,
                                shape=[self.num_graphs, batch_size])

    self.embedding = tf.get_variable('target_embedding',
        [self.num_of_nodes, args.embedding_dim],
        initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
    graph_axis = np.ones([self.num_graphs,1,1])
    query_init = graph_axis * np.expand_dims(np.eye(args.embedding_dim), axis=0)
    query_init = tf.constant(query_init, dtype=tf.float32)
    self.query = tf.get_variable('query', initializer=query_init, dtype=tf.float32)

    self.u_i_embedding = tf.matmul(
        tf.one_hot(self.u_i, depth=self.num_of_nodes), self.embedding)
    self.u_j_embedding = tf.matmul(
        tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
    embedding_i_query = tf.matmul(self.u_i_embedding, self.query)
    self.inner_product = tf.reduce_sum(
        embedding_i_query * self.u_j_embedding, axis=2)

    self.l2_loss = tf.nn.l2_loss(self.embedding) * args.embedding_penalty
    self.loss = tf.losses.sigmoid_cross_entropy(self.label, self.inner_product)
    self.loss = self.loss + self.l2_loss
    self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
    # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)


  def train(
      self,
      args,
      print_step=None):
    """Train the model."""
    if print_step is None:
      print_step = int(args.num_epochs / 10)
    with tf.Session() as sess:
      print('itrs\tloss')
      tf.global_variables_initializer().run()
      self.embedding_init = sess.run(self.embedding)
      learning_rate = args.learning_rate
      u_i, u_j, label = self.data_loader.fetch_binary()

      for itr in range(args.num_epochs):
        feed_dict = {self.u_i: u_i, self.u_j: u_j, self.label: label,
            self.learning_rate: learning_rate}
        sess.run(self.train_op, feed_dict=feed_dict)
        learning_rate = max(args.learning_rate * (1 - itr / args.num_epochs),
                            args.learning_rate * 0.0001)
        if itr % print_step == 0 or itr == args.num_epochs-1:
          loss = sess.run(self.loss, feed_dict=feed_dict)
          print(f'{itr}\t{loss:.3e}')
          if loss < 1e-5:  # Good enough to converge.
            break

      self.embedding_trained = sess.run(self.embedding)
      self.query_trained = sess.run(self.query)

