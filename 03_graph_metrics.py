"""Graph-theoretic network metrics.

Global and nodal metrics integrated over a range of graph densities with the
area-under-the-curve (AUC) strategy to remove single-threshold dependence.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

try:
    from community import best_partition
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("[!] python-louvain paketi bulunamadi: pip install python-louvain")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")


def get_largest_component(G):
    if nx.is_connected(G):
        return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def binary_to_graph(binary_matrix):
    G = nx.from_numpy_array(binary_matrix)
    return G


def compute_global_metrics(G):
    """Compute global graph metrics for a binary graph."""
    G_lc = get_largest_component(G)
    n = G_lc.number_of_nodes()
    metrics = {}

    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_nodes_lc"] = n
    metrics["n_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)

    metrics["clustering"] = nx.average_clustering(G_lc)

    if n > 1:
        metrics["path_length"] = nx.average_shortest_path_length(G_lc)
    else:
        metrics["path_length"] = 0.0

    metrics["global_eff"] = nx.global_efficiency(G_lc)

    metrics["local_eff"] = nx.local_efficiency(G_lc)

    metrics["transitivity"] = nx.transitivity(G_lc)

    return metrics


def compute_small_world(G, n_random=100):
    """Compute the small-world index sigma against random graphs."""
    G_lc = get_largest_component(G)
    n = G_lc.number_of_nodes()
    p = nx.density(G_lc)

    if n < 3 or p == 0:
        return {"sigma": 0, "gamma": 0, "lambda_norm": 0}

    C_real = nx.average_clustering(G_lc)
    L_real = nx.average_shortest_path_length(G_lc)

    rand_C, rand_L = [], []
    for _ in range(n_random):
        G_rand = nx.erdos_renyi_graph(n, p)
        G_rand_lc = get_largest_component(G_rand)
        if G_rand_lc.number_of_nodes() > 2:
            rand_C.append(nx.average_clustering(G_rand_lc))
            try:
                rand_L.append(nx.average_shortest_path_length(G_rand_lc))
            except nx.NetworkXError:
                pass

    if not rand_C or not rand_L:
        return {"sigma": 0, "gamma": 0, "lambda_norm": 0}

    C_rand = np.mean(rand_C)
    L_rand = np.mean(rand_L)

    gamma = C_real / C_rand if C_rand > 0 else 0

    lambda_norm = L_real / L_rand if L_rand > 0 else 0

    sigma = gamma / lambda_norm if lambda_norm > 0 else 0

    return {
        "sigma": sigma,
        "gamma": gamma,
        "lambda_norm": lambda_norm,
        "C_real": C_real,
        "L_real": L_real,
        "C_rand": C_rand,
        "L_rand": L_rand,
    }


def compute_dmn_clustering(G, dmn_indices=None):
    if dmn_indices is None:
        dmn_indices = config.DMN_ROI_INDICES

    cc = nx.clustering(G)

    dmn_cc = [cc.get(i, 0) for i in dmn_indices if i in G.nodes()]

    if not dmn_cc:
        return {"dmn_clustering_mean": 0, "dmn_clustering_std": 0}

    return {
        "dmn_clustering_mean": np.mean(dmn_cc),
        "dmn_clustering_std": np.std(dmn_cc),
        "dmn_clustering_values": dmn_cc,
    }


def compute_rich_club(G, k_range=None):
    G_lc = get_largest_component(G)

    if G_lc.number_of_nodes() < 3:
        return {"rich_club_mean": 0, "rich_club_max": 0}

    try:
        rc = nx.rich_club_coefficient(G_lc, normalized=False)
    except Exception:
        return {"rich_club_mean": 0, "rich_club_max": 0}

    if not rc:
        return {"rich_club_mean": 0, "rich_club_max": 0}

    rc_values = list(rc.values())

    degrees = [d for _, d in G_lc.degree()]
    mean_degree = np.mean(degrees)

    return {
        "rich_club_mean": np.mean(rc_values),
        "rich_club_max": np.max(rc_values) if rc_values else 0,
        "rich_club_at_mean_k": rc.get(int(mean_degree), 0),
        "rich_club_dict": rc,
    }


def compute_cerebellar_connectivity(conn_matrix, binary_matrix,
                                     cerebellar_indices=None):
    if cerebellar_indices is None:
        cerebellar_indices = config.CEREBELLAR_ROI_INDICES

    n = conn_matrix.shape[0]
    cortical_indices = [i for i in range(n) if i not in cerebellar_indices]

    cb_cx_raw = conn_matrix[np.ix_(cerebellar_indices, cortical_indices)]
    cb_cx_binary = binary_matrix[np.ix_(cerebellar_indices, cortical_indices)]

    return {
        "cb_cx_mean_corr": np.mean(np.abs(cb_cx_raw)),
        "cb_cx_density": np.mean(cb_cx_binary),
        "cb_cx_max_corr": np.max(np.abs(cb_cx_raw)),
        "cb_cx_n_connections": int(np.sum(cb_cx_binary)),
    }


def compute_louvain_modularity(G, n_repetitions=None):
    if not HAS_LOUVAIN:
        return {"modularity": 0, "n_communities": 0}

    if n_repetitions is None:
        n_repetitions = config.LOUVAIN_N_REPETITIONS

    G_lc = get_largest_component(G)
    n = G_lc.number_of_nodes()

    if n < 3:
        return {"modularity": 0, "n_communities": 0}

    modularities = []
    all_partitions = []
    community_counts = []

    for rep in range(n_repetitions):
        partition = best_partition(G_lc, random_state=rep)

        communities = [
            {node for node, comm in partition.items() if comm == c}
            for c in set(partition.values())
        ]

        Q = nx.community.modularity(G_lc, communities)
        modularities.append(Q)
        all_partitions.append(partition)
        community_counts.append(len(communities))

    median_Q = np.median(modularities)

    median_idx = np.argmin(np.abs(np.array(modularities) - median_Q))

    co_assignment = np.zeros((n, n))
    node_list = sorted(G_lc.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    for partition in all_partitions:
        for ni in node_list:
            for nj in node_list:
                if partition.get(ni) == partition.get(nj):
                    co_assignment[node_to_idx[ni], node_to_idx[nj]] += 1

    co_assignment /= n_repetitions

    upper_tri = co_assignment[np.triu_indices(n, k=1)]
    consensus_stability = np.mean(upper_tri)

    return {
        "modularity": median_Q,
        "modularity_mean": np.mean(modularities),
        "modularity_std": np.std(modularities),
        "n_communities": int(np.median(community_counts)),
        "consensus_stability": consensus_stability,
        "best_partition": all_partitions[median_idx],
        "co_assignment": co_assignment,
    }


def compute_nodal_metrics(G):
    """Compute nodal centrality metrics for each region."""
    G_lc = get_largest_component(G)
    n_total = G.number_of_nodes()

    dc = nx.degree_centrality(G_lc)
    bc = nx.betweenness_centrality(G_lc, normalized=True)
    cc = nx.clustering(G_lc)

    try:
        ec = nx.eigenvector_centrality(G_lc, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        ec = {n: 0 for n in G_lc.nodes()}

    nodal = {
        "degree_centrality": np.zeros(n_total),
        "betweenness_centrality": np.zeros(n_total),
        "clustering": np.zeros(n_total),
        "eigenvector_centrality": np.zeros(n_total),
    }

    for node in G_lc.nodes():
        if node < n_total:
            nodal["degree_centrality"][node] = dc.get(node, 0)
            nodal["betweenness_centrality"][node] = bc.get(node, 0)
            nodal["clustering"][node] = cc.get(node, 0)
            nodal["eigenvector_centrality"][node] = ec.get(node, 0)

    return nodal


def compute_metrics_across_densities(conn_matrix, density_range=None):
    """Integrate graph metrics over densities using the AUC strategy."""
    if density_range is None:
        density_range = config.DENSITY_RANGE

    metrics_per_density = defaultdict(list)
    densities_used = []

    for d in density_range:
        from importlib import import_module
        conn_mod = import_module("02_connectivity")
        binary = conn_mod.threshold_by_density(conn_matrix, d)
        G = binary_to_graph(binary)

        if not nx.is_connected(G):
            lc_size = len(max(nx.connected_components(G), key=len))
            if lc_size < 10:
                continue

        densities_used.append(d)

        gm = compute_global_metrics(G)
        for k, v in gm.items():
            if isinstance(v, (int, float)):
                metrics_per_density[k].append(v)

        sw = compute_small_world(G, n_random=50)
        for k, v in sw.items():
            if isinstance(v, (int, float)):
                metrics_per_density[k].append(v)

        dmn = compute_dmn_clustering(G)
        metrics_per_density["dmn_clustering"].append(dmn["dmn_clustering_mean"])

        rc = compute_rich_club(G)
        metrics_per_density["rich_club"].append(rc["rich_club_mean"])

    auc_metrics = {}
    densities_arr = np.array(densities_used)

    for metric_name, values in metrics_per_density.items():
        if len(values) == len(densities_used) and len(values) > 1:
            trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            auc = trapz_fn(values, densities_arr)
            auc_metrics[f"auc_{metric_name}"] = auc

    return auc_metrics, metrics_per_density, densities_used


def compute_all_for_subject(conn_matrix, binary_matrix=None, density=0.15):
    """Compute all graph metrics for one subject."""
    from importlib import import_module
    conn_mod = import_module("02_connectivity")

    if binary_matrix is None:
        binary_matrix = conn_mod.threshold_by_density(conn_matrix, density)

    G = binary_to_graph(binary_matrix)

    metrics = {}

    metrics.update(compute_global_metrics(G))

    sw = compute_small_world(G, n_random=config.N_NULL_ITERATIONS)
    metrics.update(sw)

    dmn = compute_dmn_clustering(G)
    metrics["dmn_clustering"] = dmn["dmn_clustering_mean"]
    metrics["dmn_clustering_std"] = dmn["dmn_clustering_std"]

    rc = compute_rich_club(G)
    metrics["rich_club_mean"] = rc["rich_club_mean"]
    metrics["rich_club_max"] = rc["rich_club_max"]

    cb = compute_cerebellar_connectivity(conn_matrix, binary_matrix)
    metrics.update(cb)

    louv = compute_louvain_modularity(G)
    metrics["modularity"] = louv["modularity"]
    metrics["n_communities"] = louv["n_communities"]
    metrics["consensus_stability"] = louv["consensus_stability"]

    auc_metrics, _, _ = compute_metrics_across_densities(conn_matrix)
    metrics.update(auc_metrics)

    nodal = compute_nodal_metrics(G)

    return metrics, nodal


def compute_all_subjects(matrices_dict):
    """Compute graph metrics for every subject."""
    all_global = []
    all_nodal = []

    n_subj = len(matrices_dict)
    print(f"\nGraf metrikleri hesaplaniyor ({n_subj} denek)...")

    for i, (subj_id, data) in enumerate(matrices_dict.items()):
        print(f"  [{i+1}/{n_subj}] {subj_id} ({data['group']})", end="")

        try:
            binary = data["thresholded"].get(0.15)
            if binary is None:
                from importlib import import_module
                conn_mod = import_module("02_connectivity")
                binary = conn_mod.threshold_by_density(data["raw"], 0.15)

            gm, nm = compute_all_for_subject(data["raw"], binary)
            gm["subject_id"] = subj_id
            gm["label"] = data["label"]
            gm["group"] = data["group"]

            all_global.append(gm)
            nm["subject_id"] = subj_id
            nm["label"] = data["label"]
            nm["group"] = data["group"]
            all_nodal.append(nm)

            print(f" -- sigma={gm.get('sigma', 0):.3f}, Q={gm.get('modularity', 0):.3f}")

        except Exception as e:
            print(f" -- HATA: {e}")

    global_df = pd.DataFrame(all_global)

    save_path = os.path.join(config.METRICS_DIR, "global_metrics.csv")
    global_df.to_csv(save_path, index=False)
    print(f"\nGlobal metrikler kaydedildi: {save_path}")

    return global_df, all_nodal


if __name__ == "__main__":
    import pickle

    pkl_path = os.path.join(config.PREPROCESSED_DIR, "connectivity_matrices.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            matrices = pickle.load(f)
        global_df, all_nodal = compute_all_subjects(matrices)
        print(f"\n{len(global_df)} denek icin metrikler hesaplandi.")
        print(global_df.describe())
    else:
        print("Baglanti matrisleri bulunamadi. Once 02_connectivity.py calistirin.")
