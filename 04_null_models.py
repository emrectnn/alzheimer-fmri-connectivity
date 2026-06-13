"""Null-model analysis.

Compares observed graph metrics against Erdos-Renyi random networks and reports
deviation (z-score) features per subject.
"""

import os
import sys
import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats as sp_stats
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")
graph_metrics = import_module("03_graph_metrics")


def generate_erdos_renyi(n, p, seed=None):
    """Generate an Erdos-Renyi random graph with a given edge probability."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return graph_metrics.get_largest_component(G)


def generate_configuration_model(degree_sequence, seed=None):
    """Generate a random graph preserving the given degree sequence."""
    deg_seq = list(degree_sequence)
    if sum(deg_seq) % 2 != 0:
        deg_seq[0] += 1

    G = nx.configuration_model(deg_seq, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return graph_metrics.get_largest_component(G)


def generate_rth_like(n, density, seed=None):
    """Generate a random hyperbolic (geometric) graph at a target density."""
    rng = np.random.RandomState(seed)

    radii = np.arccosh(1 + rng.exponential(scale=1.0, size=n))
    angles = rng.uniform(0, 2 * np.pi, size=n)

    def hyperbolic_distance(r1, a1, r2, a2):
        """Hyperbolic distance between two points in the Poincare disk."""
        delta_a = np.abs(a1 - a2)
        if delta_a > np.pi:
            delta_a = 2 * np.pi - delta_a
        return np.arccosh(
            np.cosh(r1) * np.cosh(r2) -
            np.sinh(r1) * np.sinh(r2) * np.cos(delta_a)
        )

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hyperbolic_distance(radii[i], angles[i], radii[j], angles[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    n_target_edges = int(density * n * (n - 1) / 2)
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    sorted_dists = np.sort(upper_tri)

    if n_target_edges >= len(sorted_dists):
        threshold = sorted_dists[-1] + 1
    else:
        threshold = sorted_dists[n_target_edges]

    adj = (dist_matrix <= threshold).astype(int)
    np.fill_diagonal(adj, 0)

    G = nx.from_numpy_array(adj)
    return graph_metrics.get_largest_component(G)


def _compute_null_metrics(args):
    """Compute graph metrics for one random null graph (worker helper)."""
    G_null, metrics_to_compute = args
    result = {}

    if G_null.number_of_nodes() < 5:
        return None

    try:
        if "clustering" in metrics_to_compute:
            result["clustering"] = nx.average_clustering(G_null)

        if "path_length" in metrics_to_compute:
            result["path_length"] = nx.average_shortest_path_length(G_null)

        if "global_eff" in metrics_to_compute:
            result["global_eff"] = nx.global_efficiency(G_null)

        if "sigma" in metrics_to_compute:
            n = G_null.number_of_nodes()
            p = nx.density(G_null)
            G_rand = nx.erdos_renyi_graph(n, p, seed=42)
            G_rand_lc = graph_metrics.get_largest_component(G_rand)
            if G_rand_lc.number_of_nodes() > 2:
                C_r = nx.average_clustering(G_rand_lc)
                L_r = nx.average_shortest_path_length(G_rand_lc)
                C_o = nx.average_clustering(G_null)
                L_o = nx.average_shortest_path_length(G_null)
                gamma = C_o / C_r if C_r > 0 else 0
                lam = L_o / L_r if L_r > 0 else 0
                result["sigma"] = gamma / lam if lam > 0 else 0

    except Exception:
        return None

    return result


def compute_null_distribution(G_real, n_null=None, model="erdos_renyi",
                              metrics=None, parallel=True):
    """Build the null distribution of metrics from random graphs."""
    if n_null is None:
        n_null = config.N_NULL_ITERATIONS

    if metrics is None:
        metrics = ["clustering", "path_length", "global_eff", "sigma"]

    G_lc = graph_metrics.get_largest_component(G_real)
    n = G_lc.number_of_nodes()
    p = nx.density(G_lc)
    degrees = [d for _, d in G_lc.degree()]

    observed = {}
    if "clustering" in metrics:
        observed["clustering"] = nx.average_clustering(G_lc)
    if "path_length" in metrics:
        observed["path_length"] = nx.average_shortest_path_length(G_lc)
    if "global_eff" in metrics:
        observed["global_eff"] = nx.global_efficiency(G_lc)
    if "sigma" in metrics:
        sw = graph_metrics.compute_small_world(G_lc, n_random=50)
        observed["sigma"] = sw["sigma"]

    null_graphs = []
    for i in range(n_null):
        if model == "erdos_renyi":
            G_null = generate_erdos_renyi(n, p, seed=i)
        elif model == "configuration":
            G_null = generate_configuration_model(degrees, seed=i)
        elif model == "rth":
            G_null = generate_rth_like(n, p, seed=i)
        else:
            raise ValueError(f"Bilinmeyen model: {model}")
        null_graphs.append(G_null)

    args_list = [(G_null, metrics) for G_null in null_graphs]

    if parallel and n_null > 10:
        n_workers = min(cpu_count(), 4)
        with Pool(n_workers) as pool:
            results = pool.map(_compute_null_metrics, args_list)
    else:
        results = [_compute_null_metrics(a) for a in args_list]

    null_dist = {m: [] for m in metrics}
    for res in results:
        if res is not None:
            for m in metrics:
                if m in res:
                    null_dist[m].append(res[m])

    null_dist = {m: np.array(v) for m, v in null_dist.items() if v}

    return null_dist, observed


def compute_deviation_scores(null_dist, observed):
    """Compute z-scored deviations of observed metrics from the null distribution."""
    scores = {}

    for metric, null_values in null_dist.items():
        if len(null_values) < 2:
            scores[f"dev_{metric}"] = 0
            continue

        obs = observed.get(metric, 0)
        median_null = np.median(null_values)
        std_null = np.std(null_values)

        if std_null > 0:
            dev = (obs - median_null) / std_null
        else:
            dev = 0

        scores[f"dev_{metric}"] = dev

        p_value = sp_stats.percentileofscore(null_values, obs) / 100
        p_value = 2 * min(p_value, 1 - p_value)
        scores[f"pval_{metric}"] = p_value

    return scores


def null_model_analysis_all(matrices_dict, model="erdos_renyi",
                             n_null=None):
    """Run null-model deviation analysis for every subject."""
    if n_null is None:
        n_null = config.N_NULL_ITERATIONS

    results = []
    n_subj = len(matrices_dict)

    print(f"\nNull model analizi ({model}, n_null={n_null})...")

    for i, (subj_id, data) in enumerate(matrices_dict.items()):
        print(f"  [{i+1}/{n_subj}] {subj_id} ({data['group']})", end="")

        try:
            binary = data["thresholded"].get(0.15)
            if binary is None:
                from importlib import import_module
                conn_mod = import_module("02a_connectivity")
                binary = conn_mod.threshold_by_density(data["raw"], 0.15)

            G = graph_metrics.binary_to_graph(binary)

            null_dist, observed = compute_null_distribution(
                G, n_null=n_null, model=model, parallel=False
            )

            scores = compute_deviation_scores(null_dist, observed)
            scores["subject_id"] = subj_id
            scores["label"] = data["label"]
            scores["group"] = data["group"]

            results.append(scores)

            dev_sigma = scores.get("dev_sigma", 0)
            print(f" -- dev_σ={dev_sigma:.2f}")

        except Exception as e:
            print(f" -- HATA: {e}")

    deviation_df = pd.DataFrame(results)

    save_path = os.path.join(config.METRICS_DIR, f"null_model_{model}.csv")
    deviation_df.to_csv(save_path, index=False)
    print(f"\nDeviation scores kaydedildi: {save_path}")

    return deviation_df


def compare_null_models(G_real, n_null=100):
    """Compare metric deviations across the available null-model families."""
    models = ["erdos_renyi", "configuration", "rth"]
    comparison = {}

    for model in models:
        print(f"  Model: {model}...", end="")
        null_dist, observed = compute_null_distribution(
            G_real, n_null=n_null, model=model, parallel=False
        )
        scores = compute_deviation_scores(null_dist, observed)
        comparison[model] = {
            "observed": observed,
            "null_dist": null_dist,
            "deviation": scores,
        }
        dev_sigma = scores.get("dev_sigma", 0)
        print(f" dev_σ={dev_sigma:.2f}")

    return comparison


if __name__ == "__main__":
    import pickle

    pkl_path = os.path.join(config.PREPROCESSED_DIR, "connectivity_matrices.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            matrices = pickle.load(f)

        dev_df = null_model_analysis_all(matrices, model="erdos_renyi")
        print("\nDeviation scores ozeti:")
        print(dev_df.groupby("group")[
            [c for c in dev_df.columns if c.startswith("dev_")]
        ].mean())
    else:
        print("Baglanti matrisleri bulunamadi.")
