"""Tests for graph-theoretic metric computation."""

import numpy as np
from importlib import import_module

gm_mod = import_module("03_graph_metrics")


def test_binary_to_graph_edge_count(synthetic_binary_matrix):
    G = gm_mod.binary_to_graph(synthetic_binary_matrix)
    n = synthetic_binary_matrix.shape[0]
    assert G.number_of_nodes() == n
    expected_edges = synthetic_binary_matrix.sum() // 2
    assert G.number_of_edges() == expected_edges


def test_compute_global_metrics_no_nan(synthetic_graph):
    metrics = gm_mod.compute_global_metrics(synthetic_graph)
    assert 0.0 <= metrics["clustering"] <= 1.0
    assert 0.0 <= metrics["density"] <= 1.0
    assert metrics["global_eff"] >= 0.0
    for k, v in metrics.items():
        assert not (isinstance(v, float) and np.isnan(v)), f"{k} NaN uretti"


def test_compute_nodal_metrics_length(synthetic_graph):
    nodal = gm_mod.compute_nodal_metrics(synthetic_graph)
    n = synthetic_graph.number_of_nodes()
    for key in ("degree_centrality", "betweenness_centrality",
                "clustering", "eigenvector_centrality"):
        assert key in nodal, f"{key} eksik"
        assert len(nodal[key]) == n


def test_density_monotonic_edge_count(synthetic_corr_matrix):
    conn_mod = import_module("02_connectivity")
    prev_edges = -1
    for d in (0.05, 0.10, 0.20, 0.30):
        binary = conn_mod.threshold_by_density(synthetic_corr_matrix, d)
        G = gm_mod.binary_to_graph(binary)
        assert G.number_of_edges() >= prev_edges, (
            f"density={d}'de kenar azaldi (monoton olmayan): "
            f"prev={prev_edges}, now={G.number_of_edges()}"
        )
        prev_edges = G.number_of_edges()
