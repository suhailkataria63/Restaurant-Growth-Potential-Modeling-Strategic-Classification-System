from __future__ import annotations

from typing import Dict

try:
    from .dimensionality_reduction import perform_dimensionality_reduction
except ImportError:
    from dimensionality_reduction import perform_dimensionality_reduction


def run_clustering_preparation(
    feature_matrix_path: str = "data/processed/feature_matrix.csv",
) -> Dict[str, object]:
    """
    Prepare clustering inputs by generating PCA (and optional UMAP) embeddings.
    """
    outputs = perform_dimensionality_reduction(feature_matrix_path=feature_matrix_path)
    print("Clustering preparation completed.")
    return outputs


if __name__ == "__main__":
    run_clustering_preparation()
