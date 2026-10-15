"""Unsupervised learning for the placement dataset.

The algorithms deliberately exclude identifiers, the supervised target,
pre-existing anomaly labels, and salary because those columns would leak
outcomes into an unsupervised structure-discovery task.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram

try:
    import umap.umap_ as umap
except ImportError:  # pragma: no cover
    umap = None

ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "static" / "plots" / "unsupervised"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED = {
    "StudentID", "PlacementStatus", "IsAnomaly", "Salary Package"
}


def _numeric_frame(df):
    cols = [c for c in df.select_dtypes(include=np.number).columns if c not in EXCLUDED]
    x = df[cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0)
    return x, cols


def _sample_indices(n, max_n, seed=42):
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def _save(fig, name):
    path = PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"plots/unsupervised/{name}"


def _fmt(v):
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    return v


def run_unsupervised(df):
    x_df, feature_names = _numeric_frame(df)
    scaler = StandardScaler()
    x = scaler.fit_transform(x_df)
    n = len(x)

    # K-Means on the complete dataset.
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(x)
    km_sil = silhouette_score(x, km_labels, sample_size=min(10000, n), random_state=42)
    km_counts = pd.Series(km_labels).value_counts().sort_index().to_dict()

    fig, ax = plt.subplots(figsize=(9, 5))
    pca2 = PCA(n_components=2, random_state=42).fit_transform(x)
    sc = ax.scatter(pca2[:, 0], pca2[:, 1], c=km_labels, s=8, alpha=0.45)
    ax.set_title("K-Means Clusters (shown in PCA space)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="Cluster")
    km_plot = _save(fig, "kmeans_clusters.png")

    # Hierarchical clustering is intentionally sampled to avoid O(n^2) memory.
    hi_idx = _sample_indices(n, 1500)
    x_hi = x[hi_idx]
    hi_model = AgglomerativeClustering(n_clusters=4, linkage="ward")
    hi_labels = hi_model.fit_predict(x_hi)
    hi_sil = silhouette_score(x_hi, hi_labels)
    hi_counts = pd.Series(hi_labels).value_counts().sort_index().to_dict()

    dendro_idx = _sample_indices(n, 250)
    z = linkage(x[dendro_idx], method="ward")
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(z, ax=ax, no_labels=True, truncate_mode="lastp", p=30)
    ax.set_title("Hierarchical Clustering Dendrogram (sample)")
    ax.set_xlabel("Sample / merged cluster")
    ax.set_ylabel("Ward distance")
    hi_plot = _save(fig, "hierarchical_dendrogram.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    hi_pca = PCA(n_components=2, random_state=42).fit_transform(x_hi)
    sc = ax.scatter(hi_pca[:, 0], hi_pca[:, 1], c=hi_labels, s=12, alpha=0.65)
    ax.set_title("Hierarchical Clusters (sample, PCA space)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="Cluster")
    hi_cluster_plot = _save(fig, "hierarchical_clusters.png")

    # DBSCAN is sampled because its worst-case memory/time can be quadratic.
    db_idx = _sample_indices(n, 5000)
    x_db = x[db_idx]
    db = DBSCAN(eps=2.0, min_samples=12, n_jobs=-1)
    db_labels = db.fit_predict(x_db)
    db_cluster_ids = sorted(c for c in set(db_labels) if c != -1)
    db_noise = int(np.sum(db_labels == -1))
    db_sil = None
    if len(db_cluster_ids) >= 2:
        mask = db_labels != -1
        if mask.sum() > 2:
            db_sil = silhouette_score(x_db[mask], db_labels[mask])
    fig, ax = plt.subplots(figsize=(9, 5))
    db_pca = PCA(n_components=2, random_state=42).fit_transform(x_db)
    sc = ax.scatter(db_pca[:, 0], db_pca[:, 1], c=db_labels, s=10, alpha=0.65)
    ax.set_title("DBSCAN Clusters (sample, PCA space; -1 = noise)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="Cluster")
    db_plot = _save(fig, "dbscan_clusters.png")

    # PCA on all rows.
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(x)
    pca_var = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(pca_data[:, 0], pca_data[:, 1], s=8, alpha=0.4)
    ax.set_title("PCA: 2D Projection")
    ax.set_xlabel(f"PC1 ({pca_var[0]:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_var[1]:.2f}% variance)")
    pca_plot = _save(fig, "pca_2d.png")

    # t-SNE and UMAP use bounded samples for browser-friendly runtime.
    dr_idx = _sample_indices(n, 2000)
    x_dr = x[dr_idx]
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42, max_iter=750)
    tsne_data = tsne.fit_transform(x_dr)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(tsne_data[:, 0], tsne_data[:, 1], s=10, alpha=0.55)
    ax.set_title("t-SNE: 2D Projection (2,000-row sample)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    tsne_plot = _save(fig, "tsne_2d.png")

    umap_plot = None
    umap_status = "UMAP package available"
    if umap is not None:
        u_idx = _sample_indices(n, 3000)
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        u_data = reducer.fit_transform(x[u_idx])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(u_data[:, 0], u_data[:, 1], s=9, alpha=0.5)
        ax.set_title("UMAP: 2D Projection (3,000-row sample)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        umap_plot = _save(fig, "umap_2d.png")
    else:
        umap_status = "Install umap-learn to enable UMAP"

    # Isolation Forest anomaly detection on all rows.
    iso = IsolationForest(n_estimators=150, contamination=0.02, random_state=42, n_jobs=-1)
    anomaly_pred = iso.fit_predict(x)
    anomaly_score = -iso.score_samples(x)
    anomaly_count = int(np.sum(anomaly_pred == -1))
    anomaly_rate = anomaly_count / n * 100 if n else 0
    top_idx = np.argsort(anomaly_score)[-10:][::-1]
    anomaly_examples = []
    for idx in top_idx:
        row = df.iloc[int(idx)]
        anomaly_examples.append({
            "student_id": int(row["StudentID"]) if "StudentID" in df.columns else int(idx),
            "score": round(float(anomaly_score[idx]), 4),
            "cgpa": round(float(row["CGPA"]), 2) if pd.notna(row.get("CGPA")) else None,
            "attendance": round(float(row["AttendancePercent"]), 2) if pd.notna(row.get("AttendancePercent")) else None,
            "coding": round(float(row["CodingTestScore"]), 2) if pd.notna(row.get("CodingTestScore")) else None,
        })
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(anomaly_score, bins=40)
    ax.axvline(np.quantile(anomaly_score, 0.98), linestyle="--", label="Approx. anomaly cutoff")
    ax.set_title("Isolation Forest Anomaly Score Distribution")
    ax.set_xlabel("Anomaly score (higher = more unusual)")
    ax.set_ylabel("Students")
    ax.legend()
    anomaly_plot = _save(fig, "anomaly_scores.png")

    # Compact cluster profiles.
    profile = x_df.copy()
    profile["cluster"] = km_labels
    cluster_profile = profile.groupby("cluster")[feature_names].mean().round(2).reset_index()
    cluster_profile = cluster_profile.to_dict(orient="records")

    return {
        "n_rows": n,
        "n_features": len(feature_names),
        "features": feature_names,
        "kmeans": {"clusters": 4, "silhouette": _fmt(km_sil), "counts": {str(k): int(v) for k, v in km_counts.items()}, "plot": km_plot},
        "hierarchical": {"clusters": 4, "sample_size": len(x_hi), "silhouette": _fmt(hi_sil), "counts": {str(k): int(v) for k, v in hi_counts.items()}, "dendrogram": hi_plot, "plot": hi_cluster_plot},
        "dbscan": {"sample_size": len(x_db), "clusters": len(db_cluster_ids), "noise": db_noise, "noise_rate": round(db_noise / len(x_db) * 100, 2), "silhouette": _fmt(db_sil) if db_sil is not None else "N/A", "plot": db_plot},
        "pca": {"variance_pc1": round(float(pca_var[0]), 2), "variance_pc2": round(float(pca_var[1]), 2), "total_variance": round(float(pca_var.sum()), 2), "plot": pca_plot},
        "tsne": {"sample_size": len(x_dr), "plot": tsne_plot},
        "umap": {"sample_size": 3000 if umap is not None else 0, "plot": umap_plot, "status": umap_status},
        "anomaly": {"method": "Isolation Forest", "count": anomaly_count, "rate": round(anomaly_rate, 2), "contamination": 0.02, "plot": anomaly_plot, "examples": anomaly_examples},
        "cluster_profile": cluster_profile,
    }
