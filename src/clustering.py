import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

from preprocessing import build_client_features

NUMERIC_FEATURES = ["num_properties", "investment_tenure_days", "tenure_per_property"]
BINARY_FEATURES = ["loan_applied_flag", "is_company"]


def encode_features(df: pd.DataFrame):
    work = df.copy()
    label_encoders = {}
    for col in ["client_type", "acquisition_purpose", "gender"]:
        le = LabelEncoder()
        work[col + "_label"] = le.fit_transform(work[col])
        label_encoders[col] = le

    model_df = pd.concat([work[["client_id"]], work[NUMERIC_FEATURES], work[BINARY_FEATURES]], axis=1)
    return model_df, label_encoders


def scale_features(model_df: pd.DataFrame):
    scaler = StandardScaler()
    feature_cols = [c for c in model_df.columns if c != "client_id"]
    scaled = scaler.fit_transform(model_df[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=model_df.index)
    scaled_df.insert(0, "client_id", model_df["client_id"].values)
    return scaled_df, scaler, feature_cols


def evaluate_k_range(X: np.ndarray, k_range=range(2, 11)):
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(results)


def fit_kmeans(X, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km, km.fit_predict(X)


def fit_hierarchical(X, n_clusters):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return agg, agg.fit_predict(X)


def run_pipeline(n_clusters: int = 5, k_search_range=range(2, 11)):
    raw = build_client_features()
    model_df, label_encoders = encode_features(raw)
    scaled_df, scaler, feature_cols = scale_features(model_df)
    X = scaled_df[feature_cols].values

    eval_df = evaluate_k_range(X, k_search_range)
    kmeans_model, kmeans_labels = fit_kmeans(X, n_clusters)
    hier_model, hier_labels = fit_hierarchical(X, n_clusters)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    result = raw.copy()
    result["kmeans_cluster"] = kmeans_labels
    result["hierarchical_cluster"] = hier_labels
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]

    artifacts = {"scaler": scaler, "kmeans_model": kmeans_model, "feature_cols": feature_cols, "eval_df": eval_df}
    return result, artifacts


def name_clusters(result: pd.DataFrame, cluster_col: str = "kmeans_cluster") -> dict:

    profile = result.groupby(cluster_col).agg(
        loan_rate=("loan_applied_flag", "mean"),
        company_rate=("is_company", "mean"),
        avg_properties=("num_properties", "mean"),
        avg_tenure=("investment_tenure_days", "mean"),
        avg_pace=("tenure_per_property", "mean"),
    )

    def z(s):
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    z_properties = z(profile["avg_properties"])
    z_company = z(profile["company_rate"])
    z_loan = z(profile["loan_rate"])
    z_tenure = z(profile["avg_tenure"])

    names = {}
    for cid in profile.index:
        if z_properties[cid] > 1.5:
            names[cid] = "Large-Portfolio Buyers"
        elif z_company[cid] > 1.0:
            names[cid] = "Corporate Buyers"
        elif z_loan[cid] > 1.0:
            names[cid] = "Loan-Backed Buyers"
        elif z_tenure[cid] > 0.3:
            names[cid] = "Slow-Accumulating Buyers"
        elif z_tenure[cid] < -0.3:
            names[cid] = "Fast-Accumulating Buyers"
        else:
            names[cid] = "Moderate-Pace Cash Buyers"

    seen = {}
    for cid in list(names.keys()):
        label = names[cid]
        seen.setdefault(label, []).append(cid)
    for label, cids in seen.items():
        if len(cids) > 1:
            for i, cid in enumerate(cids, start=1):
                names[cid] = f"{label} ({i})"

    return names



if __name__ == "__main__":
    result, artifacts = run_pipeline(n_clusters=5)
    names = name_clusters(result)
    result["segment_name"] = result["kmeans_cluster"].map(names)

    result.to_csv("outputs/clustered_clients.csv", index=False)
    artifacts["eval_df"].to_csv("outputs/k_evaluation.csv", index=False)
    joblib.dump(artifacts["scaler"], "outputs/scaler.joblib")
    joblib.dump(artifacts["kmeans_model"], "outputs/kmeans_model.joblib")
    joblib.dump(artifacts["feature_cols"], "outputs/feature_cols.joblib")

    print(result.segment_name.value_counts())
    print()
    print(result.groupby("segment_name")[["num_properties", "investment_tenure_days",
                                           "tenure_per_property", "loan_applied_flag",
                                           "is_company"]].mean().round(2))
    print()
    print(artifacts["eval_df"])