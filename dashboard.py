import base64
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

EXPERIMENTS_ROOT_DEFAULT = str(Path(__file__).parent / "resources" / "experiments")
RESERVED_DIRS = frozenset({"shared", "processed_data", "tb"})
FIGURE_EXTENSIONS = (".png", ".pdf")

HEATMAP_METRICS: dict[str, str] = {
    "f1_macro": "Test F1 macro",
    "f1_weighted": "Test F1 weighted",
    "accuracy": "Test accuracy",
    "precision_macro": "Test precision macro",
    "recall_macro": "Test recall macro",
    "f1_extended": "F1 extended (transductive)",
    "fc_spearman": "Failure-regressor Spearman ρ",
    "fc_r2": "Failure-regressor R²",
    "fc_mae": "Failure-regressor MAE",
}

METRIC_COLORSCALE_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "f1_macro": (0.0, 1.0),
    "f1_weighted": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "precision_macro": (0.0, 1.0),
    "recall_macro": (0.0, 1.0),
    "f1_extended": (0.0, 1.0),
    "fc_spearman": (-1.0, 1.0),
    "fc_r2": (-1.0, 1.0),
    "fc_mae": (None, None),  # auto-range; lower-is-better
}

GALLERY_CATEGORIES: list[str] = [
    "testing",
    "training",
    "summary",
    "summary/correlation",
    "summary/global",
    "explain",
]

CLUSTER_NON_FEATURE_COLS = frozenset(
    {"cluster_id", "cluster_class", "is_noise_cluster", "failure_rate"}
)


@dataclass(frozen=True)
class ExperimentRecord:
    """Lightweight handle to one (variant, dataset, seed, classifier) experiment."""

    variant: str
    dataset_dir: str
    file_name: str
    seed: int
    classifier: str
    family: Literal["ml", "dl"]
    root: Path
    shared: Path
    accuracy: float | None
    f1_macro: float | None
    f1_weighted: float | None
    precision_macro: float | None
    recall_macro: float | None
    f1_extended: float | None
    fc_spearman: float | None
    fc_r2: float | None
    fc_mae: float | None
    fc_skipped: bool = False

    @property
    def key(self) -> str:
        return f"{self.variant}|{self.dataset_dir}|{self.classifier}"

    @property
    def label(self) -> str:
        return f"{self.variant} · {self.file_name} · {self.classifier}"


@dataclass
class ExperimentDetail:
    """All heavy artifacts for one experiment, loaded lazily."""

    testing: dict | None = None
    classifier_results: dict | None = None
    cluster_summary: pd.DataFrame | None = None
    predictions: dict | None = None
    confusion_matrix: np.ndarray | None = None
    grid_search: dict | None = None
    df_meta: dict = field(default_factory=dict)
    df_info: dict = field(default_factory=dict)
    complexity: dict = field(default_factory=dict)
    clusters_meta: dict = field(default_factory=dict)


def _read_json(path: Path) -> dict | list | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _parse_dataset_dir(name: str) -> tuple[str, int] | None:
    """Split `{file_name}_{seed}` → (file_name, seed). Returns None on failure."""
    head, _, tail = name.rpartition("_")
    if not head or not tail.isdigit():
        return None
    return head, int(tail)


def _classifier_family(classifier_dir: Path) -> Literal["ml", "dl"]:
    """Infer ML vs DL from config_composed.json::classifier.kind (fallback: model file)."""
    cfg = _read_json(classifier_dir / "configs" / "config_composed.json")
    if isinstance(cfg, dict):
        kind = cfg.get("classifier", {}).get("kind")
        if kind in {"ml", "dl"}:
            return kind  # type: ignore[return-value]
    if (classifier_dir / "models" / "model.joblib").exists():
        return "ml"
    if any((classifier_dir / "models").glob("*.pt")):
        return "dl"
    return "ml"


def _extract_headline_metrics(classifier_dir: Path) -> dict[str, float | bool | None]:
    summary = _read_json(classifier_dir / "outputs" / "testing" / "summary.json") or {}
    fc = (
        _read_json(classifier_dir / "outputs" / "analysis" / "classifier_results.json")
        or {}
    )
    extended = (
        _read_json(classifier_dir / "outputs" / "testing" / "summary_extended.json")
        or {}
    )
    return {
        "accuracy": _safe_float(summary.get("accuracy")),
        "f1_macro": _safe_float(summary.get("f1_macro")),
        "f1_weighted": _safe_float(summary.get("f1_weighted")),
        "precision_macro": _safe_float(summary.get("precision_macro")),
        "recall_macro": _safe_float(summary.get("recall_macro")),
        "f1_extended": _safe_float(extended.get("f1_macro")),
        "fc_spearman": _safe_float(fc.get("spearman")),
        "fc_r2": _safe_float(fc.get("r2")),
        "fc_mae": _safe_float(fc.get("mae")),
        "fc_skipped": bool(fc.get("skipped", False)),
    }


def _is_classifier_dir(path: Path) -> bool:
    """Heuristic: a classifier dir has either an outputs/ or a configs/config_composed.json."""
    if not path.is_dir():
        return False
    if path.name in RESERVED_DIRS or path.name.startswith("."):
        return False
    return (path / "outputs").exists() or (
        path / "configs" / "config_composed.json"
    ).exists()


@st.cache_data(show_spinner="Scanning experiments…")
def discover_experiments(root: str) -> tuple[list[ExperimentRecord], int]:
    """Walk `root/<variant>/<dataset_dir>/` and emit one record per classifier.

    Returns `(records, n_skipped_legacy)`.
    """
    root_path = Path(root)
    records: list[ExperimentRecord] = []
    skipped = 0

    if not root_path.is_dir():
        return [], 0

    for variant_dir in sorted(root_path.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("."):
            continue
        for dataset_dir in sorted(variant_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            parsed = _parse_dataset_dir(dataset_dir.name)
            if parsed is None:
                continue
            file_name, seed = parsed
            shared = dataset_dir / "shared"
            if not (shared / "metadata/df_meta.json").exists():
                skipped += 1
                continue
            for classifier_dir in sorted(dataset_dir.iterdir()):
                if not _is_classifier_dir(classifier_dir):
                    continue
                records.append(
                    ExperimentRecord(
                        variant=variant_dir.name,
                        dataset_dir=dataset_dir.name,
                        file_name=file_name,
                        seed=seed,
                        classifier=classifier_dir.name,
                        family=_classifier_family(classifier_dir),
                        root=classifier_dir,
                        shared=shared,
                        **_extract_headline_metrics(classifier_dir),
                    )
                )

    return records, skipped


def _cluster_summary_df(data: dict | None) -> pd.DataFrame | None:
    if not data:
        return None
    df = pd.DataFrame([{"cluster_id": cid, **row} for cid, row in data.items()])
    if "failure_rate" in df.columns:
        df = df.sort_values("failure_rate", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_experiment_detail(record_root: str, record_shared: str) -> ExperimentDetail:
    """Load every heavy artifact for one experiment. Returns empty fields when missing."""
    root = Path(record_root)
    shared = Path(record_shared)
    detail = ExperimentDetail(
        testing=_read_json(root / "outputs" / "testing" / "summary.json"),
        classifier_results=_read_json(
            root / "outputs" / "analysis" / "classifier_results.json"
        ),
        predictions=_read_json(
            root / "outputs" / "analysis" / "predictions" / "test.json"
        ),
        grid_search=_read_json(root / "outputs" / "training" / "grid_search.json"),
        df_meta=_read_json(shared / "metadata/df_meta.json") or {},
        df_info=_read_json(shared / "metadata/df_info.json") or {},
        complexity=_read_json(shared / "complexity.json") or {},
        clusters_meta=_read_json(shared / "metadata/clusters_meta.json") or {},
    )
    cs = _read_json(root / "outputs" / "analysis" / "cluster_summary.json")
    detail.cluster_summary = _cluster_summary_df(cs if isinstance(cs, dict) else None)
    cm = _read_pickle(root / "pickle" / "analysis" / "confusion_matrices" / "test.pkl")
    if isinstance(cm, np.ndarray):
        detail.confusion_matrix = cm
    elif isinstance(cm, list):
        detail.confusion_matrix = np.asarray(cm)
    return detail


@st.cache_data(show_spinner=False)
def load_figure_index(record_root: str) -> dict[str, str]:
    """Return `{relative_posix_path: absolute_path}` for every figure under `figures/`."""
    figures_dir = Path(record_root) / "figures"
    if not figures_dir.is_dir():
        return {}
    return {
        fig.relative_to(figures_dir).as_posix(): str(fig)
        for fig in sorted(figures_dir.rglob("*"))
        if fig.suffix.lower() in FIGURE_EXTENSIONS
    }


@st.cache_data(show_spinner=False)
def count_clusters(record_shared: str) -> int | None:
    """Total number of clusters for a dataset, read from shared clusters_meta.json."""
    meta = _read_json(Path(record_shared) / "metadata/clusters_meta.json")
    if not isinstance(meta, dict):
        return None
    dist = meta.get("clusters_distribution")
    if isinstance(dist, dict) and dist:
        return len(dist)
    centroids = meta.get("centroids")
    return len(centroids) if isinstance(centroids, dict) and centroids else None


@st.cache_data(show_spinner=False)
def _dataset_test_size(shared: str) -> int:
    """Test-set row count for a dataset, read from shared df_meta.json."""
    meta = _read_json(Path(shared) / "metadata/df_meta.json") or {}
    return (meta.get("dataset_sizes") or {}).get("test", 0)


def records_to_df(records: list[ExperimentRecord]) -> pd.DataFrame:
    """Flat DataFrame view of records (one row per (variant, dataset, classifier))."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "variant": r.variant,
                "dataset": r.file_name,
                "seed": r.seed,
                "classifier": r.classifier,
                "family": r.family,
                "accuracy": r.accuracy,
                "f1_macro": r.f1_macro,
                "f1_weighted": r.f1_weighted,
                "precision_macro": r.precision_macro,
                "recall_macro": r.recall_macro,
                "f1_extended": r.f1_extended,
                "fc_spearman": r.fc_spearman,
                "fc_r2": r.fc_r2,
                "fc_mae": r.fc_mae,
                "key": r.key,
            }
            for r in records
        ]
    )


def filter_records(
    records: list[ExperimentRecord],
    *,
    variants: list[str] | None = None,
    seed: int | None = None,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
) -> list[ExperimentRecord]:
    out = records
    if variants is not None:
        out = [r for r in out if r.variant in variants]
    if seed is not None:
        out = [r for r in out if r.seed == seed]
    if datasets is not None:
        out = [r for r in out if r.file_name in datasets]
    if classifiers is not None:
        out = [r for r in out if r.classifier in classifiers]
    return out


def find_record(records: list[ExperimentRecord], key: str) -> ExperimentRecord | None:
    for r in records:
        if r.key == key:
            return r
    return None


def _show_figure(path: str | Path, caption: str | None = None) -> None:
    """Display a figure file: st.image for raster formats, embedded viewer for PDF."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        b64 = base64.b64encode(p.read_bytes()).decode()
        st.markdown(
            f'<embed src="data:application/pdf;base64,{b64}" '
            'width="100%" height="480" type="application/pdf">',
            unsafe_allow_html=True,
        )
        if caption:
            st.caption(caption)
    else:
        st.image(str(p), caption=caption, width="stretch")


def find_figure(record: ExperimentRecord, relative: str) -> Path | None:
    """Resolve a figure path, falling back across the supported extensions."""
    base = record.root / "figures" / relative
    for candidate in (base, *(base.with_suffix(ext) for ext in FIGURE_EXTENSIONS)):
        if candidate.is_file():
            return candidate
    return None


def render_figure_if_present(
    record: ExperimentRecord, relative: str, caption: str
) -> bool:
    path = find_figure(record, relative)
    if path is None:
        return False
    _show_figure(path, caption)
    return True


def _apply_matrix_layout(
    fig: go.Figure,
    pivot: pd.DataFrame,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str = "Dataset",
) -> None:
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=max(380, 60 + 46 * len(pivot.index)),
        margin=dict(l=120, r=40, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(pivot.index),
        ticktext=[str(v) for v in pivot.index],
    )


def heatmap_fig(
    pivot: pd.DataFrame,
    *,
    title: str,
    metric_label: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    z = pivot.values.astype(float)
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.3f}")(z))
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            hovertemplate="dataset=%{y}<br>classifier=%{x}<br>"
            + metric_label
            + "=%{z:.4f}<extra></extra>",
            colorbar=dict(title=metric_label),
        )
    )
    _apply_matrix_layout(fig, pivot, title=title, xaxis_title="Classifier")
    return fig


def count_heatmap_fig(
    pivot: pd.DataFrame,
    *,
    title: str,
    value_label: str = "clusters",
) -> go.Figure:
    """Integer-valued heatmap (auto colour range), for counts like clusters-per-dataset."""
    z = pivot.values.astype(float)
    text = np.vectorize(lambda v: "" if not np.isfinite(v) else f"{int(round(v)):d}")(z)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertemplate="dataset=%{y}<br>variant=%{x}<br>"
            + value_label
            + "=%{z:.0f}<extra></extra>",
            colorbar=dict(title=value_label),
        )
    )
    _apply_matrix_layout(fig, pivot, title=title, xaxis_title="Variant")
    return fig


def confusion_matrix_fig(
    cm: np.ndarray, labels: list[str], *, title: str = ""
) -> go.Figure:
    is_normalized = cm.dtype.kind == "f" and cm.max() <= 1.0 + 1e-6
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{z:.2f}" if is_normalized else "%{z:d}",
            hovertemplate="true=%{y}<br>pred=%{x}<br>value=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        height=360,
    )
    return fig


def per_class_bar_fig(
    *,
    classes: list[str],
    f1: list[float],
    precision: list[float],
    recall: list[float],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="F1", x=classes, y=f1, marker_color="#1f77b4"))
    fig.add_trace(
        go.Bar(name="Precision", x=classes, y=precision, marker_color="#9ecae1")
    )
    fig.add_trace(go.Bar(name="Recall", x=classes, y=recall, marker_color="#6baed6"))
    fig.update_layout(
        barmode="group",
        height=360,
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(tickangle=-30)
    return fig


def feature_importance_bar(importances: dict, *, top_k: int = 20) -> go.Figure:
    pairs = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#2ca02c"))
    fig.update_layout(
        height=max(280, 22 * len(pairs)),
        xaxis_title="Importance",
        yaxis_title="",
        margin=dict(l=160, r=40, t=20, b=40),
    )
    return fig


def pred_vs_actual_fig(fc: dict, cluster_df: pd.DataFrame) -> go.Figure:
    predicted = fc.get("oof_predicted_rate", {})
    df = cluster_df[["cluster_id", "cluster_class", "failure_rate"]].copy()
    df["predicted"] = df["cluster_id"].astype(str).map(predicted)
    df = df.dropna(subset=["failure_rate", "predicted"])
    fig = px.scatter(
        df,
        x="failure_rate",
        y="predicted",
        color="failure_rate",
        color_continuous_scale="Viridis",
        hover_data=["cluster_id", "cluster_class"],
    )
    lo = float(min(df["failure_rate"].min(), df["predicted"].min())) if len(df) else 0.0
    hi = float(max(df["failure_rate"].max(), df["predicted"].max())) if len(df) else 1.0
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y=x",
            line=dict(dash="dash", color="grey"),
        )
    )
    fig.update_layout(
        height=400,
        xaxis_title="Observed failure rate",
        yaxis_title="Predicted failure rate (OOF)",
        showlegend=False,
    )
    return fig


def complexity_vs_failure_scatter(cluster_df: pd.DataFrame, feature: str) -> go.Figure:
    df = cluster_df.dropna(subset=[feature, "failure_rate"])
    fig = px.scatter(
        df,
        x=feature,
        y="failure_rate",
        color="failure_rate",
        color_continuous_scale="Viridis",
        hover_data=["cluster_id", "cluster_class"],
    )
    fig.update_layout(height=400, xaxis_title=feature, yaxis_title="Failure rate")
    return fig


def failure_rate_strip(cluster_df: pd.DataFrame, label_map: dict) -> go.Figure:
    plot_df = cluster_df[["cluster_id", "cluster_class", "failure_rate"]].copy()
    plot_df["class_name"] = plot_df["cluster_class"].map(
        lambda c: label_map.get(str(int(c)), str(int(c)))
    )
    fig = px.strip(
        plot_df,
        x="class_name",
        y="failure_rate",
        color="class_name",
        hover_data=["cluster_id"],
        stripmode="overlay",
    )
    fig.update_layout(height=380, showlegend=False, yaxis_title="Failure rate")
    fig.update_xaxes(tickangle=-30, title_text="Class")
    return fig


def _wkey(prefix: str, name: str, record: ExperimentRecord) -> str:
    """Unique widget key scoped to a view prefix and record, preventing duplicate-id errors."""
    return f"{prefix}_{name}_{record.key}"


def _cluster_feature_candidates(cdf: pd.DataFrame) -> tuple[list[str], str]:
    """Return (candidates, default) for complexity feature selectors."""
    candidates = [c for c in cdf.columns if c not in CLUSTER_NON_FEATURE_COLS]
    default = (
        "cluster_p5_silhouette"
        if "cluster_p5_silhouette" in candidates
        else (candidates[0] if candidates else "")
    )
    return candidates, default


def _filter_figures_by_category(
    figures: dict[str, str], category: str
) -> dict[str, str]:
    if category == "all":
        return figures
    return {
        rel: abs_
        for rel, abs_ in figures.items()
        if rel.startswith(category + "/") or rel.rsplit(".", 1)[0] == category
    }


def panel_test_performance(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Test performance**")
    if detail.testing is None:
        st.caption("No `outputs/testing/summary.json` for this run.")
        return
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{detail.testing.get('accuracy', float('nan')):.4f}")
    cols[1].metric("F1 macro", f"{detail.testing.get('f1_macro', float('nan')):.4f}")
    cols[2].metric(
        "F1 weighted", f"{detail.testing.get('f1_weighted', float('nan')):.4f}"
    )
    cols[3].metric(
        "Recall macro", f"{detail.testing.get('recall_macro', float('nan')):.4f}"
    )
    st.caption(f"eval mode: `{detail.testing.get('eval_mode', '?')}`")
    render_figure_if_present(record, "testing/f1_per_class.png", "f1_per_class.png")


def panel_confusion_matrix(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Confusion matrix**")
    labels = list((detail.df_meta.get("label_mapping") or {}).values())
    if detail.confusion_matrix is not None:
        if not labels or len(labels) != detail.confusion_matrix.shape[0]:
            labels = [str(i) for i in range(detail.confusion_matrix.shape[0])]
        st.plotly_chart(
            confusion_matrix_fig(detail.confusion_matrix, labels),
            width="stretch",
            key=_wkey(key_prefix, "cm", record),
        )
        return
    if not render_figure_if_present(
        record, "testing/confusion_matrix.png", "confusion_matrix.png"
    ):
        st.caption("No confusion matrix available.")


def panel_failure_classifier(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Failure regressor (RF on cluster complexity)**")
    fc = detail.classifier_results
    if fc is None:
        st.caption("⚠️ Not computed — run `make failure-classify` to generate.")
        return
    if fc.get("skipped"):
        st.warning(
            f"🚫 **Stage skipped** — {fc.get('message', fc.get('reason'))}\n\n"
            f"Clusters used: {fc.get('n_clusters_used', '?')}"
        )
        return
    cols = st.columns(4)
    cols[0].metric("Spearman ρ", f"{fc.get('spearman', float('nan')):.4f}")
    cols[1].metric(
        "R²", f"{fc.get('r2', float('nan')):.4f}", delta=f"± {fc.get('r2_std', 0):.3f}"
    )
    cols[2].metric(
        "MAE",
        f"{fc.get('mae', float('nan')):.4f}",
        delta=f"± {fc.get('mae_std', 0):.3f}",
    )
    cols[3].metric("CV folds", f"{len(fc.get('r2_per_fold', []))}")
    if fc.get("oof_predicted_rate") and detail.cluster_summary is not None:
        st.plotly_chart(
            pred_vs_actual_fig(fc, detail.cluster_summary),
            width="stretch",
            key=_wkey(key_prefix, "pred_vs_actual", record),
        )


def panel_feature_importances(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Feature importances (failure classifier)**")
    fc = detail.classifier_results
    if fc is None:
        st.caption("⚠️ Not computed — run `make failure-classify` to generate.")
        return
    if fc.get("skipped"):
        st.warning(f"🚫 Skipped — {fc.get('message', fc.get('reason'))}")
        return
    if "feature_importances" not in fc:
        st.caption("Malformed `classifier_results.json` (no feature_importances).")
        return
    top_k = st.slider(
        "Top-K features", 5, 50, 20, key=_wkey(key_prefix, "fi_topk", record)
    )
    st.plotly_chart(
        feature_importance_bar(fc["feature_importances"], top_k=top_k),
        width="stretch",
        key=_wkey(key_prefix, "fi_chart", record),
    )


def panel_feature_distribution(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Feature distribution across clusters**")
    cdf = detail.cluster_summary
    if cdf is None or cdf.empty:
        st.caption("No cluster_summary.json for this run.")
        return
    candidates, default = _cluster_feature_candidates(cdf)
    if not candidates:
        st.caption("No complexity features in cluster_summary.")
        return
    feature = st.selectbox(
        "Feature",
        candidates,
        index=candidates.index(default),
        key=_wkey(key_prefix, "fd_feat", record),
    )
    fig = px.violin(cdf.dropna(subset=[feature]), y=feature, box=True, points="all")
    fig.update_layout(height=380, yaxis_title=feature)
    st.plotly_chart(fig, width="stretch", key=_wkey(key_prefix, "fd_chart", record))
    render_figure_if_present(
        record, f"summary/global/{feature}.png", f"summary/global/{feature}"
    )


def panel_complexity_vs_failure(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Complexity feature → failure rate**")
    cdf = detail.cluster_summary
    if cdf is None or cdf.empty or "failure_rate" not in cdf.columns:
        st.caption("Need cluster_summary with `failure_rate`.")
        return
    candidates, default = _cluster_feature_candidates(cdf)
    feature = st.selectbox(
        "X axis",
        candidates,
        index=candidates.index(default),
        key=_wkey(key_prefix, "sc_feat", record),
    )
    st.plotly_chart(
        complexity_vs_failure_scatter(cdf, feature),
        width="stretch",
        key=_wkey(key_prefix, "sc_chart", record),
    )


def panel_failure_rate_distribution(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Failure rate distribution**")
    cdf = detail.cluster_summary
    if cdf is None or cdf.empty or "failure_rate" not in cdf.columns:
        st.caption("No cluster_summary with `failure_rate`.")
        return
    label_map = detail.df_meta.get("label_mapping") or {}
    st.plotly_chart(
        failure_rate_strip(cdf, label_map),
        width="stretch",
        key=_wkey(key_prefix, "fr_chart", record),
    )
    render_figure_if_present(
        record, "summary/failure_rate_strip_box.png", "summary/failure_rate_strip_box"
    )


def panel_per_class_breakdown(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Per-class breakdown**")
    if detail.testing is None:
        st.caption("No `summary.json` for per-class metrics.")
        return
    f1 = detail.testing.get("f1_per_class") or []
    prec = detail.testing.get("precision_per_class") or []
    rec = detail.testing.get("recall_per_class") or []
    label_map = detail.df_meta.get("label_mapping") or {}
    classes = [label_map.get(str(i), str(i)) for i in range(len(f1))]
    st.plotly_chart(
        per_class_bar_fig(classes=classes, f1=f1, precision=prec, recall=rec),
        width="stretch",
        key=_wkey(key_prefix, "pc_chart", record),
    )


def panel_cluster_table(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Cluster table** (sorted by failure rate desc)")
    cdf = detail.cluster_summary
    if cdf is None or cdf.empty:
        st.caption("No cluster_summary.json.")
        return
    label_map = detail.df_meta.get("label_mapping") or {}
    show_cols = ["cluster_id", "cluster_class", "failure_rate", "is_noise_cluster"]
    show_cols += [
        c
        for c in [
            "cluster_p5_silhouette",
            "cluster_frac_at_risk",
            "class_f1_min",
            "class_n1_max",
        ]
        if c in cdf.columns
    ]
    view = cdf[[c for c in show_cols if c in cdf.columns]].copy()
    if "cluster_class" in view.columns:
        view["class_name"] = view["cluster_class"].map(
            lambda c: label_map.get(str(int(c)), str(int(c)))
        )
    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        key=_wkey(key_prefix, "cluster_tbl", record),
    )


def panel_sibling_classifiers(
    record: ExperimentRecord,
    all_records: list[ExperimentRecord],
    key_prefix: str = "drill",
) -> None:
    st.markdown(f"**Other classifiers on {record.file_name} (seed {record.seed})**")
    siblings = [
        r
        for r in all_records
        if r.variant == record.variant
        and r.file_name == record.file_name
        and r.seed == record.seed
    ]
    df = records_to_df(siblings)
    if df.empty:
        st.caption("No sibling classifiers found.")
        return
    view_cols = [
        "classifier",
        "family",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "fc_spearman",
        "fc_r2",
        "fc_mae",
    ]
    view = df[view_cols].sort_values("f1_macro", ascending=False, na_position="last")
    styled = view.style.highlight_max(
        subset=[
            c
            for c in ["accuracy", "f1_macro", "f1_weighted", "fc_spearman", "fc_r2"]
            if c in view.columns
        ],
        color="rgba(46, 160, 67, 0.25)",
    ).format(
        {c: "{:.4f}" for c in view.columns if view[c].dtype.kind == "f"}, na_rep="—"
    )
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        key=_wkey(key_prefix, "siblings_tbl", record),
    )


def panel_explain(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    """Extended-classifier metrics + SHAP figures of the extend variant."""
    st.markdown("**Explain — extended classifier (transductive)**")
    summary = _read_json(record.root / "outputs" / "testing" / "summary_extended.json")
    figures_dir = record.root / "figures" / "explain"
    figures = (
        sorted(
            p for p in figures_dir.rglob("*") if p.suffix.lower() in FIGURE_EXTENSIONS
        )
        if figures_dir.is_dir()
        else []
    )
    if summary is None and not figures:
        st.caption("No extend run for this classifier — run with `EXTEND=1`.")
        return
    if summary:
        cols = st.columns(3)
        ext_f1 = summary.get("f1_macro")
        cols[0].metric(
            "F1 extended (transductive)",
            f"{ext_f1:.4f}" if ext_f1 is not None else "—",
        )
        if record.f1_macro is not None:
            cols[1].metric("F1 base", f"{record.f1_macro:.4f}")
            if ext_f1 is not None:
                cols[2].metric("Δ vs base", f"{ext_f1 - record.f1_macro:+.4f}")
        st.caption(
            "⚠️ The extended F1 is a transductive upper bound: complexity "
            "features are constant per cluster and clusters are class-pure, "
            "so this measures how strongly the geometry encodes the label — "
            "not a generalisation gain over the base classifier."
        )
    for start in range(0, len(figures), 2):
        cols = st.columns(2)
        for col, fig_path in zip(cols, figures[start : start + 2]):
            with col:
                _show_figure(
                    fig_path, caption=fig_path.relative_to(figures_dir).as_posix()
                )


def panel_training_curve(record: ExperimentRecord) -> None:
    st.markdown("**Training curve (DL)**")
    if not render_figure_if_present(
        record, "training/loss_curve.png", "training/loss_curve"
    ):
        st.caption("Training figure not produced for this run.")


def panel_grid_search(record: ExperimentRecord, detail: ExperimentDetail) -> None:
    st.markdown("**Grid search (ML)**")
    if not detail.grid_search:
        st.caption("Grid search not run for this classifier.")
        return
    st.json(detail.grid_search, expanded=False)


def _render_hypothesis_scoreboard(
    records: list[ExperimentRecord],
    selected_variants: list[str],
    seed: int,
) -> None:
    """Aggregate Spearman ρ across all selected experiments as a hypothesis check."""
    rs = filter_records(records, variants=selected_variants, seed=seed)
    rho_vals = [r.fc_spearman for r in rs if r.fc_spearman is not None]
    n_skipped = sum(1 for r in rs if r.fc_skipped)

    st.subheader("Hypothesis validation — failure-rate predictability")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments with ρ", f"{len(rho_vals)} / {len(rs)}")
    c2.metric("Mean ρ", f"{float(np.mean(rho_vals)):.3f}" if rho_vals else "—")
    pct = sum(v >= 0.6 for v in rho_vals) / len(rho_vals) if rho_vals else None
    c3.metric("ρ ≥ 0.6", f"{pct:.0%}" if pct is not None else "—")
    c4.metric("FC skipped", n_skipped)

    if rho_vals:
        hist_fig = px.histogram(
            x=rho_vals,
            nbins=20,
            labels={"x": "Spearman ρ"},
            title="Distribution of Spearman ρ across experiments",
        )
        hist_fig.add_vline(
            x=0.6, line_dash="dash", line_color="green", annotation_text="ρ = 0.6"
        )
        hist_fig.update_layout(
            height=220, margin=dict(t=40, b=30, l=40, r=20), showlegend=False
        )
        st.plotly_chart(hist_fig, width="stretch", key="hyp_rho_hist")
    st.divider()


def render_overview(
    records: list[ExperimentRecord],
    selected_variants: list[str],
    seed: int,
    metric: str,
) -> None:
    if not selected_variants:
        st.info("Pick at least one variant in the sidebar.")
        return

    _render_hypothesis_scoreboard(records, selected_variants, seed)

    size_by_ds: dict[str, int] = {}
    for r in filter_records(records, seed=seed):
        if r.file_name not in size_by_ds:
            size_by_ds[r.file_name] = _dataset_test_size(str(r.shared))
    ordered_datasets = sorted(size_by_ds, key=lambda ds: size_by_ds[ds], reverse=True)

    cluster_rows = [
        {
            "dataset": r.file_name,
            "variant": variant,
            "n_clusters": count_clusters(str(r.shared)),
        }
        for variant in selected_variants
        for r in filter_records(records, variants=[variant], seed=seed)
    ]
    if cluster_rows:
        cdf = pd.DataFrame(cluster_rows).drop_duplicates(subset=["dataset", "variant"])
        present = [ds for ds in ordered_datasets if ds in cdf["dataset"].values]
        cluster_pivot = cdf.pivot_table(
            index="dataset", columns="variant", values="n_clusters", aggfunc="first"
        ).reindex(index=present, columns=selected_variants)
        st.subheader("Clusters per dataset")
        st.plotly_chart(
            count_heatmap_fig(cluster_pivot, title="Total clusters per dataset"),
            width="stretch",
            key="cluster_count_heatmap",
        )
        st.divider()

    metric_label = HEATMAP_METRICS[metric]
    zmin, zmax = METRIC_COLORSCALE_BOUNDS.get(metric, (0.0, 1.0))
    for variant in selected_variants:
        rs = filter_records(records, variants=[variant], seed=seed)
        if not rs:
            st.warning(f"No records for variant `{variant}` at seed {seed}.")
            continue

        df = records_to_df(rs)
        present_ds = [ds for ds in ordered_datasets if ds in df["dataset"].values]
        all_classifiers = sorted(df["classifier"].unique())
        pivot = df.pivot_table(
            index="dataset", columns="classifier", values=metric, aggfunc="first"
        )
        pivot = pivot.reindex(index=present_ds, columns=all_classifiers)

        st.subheader(f"Variant: {variant}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Datasets", len(present_ds))
        c2.metric("Classifiers", len(all_classifiers))
        c3.metric("Empty cells", int(pivot.isna().sum().sum()))

        event = st.plotly_chart(
            heatmap_fig(
                pivot, title=metric_label, metric_label=metric, zmin=zmin, zmax=zmax
            ),
            width="stretch",
            key=f"heatmap_{variant}",
            on_select="rerun",
            selection_mode="points",
        )

        clicked = _heatmap_click_target(event, pivot, variant, rs)
        if clicked is not None:
            st.session_state["drill_target"] = clicked
            st.toast(
                f"Drill-down armed → {clicked['variant']} · {clicked['dataset']} · {clicked['classifier']}"
            )

        with st.expander("Show data", expanded=False):
            st.dataframe(
                pivot.style.background_gradient(cmap="viridis", vmin=0, vmax=1).format(
                    "{:.4f}", na_rep="—"
                ),
                width="stretch",
            )


def _heatmap_click_target(
    event: Any,
    pivot: pd.DataFrame,
    variant: str,
    records: list[ExperimentRecord],
) -> dict | None:
    """Extract (variant, dataset, classifier) from a plotly_chart selection event."""
    try:
        selection = event.selection  # type: ignore[attr-defined]
        points = selection.get("points") if isinstance(selection, dict) else None
        if not points:
            return None
        p = points[0]
        dataset = (
            pivot.index[int(p["y"])]
            if isinstance(p.get("y"), (int, np.integer))
            else p.get("y")
        )
        classifier = (
            pivot.columns[int(p["x"])]
            if isinstance(p.get("x"), (int, np.integer))
            else p.get("x")
        )
        if dataset is None or classifier is None:
            return None
        match = next(
            (
                r
                for r in records
                if r.file_name == dataset and r.classifier == classifier
            ),
            None,
        )
        if match is None:
            return None
        return {
            "variant": variant,
            "dataset": match.file_name,
            "classifier": classifier,
            "key": match.key,
        }
    except (AttributeError, KeyError, IndexError, ValueError, TypeError):
        return None


def render_drilldown(records: list[ExperimentRecord], seed: int) -> None:
    if not records:
        st.info("No records to drill into.")
        return

    target = st.session_state.get("drill_target")
    variants = sorted({r.variant for r in records})
    pre_variant = (
        target["variant"] if target and target["variant"] in variants else variants[0]
    )

    col_v, col_d, col_c = st.columns(3)
    variant = col_v.selectbox(
        "Variant", variants, index=variants.index(pre_variant), key="dd_variant"
    )

    rs_v = filter_records(records, variants=[variant], seed=seed)
    datasets = sorted({r.file_name for r in rs_v})
    if not datasets:
        st.warning("No datasets for this variant at the selected seed.")
        return
    pre_dataset = (
        target["dataset"]
        if target and target.get("dataset") in datasets
        else datasets[0]
    )
    dataset = col_d.selectbox(
        "Dataset", datasets, index=datasets.index(pre_dataset), key="dd_dataset"
    )

    rs_vd = filter_records(rs_v, datasets=[dataset])
    classifiers = sorted({r.classifier for r in rs_vd})
    if not classifiers:
        st.warning("No classifier for this (variant, dataset, seed).")
        return
    pre_clf = (
        target["classifier"]
        if target and target.get("classifier") in classifiers
        else classifiers[0]
    )
    classifier = col_c.selectbox(
        "Classifier", classifiers, index=classifiers.index(pre_clf), key="dd_classifier"
    )

    record = next(r for r in rs_vd if r.classifier == classifier)
    detail = load_experiment_detail(str(record.root), str(record.shared))

    st.caption(f"`{record.root}` · family `{record.family}` · seed `{record.seed}`")

    row_pairs = [
        (panel_test_performance, panel_confusion_matrix),
        (panel_failure_classifier, panel_feature_importances),
        (panel_complexity_vs_failure, panel_failure_rate_distribution),
        (panel_per_class_breakdown, panel_feature_distribution),
    ]
    for left_panel, right_panel in row_pairs:
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                left_panel(record, detail)
        with right:
            with st.container(border=True):
                right_panel(record, detail)

    st.divider()
    with st.container(border=True):
        panel_sibling_classifiers(record, records)

    st.divider()
    with st.container(border=True):
        if record.family == "dl":
            panel_training_curve(record)
        else:
            panel_grid_search(record, detail)

    st.divider()
    with st.container(border=True):
        panel_explain(record, detail)

    st.divider()
    with st.container(border=True):
        panel_cluster_table(record, detail)


def render_side_by_side(records: list[ExperimentRecord], seed: int) -> None:
    rs = [r for r in records if r.seed == seed]
    if not rs:
        st.info("No records at the selected seed.")
        return
    options = {r.key: r.label for r in rs}
    selected_keys = st.multiselect(
        "Pick 2–4 experiments to compare",
        options=list(options.keys()),
        format_func=lambda k: options[k],
        max_selections=4,
        key="sbs_keys",
    )
    if len(selected_keys) < 2:
        st.caption("Select at least two experiments to start comparing.")
        return

    selected = [
        r for r in (find_record(records, k) for k in selected_keys) if r is not None
    ]
    details = [load_experiment_detail(str(r.root), str(r.shared)) for r in selected]

    cols = st.columns(len(selected))
    for i, (col, record, detail) in enumerate(zip(cols, selected, details)):
        prefix = f"sbs{i}"
        with col:
            st.markdown(f"### {record.label}")
            st.caption(f"family `{record.family}` · seed `{record.seed}`")
            panel_test_performance(record, detail, key_prefix=prefix)
            panel_confusion_matrix(record, detail, key_prefix=prefix)
            panel_failure_classifier(record, detail, key_prefix=prefix)
            panel_per_class_breakdown(record, detail, key_prefix=prefix)
            st.divider()
            panel_cluster_table(record, detail, key_prefix=prefix)


def render_gallery(records: list[ExperimentRecord], seed: int) -> None:
    rs = [r for r in records if r.seed == seed]
    if not rs:
        st.info("No records at the selected seed.")
        return

    col_v, col_mode = st.columns([1, 2])
    variants = sorted({r.variant for r in rs})
    variant = col_v.selectbox("Variant", variants, key="gal_variant")
    rs_v = [r for r in rs if r.variant == variant]

    mode = col_mode.radio(
        "Mode",
        ["Single experiment", "Cross-experiment"],
        horizontal=True,
        key="gal_mode",
    )
    if mode == "Single experiment":
        _render_gallery_single(rs_v)
    else:
        _render_gallery_cross(rs_v)


def _render_gallery_single(rs_v: list[ExperimentRecord]) -> None:
    col_d, col_c, col_cat = st.columns([1, 1, 1])
    datasets = sorted({r.file_name for r in rs_v})
    dataset = col_d.selectbox("Dataset", datasets, key="gal_dataset")
    rs_vd = [r for r in rs_v if r.file_name == dataset]
    classifiers = sorted({r.classifier for r in rs_vd})
    if not classifiers:
        st.warning("No classifier outputs for this selection.")
        return
    classifier = col_c.selectbox("Classifier", classifiers, key="gal_classifier")
    category = col_cat.selectbox(
        "Category", ["all", *GALLERY_CATEGORIES], key="gal_category"
    )

    record = next(r for r in rs_vd if r.classifier == classifier)
    all_figures = load_figure_index(str(record.root))
    if not all_figures:
        st.info(f"No figures under `{record.root / 'figures'}`.")
        return
    figures = _filter_figures_by_category(all_figures, category)
    if not figures:
        st.info(f"No figures matching category `{category}`.")
        return

    st.caption(f"{len(figures)} figure(s) in `{record.root.name}/figures/`")
    rels = sorted(figures.keys())
    for i in range(0, len(rels), 3):
        cols = st.columns(3)
        for col, rel in zip(cols, rels[i : i + 3]):
            with col:
                _show_figure(figures[rel], caption=rel)


def _render_gallery_cross(rs_v: list[ExperimentRecord]) -> None:
    all_datasets = sorted({r.file_name for r in rs_v})
    all_classifiers = sorted({r.classifier for r in rs_v})

    col_d, col_c = st.columns([1, 1])
    sel_datasets = col_d.multiselect(
        "Datasets", all_datasets, default=all_datasets, key="gal_x_datasets"
    )
    sel_classifiers = col_c.multiselect(
        "Classifiers", all_classifiers, default=all_classifiers, key="gal_x_classifiers"
    )

    if not sel_datasets or not sel_classifiers:
        st.warning("Select at least one dataset and one classifier.")
        return

    rs_filtered = [
        r
        for r in rs_v
        if r.file_name in sel_datasets and r.classifier in sel_classifiers
    ]
    if not rs_filtered:
        st.warning("No experiments match the current selection.")
        return

    fig_index: dict[str, dict[str, str]] = {}
    for r in rs_filtered:
        for rel, abs_ in load_figure_index(str(r.root)).items():
            fig_index.setdefault(rel, {})[f"{r.file_name}|{r.classifier}"] = abs_

    if not fig_index:
        st.info("No figures found in the selected experiments.")
        return

    col_fig, col_cat = st.columns([2, 1])
    category = col_cat.selectbox(
        "Category", ["all", *GALLERY_CATEGORIES], key="gal_x_category"
    )
    rels_all = sorted(
        rel
        for rel in fig_index
        if category == "all"
        or rel.startswith(category + "/")
        or rel.rsplit(".", 1)[0] == category
    )
    if not rels_all:
        st.info(f"No figures matching category `{category}`.")
        return

    figure_path = col_fig.selectbox("Figure", rels_all, key="gal_x_figure")
    st.markdown(f"**`{figure_path}`**")
    cell_map = fig_index.get(figure_path, {})

    header_cols = st.columns([1] + [2] * len(sel_classifiers))
    header_cols[0].markdown("**Dataset \\ Classifier**")
    for i, clf in enumerate(sel_classifiers):
        header_cols[i + 1].markdown(f"**{clf}**")

    for ds in sel_datasets:
        row_cols = st.columns([1] + [2] * len(sel_classifiers))
        row_cols[0].markdown(f"`{ds}`")
        for i, clf in enumerate(sel_classifiers):
            abs_path = cell_map.get(f"{ds}|{clf}")
            with row_cols[i + 1]:
                if abs_path:
                    _show_figure(abs_path)
                else:
                    st.caption("—")


def render_sidebar(
    records: list[ExperimentRecord], n_skipped: int
) -> tuple[str, list[str], int, str]:
    st.sidebar.title("Filters")

    root = st.sidebar.text_input("Experiments root", value=EXPERIMENTS_ROOT_DEFAULT)

    variants = sorted({r.variant for r in records})
    selected_variants = st.sidebar.multiselect(
        "Variants", variants, default=variants, key="sb_variants"
    )

    seeds = sorted({r.seed for r in records if r.variant in selected_variants})
    default_seed = 42 if 42 in seeds else (seeds[0] if seeds else 42)
    if seeds:
        seed = st.sidebar.selectbox(
            "Seed", seeds, index=seeds.index(default_seed), key="sb_seed"
        )
    else:
        seed = default_seed
        st.sidebar.warning("No seeds found.")

    metric = st.sidebar.selectbox(
        "Heatmap metric",
        list(HEATMAP_METRICS.keys()),
        format_func=lambda k: HEATMAP_METRICS[k],
        index=0,
        key="sb_metric",
    )

    st.sidebar.divider()
    if st.sidebar.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    with st.sidebar.expander("Debug", expanded=False):
        st.write(f"Valid records: **{len(records)}**")
        st.write(f"Legacy folders skipped: **{n_skipped}**")
        st.write(
            f"Variants: {len(variants)} · Datasets: {len({r.file_name for r in records})}"
        )

    return root, selected_variants, seed, metric


def main() -> None:
    st.set_page_config(page_title="Intrusion Forge — Experiments", layout="wide")
    st.title("Experiment Dashboard")

    initial_root = EXPERIMENTS_ROOT_DEFAULT
    records, n_skipped = discover_experiments(initial_root)

    root, selected_variants, seed, metric = render_sidebar(records, n_skipped)

    if root != initial_root:
        records, n_skipped = discover_experiments(root)

    if not records:
        st.error(
            "No valid experiments found. The dashboard probes for "
            "`shared/metadata/df_meta.json` under each `<variant>/<dataset>_<seed>/`. "
            f"Scanned root: `{root}`"
        )
        st.stop()

    tab_overview, tab_drill, tab_sbs, tab_gallery = st.tabs(
        ["Overview", "Drill-Down", "Side-by-Side", "Gallery"]
    )

    with tab_overview:
        render_overview(records, selected_variants, seed, metric)
    with tab_drill:
        render_drilldown(filter_records(records, variants=selected_variants), seed)
    with tab_sbs:
        render_side_by_side(filter_records(records, variants=selected_variants), seed)
    with tab_gallery:
        render_gallery(filter_records(records, variants=selected_variants), seed)


main()
