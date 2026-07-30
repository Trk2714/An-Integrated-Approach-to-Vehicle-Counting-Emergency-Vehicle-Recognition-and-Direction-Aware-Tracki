from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_METRIC_FILES = {
    "Vehicle Detector (YOLOv8)": PROJECT_ROOT / "models" / "vehicle_detector_metrics.json",
    "Emergency Visual Detector (YOLOv8)": PROJECT_ROOT / "models" / "emergency_visual_metrics.json",
    "Siren Audio Classifier (ResNet18)": PROJECT_ROOT / "models" / "audio_siren_metrics.json",
}

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Manrope", "Trebuchet MS", sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #1f2a44 0%, #101522 45%, #0b0f17 100%);
    color: #f4f6fb;
}

.block-container {
    padding-top: 2rem;
}

.panel {
    background: linear-gradient(145deg, rgba(21, 31, 52, 0.92), rgba(11, 18, 31, 0.92));
    border: 1px solid rgba(100, 129, 178, 0.35);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}

.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.28rem 0.8rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.low { background: rgba(35, 181, 126, 0.18); color: #6af0be; border: 1px solid #2ea979; }
.medium { background: rgba(236, 184, 82, 0.16); color: #ffd37f; border: 1px solid #d4a24b; }
.high { background: rgba(239, 88, 88, 0.16); color: #ff9a9a; border: 1px solid #d25d5d; }
.ok { background: rgba(84, 175, 255, 0.16); color: #8dccff; border: 1px solid #4d9de0; }

.alert-on {
    background: linear-gradient(90deg, #ff5c5c, #ff964b);
    color: #fff;
    padding: 0.5rem 0.85rem;
    border-radius: 12px;
    font-weight: 800;
    animation: pulse 1.4s ease-in-out infinite;
}
.alert-off {
    background: rgba(74, 103, 152, 0.26);
    color: #d2def2;
    padding: 0.5rem 0.85rem;
    border-radius: 12px;
    font-weight: 700;
}
@keyframes pulse {
  0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 90, 90, 0.5); }
  50% { transform: scale(1.03); box-shadow: 0 0 0 10px rgba(255, 90, 90, 0.0); }
}
</style>
""",
    unsafe_allow_html=True,
)


def api_get(backend_url: str, path: str, timeout: int = 10) -> tuple[dict[str, Any] | None, str | None]:
    try:
        resp = requests.get(f"{backend_url.rstrip('/')}{path}", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if not resp.ok:
        return None, resp.text
    try:
        return resp.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Invalid JSON response: {exc}"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def to_backend_video_url(backend_url: str, video_ref: str | None, run_id: int | None) -> str | None:
    if video_ref:
        if video_ref.startswith("http://") or video_ref.startswith("https://"):
            return video_ref
        if video_ref.startswith("/"):
            return f"{backend_url.rstrip('/')}{video_ref}"
    if run_id is not None:
        return f"{backend_url.rstrip('/')}/runs/{run_id}/video"
    return None


def render_summary(summary: dict[str, Any]) -> None:
    congestion = summary["congestion"]["level"]
    emergency = summary["emergency"]["confirmed"]
    direction_counts = summary["direction_counts"]

    cols = st.columns(4)
    cols[0].metric("N->S", direction_counts["N->S"])
    cols[1].metric("S->N", direction_counts["S->N"])
    cols[2].metric("E->W", direction_counts["E->W"])
    cols[3].metric("W->E", direction_counts["W->E"])

    st.markdown(
        f'<div class="badge {congestion}">Congestion: {congestion.upper()}</div>',
        unsafe_allow_html=True,
    )
    if emergency:
        st.markdown('<div class="alert-on">Emergency Confirmed (Visual + Siren)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-off">No Confirmed Emergency</div>', unsafe_allow_html=True)

    st.write("Signal Priority:", summary["signal_priority"])
    st.json(summary)


st.title("Integrated Smart Traffic Management Dashboard")
st.caption("Vehicle counting, direction-aware tracking, emergency confirmation, congestion analysis, and signal priority.")

default_backend = "http://127.0.0.1:8000"
backend_url = st.text_input("Backend URL", value=st.session_state.get("backend_url", default_backend))
st.session_state["backend_url"] = backend_url

health_data, _health_err = api_get(backend_url, "/health", timeout=5)
if health_data:
    st.markdown('<div class="badge ok">Backend Online</div>', unsafe_allow_html=True)
    st.caption(
        "Inference devices - requested: "
        f"`{health_data.get('inference_device_requested', 'auto')}`, "
        f"video: `{health_data.get('video_inference_device', 'unknown')}`, "
        f"audio: `{health_data.get('audio_inference_device', 'unknown')}`"
    )
else:
    st.markdown('<div class="badge high">Backend Offline</div>', unsafe_allow_html=True)

inference_tab, stats_tab = st.tabs(["Inference", "Stats & Visualizations"])

with inference_tab:
    left, right = st.columns([1.05, 1.45], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Run Inference")
        video_file = st.file_uploader(
            "Traffic video with embedded audio",
            type=["mp4", "mkv", "avi", "mov", "webm"],
            key="video",
            help="Upload a single traffic video file. The backend extracts its audio track automatically if present.",
        )
        run_clicked = st.button("Start Video Inference", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Latest Summary")
        latest_payload = st.session_state.get("latest_inference_payload")
        if latest_payload and "summary" in latest_payload:
            render_summary(latest_payload["summary"])
        else:
            st.info("Run a video inference to view live summary.")
        st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked:
        if video_file is None:
            st.error("Please upload a traffic video.")
        else:
            with st.spinner("Processing video on backend..."):
                files = {"video": (video_file.name, video_file.getvalue(), video_file.type or "video/mp4")}
                try:
                    resp = requests.post(f"{backend_url.rstrip('/')}/infer/video", files=files, timeout=7200)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Backend request failed: {exc}")
                else:
                    if not resp.ok:
                        st.error(f"Inference failed: {resp.text}")
                    else:
                        payload = resp.json()
                        st.session_state["latest_inference_payload"] = payload
                        st.success(f"Inference complete. Run ID: {payload.get('run_id')}")

    latest_payload = st.session_state.get("latest_inference_payload")
    if latest_payload and "summary" in latest_payload:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Processed Video")
        summary = latest_payload["summary"]
        run_id = latest_payload.get("run_id")
        video_ref = summary.get("output_video_url") or latest_payload.get("output_video_url")
        video_url = to_backend_video_url(backend_url, video_ref, run_id)
        if video_url:
            st.video(video_url)
            st.caption(f"Streaming from `{video_url}`")
        else:
            output_path = Path(summary.get("output_video", ""))
            if output_path.exists():
                st.video(output_path.read_bytes())
            else:
                st.warning("Processed video path is unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recent Runs")
    runs_payload, runs_err = api_get(backend_url, "/vehicle-counts?limit=10")
    alerts_payload, alerts_err = api_get(backend_url, "/emergency-alerts?limit=10")

    if runs_payload:
        runs = runs_payload.get("runs", [])
        if runs:
            run_rows = []
            for run in runs:
                run_rows.append(
                    {
                        "run_id": run["id"],
                        "created_at": run["created_at"],
                        "congestion": run["congestion"]["level"],
                        "signal_priority": run["signal_priority"],
                        "N->S": run["direction_counts"]["N->S"],
                        "S->N": run["direction_counts"]["S->N"],
                        "E->W": run["direction_counts"]["E->W"],
                        "W->E": run["direction_counts"]["W->E"],
                    }
                )
            st.dataframe(pd.DataFrame(run_rows), use_container_width=True)
        else:
            st.info("No runs yet.")
    elif runs_err:
        st.warning(f"Could not fetch runs: {runs_err}")

    if alerts_payload:
        alerts = alerts_payload.get("alerts", [])
        if alerts:
            st.write("Emergency Alerts")
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    elif alerts_err:
        st.warning(f"Could not fetch alerts: {alerts_err}")
    st.markdown("</div>", unsafe_allow_html=True)

with stats_tab:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Achieved Metrics")

    model_tables = []
    for model_name, metrics_path in MODEL_METRIC_FILES.items():
        metrics = load_json(metrics_path)
        if not metrics:
            continue

        if "val_metrics" in metrics:
            val = metrics.get("val_metrics", {})
            test = metrics.get("test_metrics", {})
            model_tables.append(
                {
                    "model": model_name,
                    "val_precision": round(float(val.get("metrics/precision(B)", 0.0)), 4),
                    "val_recall": round(float(val.get("metrics/recall(B)", 0.0)), 4),
                    "val_mAP50": round(float(val.get("metrics/mAP50(B)", 0.0)), 4),
                    "val_mAP50_95": round(float(val.get("metrics/mAP50-95(B)", 0.0)), 4),
                    "test_precision": round(float(test.get("metrics/precision(B)", 0.0)), 4),
                    "test_recall": round(float(test.get("metrics/recall(B)", 0.0)), 4),
                    "test_mAP50": round(float(test.get("metrics/mAP50(B)", 0.0)), 4),
                    "test_mAP50_95": round(float(test.get("metrics/mAP50-95(B)", 0.0)), 4),
                }
            )

        elif model_name == "Siren Audio Classifier (ResNet18)" and "test_metrics" in metrics:
            test = metrics["test_metrics"]
            model_tables.append(
                {
                    "model": model_name,
                    "test_accuracy": round(float(test.get("accuracy", 0.0)), 4),
                    "test_precision": round(float(test.get("precision", 0.0)), 4),
                    "test_recall": round(float(test.get("recall", 0.0)), 4),
                    "test_f1": round(float(test.get("f1", 0.0)), 4),
                    "test_loss": round(float(test.get("loss", 0.0)), 4),
                }
            )

    if model_tables:
        st.dataframe(pd.DataFrame(model_tables), use_container_width=True)
    else:
        st.info("Model metrics are not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Run Analytics")
    runs_payload, runs_err = api_get(backend_url, "/vehicle-counts?limit=200")
    alerts_payload, alerts_err = api_get(backend_url, "/emergency-alerts?limit=200")

    if not runs_payload:
        if runs_err:
            st.warning(f"Could not load run analytics: {runs_err}")
        else:
            st.info("No run data available.")
    else:
        runs = runs_payload.get("runs", [])
        if not runs:
            st.info("No run data available.")
        else:
            rows: list[dict[str, Any]] = []
            direction_total = {"N->S": 0, "S->N": 0, "E->W": 0, "W->E": 0}
            class_total: dict[str, int] = {}
            for run in runs:
                direction_counts = run["direction_counts"]
                class_counts = run["class_counts"]
                for k in direction_total:
                    direction_total[k] += int(direction_counts.get(k, 0))
                for cls_name, cls_count in class_counts.items():
                    class_total[cls_name] = class_total.get(cls_name, 0) + int(cls_count)

                rows.append(
                    {
                        "run_id": run["id"],
                        "created_at": run["created_at"],
                        "congestion_level": run["congestion"]["level"],
                        "avg_vehicles_per_frame": float(run["congestion"]["average_vehicles_per_frame"]),
                        "peak_vehicles_in_frame": int(run["congestion"]["peak_vehicles_in_frame"]),
                        "signal_priority": run["signal_priority"],
                        "N->S": int(direction_counts.get("N->S", 0)),
                        "S->N": int(direction_counts.get("S->N", 0)),
                        "E->W": int(direction_counts.get("E->W", 0)),
                        "W->E": int(direction_counts.get("W->E", 0)),
                    }
                )

            run_df = pd.DataFrame(rows)
            run_df["created_at"] = pd.to_datetime(run_df["created_at"])
            run_df = run_df.sort_values("created_at")
            run_df["congestion_score"] = run_df["congestion_level"].map({"low": 1, "medium": 2, "high": 3})

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Inference Runs", len(run_df))
            c2.metric("Average Peak Vehicles", f"{run_df['peak_vehicles_in_frame'].mean():.2f}")
            c3.metric("Average Vehicles/Frame", f"{run_df['avg_vehicles_per_frame'].mean():.2f}")

            st.write("Congestion Trend")
            st.line_chart(
                run_df.set_index("created_at")[["congestion_score", "peak_vehicles_in_frame"]],
                use_container_width=True,
            )

            direction_df = pd.DataFrame(
                {"direction": list(direction_total.keys()), "count": list(direction_total.values())}
            ).set_index("direction")
            st.write("Direction Totals")
            st.bar_chart(direction_df, use_container_width=True)

            congestion_dist = (
                run_df["congestion_level"]
                .value_counts()
                .reindex(["low", "medium", "high"], fill_value=0)
                .rename_axis("level")
                .to_frame("count")
            )
            st.write("Congestion Distribution")
            st.bar_chart(congestion_dist, use_container_width=True)

            if class_total:
                class_df = (
                    pd.DataFrame({"class": list(class_total.keys()), "count": list(class_total.values())})
                    .sort_values("count", ascending=False)
                    .set_index("class")
                )
                st.write("Tracked Vehicle Classes")
                st.bar_chart(class_df, use_container_width=True)

    if alerts_payload:
        alerts = alerts_payload.get("alerts", [])
        if alerts:
            alert_df = pd.DataFrame(alerts)
            confirmed_ratio = float(alert_df["confirmed"].mean()) if "confirmed" in alert_df else 0.0
            st.metric("Emergency Confirmation Rate", f"{(confirmed_ratio * 100):.2f}%")

            if "siren_probability" in alert_df:
                prob_df = alert_df[["created_at", "siren_probability"]].copy()
                prob_df["created_at"] = pd.to_datetime(prob_df["created_at"])
                prob_df = prob_df.sort_values("created_at").set_index("created_at")
                st.write("Siren Probability Over Time")
                st.line_chart(prob_df, use_container_width=True)
    elif alerts_err:
        st.warning(f"Could not load emergency alerts: {alerts_err}")

    audio_metrics = load_json(MODEL_METRIC_FILES["Siren Audio Classifier (ResNet18)"])
    history = (audio_metrics or {}).get("history", [])
    if history:
        history_df = pd.DataFrame(history).set_index("epoch")
        chart_cols = [c for c in ["train_accuracy", "val_accuracy", "train_f1", "val_f1"] if c in history_df.columns]
        if chart_cols:
            st.write("Audio Training Curves")
            st.line_chart(history_df[chart_cols], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
