"""Streamlit dashboard for Student Pass/Fail Prediction."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import predict_student
from utils.preprocessing import FEATURE_COLUMNS, load_dataset
from utils.ml_insights import (
    create_student_report_pdf,
    feature_based_explanation,
    get_feature_importance,
    get_roc_data,
    load_trained_model,
    model_prediction,
    predict_batch,
    recommendations_for_inputs,
    score_dataset,
)

DATA_PATH = Path("data/students.csv")
METRICS_PATH = Path("reports/model_metrics.json")
CONFUSION_MATRIX_PATH = Path("reports/confusion_matrix.png")
BEST_MODEL_PATH = Path("models/best_model.pkl")
LOGISTIC_MODEL_PATH = Path("models/logistic_regression.pkl")
TREE_MODEL_PATH = Path("models/decision_tree.pkl")
PASS_COLOR = "#22c55e"
FAIL_COLOR = "#ff4b5c"
ACCENT_COLOR = "#38bdf8"
PLOT_TEMPLATE = "plotly_dark"

st.set_page_config(
    page_title="Student Pass/Fail Prediction System",
    page_icon="SP",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Hallmark · pre-emit critique: P4 H4 E4 S4 R4 V4 */
    :root {
        --color-bg: #090d16;
        --color-panel: #111827;
        --color-panel-2: #172033;
        --color-line: rgba(148, 163, 184, 0.22);
        --color-text: #f8fafc;
        --color-muted: #a8b3c7;
        --color-accent: #38bdf8;
        --color-accent-2: #ff4b5c;
        --color-pass: #22c55e;
        --color-fail: #ff4b5c;
        --color-warn: #f59e0b;
        --font-display: "Segoe UI", sans-serif;
        --font-body: "Segoe UI", sans-serif;
        --shadow-soft: 0 18px 50px rgba(0, 0, 0, 0.26);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 30rem),
            linear-gradient(135deg, #090d16 0%, #0c1220 42%, #111827 100%);
        color: var(--color-text);
        font-family: var(--font-body);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }
    h1, h2, h3 {
        font-family: var(--font-display);
        letter-spacing: 0;
        color: var(--color-text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid var(--color-line);
        font-weight: 650;
    }
    [data-testid="stSidebar"] * {
        font-family: var(--font-body);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        font-weight: 750;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-weight: 850;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 8px;
        padding: 0.28rem 0.45rem;
        margin-bottom: 0.2rem;
        font-weight: 850;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(56, 189, 248, 0.09);
    }
    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        border-radius: 8px;
        border-color: var(--color-line);
        background-color: rgba(17, 24, 39, 0.88);
    }
    .stButton button,
    .stDownloadButton button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 8px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        background: linear-gradient(135deg, var(--color-accent-2), var(--color-accent));
        color: var(--color-text);
        font-weight: 700;
        min-height: 2.8rem;
        box-shadow: 0 12px 28px rgba(56, 189, 248, 0.15);
    }
    .stButton button:hover,
    .stDownloadButton button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        border-color: rgba(248, 250, 252, 0.62);
        filter: brightness(1.05);
    }
    .nav-footer {
        color: var(--color-muted);
        font-size: 0.78rem;
        line-height: 1.45;
        border-top: 1px solid var(--color-line);
        margin-top: 1.2rem;
        padding-top: 0.9rem;
        font-weight: 750;
    }
    .nav-footer strong {
        color: var(--color-text);
        font-weight: 950;
    }
    .brand-card {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.72);
        margin: 0.75rem 0 1rem;
    }
    .brand-mark {
        width: 2.4rem;
        height: 2.4rem;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: linear-gradient(135deg, var(--color-accent-2), var(--color-accent));
        font-weight: 900;
        color: var(--color-text);
        margin-bottom: 0.7rem;
    }
    .brand-title {
        font-size: 1.08rem;
        font-weight: 950;
        color: var(--color-text);
        line-height: 1.2;
    }
    .brand-subtitle {
        color: var(--color-muted);
        font-size: 0.82rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }
    .page-hero {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        padding: 1.6rem 1.7rem;
        margin-bottom: 1.35rem;
        background:
            linear-gradient(135deg, rgba(255, 75, 92, 0.12), rgba(56, 189, 248, 0.10)),
            rgba(17, 24, 39, 0.72);
        box-shadow: var(--shadow-soft);
    }
    .eyebrow {
        color: var(--color-accent);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.76rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }
    .page-title {
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 900;
        line-height: 1.05;
        color: var(--color-text);
        overflow-wrap: anywhere;
    }
    .page-subtitle {
        color: var(--color-muted);
        font-size: 1rem;
        margin-top: 0.7rem;
        max-width: 62rem;
    }
    .metric-card {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        padding: 1.05rem;
        background: linear-gradient(180deg, rgba(23, 32, 51, 0.92), rgba(17, 24, 39, 0.92));
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.20);
        min-height: 7.1rem;
    }
    .metric-label {
        font-size: 0.82rem;
        color: var(--color-muted);
        margin-bottom: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 900;
        color: var(--color-text);
        line-height: 1;
    }
    .metric-note {
        color: var(--color-muted);
        font-size: 0.8rem;
        margin-top: 0.55rem;
    }
    .panel {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        padding: 1.15rem;
        background: rgba(17, 24, 39, 0.70);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        margin-bottom: 1rem;
    }
    .glass-card {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        padding: 1rem;
        background: linear-gradient(145deg, rgba(23, 32, 51, 0.72), rgba(9, 13, 22, 0.56));
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.18);
        backdrop-filter: blur(12px);
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 850;
        margin: 0 0 0.8rem;
        color: var(--color-text);
    }
    .pass-card {
        border: 1px solid rgba(34, 197, 94, 0.38);
        border-left: 6px solid var(--color-pass);
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.16), rgba(17, 24, 39, 0.90));
        padding: 1.25rem;
        border-radius: 8px;
        box-shadow: 0 16px 38px rgba(34, 197, 94, 0.11);
    }
    .fail-card {
        border: 1px solid rgba(255, 75, 92, 0.40);
        border-left: 6px solid var(--color-fail);
        background: linear-gradient(135deg, rgba(255, 75, 92, 0.17), rgba(17, 24, 39, 0.90));
        padding: 1.25rem;
        border-radius: 8px;
        box-shadow: 0 16px 38px rgba(255, 75, 92, 0.10);
    }
    .result-title {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 950;
    }
    .result-copy {
        color: var(--color-muted);
        margin: 0.35rem 0 0;
    }
    .student-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.7rem 0 1.05rem;
    }
    .student-chip {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        padding: 0.8rem;
        background: rgba(9, 13, 22, 0.42);
    }
    .student-chip span {
        display: block;
        color: var(--color-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .student-chip strong {
        color: var(--color-text);
        font-size: 1rem;
    }
    .risk-low {color: var(--color-pass); font-weight: 900;}
    .risk-moderate {color: var(--color-warn); font-weight: 900;}
    .risk-high {color: var(--color-fail); font-weight: 900;}
    .workflow {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        align-items: center;
        margin: 1rem 0;
    }
    .workflow-step {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        background: rgba(17, 24, 39, 0.74);
        color: var(--color-text);
        font-weight: 800;
    }
    .workflow-arrow {color: var(--color-accent);}
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        overflow: hidden;
    }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }
    .info-box {
        border: 1px solid var(--color-line);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(17, 24, 39, 0.70);
    }
    .info-box h3 {
        font-size: 1rem;
        margin: 0 0 0.5rem;
    }
    .info-box p, .info-box li {
        color: var(--color-muted);
    }
    @media (max-width: 768px) {
        .main .block-container {padding-left: 1rem; padding-right: 1rem;}
        .student-strip, .info-grid {grid-template-columns: 1fr;}
        .page-hero {padding: 1.25rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_data
def get_metrics() -> dict:
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def get_best_model():
    return load_trained_model(BEST_MODEL_PATH)


@st.cache_resource
def get_logistic_model():
    return load_trained_model(LOGISTIC_MODEL_PATH)


@st.cache_resource
def get_tree_model():
    return load_trained_model(TREE_MODEL_PATH)


@st.cache_data
def get_scored_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return score_dataset(df, get_best_model())


def save_dataset(df: pd.DataFrame) -> None:
    """Persist student records to the CSV dataset used by the app."""
    ordered_columns = ["student_id", "student_name", "attendance", "internal_marks", "test_score", "result"]
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[ordered_columns].to_csv(DATA_PATH, index=False)


def get_next_student_id(df: pd.DataFrame) -> str:
    """Generate the next ST### identifier from the current dataset."""
    numeric_ids = pd.to_numeric(df["student_id"].str.replace("ST", "", regex=False), errors="coerce")
    next_number = int(numeric_ids.max()) + 1 if not numeric_ids.dropna().empty else 1
    return f"ST{next_number:03d}"


def refresh_dataset_view() -> None:
    """Clear Streamlit cached data and rerun the page after CSV changes."""
    st.cache_data.clear()
    st.rerun()


def init_session_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def selected_model_name(metrics: dict) -> str:
    return str(metrics.get("selected_model", "Best Model"))


def risk_class(category: str) -> str:
    if category == "Low Risk":
        return "risk-low"
    if category == "Moderate Risk":
        return "risk-moderate"
    return "risk-high"


def add_prediction_history(student_id: str, student_name: str, attendance: float, internal_marks: float, test_score: float, result: dict) -> None:
    st.session_state.prediction_history.append(
        {
            "date_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_id": student_id,
            "student_name": student_name,
            "attendance": round(float(attendance), 1),
            "internal_marks": round(float(internal_marks), 1),
            "test_score": round(float(test_score), 1),
            "prediction": result["prediction"],
            "pass_probability": result["pass_probability"],
            "fail_probability": result["fail_probability"],
            "risk_category": result["risk_category"],
        }
    )


def render_metric_card(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="metric-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def header(page_label: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="page-hero">
            <div class="eyebrow">{html.escape(page_label)}</div>
            <div class="page-title">Student Pass/Fail Prediction System</div>
            <div class="page-subtitle">{html.escape(subtitle)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def panel_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def style_chart(fig):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dbeafe"},
        title={"font": {"size": 20}},
        margin={"l": 20, "r": 20, "t": 58, "b": 20},
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.18)")
    return fig


def dashboard_page(df: pd.DataFrame) -> None:
    header("Academic Intelligence Dashboard", "AI-powered analysis of student academic performance")
    scored_df = get_scored_dataset(df)
    pass_count = int((df["result"] == "Pass").sum())
    fail_count = int((df["result"] == "Fail").sum())
    at_risk_count = int((scored_df["risk_category"] == "High Risk").sum())
    total = len(df)

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Total Students", f"{total}", "Records in dataset")
    with cols[1]:
        render_metric_card("Pass Students", f"{pass_count}", "Actual dataset result")
    with cols[2]:
        render_metric_card("Fail Students", f"{fail_count}", "Actual dataset result")
    with cols[3]:
        render_metric_card("Pass Rate", f"{pass_count / total * 100:.1f}%", "No historical trend shown")

    cols2 = st.columns(4)
    with cols2[0]:
        render_metric_card("Average Attendance", f"{df['attendance'].mean():.1f}%", "Class average")
    with cols2[1]:
        render_metric_card("Average Internal Marks", f"{df['internal_marks'].mean():.1f}/50", "Class average")
    with cols2[2]:
        render_metric_card("Average Test Score", f"{df['test_score'].mean():.1f}/100", "Class average")
    with cols2[3]:
        render_metric_card("At-Risk Students", f"{at_risk_count}", "High risk by model failure probability")

    chart_col, avg_col = st.columns([2, 1])
    with chart_col:
        panel_title("Result Overview")
        result_counts = df["result"].value_counts().reset_index()
        result_counts.columns = ["Result", "Students"]
        fig = px.bar(
            result_counts,
            x="Result",
            y="Students",
            color="Result",
            color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR},
            title="Pass vs Fail Students",
            text="Students",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig = style_chart(fig)
        st.plotly_chart(fig, width='stretch')
    with avg_col:
        panel_title("Student Risk")
        risk_counts = scored_df["risk_category"].value_counts().reindex(["Low Risk", "Moderate Risk", "High Risk"], fill_value=0).reset_index()
        risk_counts.columns = ["Risk Category", "Students"]
        risk_fig = px.bar(
            risk_counts,
            x="Students",
            y="Risk Category",
            orientation="h",
            color="Risk Category",
            color_discrete_map={"Low Risk": PASS_COLOR, "Moderate Risk": "#f59e0b", "High Risk": FAIL_COLOR},
            title="Risk Category Distribution",
            text="Students",
        )
        st.plotly_chart(style_chart(risk_fig), width='stretch')


def predict_page(df: pd.DataFrame, metrics: dict) -> None:
    header("Predict Student", "Search a student, review academic inputs, and run the trained ML model.")
    model = get_best_model()
    model_name = selected_model_name(metrics)

    student_names = ["Select student name"] + df["student_name"].sort_values().tolist()
    student_ids = ["Select student ID"] + df["student_id"].tolist()
    selected_record = None

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        selected_name = st.selectbox("Student Name", student_names)
    with search_col2:
        selected_id = st.selectbox("Student ID", student_ids)

    if selected_id != "Select student ID":
        selected_record = df[df["student_id"] == selected_id].iloc[0]
    elif selected_name != "Select student name":
        selected_record = df[df["student_name"] == selected_name].iloc[0]

    default_name = "" if selected_record is None else str(selected_record["student_name"])
    default_id = "" if selected_record is None else str(selected_record["student_id"])
    default_attendance = 75.0 if selected_record is None else float(selected_record["attendance"])
    default_internal_marks = 35.0 if selected_record is None else float(selected_record["internal_marks"])
    default_test_score = 70.0 if selected_record is None else float(selected_record["test_score"])

    if selected_record is not None:
        st.markdown(
            f"""
            <div class="student-strip">
                <div class="student-chip"><span>Name</span><strong>{html.escape(default_name)}</strong></div>
                <div class="student-chip"><span>ID</span><strong>{html.escape(default_id)}</strong></div>
                <div class="student-chip"><span>Saved Result</span><strong>{html.escape(str(selected_record["result"]))}</strong></div>
                <div class="student-chip"><span>Dataset Row</span><strong>{int(selected_record.name) + 1}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form("prediction_form"):
        panel_title("Prediction Form")
        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input("Selected Student Name", value=default_name)
            student_id = st.text_input("Selected Student ID", value=default_id)
            attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=default_attendance, step=0.5)
        with col2:
            internal_marks = st.number_input("Internal Marks (0-50)", min_value=0.0, max_value=50.0, value=default_internal_marks, step=0.5)
            test_score = st.number_input("Test Score (0-100)", min_value=0.0, max_value=100.0, value=default_test_score, step=0.5)
            submitted = st.form_submit_button("Predict Result", width='stretch')

    if submitted:
        try:
            result = model_prediction(attendance, internal_marks, test_score, model)
            recommendations = recommendations_for_inputs(attendance, internal_marks, test_score)
            explanations = feature_based_explanation(attendance, internal_marks, test_score, model, model_name)
            add_prediction_history(student_id, student_name, attendance, internal_marks, test_score, result)
            card_class = "pass-card" if result["prediction"] == "PASS" else "fail-card"
            st.markdown(
                f"""
                <div class="{card_class}">
                    <h2 class="result-title">{result['prediction']}</h2>
                    <p class="result-copy">Prediction for {html.escape(student_name or 'student')} ({html.escape(student_id or 'ID not provided')}).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            result_cols = st.columns(4)
            with result_cols[0]:
                render_metric_card("Pass Probability", f"{result['pass_probability']}%", "Model probability")
            with result_cols[1]:
                render_metric_card("Fail Probability", f"{result['fail_probability']}%", "Model probability")
            with result_cols[2]:
                render_metric_card("Risk Score", f"{result['risk_score']}/100", "Equals failure probability")
            with result_cols[3]:
                render_metric_card("Confidence", f"{result['confidence']}%", f"Model: {model_name}")

            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="eyebrow">Risk Category</div>
                    <div class="{risk_class(result['risk_category'])}" style="font-size: 1.4rem;">{html.escape(result['risk_category'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("Student Risk")
            st.progress(result["risk_score"] / 100, text=f"{result['risk_score']} / 100")
            st.write("PASS Probability")
            st.progress(result["pass_probability"] / 100, text=f"{result['pass_probability']}%")
            st.write("FAIL Probability")
            st.progress(result["fail_probability"] / 100, text=f"{result['fail_probability']}%")

            explain_col, recommend_col = st.columns(2)
            with explain_col:
                panel_title("Why This Prediction?")
                for item in explanations:
                    st.write(f"- {item}")
                st.caption("Feature importance reflects how the trained model uses available features; it does not prove causation.")
            with recommend_col:
                panel_title("Personalized Recommendations")
                for item in recommendations:
                    st.write(f"- {item}")

            report_student = {
                "student_name": student_name or "Student",
                "student_id": student_id or "ID not provided",
                "attendance": attendance,
                "internal_marks": internal_marks,
                "test_score": test_score,
                **result,
            }
            pdf_bytes = create_student_report_pdf(report_student, recommendations, model_name)
            st.download_button(
                "Download Student Report PDF",
                data=pdf_bytes,
                file_name=f"{student_id or 'student'}_prediction_report.pdf",
                mime="application/pdf",
            )
            st.info("Prediction is based on attendance, internal marks, and test score.")
            st.warning("This prediction is an estimate and should not be treated as a definitive academic decision.")
        except ValueError as exc:
            st.error(str(exc))


def student_profiles_page(df: pd.DataFrame, metrics: dict) -> None:
    header("Student Profiles", "Inspect a student profile and compare it against dataset averages.")
    scored_df = get_scored_dataset(df)
    model_name = selected_model_name(metrics)

    search = st.text_input("Search by Student ID or Student Name", placeholder="Type ST032 or Mansi")
    view_df = scored_df
    if search.strip():
        query = search.strip().lower()
        view_df = scored_df[
            scored_df["student_id"].str.lower().str.contains(query, na=False)
            | scored_df["student_name"].str.lower().str.contains(query, na=False)
        ]
    if view_df.empty:
        st.warning("No matching student found.")
        return

    options = [f"{row.student_id} | {row.student_name}" for row in view_df[["student_id", "student_name"]].itertuples(index=False)]
    selected = st.selectbox("Select Student", options)
    selected_id = selected.split(" | ", 1)[0]
    student = scored_df[scored_df["student_id"] == selected_id].iloc[0]

    st.markdown(
        f"""
        <div class="student-strip">
            <div class="student-chip"><span>Name</span><strong>{html.escape(str(student['student_name']))}</strong></div>
            <div class="student-chip"><span>ID</span><strong>{html.escape(str(student['student_id']))}</strong></div>
            <div class="student-chip"><span>Actual Result</span><strong>{html.escape(str(student['result']))}</strong></div>
            <div class="student-chip"><span>Model Prediction</span><strong>{html.escape(str(student['model_prediction']))}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Risk Score", f"{student['risk_score']:.2f}/100", student["risk_category"])
    with cols[1]:
        render_metric_card("Failure Probability", f"{student['fail_probability']:.2f}%", model_name)
    with cols[2]:
        render_metric_card("Attendance", f"{student['attendance']:.1f}%", "Student value")
    with cols[3]:
        render_metric_card("Test Score", f"{student['test_score']:.1f}/100", "Student value")

    averages = df[FEATURE_COLUMNS].mean()
    comparison = pd.DataFrame(
        {
            "Feature": ["Attendance", "Internal Marks", "Test Score"],
            "Student": [student["attendance"], student["internal_marks"], student["test_score"]],
            "Dataset Average": [averages["attendance"], averages["internal_marks"], averages["test_score"]],
        }
    )
    comparison_fig = px.bar(
        comparison,
        x="Feature",
        y=["Student", "Dataset Average"],
        barmode="group",
        title="Student Performance vs Dataset Average",
        color_discrete_sequence=[ACCENT_COLOR, "#94a3b8"],
    )
    st.plotly_chart(style_chart(comparison_fig), width='stretch')
    st.caption("Actual dataset result is the stored CSV label. Model prediction is calculated by the trained model.")


def risk_analysis_page(df: pd.DataFrame) -> None:
    header("Risk Analysis", "Analyze student academic risk using actual model failure probabilities.")
    scored_df = get_scored_dataset(df)
    risk_counts = scored_df["risk_category"].value_counts().reindex(["Low Risk", "Moderate Risk", "High Risk"], fill_value=0)

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Total Students", f"{len(scored_df)}", "Scored by best model")
    with cols[1]:
        render_metric_card("Low Risk", f"{risk_counts['Low Risk']}", "Failure probability < 35%")
    with cols[2]:
        render_metric_card("Moderate Risk", f"{risk_counts['Moderate Risk']}", "35% to < 65%")
    with cols[3]:
        render_metric_card("High Risk", f"{risk_counts['High Risk']}", "Failure probability >= 65%")

    col1, col2 = st.columns(2)
    with col1:
        risk_df = risk_counts.reset_index()
        risk_df.columns = ["Risk Category", "Students"]
        fig = px.pie(
            risk_df,
            names="Risk Category",
            values="Students",
            title="Risk Category Distribution",
            color="Risk Category",
            color_discrete_map={"Low Risk": PASS_COLOR, "Moderate Risk": "#f59e0b", "High Risk": FAIL_COLOR},
            hole=0.45,
        )
        st.plotly_chart(style_chart(fig), width='stretch')
        hist = px.histogram(scored_df, x="fail_probability", nbins=24, title="Failure Probability Distribution")
        st.plotly_chart(style_chart(hist), width='stretch')
    with col2:
        for feature, title in [
            ("attendance", "Attendance vs Failure Probability"),
            ("test_score", "Test Score vs Failure Probability"),
            ("internal_marks", "Internal Marks vs Failure Probability"),
        ]:
            fig = px.scatter(
                scored_df,
                x=feature,
                y="fail_probability",
                color="risk_category",
                hover_data=["student_id", "student_name", "result", "model_prediction"],
                title=title,
                color_discrete_map={"Low Risk": PASS_COLOR, "Moderate Risk": "#f59e0b", "High Risk": FAIL_COLOR},
            )
            st.plotly_chart(style_chart(fig), width='stretch')


def batch_prediction_page(metrics: dict) -> None:
    header("Batch Prediction", "Upload a CSV and score many students without changing the main dataset.")
    st.caption("Required columns: student_id, student_name, attendance, internal_marks, test_score")
    uploaded_file = st.file_uploader("Upload Student CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to generate batch predictions.")
        return

    try:
        upload_df = pd.read_csv(uploaded_file)
        result_df = predict_batch(upload_df, get_best_model())
        render_metric_card("Batch Records", f"{len(result_df)}", f"Model: {selected_model_name(metrics)}")
        st.dataframe(result_df, width='stretch', height=460)
        st.download_button(
            "Download Predictions CSV",
            result_df.to_csv(index=False),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")


def what_if_page() -> None:
    header("What-If Simulator", "Compare current and simulated outcomes using the actual trained model.")
    model = get_best_model()
    panel_title("Current Student Values")
    c1, c2, c3 = st.columns(3)
    with c1:
        current_attendance = st.number_input("Current Attendance (%)", 0.0, 100.0, 62.0, 0.5)
    with c2:
        current_marks = st.number_input("Current Internal Marks", 0.0, 50.0, 27.0, 0.5)
    with c3:
        current_score = st.number_input("Current Test Score", 0.0, 100.0, 48.0, 0.5)

    panel_title("Simulated Improvements")
    s1, s2, s3 = st.columns(3)
    with s1:
        sim_attendance = st.slider("Simulated Attendance (%)", 0.0, 100.0, current_attendance, 0.5)
    with s2:
        sim_marks = st.slider("Simulated Internal Marks", 0.0, 50.0, current_marks, 0.5)
    with s3:
        sim_score = st.slider("Simulated Test Score", 0.0, 100.0, current_score, 0.5)

    current = model_prediction(current_attendance, current_marks, current_score, model)
    simulated = model_prediction(sim_attendance, sim_marks, sim_score, model)
    change = round(simulated["fail_probability"] - current["fail_probability"], 2)

    cols = st.columns(5)
    with cols[0]:
        render_metric_card("Current Result", current["prediction"], current["risk_category"])
    with cols[1]:
        render_metric_card("Simulated Result", simulated["prediction"], simulated["risk_category"])
    with cols[2]:
        render_metric_card("Current Failure", f"{current['fail_probability']}%", "Actual model call")
    with cols[3]:
        render_metric_card("Simulated Failure", f"{simulated['fail_probability']}%", "Actual model call")
    with cols[4]:
        render_metric_card("Change", f"{change:+.2f}%", "Failure probability")

    comparison = pd.DataFrame(
        {
            "Scenario": ["Current", "Simulated"],
            "Failure Probability": [current["fail_probability"], simulated["fail_probability"]],
            "Pass Probability": [current["pass_probability"], simulated["pass_probability"]],
        }
    )
    fig = px.bar(comparison, x="Scenario", y=["Failure Probability", "Pass Probability"], barmode="group", title="Current vs Simulated Probability")
    st.plotly_chart(style_chart(fig), width='stretch')


def prediction_history_page() -> None:
    header("Prediction History", "Session-based history of predictions made in this app.")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    if history_df.empty:
        st.info("No predictions made in this session yet.")
        return

    st.dataframe(history_df, width='stretch', height=430)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download History CSV", history_df.to_csv(index=False), "prediction_history.csv", "text/csv")
    with c2:
        if st.button("Clear History", width='stretch'):
            st.session_state.prediction_history = []
            st.rerun()


def dataset_page(df: pd.DataFrame) -> None:
    header("Dataset", "Browse, add, delete, and download student records from the CSV dataset.")
    browse_tab, add_tab, delete_tab = st.tabs(["Browse Dataset", "Add Student", "Delete Student"])

    with browse_tab:
        panel_title("Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            search_query = st.text_input("Search Student ID or Name", placeholder="ST032 or Mansi")
            result_filter = st.selectbox("Result", ["All", "Pass", "Fail"])
            min_attendance = st.slider("Minimum Attendance", 0, 100, 0)
            max_attendance = st.slider("Maximum Attendance", 0, 100, 100)
        with col2:
            min_marks = st.slider("Minimum Internal Marks", 0, 50, 0)
            max_marks = st.slider("Maximum Internal Marks", 0, 50, 50)
        with col3:
            min_score = st.slider("Minimum Test Score", 0, 100, 0)
            max_score = st.slider("Maximum Test Score", 0, 100, 100)
            sort_by = st.selectbox("Sort By", ["student_id", "student_name", "attendance", "internal_marks", "test_score", "result"])
            sort_order = st.radio("Sort Order", ["Ascending", "Descending"], horizontal=True)

        filtered = df[
            df["attendance"].between(min_attendance, max_attendance)
            & df["internal_marks"].between(min_marks, max_marks)
            & df["test_score"].between(min_score, max_score)
        ]
        if result_filter != "All":
            filtered = filtered[filtered["result"] == result_filter]
        if search_query.strip():
            query = search_query.strip().lower()
            filtered = filtered[
                filtered["student_id"].str.lower().str.contains(query, na=False)
                | filtered["student_name"].str.lower().str.contains(query, na=False)
            ]
        filtered = filtered.sort_values(sort_by, ascending=sort_order == "Ascending")

        stats = st.columns(4)
        with stats[0]:
            render_metric_card("Total Records", f"{len(df)}", "Complete dataset")
        with stats[1]:
            render_metric_card("Filtered Records", f"{len(filtered)}", "Rows matching filters")
        with stats[2]:
            render_metric_card("Pass Count", f"{int((filtered['result'] == 'Pass').sum())}", "Filtered data")
        with stats[3]:
            render_metric_card("Fail Count", f"{int((filtered['result'] == 'Fail').sum())}", "Filtered data")
        st.dataframe(filtered, width='stretch', height=460)
        st.download_button("Download Filtered CSV", filtered.to_csv(index=False), file_name="filtered_students.csv", mime="text/csv")

    with add_tab:
        panel_title("Add New Student")
        st.caption("New records are saved directly in data/students.csv. The result can be manually selected or filled from the trained model prediction.")
        suggested_id = get_next_student_id(df)

        with st.form("add_student_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_student_id = st.text_input("Student ID", value=suggested_id)
                new_student_name = st.text_input("Student Name", placeholder="Example: Rahul Sharma")
                new_attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
            with col2:
                new_internal_marks = st.number_input("Internal Marks (0-50)", min_value=0.0, max_value=50.0, value=35.0, step=0.5)
                new_test_score = st.number_input("Test Score (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
                result_mode = st.radio("Result Save Mode", ["Use ML Prediction", "Select Manually"], horizontal=True)
                manual_result = st.selectbox("Manual Result", ["Pass", "Fail"], disabled=result_mode == "Use ML Prediction")
            add_submitted = st.form_submit_button("Add Student", width='stretch')

        if add_submitted:
            cleaned_id = new_student_id.strip().upper()
            cleaned_name = new_student_name.strip()

            if not cleaned_id:
                st.error("Student ID required hai.")
            elif not cleaned_name:
                st.error("Student Name required hai.")
            elif cleaned_id in set(df["student_id"]):
                st.error("Ye Student ID already dataset me exists karti hai. Dusri ID use karo.")
            else:
                if result_mode == "Use ML Prediction":
                    predicted = predict_student(new_attendance, new_internal_marks, new_test_score)
                    saved_result = "Pass" if predicted["prediction"] == "PASS" else "Fail"
                else:
                    saved_result = manual_result

                new_row = pd.DataFrame(
                    [
                        {
                            "student_id": cleaned_id,
                            "student_name": cleaned_name,
                            "attendance": round(float(new_attendance), 1),
                            "internal_marks": round(float(new_internal_marks), 1),
                            "test_score": round(float(new_test_score), 1),
                            "result": saved_result,
                        }
                    ]
                )
                updated_df = pd.concat([df, new_row], ignore_index=True)
                save_dataset(updated_df)
                st.success(f"{cleaned_name} ({cleaned_id}) dataset me add ho gaya. Saved result: {saved_result}.")
                refresh_dataset_view()

    with delete_tab:
        panel_title("Delete Student")
        st.caption("Delete karne par record data/students.csv se remove ho jayega.")

        delete_options = [
            f"{row.student_id} | {row.student_name} | {row.result}"
            for row in df[["student_id", "student_name", "result"]].itertuples(index=False)
        ]
        selected_delete = st.selectbox("Student Select Karo", delete_options)
        selected_delete_id = selected_delete.split(" | ", 1)[0]
        selected_delete_row = df[df["student_id"] == selected_delete_id].iloc[0]

        st.markdown(
            f"""
            <div class="student-strip">
                <div class="student-chip"><span>Name</span><strong>{html.escape(str(selected_delete_row["student_name"]))}</strong></div>
                <div class="student-chip"><span>ID</span><strong>{html.escape(str(selected_delete_row["student_id"]))}</strong></div>
                <div class="student-chip"><span>Attendance</span><strong>{float(selected_delete_row["attendance"]):.1f}%</strong></div>
                <div class="student-chip"><span>Result</span><strong>{html.escape(str(selected_delete_row["result"]))}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        confirm_delete = st.checkbox("Haan, main is student ko delete karna chahta/chahti hoon")
        if st.button("Delete Selected Student", width='stretch', disabled=not confirm_delete):
            updated_df = df[df["student_id"] != selected_delete_id].reset_index(drop=True)
            save_dataset(updated_df)
            st.success(f"{selected_delete_id} delete ho gaya.")
            refresh_dataset_view()


def performance_page(metrics: dict) -> None:
    header("Model Performance", "Compare trained algorithms using real held-out test metrics.")
    selected = metrics["selected_model"]
    selected_metrics = metrics["models"][selected]
    st.markdown(
        f"""
        <div class="panel">
            <div class="eyebrow">Selected Model</div>
            <div class="page-title" style="font-size: 2rem;">{html.escape(selected)}</div>
            <div class="page-subtitle">{html.escape(metrics.get("selected_model_reason", ""))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score"], ["accuracy", "precision", "recall", "f1_score"]):
        with col:
            render_metric_card(label, f"{selected_metrics[key] * 100:.2f}%")

    rows = []
    for model_name, model_metrics in metrics["models"].items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": model_metrics["accuracy"],
                "Precision": model_metrics["precision"],
                "Recall": model_metrics["recall"],
                "F1 Score": model_metrics["f1_score"],
            }
        )
    panel_title("Model Comparison")
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    comparison_fig = px.bar(
        pd.DataFrame(rows).melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Comparison Chart",
        color_discrete_sequence=[ACCENT_COLOR, FAIL_COLOR],
    )
    st.plotly_chart(style_chart(comparison_fig), width='stretch')

    roc_data = get_roc_data(get_best_model())
    render_metric_card("ROC-AUC", f"{roc_data['roc_auc']:.4f}", "Calculated on held-out test set")
    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(
            x=roc_data["false_positive_rate"],
            y=roc_data["true_positive_rate"],
            mode="lines",
            name=f"ROC Curve (AUC={roc_data['roc_auc']:.4f})",
            line={"color": ACCENT_COLOR, "width": 3},
        )
    )
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline", line={"dash": "dash", "color": "#94a3b8"}))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(style_chart(roc_fig), width='stretch')

    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), caption="Actual confusion matrix generated during evaluation")

    panel_title("Feature Importance / Interpretability")
    tree_importance = pd.DataFrame(get_feature_importance(get_tree_model(), "Decision Tree"))
    logistic_importance = pd.DataFrame(get_feature_importance(get_logistic_model(), "Logistic Regression"))
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Decision Tree: actual feature_importances_")
        if not tree_importance.empty:
            fig = px.bar(tree_importance, x="feature", y="value", title="Decision Tree Feature Importance")
            st.plotly_chart(style_chart(fig), width='stretch')
            st.dataframe(tree_importance, width='stretch', hide_index=True)
    with c2:
        st.caption("Logistic Regression: actual coefficients after model training")
        if not logistic_importance.empty:
            fig = px.bar(logistic_importance, x="feature", y="value", title="Logistic Regression Coefficients")
            st.plotly_chart(style_chart(fig), width='stretch')
            st.dataframe(logistic_importance, width='stretch', hide_index=True)
    st.info("Feature importance shows how the trained model uses the available features; it does not prove causation.")
    st.download_button("Download Model Metrics JSON", METRICS_PATH.read_text(encoding="utf-8"), "model_metrics.json", "application/json")

    st.markdown(
        """
        **Accuracy:** Percentage of predictions that were correct.

        **Precision:** How many predicted pass students were actually pass.

        **Recall:** How many actual pass students were correctly identified.

        **F1 Score:** Balance between precision and recall.
        """
    )


def analytics_page(df: pd.DataFrame) -> None:
    header("Analytics", "Visual analysis of attendance, marks, test scores, and pass/fail outcomes.")
    pie_fig = px.pie(
        df,
        names="result",
        title="Pass vs Fail Distribution",
        color="result",
        color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR},
        hole=0.48,
    )
    st.plotly_chart(style_chart(pie_fig), width='stretch')
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(style_chart(px.box(df, x="result", y="attendance", color="result", color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR}, title="Attendance vs Result")), width='stretch')
        st.plotly_chart(style_chart(px.box(df, x="result", y="internal_marks", color="result", color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR}, title="Internal Marks vs Result")), width='stretch')
        st.plotly_chart(style_chart(px.scatter(df, x="attendance", y="test_score", color="result", color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR}, title="Attendance vs Test Score")), width='stretch')
    with col2:
        st.plotly_chart(style_chart(px.box(df, x="result", y="test_score", color="result", color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR}, title="Test Score vs Result")), width='stretch')
        melted = df.melt(id_vars="result", value_vars=FEATURE_COLUMNS, var_name="Feature", value_name="Value")
        st.plotly_chart(style_chart(px.histogram(melted, x="Value", color="result", color_discrete_map={"Pass": PASS_COLOR, "Fail": FAIL_COLOR}, facet_row="Feature", title="Feature Distributions", height=700)), width='stretch')


def about_page() -> None:
    header("About Project", "A complete machine learning workflow designed for academic demonstration.")
    st.markdown(
        """
        <div class="workflow">
            <div class="workflow-step">Dataset</div><div class="workflow-arrow">-></div>
            <div class="workflow-step">Preprocessing</div><div class="workflow-arrow">-></div>
            <div class="workflow-step">Feature Selection</div><div class="workflow-arrow">-></div>
            <div class="workflow-step">Training</div><div class="workflow-arrow">-></div>
            <div class="workflow-step">Evaluation</div><div class="workflow-arrow">-></div>
            <div class="workflow-step">Best Model</div><div class="workflow-arrow">-></div>
            <div class="workflow-step">Prediction</div><div class="workflow-arrow">-></div>
            <div class="workflow-step">Risk Analysis</div>
        </div>
        <div class="info-grid">
            <div class="info-box">
                <h3>Project Overview</h3>
                <p>This platform predicts pass/fail outcomes and academic risk using attendance percentage, internal marks, test score, and trained machine learning models.</p>
            </div>
            <div class="info-box">
                <h3>Problem Statement</h3>
                <p>Early prediction of academic risk helps teachers identify students who may need additional support before final outcomes are decided.</p>
            </div>
            <div class="info-box">
                <h3>Objectives</h3>
                <ul>
                    <li>Build a real machine learning model for pass/fail classification.</li>
                    <li>Compare Logistic Regression and Decision Tree classifiers.</li>
                    <li>Provide dashboard, analytics, and prediction views.</li>
                </ul>
            </div>
            <div class="info-box">
                <h3>Dataset & Features</h3>
                <p>The dataset contains student ID, name, attendance, internal marks, test score, and actual result. The model uses attendance, internal marks, and test score as features.</p>
            </div>
            <div class="info-box">
                <h3>Machine Learning Workflow</h3>
                <p>CSV loading, cleaning, validation, feature selection, train/test split, model training, evaluation, best-model selection, prediction, and risk scoring.</p>
            </div>
            <div class="info-box">
                <h3>Machine Learning Algorithms</h3>
                <p>Logistic Regression estimates class probability. Decision Tree creates rule-based splits from feature values.</p>
            </div>
            <div class="info-box">
                <h3>Model Evaluation</h3>
                <p>Accuracy, precision, recall, F1 score, confusion matrix, ROC curve, and ROC-AUC are calculated from actual model outputs.</p>
            </div>
            <div class="info-box">
                <h3>Prediction Process</h3>
                <p>User inputs are validated, passed to the trained model, and converted into prediction, probabilities, confidence, recommendations, and risk score.</p>
            </div>
            <div class="info-box">
                <h3>Risk Analysis</h3>
                <p>Risk score is based on model failure probability. Low, moderate, and high categories are assigned using documented probability thresholds.</p>
            </div>
            <div class="info-box">
                <h3>Limitations</h3>
                <p>The dataset is synthetic and limited to three academic indicators. Predictions are estimates, not official academic decisions.</p>
            </div>
            <div class="info-box">
                <h3>Future Scope</h3>
                <p>More academic features, historical performance, attendance trends, early warning systems, teacher dashboard, intervention tracking, database integration, authentication, explainable AI, model retraining, and advanced ML algorithms.</p>
            </div>
            <div class="info-box">
                <h3>Academic Note</h3>
                <p>The system supports academic analysis and viva demonstration. It should assist decision-making but should not replace teacher judgment or official evaluation rules.</p>
            </div>
        </div>
        """
        ,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_session_state()
    df = get_dataset()
    metrics = get_metrics()
    st.sidebar.markdown(
        """
        <div class="brand-card">
            <div class="brand-mark">&rarr;</div>
            <div class="brand-title"><strong>Academic Predictor</strong></div>
            <div class="brand-subtitle"><strong>AI-powered student academic analytics</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Student Prediction",
            "Student Profiles",
            "Risk Analysis",
            "Batch Prediction",
            "What-If Simulator",
            "Dataset",
            "Model Performance",
            "Analytics",
            "Prediction History",
            "About Project",
        ],
    )
    st.sidebar.markdown(
        """
        <div class="nav-footer">
            <strong>Student Pass/Fail Prediction System</strong><br>
            Machine Learning Academic Analytics
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Dashboard":
        dashboard_page(df)
    elif page == "Student Prediction":
        predict_page(df, metrics)
    elif page == "Student Profiles":
        student_profiles_page(df, metrics)
    elif page == "Risk Analysis":
        risk_analysis_page(df)
    elif page == "Batch Prediction":
        batch_prediction_page(metrics)
    elif page == "What-If Simulator":
        what_if_page()
    elif page == "Dataset":
        dataset_page(df)
    elif page == "Model Performance":
        performance_page(metrics)
    elif page == "Analytics":
        analytics_page(df)
    elif page == "Prediction History":
        prediction_history_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
