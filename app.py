import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import joblib

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="TeacherOn Subject Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"], .stMarkdown, p, li, span, label {
    font-family: 'Outfit', sans-serif;
}

/* hide default chrome */
#MainMenu, header, footer, .stDeployButton, [data-testid="stToolbar"] {
    display: none !important;
}

/* gradient hero title */
.hero-title {
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.6rem;
    line-height: 1.15;
    margin-bottom: 4px;
}

.hero-sub {
    color: #64748b;
    font-size: 1.1rem;
    margin-bottom: 28px;
}

/* stat cards */
.stat-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.06));
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
}
.stat-number {
    font-size: 2rem;
    font-weight: 800;
    color: #6366f1;
}
.stat-label {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 2px;
}

/* insight callout */
.insight {
    background: #ffffff;
    border-left: 4px solid #10b981;
    border-radius: 6px;
    padding: 14px 20px;
    margin: 12px 0 24px 0;
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 500;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
    border-top: 1px solid rgba(0,0,0,0.03);
    border-right: 1px solid rgba(0,0,0,0.03);
    border-bottom: 1px solid rgba(0,0,0,0.03);
}

/* winner card */
.winner-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(168,85,247,0.08));
    border: 2px solid #6366f1;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

/* prediction result */
.pred-result {
    text-align: center;
    padding: 30px;
    border-radius: 18px;
    border: 2px solid;
}

/* subject badge */
.sbadge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.15rem;
    color: white;
    margin-top: 8px;
    text-transform: capitalize;
}

/* custom button */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    transition: transform 0.2s ease;
}
div.stButton > button:first-child:hover {
    transform: scale(1.03);
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e1b4b) !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── DATA & MODEL LOADING ──────────────────────────────────
DATA_PATH = "tutors_ml_dataset.csv"
MODEL_PATH = "tutor_model.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["search_subject"] = df["search_subject"].str.lower().str.strip()
    for c in ["total_experience", "online_experience", "price_avg_usd",
              "price_min_usd", "price_max_usd", "desc_length"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["num_subjects"] = pd.to_numeric(df["num_subjects"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

df = load_data()
bundle = load_bundle()

NUM_COLS = ["total_experience", "online_experience", "price_min_usd",
            "price_max_usd", "price_avg_usd", "num_subjects", "desc_length"]

SUBJECT_META = {
    "math":             {"icon": "🧮", "clr": "#6366f1"},
    "physics":          {"icon": "⚛️",  "clr": "#a855f7"},
    "chemistry":        {"icon": "🧪", "clr": "#ec4899"},
    "biology":          {"icon": "🧬", "clr": "#10b981"},
    "english":          {"icon": "📚", "clr": "#f59e0b"},
    "computer science": {"icon": "💻", "clr": "#06b6d4"},
}

def style_plotly_fig(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Outfit",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Outfit",
            bordercolor="#6366f1",
            font_color="#0f172a",
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(0,0,0,0.15)",
        title_font=dict(size=12, family="Outfit", color="#475569"),
        tickfont=dict(size=11, family="Outfit", color="#64748b")
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0.08)",
        linecolor="rgba(0,0,0,0.15)",
        title_font=dict(size=12, family="Outfit", color="#475569"),
        tickfont=dict(size=11, family="Outfit", color="#64748b")
    )
    return fig

# ─── SIDEBAR ────────────────────────────────────────────────
st.sidebar.markdown("## 🎓 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Data Insights", "🤖 The Model", "🔮 Predict a Subject"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on 1 200 tutor profiles scraped from TeacherOn.com · "
    "Gradient Boosting Classifier · 79.1 % accuracy"
)

# ════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("<h1 class='hero-title'>Can We Guess What a Tutor Teaches?</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>"
        "We scraped 1 200 tutor profiles from <b>TeacherOn.com</b>, explored the data, "
        "and trained a machine-learning model that predicts a tutor's <b>subject specialty</b> "
        "from their profile alone — no subject label needed."
        "</p>",
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{len(df):,}", "Tutor Profiles Scraped"),
        ("6", "Subject Categories"),
        (str(df["country"].nunique()), "Countries Represented"),
        ("79.1 %", "Model Accuracy"),
    ]
    for col, (num, lbl) in zip([c1, c2, c3, c4], stats):
        col.markdown(
            f"<div class='stat-card'><div class='stat-number'>{num}</div>"
            f"<div class='stat-label'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # The story
    st.markdown("### 📖 The Story")
    st.markdown(
        "**TeacherOn** is a global marketplace where students find private tutors. "
        "We searched for tutors in **6 subjects** — Math, Physics, Chemistry, Biology, "
        "English and Computer Science — and saved their public profiles.\n\n"
        "Each profile contains the tutor's **name, country, years of experience, "
        "hourly rate, listed subjects, and a self-written description**. "
        "The question we asked:\n"
    )
    st.info(
        "💡 **Given only a tutor's profile information, can a machine-learning model "
        "correctly predict which subject they primarily teach?**"
    )

    # Column reference
    st.markdown("### 📋 Dataset Columns at a Glance")
    col_desc = pd.DataFrame({
        "Column": [
            "search_subject", "name", "country", "total_experience",
            "online_experience", "price_min_usd", "price_max_usd",
            "price_avg_usd", "price_unit", "num_subjects",
            "subjects_taught", "desc_length", "description",
        ],
        "What it means": [
            "The subject searched to find this tutor (our TARGET)",
            "Tutor's display name",
            "Country of the tutor",
            "Total years of teaching experience",
            "Years of online teaching experience",
            "Lowest listed price (USD)",
            "Highest listed price (USD)",
            "Average of min & max price (USD)",
            "Billing cycle — hour / month / day",
            "How many subjects the tutor lists",
            "Comma-separated list of all subjects they teach",
            "Character count of their profile description",
            "The tutor's self-written profile text",
        ],
    })
    st.dataframe(col_desc, use_container_width=True, hide_index=True)

    st.markdown("### 👀 Sample Profiles")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
#  PAGE 2 — DATA INSIGHTS (EDA)
# ════════════════════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.markdown("<h1 class='hero-title'>What Does the Data Tell Us?</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>Nine visualizations that shaped our understanding before building the model.</p>",
        unsafe_allow_html=True,
    )

    # ── 1. Tutors per subject ───────────────────────────────
    st.markdown("### 1 · How Many Tutors per Subject?")
    counts = df["search_subject"].value_counts().sort_index()
    counts_df = pd.DataFrame({
        "Subject": counts.index.str.title(),
        "Tutor Count": counts.values
    })
    fig1 = px.bar(
        counts_df,
        x="Subject",
        y="Tutor Count",
        labels={"Subject": "Subject", "Tutor Count": "Number of Tutors"},
        color="Subject",
        color_discrete_map={
            "Math": "#6366f1",
            "Physics": "#a855f7",
            "Chemistry": "#ec4899",
            "Biology": "#10b981",
            "English": "#f59e0b",
            "Computer Science": "#06b6d4",
        }
    )
    fig1.update_yaxes(title="Number of Tutors")
    fig1.update_xaxes(title="Subject")
    fig1.update_layout(showlegend=False, height=350)
    st.plotly_chart(style_plotly_fig(fig1), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>✅ <b>Perfectly balanced</b> — every subject has the same number of tutors. "
        "This is ideal for classification because the model won't be biased toward any one subject.</div>",
        unsafe_allow_html=True,
    )

    # ── 2. Numeric distributions ────────────────────────────
    st.markdown("### 2 · How Are the Numbers Distributed?")
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig2 = make_subplots(
        rows=3, cols=3, 
        subplot_titles=[c.replace("_", " ").title() for c in NUM_COLS],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    for i, col in enumerate(NUM_COLS):
        row_idx = (i // 3) + 1
        col_idx = (i % 3) + 1
        fig2.add_trace(
            go.Histogram(
                x=df[col], 
                name=col.replace("_", " ").title(), 
                marker_color="#10b981", 
                nbinsx=20,
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
            ),
            row=row_idx, col=col_idx
        )
    # Hide empty subplots
    fig2.update_xaxes(visible=False, row=3, col=2)
    fig2.update_yaxes(visible=False, row=3, col=2)
    fig2.update_xaxes(visible=False, row=3, col=3)
    fig2.update_yaxes(visible=False, row=3, col=3)
    
    fig2.update_layout(height=650, showlegend=False)
    st.plotly_chart(style_plotly_fig(fig2), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>📌 Most tutors have <b>5–15 years</b> of experience and charge "
        "<b>low hourly rates</b>. Price columns are right-skewed with some expensive outliers.</div>",
        unsafe_allow_html=True,
    )

    # ── 3. Top countries ────────────────────────────────────
    st.markdown("### 3 · Where Are the Tutors From?")
    top10 = df["country"].value_counts().head(10).reset_index()
    top10.columns = ["Country", "Tutor Count"]
    top10 = top10.sort_values(by="Tutor Count", ascending=True)
    fig3 = px.bar(
        top10,
        y="Country",
        x="Tutor Count",
        orientation="h",
        color="Tutor Count",
        color_continuous_scale="Oranges",
        labels={"Country": "Country", "Tutor Count": "Number of Tutors"}
    )
    fig3.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(style_plotly_fig(fig3), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>🌏 <b>India dominates</b> the dataset with ~65 % of all profiles, "
        "followed by Pakistan. This geographic skew is something the model learns to account for.</div>",
        unsafe_allow_html=True,
    )

    # ── 4. Price unit pie chart ─────────────────────────────
    st.markdown("### 4 · How Do Tutors Charge?")
    unit_counts = df["price_unit"].value_counts()
    unit_df = pd.DataFrame({
        "Price Unit": unit_counts.index.str.title(),
        "Count": unit_counts.values
    })
    fig4 = px.pie(
        unit_df,
        names="Price Unit",
        values="Count",
        hole=0.4,
        color="Price Unit",
        color_discrete_map={
            "Hour": "#6366f1",
            "Month": "#a855f7",
            "Day": "#ec4899",
            "Unknown": "#94a3b8"
        }
    )
    fig4.update_traces(
        textposition="inside",
        textinfo="percent+label",
        insidetextorientation="horizontal",
    )
    fig4.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(style_plotly_fig(fig4), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>⏱️ Nearly <b>90 % charge by the hour</b>. "
        "Monthly and daily billing are rare edge cases.</div>",
        unsafe_allow_html=True,
    )

    # ── 5. Correlation heatmap ──────────────────────────────
    st.markdown("### 5 · Which Features Move Together?")
    corr = df[NUM_COLS].corr()
    fig5 = px.imshow(
        corr,
        x=[c.replace("_", " ").title() for c in NUM_COLS],
        y=[c.replace("_", " ").title() for c in NUM_COLS],
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels=dict(x="Feature", y="Feature", color="Correlation")
    )
    fig5.update_traces(
        text=corr.round(2).values,
        texttemplate="%{text}",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
    )
    fig5.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(style_plotly_fig(fig5), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>🔗 <b>Price max ↔ Price avg</b> are almost perfectly correlated (0.99). "
        "<b>Total experience ↔ Online experience</b> share a moderate link (0.58).</div>",
        unsafe_allow_html=True,
    )

    # ── 6. Avg hourly price by subject ──────────────────────
    st.markdown("### 6 · Does the Subject Affect the Price?")
    hourly = df[df["price_unit"] == "hour"]
    avg_by_subj = hourly.groupby("search_subject")["price_avg_usd"].mean().sort_values()
    avg_df = pd.DataFrame({
        "Subject": avg_by_subj.index.str.title(),
        "Avg Hourly Price": avg_by_subj.values
    })
    fig6 = px.bar(
        avg_df,
        x="Subject",
        y="Avg Hourly Price",
        labels={"Subject": "Subject", "Avg Hourly Price": "Average Hourly Price (USD)"},
        color="Subject",
        color_discrete_map={
            "Math": "#6366f1",
            "Physics": "#a855f7",
            "Chemistry": "#ec4899",
            "Biology": "#10b981",
            "English": "#f59e0b",
            "Computer Science": "#06b6d4",
        }
    )
    fig6.update_yaxes(title="Average Hourly Price (USD)")
    fig6.update_xaxes(title="Subject")
    fig6.update_layout(showlegend=False, height=350)
    st.plotly_chart(style_plotly_fig(fig6), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>💲 Prices are fairly similar across subjects. "
        "<b>Math</b> tutors charge slightly more on average, while <b>English</b> tutors charge the least.</div>",
        unsafe_allow_html=True,
    )

    # ── 7. Scatter plots of correlated pairs ────────────────
    st.markdown("### 7 · Zooming Into the Strongest Relationships")
    fig7 = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Max Price vs Avg Price",
            "Total Exp vs Online Exp",
            "Min Price vs Avg Price"
        ],
        horizontal_spacing=0.08
    )
    fig7.add_trace(
        go.Scatter(
            x=df["price_max_usd"], y=df["price_avg_usd"],
            mode="markers",
            marker=dict(color="#c44e52", opacity=0.6, size=6),
            name="Max vs Avg",
            hovertemplate="Max Price: $%{x:.2f}<br>Avg Price: $%{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    fig7.update_xaxes(title_text="Max Price (USD)", row=1, col=1)
    fig7.update_yaxes(title_text="Avg Price (USD)", row=1, col=1)
    
    fig7.add_trace(
        go.Scatter(
            x=df["total_experience"], y=df["online_experience"],
            mode="markers",
            marker=dict(color="#c44e52", opacity=0.6, size=6),
            name="Total vs Online",
            hovertemplate="Total Exp: %{x} yrs<br>Online Exp: %{y} yrs<extra></extra>"
        ),
        row=1, col=2
    )
    fig7.update_xaxes(title_text="Total Experience (Years)", row=1, col=2)
    fig7.update_yaxes(title_text="Online Experience (Years)", row=1, col=2)
    
    fig7.add_trace(
        go.Scatter(
            x=df["price_min_usd"], y=df["price_avg_usd"],
            mode="markers",
            marker=dict(color="#c44e52", opacity=0.6, size=6),
            name="Min vs Avg",
            hovertemplate="Min Price: $%{x:.2f}<br>Avg Price: $%{y:.2f}<extra></extra>"
        ),
        row=1, col=3
    )
    fig7.update_xaxes(title_text="Min Price (USD)", row=1, col=3)
    fig7.update_yaxes(title_text="Avg Price (USD)", row=1, col=3)
    
    fig7.update_layout(height=400, showlegend=False)
    st.plotly_chart(style_plotly_fig(fig7), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>📈 The tight diagonal in the first plot confirms max and avg price "
        "carry nearly identical information — the model only really needs one of them.</div>",
        unsafe_allow_html=True,
    )

    # ── 8. Line plot: avg price vs max price ────────────────
    st.markdown("### 8 · How Does Average Price Rise With Max Price?")
    line1 = df.groupby("price_max_usd")["price_avg_usd"].mean().reset_index()
    fig8 = px.line(
        line1,
        x="price_max_usd",
        y="price_avg_usd",
        labels={"price_max_usd": "Max Price (USD)", "price_avg_usd": "Avg Price (USD)"}
    )
    fig8.update_traces(
        line_color="#6366f1",
        line_width=2.5,
        hovertemplate="Max Price: $%{x:.2f}<br>Avg Price: $%{y:.2f}<extra></extra>"
    )
    fig8.update_layout(height=350)
    st.plotly_chart(style_plotly_fig(fig8), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>📐 A clear <b>upward trend</b> — as max price increases, "
        "average price follows almost perfectly.</div>",
        unsafe_allow_html=True,
    )

    # ── 9. Line plot: avg price by experience ───────────────
    st.markdown("### 9 · Do More Experienced Tutors Charge More?")
    line2 = df.groupby("total_experience")["price_avg_usd"].mean().reset_index()
    fig9 = px.line(
        line2,
        x="total_experience",
        y="price_avg_usd",
        markers=True,
        labels={"total_experience": "Total Experience (Years)", "price_avg_usd": "Avg Price (USD)"}
    )
    fig9.update_traces(
        line_color="#ec4899",
        line_width=2,
        marker=dict(size=6, color="#ec4899"),
        hovertemplate="Experience: %{x} yrs<br>Avg Price: $%{y:.2f}<extra></extra>"
    )
    fig9.update_layout(height=350)
    st.plotly_chart(style_plotly_fig(fig9), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div class='insight'>🤷 <b>No clear trend</b> — more experience does NOT reliably mean higher prices. "
        "The line is bumpy and flat, suggesting other factors matter more.</div>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════
#  PAGE 3 — THE MODEL
# ════════════════════════════════════════════════════════════
elif page == "🤖 The Model":
    st.markdown("<h1 class='hero-title'>How We Built the Classifier</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>From raw profiles to a working prediction engine — here's the journey.</p>",
        unsafe_allow_html=True,
    )

    # Preprocessing summary
    st.markdown("### 🔧 Step 1 — Preprocessing")
    st.markdown(
        "Before training, the raw data went through four cleaning stages:"
    )
    steps = {
        "Outlier Removal": "Removed extreme prices using the IQR method (1 200 → 1 055 rows).",
        "Price Capping": "Clipped remaining prices at the 99th percentile to tame outliers.",
        "One-Hot Encoding": "Converted country (grouped rare ones into 'Other') and price unit into 18 binary columns.",
        "TF-IDF Vectorization": "Turned subjects taught + profile description into 250 numerical text features.",
    }
    for title, desc in steps.items():
        st.markdown(f"- **{title}:** {desc}")

    st.markdown(
        f"\n> **Final feature matrix:** 25 base features + 250 text features = **275 total features** "
        f"· Training set: 844 profiles · Test set: 211 profiles"
    )

    st.markdown("---")

    # Model comparison
    st.markdown("### 🏁 Step 2 — Model Comparison")
    st.markdown("We trained three different classifiers and compared their accuracy on unseen test data:")

    model_results = {
        "Logistic Regression": 0.720,
        "K-Nearest Neighbors": 0.284,
        "Gradient Boosting": 0.791,
    }
    model_df = pd.DataFrame({
        "Model": list(model_results.keys()),
        "Test Accuracy": list(model_results.values())
    })
    fig_m = px.bar(
        model_df,
        x="Model",
        y="Test Accuracy",
        labels={"Model": "Model Classifier", "Test Accuracy": "Test Accuracy"},
        color="Model",
        color_discrete_map={
            "Logistic Regression": "#94a3b8",
            "K-Nearest Neighbors": "#94a3b8",
            "Gradient Boosting": "#6366f1"
        }
    )
    fig_m.update_yaxes(range=[0, 1.0], title="Test Accuracy")
    fig_m.update_xaxes(title="")
    
    # Add random guess line
    fig_m.add_shape(
        type="line",
        x0=-0.5, x1=2.5,
        y0=0.167, y1=0.167,
        line=dict(color="#e11d48", width=1.5, dash="dash")
    )
    
    # Add a text annotation for the random guess line
    fig_m.add_annotation(
        x=2.0, y=0.19,
        text="Random Guess (16.7%)",
        showarrow=False,
        font=dict(size=10, family="Outfit", color="#e11d48"),
        align="right"
    )
    
    fig_m.update_traces(
        hovertemplate="Model: %{x}<br>Accuracy: %{y:.1%}<extra></extra>"
    )
    fig_m.update_layout(showlegend=False, height=350)
    st.plotly_chart(style_plotly_fig(fig_m), use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "<div class='insight'>🏆 <b>Gradient Boosting</b> is the clear winner — nearly 4× better than "
        "K-Nearest Neighbors and 7 percentage points above Logistic Regression. "
        "The dashed red line shows what random guessing (1 in 6) would score.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Winner metrics
    st.markdown("### 📊 Step 3 — Evaluating the Winner")

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("79.1 %", "Accuracy", "How often the model is correct overall"),
        ("79.7 %", "Precision", "When it predicts a subject, how often it's right"),
        ("79.1 %", "Recall", "Of all tutors in a subject, how many it finds"),
        ("79.3 %", "F1 Score", "Balance between precision & recall"),
    ]
    for col, (val, name, tip) in zip([m1, m2, m3, m4], metrics):
        col.metric(name, val, help=tip)

    st.markdown(
        "<div class='insight'>📝 <b>In plain English:</b> If you hand the model a tutor profile it has "
        "never seen before, it will correctly guess their subject <b>about 4 out of 5 times</b>.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # What the model learned
    st.markdown("### 🧠 What the Model Learns From")
    left, right = st.columns(2)
    with left:
        st.markdown("**Numerical features (scaled)**")
        st.markdown(
            "- Total teaching experience\n"
            "- Online teaching experience\n"
            "- Min / Max / Avg price (USD)\n"
            "- Number of subjects listed\n"
            "- Profile description length"
        )
    with right:
        st.markdown("**Text features (TF-IDF)**")
        st.markdown(
            "- All listed subjects the tutor teaches\n"
            "- The tutor's self-written description\n\n"
            "These are converted into **250 word-importance scores** "
            "that capture which words appear most for each subject."
        )

    st.markdown(
        "**Categorical features (one-hot):** Tutor's country (14 groups) "
        "and billing unit (hour / month / day / unknown) → 18 binary columns."
    )

# ════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICT A SUBJECT
# ════════════════════════════════════════════════════════════
elif page == "🔮 Predict a Subject":
    st.markdown("<h1 class='hero-title'>Try It Yourself</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>"
        "Fill in a tutor's profile details and the model will predict their primary subject."
        "</p>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### 👤 Experience & Location")
        total_exp = st.slider("Total teaching experience (years)", 0.0, 40.0, 8.0, 0.5)
        online_exp = st.slider("Online teaching experience (years)", 0.0, 40.0, 5.0, 0.5)

        countries = [c.replace("country_", "") for c in bundle["cat_columns"] if c.startswith("country_")]
        country = st.selectbox("Country", sorted(countries))

        st.markdown("##### 💵 Pricing")
        price_min = st.number_input("Min price (USD)", 0.0, 500.0, 10.0, 1.0)
        price_max = st.number_input("Max price (USD)", 0.0, 1000.0, 30.0, 1.0)
        price_avg = (price_min + price_max) / 2

        units = [u.replace("unit_", "") for u in bundle["cat_columns"] if u.startswith("unit_")]
        price_unit = st.selectbox("Billing cycle", units,
                                  index=units.index("hour") if "hour" in units else 0)

    with col_r:
        st.markdown("##### 📝 Profile Text")
        
        # Predefined common subjects in the dataset to prevent typos
        COMMON_SUBJECTS = [
            "Chemistry", "English", "Physics", "Maths", "Biology", "Science", 
            "Computer Science", "Math", "Mathematics", "Hindi", "Python", 
            "French", "Java", "Economics", "Calculus", "Geometry", "Algebra", 
            "Trigonometry", "Linear Algebra", "Organic Chemistry", "Inorganic Chemistry", 
            "Mechanics", "Electromagnetism", "Thermodynamics", "Genetics", "Botany", 
            "Zoology", "Grammar", "Literature", "Writing", "Spoken English", 
            "Data Structures", "Algorithms", "Web Development", "Machine Learning", "C++"
        ]
        
        selected_subjects = st.multiselect(
            "Select subjects taught (prevents typos)",
            options=COMMON_SUBJECTS,
            default=["Algebra", "Calculus", "Geometry", "Mathematics"]
        )
        
        subjects_text = ", ".join(selected_subjects)
        num_subj = max(len(selected_subjects), 1)
        
        description_text = st.text_area(
            "Profile description",
            "I am a passionate mathematics teacher with 8 years of classroom "
            "and online experience. I focus on breaking down complex calculus, "
            "equations, and algebra theorems into simpler modules for school "
            "and college test prep.",
            height=175,
        )
        desc_len = len(description_text)

    st.markdown("")
    if st.button("🚀  Predict Subject"):

        # 1 — numeric
        numeric = pd.DataFrame(
            [[total_exp, online_exp, price_min, price_max, price_avg, num_subj, desc_len]],
            columns=bundle["num_cols"],
        )

        # 2 — categorical dummies
        cat = pd.DataFrame(0, index=[0], columns=bundle["cat_columns"])
        c_col = f"country_{country}"
        if c_col in cat.columns:
            cat[c_col] = 1
        elif "country_Other" in cat.columns:
            cat["country_Other"] = 1
        u_col = f"unit_{price_unit}"
        if u_col in cat.columns:
            cat[u_col] = 1

        # 3 — scale
        base = pd.concat([numeric, cat], axis=1)
        base_scaled = bundle["scaler"].transform(base)

        # 4 — TF-IDF
        text_vec = bundle["tfidf"].transform([subjects_text + " " + description_text]).toarray()

        # 5 — combine & predict
        X_new = np.hstack([base_scaled, text_vec])
        pred_num = bundle["model"].predict(X_new)[0]
        pred_subject = bundle["le"].inverse_transform([pred_num])[0]

        # Confidence bars
        probs = bundle["model"].predict_proba(X_new)[0] if hasattr(bundle["model"], "predict_proba") else None
        classes = bundle["le"].classes_

        st.markdown("---")

        res_l, res_r = st.columns([1, 1.5])

        with res_l:
            meta = SUBJECT_META.get(pred_subject, {"icon": "🎓", "clr": "#6366f1"})
            icon = meta['icon']
            clr = meta['clr']
            st.markdown(
                f"<div class='pred-result' style='border-color:{clr};'>"
                f"<div style='font-size:4.5rem'>{icon}</div>"
                f"<h4 style='margin:8px 0 0'>Predicted Subject</h4>"
                f"<div class='sbadge' style='background:{clr}'>{pred_subject}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with res_r:
            if probs is not None:
                st.markdown("##### Confidence Breakdown")
                sorted_idx = np.argsort(probs)[::-1]
                for idx in sorted_idx:
                    name = classes[idx]
                    prob = probs[idx]
                    m = SUBJECT_META.get(name, {"icon": "🎓", "clr": "#6366f1"})
                    st.write(f"{m['icon']} **{name.title()}** — {prob*100:.1f} %")
                    st.progress(float(prob))
