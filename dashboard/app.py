
from pathlib import Path
import html

import pandas as pd
import plotly.express as px
import streamlit as st

# ================= CONFIG =================

st.set_page_config(
    page_title="RR Pressure Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = Path(__file__).parent / "data" / "05_player_pressure_scores.csv"

# ================= DATA =================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    numeric_cols = [
        "rank", "pressure_score",
        "total_pressure_balls", "total_normal_balls"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(
        subset=["role", "player_name", "rank", "pressure_score"]
    )

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load ranking data: {error}")
    st.stop()

batters = df[df["role"] == "Batter"].sort_values("rank")
bowlers = df[df["role"] == "Bowler"].sort_values("rank")

# ================= PREMIUM DESIGN =================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #080D19;
    color: #F4F5FA;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #10182B;
    border-right: 1px solid #26334D;
}

[data-testid="stSidebar"] * {
    color: #E9ECF5;
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(120deg, #171F39, #31152E);
    border: 1px solid #59304C;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-label {
    color: #FF6B9D;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin: 10px 0;
}

.hero-subtitle {
    color: #B9C1D5;
    font-size: 14px;
}

.kpi {
    background: #131E33;
    border: 1px solid #293953;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.kpi-label {
    color: #AAB5CC;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.kpi-value {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}

.section-title {
    color: #FF6B9D;
    font-size: 20px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 12px;
}

.player-card {
    background: #131E33;
    border: 1px solid #293953;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 10px;
}

.player-rank {
    color: #FF6B9D;
    font-size: 13px;
    font-weight: 800;
}

.player-name {
    color: white;
    font-size: 18px;
    font-weight: 700;
    margin-top: 5px;
}

.player-score {
    color: #F4C66A;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}

.rr-table {
    width: 100%;
    border-collapse: collapse;
    background: #111B2E;
    border-radius: 12px;
    overflow: hidden;
}

.rr-table th {
    background: #26314A;
    color: #FF8AB0;
    padding: 13px;
    text-align: left;
    font-size: 12px;
}

.rr-table td {
    padding: 12px;
    border-bottom: 1px solid #26334D;
    color: #E8ECF5;
    font-size: 13px;
}

.rr-table tr:hover td {
    background: #1B2942;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    color: #E8ECF5;
}

.stButton button {
    background: #F04483;
    color: white;
    border: none;
    border-radius: 8px;
}

hr {
    border-color: #293953;
}
</style>
""", unsafe_allow_html=True)


# ================= HELPERS =================

def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-label">{html.escape(str(label))}</div>
        <div class="kpi-value">{html.escape(str(value))}</div>
    </div>
    """, unsafe_allow_html=True)


def show_table(data, columns):
    display = data[columns].copy()

    if "pressure_score" in display.columns:
        display["pressure_score"] = display["pressure_score"].map(
            lambda x: f"{x:.4f}"
        )

    display.columns = [
        col.replace("_", " ").title() for col in display.columns
    ]

    table_html = display.to_html(
        index=False,
        escape=True,
        classes="rr-table",
        border=0
    )

    st.markdown(table_html, unsafe_allow_html=True)


def show_chart(data, title):
    chart_data = data.sort_values("pressure_score")

    fig = px.bar(
        chart_data,
        x="pressure_score",
        y="player_name",
        orientation="h",
        color="pressure_score",
        color_continuous_scale=["#8B315B", "#F04483", "#F4C66A"],
        title=title,
        hover_data=["rank", "total_pressure_balls"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080D19",
        plot_bgcolor="#10182B",
        font=dict(color="#E8ECF5"),
        title_font_size=18,
        xaxis_title="Pressure Score",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=10, r=15, t=55, b=15)
    )

    st.plotly_chart(fig, use_container_width=True)


# ================= SIDEBAR =================

st.sidebar.markdown(
    "<h2 style='color:#FF6B9D'>RR ANALYTICS</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption("SUPERR SELECTOR SEASON 2")
st.sidebar.divider()

page = st.sidebar.radio(
    "DASHBOARD MENU",
    [
        "Overview",
        "Top 10 Batters",
        "Top 10 Bowlers",
        "Player Comparison",
        "Methodology"
    ]
)

st.sidebar.divider()
st.sidebar.caption("DATA PERIOD")
st.sidebar.markdown("2022 – 2026")
st.sidebar.caption("Historical performance analysis")


# ================= HEADER =================

st.markdown("""
<div class="hero">
    <div class="hero-label">RAJASTHAN ROYALS • DATA ANALYTICS</div>
    <div class="hero-title">RR Pressure Analytics</div>
    <div class="hero-subtitle">
        Discovering players who deliver when the game gets difficult.
        Historical IPL and T20I performance analysis.
    </div>
</div>
""", unsafe_allow_html=True)


# ================= OVERVIEW =================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Performance Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Players Ranked", df["player_name"].nunique())

    with c2:
        kpi_card("Batters", len(batters))

    with c3:
        kpi_card("Bowlers", len(bowlers))

    with c4:
        kpi_card("Data Period", "2022–2026")

    st.markdown(
        '<div class="section-title">Top Pressure Performers</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        if not batters.empty:
            p = batters.iloc[0]
            st.markdown(f"""
            <div class="player-card">
                <div class="player-rank">#1 BATTER</div>
                <div class="player-name">{html.escape(str(p["player_name"]))}</div>
                <div class="player-score">{p["pressure_score"]:.4f}</div>
                <div style="color:#AAB5CC;font-size:12px">
                    Historical pressure score
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        if not bowlers.empty:
            p = bowlers.iloc[0]
            st.markdown(f"""
            <div class="player-card">
                <div class="player-rank">#1 BOWLER</div>
                <div class="player-name">{html.escape(str(p["player_name"]))}</div>
                <div class="player-score">{p["pressure_score"]:.4f}</div>
                <div style="color:#AAB5CC;font-size:12px">
                    Historical pressure score
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Top 10 Batter Performance</div>',
        unsafe_allow_html=True
    )
    show_chart(batters.head(10), "Batter Pressure Score")

    st.markdown(
        '<div class="section-title">Top 10 Bowler Performance</div>',
        unsafe_allow_html=True
    )
    show_chart(bowlers.head(10), "Bowler Pressure Score")


# ================= BATTERS =================

elif page == "Top 10 Batters":

    st.title("Top 10 Batters")
    st.caption("Ranked by historical pressure-performance score")

    top = batters.head(10)

    show_table(
        top,
        [
            "rank", "player_name", "pressure_score",
            "total_pressure_balls", "total_normal_balls"
        ]
    )

    show_chart(top, "Top 10 Batters – Pressure Score")


# ================= BOWLERS =================

elif page == "Top 10 Bowlers":

    st.title("Top 10 Bowlers")
    st.caption("Ranked by historical pressure-performance score")

    top = bowlers.head(10)

    show_table(
        top,
        [
            "rank", "player_name", "pressure_score",
            "total_pressure_balls", "total_normal_balls"
        ]
    )

    show_chart(top, "Top 10 Bowlers – Pressure Score")


# ================= COMPARISON =================

elif page == "Player Comparison":

    st.title("Player Comparison")
    st.caption("Compare historical pressure scores between two players.")

    role = st.selectbox("Choose role", ["Batter", "Bowler"])

    role_df = df[df["role"] == role].sort_values("rank")
    names = role_df["player_name"].drop_duplicates().tolist()

    if len(names) >= 2:

        c1, c2 = st.columns(2)

        with c1:
            player1 = st.selectbox("Player 1", names, index=0)

        with c2:
            player2 = st.selectbox("Player 2", names, index=1)

        selected = role_df[
            role_df["player_name"].isin([player1, player2])
        ]

        show_table(
            selected,
            [
                "rank", "player_name", "pressure_score",
                "total_pressure_balls", "total_normal_balls"
            ]
        )

        fig = px.bar(
            selected,
            x="player_name",
            y="pressure_score",
            color="player_name",
            title="Head-to-head Pressure Score",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="#080D19",
            plot_bgcolor="#10182B",
            font=dict(color="#E8ECF5")
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough players for comparison.")


# ================= METHODOLOGY =================

elif page == "Methodology":

    st.title("Research Methodology")

    st.markdown("### 01 — Data Collection")
    st.write(
        "Historical IPL and T20I ball-by-ball data from 2022 to 2026."
    )

    st.markdown("### 02 — Pressure Identification")
    st.write(
        "Batting pressure included high required run rates, resource "
        "collapse and close finishes. Bowling pressure included "
        "death-over defence, low-target powerplays and partnership momentum."
    )

    st.markdown("### 03 — Performance Metrics")
    st.write(
        "Batting scoring and dismissal rates were compared under pressure "
        "and normal conditions. Bowling conceded rates and wicket "
        "probabilities were also compared."
    )

    st.markdown("### 04 — Statistical Ranking")
    st.write(
        "Standardised performance components were combined using "
        "exposure-based weights to create separate role rankings."
    )

    st.markdown("### 05 — Sensitivity Testing")
    st.write(
        "Alternative pressure thresholds were tested to examine "
        "how ranking results changed."
    )

    st.warning(
        "These are historical comparative rankings. They do not predict "
        "future performance or confirm IPL 2027 auction availability."
    )


# ================= FOOTER =================

st.divider()

st.markdown("""
<div style="text-align:center;color:#8995AD;font-size:12px;padding:10px">
    RR PRESSURE ANALYTICS<br>
    Built with Python • Pandas • Streamlit • Plotly<br>
    SupeRR Selector Season 2
</div>
""", unsafe_allow_html=True)
