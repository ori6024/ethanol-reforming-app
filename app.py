
import numpy as np
import streamlit as st
import cantera as ct
import plotly.graph_objects as go


@st.cache_resource
def get_gas():
    # ethanol.yaml must be in the repository root
    return ct.Solution("ethanol.yaml")


@st.cache_resource
def get_graphite():
    # Optional for carbon margin; if not available, margin indicator is disabled
    for name in ["graphite.yaml", "graphite.cti", "graphite.xml"]:
        try:
            return ct.Solution(name)
        except Exception:
            pass
    return None


def normalize_percent(y_dict, keys):
    s = 0.0
    for k in keys:
        s += max(y_dict.get(k, 0.0), 0.0)
    if s <= 0:
        return {k: 0.0 for k in keys}
    return {k: 100.0 * max(y_dict.get(k, 0.0), 0.0) / s for k in keys}


def find_regions(x, flag01):
    regions = []
    start = None
    for i in range(len(x)):
        on = (flag01[i] == 1)
        last = (i == len(x) - 1)
        if on and start is None:
            start = x[i]
        if start is not None and ((not on) or last):
            end = x[i] if (on and last) else x[i - 1]
            if end > start:
                regions.append((start, end))
            start = None
    return regions


def reaction_Kp(T, gas, graphite, nu_gas, nu_graphite=None):
    # Compute Kp from standard-state Gibbs energies (NASA polynomials inside Cantera).
    gas.TP = T, ct.one_atm
    g0_RT = gas.standard_gibbs_RT

    dg0_RT = 0.0
    for sp, nu in nu_gas.items():
        if sp not in gas.species_names:
            return np.nan
        dg0_RT += nu * g0_RT[gas.species_index(sp)]

    if nu_graphite and graphite is not None:
        graphite.TP = T, ct.one_atm
        g0g_RT = graphite.standard_gibbs_RT
        for sp, nu in nu_graphite.items():
            if sp not in graphite.species_names:
                return np.nan
            dg0_RT += nu * g0g_RT[graphite.species_index(sp)]

    return float(np.exp(-dg0_RT))


def carbon_margin(T, P_bar, y_gas_eq, gas, graphite):
    # margin = ln(Q) - ln(K). negative => risk
    eps = 1e-30

    def p(sp):
        return max(y_gas_eq.get(sp, 0.0), 0.0) * P_bar

    if graphite is None:
        return {"enabled": False, "allow_any": False, "m1": np.nan, "m2": np.nan}

    # (1) CH4 -> C + 2H2
    pCH4, pH2 = p("CH4"), p("H2")
    Q1 = (pH2**2) / (pCH4 + eps)
    K1 = reaction_Kp(
        T, gas, graphite,
        nu_gas={"CH4": -1, "H2": +2},
        nu_graphite={"C(gr)": +1},
    )
    m1 = np.log(max(Q1, eps)) - np.log(max(K1, eps)) if np.isfinite(K1) and K1 > 0 else np.nan
    allow_ch4 = (Q1 < K1) if np.isfinite(K1) else False

    # (2) 2CO -> C + CO2
    pCO, pCO2 = p("CO"), p("CO2")
    Q2 = (pCO2) / (pCO**2 + eps)
    K2 = reaction_Kp(
        T, gas, graphite,
        nu_gas={"CO": -2, "CO2": +1},
        nu_graphite={"C(gr)": +1},
    )
    m2 = np.log(max(Q2, eps)) - np.log(max(K2, eps)) if np.isfinite(K2) and K2 > 0 else np.nan
    allow_bou = (Q2 < K2) if np.isfinite(K2) else False

    return {"enabled": True, "allow_any": bool(allow_ch4 or allow_bou), "m1": m1, "m2": m2}


def apply_mobile_layout(fig, title_text, y_title, y_is_percent=True):
    fig.update_layout(
        title={"text": title_text, "font": {"size": 16}, "x": 0.5, "xanchor": "center"},
        margin={"l": 60, "r": 20, "t": 55, "b": 90},
        legend={"orientation": "h", "yanchor": "top", "y": -0.30, "xanchor": "center", "x": 0.5, "font": {"size": 11}},
    )
    fig.update_xaxes(
        title={"text": "Temperature (°C)", "standoff": 18},
        tickfont={"size": 10},
        tickmode="linear",
        tick0=400,
        dtick=100,
        fixedrange=True,
    )
    fig.update_yaxes(
        title={"text": y_title, "standoff": 10},
        tickfont={"size": 10},
        fixedrange=True,
    )
    if y_is_percent:
        fig.update_yaxes(range=[0, 100], tickmode="linear", tick0=0, dtick=20)
    return fig


def stacked_area(x, series_dict, title, ytitle, danger_regions=None):
    fig = go.Figure()
    for name, y in series_dict.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", stackgroup="one", name=name))
    if danger_regions:
        for (x0, x1) in danger_regions:
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(255,0,0,0.12)", line_width=0, layer="below")
    apply_mobile_layout(fig, title, ytitle, y_is_percent=True)
    return fig


def percent_lines(x, series_dict, title, ytitle, danger_regions=None):
    fig = go.Figure()
    for name, y in series_dict.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    if danger_regions:
        for (x0, x1) in danger_regions:
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(255,0,0,0.12)", line_width=0, layer="below")
    apply_mobile_layout(fig, title, ytitle, y_is_percent=True)
    return fig


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Ethanol Reformer Simulator", layout="wide")
st.markdown("<h2 style='margin-bottom:0.2rem;'>Ethanol Reformer Simulator (NASA thermo via Cantera)</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Inputs")
    P_atm = st.slider("Pressure (atm)", 1.0, 2.0, 1.20, 0.01)
    S_E = st.slider("Steam-to-Ethanol (mol H2O / mol EtOH)", 0.0, 10.0, 3.0, 0.1)
    CO2_E = st.slider("CO2-to-Ethanol (mol CO2 / mol EtOH)", 0.0, 5.0, 0.0, 0.1)
    npts = st.slider("Temperature points", 21, 101, 41, 2)
    st.caption("Feed basis: 1 mol EtOH + S/E mol H2O + CO2/E mol CO2, equilibrium at each T.")


# Load mechanism
try:
    gas = get_gas()
except Exception as e:
    st.error("Failed to load ethanol.yaml. Put it in the repository root with the name 'ethanol.yaml'.")
    st.exception(e)
    st.stop()

graphite = get_graphite()

# Required species check
must = ["C2H5OH", "H2O", "H2", "CO", "CO2", "CH4"]
missing = [sp for sp in must if sp not in gas.species_names]
if missing:
    st.error("ethanol.yaml is missing required species: " + ", ".join(missing))
    st.stop()

T_C = np.linspace(400, 800, npts)
P_bar = P_atm * 1.01325

dry_pref = ["H2", "CO", "CO2", "CH4", "C2H4", "C2H6", "CH3CHO", "C2H2"]
wet_pref = ["H2O", "H2", "CO", "CO2", "CH4", "C2H4", "C2H6", "CH3CHO", "C2H5OH", "C2H2"]

dry_keys = [k for k in dry_pref if k in gas.species_names and k != "H2O"]
wet_keys = [k for k in wet_pref if k in gas.species_names]

dry = {k: [] for k in dry_keys}
wet = {k: [] for k in wet_keys}

flag = []
m1_list = []
m2_list = []

for Tc in T_C:
    T = Tc + 273.15

    comp = {"C2H5OH": 1.0, "H2O": S_E, "CO2": CO2_E}
    gas.TPX = T, P_bar * 1e5, comp
    gas.equilibrate("TP")

    y = {sp: gas.X[gas.species_index(sp)] for sp in wet_keys}

    d = normalize_percent(y, dry_keys)
    w = normalize_percent(y, wet_keys)

    for k in dry_keys:
        dry[k].append(d.get(k, 0.0))
    for k in wet_keys:
        wet[k].append(w.get(k, 0.0))

    cm = carbon_margin(T, P_bar, y, gas, graphite)
    flag.append(1 if (cm["enabled"] and cm["allow_any"]) else 0)
    m1_list.append(cm["m1"])
    m2_list.append(cm["m2"])

danger_regions = find_regions(T_C, flag)
show_warn = any(flag)

# Top plots (unchanged style: stacked area)
colA, colB = st.columns([1, 1], gap="large")

with colA:
    fig1 = stacked_area(T_C, {k: dry[k] for k in dry_keys}, "Dry gas composition (normalized to 100%)", "Dry composition (%)", danger_regions)
    if show_warn:
        st.warning("Carbon deposition may be thermodynamically allowed in shaded temperature ranges (indicator based on CH4 cracking / Boudouard).")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    fig2 = stacked_area(T_C, {k: wet[k] for k in wet_keys}, "Wet gas composition (including steam, normalized to 100%)", "Wet composition (%)", danger_regions)
    st.plotly_chart(fig2, use_container_width=True)

# Carbon margin plot
st.subheader("Carbon deposition margin (ln(Q) - ln(K); negative = risk)")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=T_C, y=m1_list, mode="lines", name="CH4 → C + 2H2"))
fig3.add_trace(go.Scatter(x=T_C, y=m2_list, mode="lines", name="2CO → C + CO2"))

for (x0, x1) in danger_regions:
    fig3.add_vrect(x0=x0, x1=x1, fillcolor="rgba(255,0,0,0.12)", line_width=0, layer="below")

fig3.add_hline(y=0.0, line_dash="dash", line_width=1)
apply_mobile_layout(fig3, "Carbon deposition margin (ln(Q) - ln(K); negative = risk)", "ln(Q) - ln(K)", y_is_percent=False)
fig3.update_layout(margin={"l": 60, "r": 20, "t": 55, "b": 85})
fig3.update_xaxes(tickmode="linear", tick0=400, dtick=100)

st.plotly_chart(fig3, use_container_width=True)

# Back plots: species % lines (as requested)
st.subheader("Gas composition curves (each species as % line)")

colC, colD = st.columns([1, 1], gap="large")
with colC:
    fig4 = percent_lines(T_C, {k: dry[k] for k in dry_keys}, "Dry gas composition (each species %, 0–100% fixed)", "Dry composition (%)", danger_regions)
    st.plotly_chart(fig4, use_container_width=True)

with colD:
    fig5 = percent_lines(T_C, {k: wet[k] for k in wet_keys}, "Wet gas composition (each species %, 0–100% fixed)", "Wet composition (%)", danger_regions)
    st.plotly_chart(fig5, use_container_width=True)

if graphite is None:
    st.info("Note: graphite phase file was not found in this Cantera install, so Kp-based carbon margin may be limited.")
