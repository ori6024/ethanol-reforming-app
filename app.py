import numpy as np
import streamlit as st
import cantera as ct
import plotly.graph_objects as go


# -----------------------------
# Cantera phases
# -----------------------------
@st.cache_resource
def get_gas():
    # ethanol.yaml must be in the repository root
    return ct.Solution("ethanol.yaml")


@st.cache_resource
def get_graphite():
    # optional; used for carbon margin indicator
    for name in ["graphite.yaml", "graphite.cti", "graphite.xml"]:
        try:
            return ct.Solution(name)
        except Exception:
            pass
    return None


# -----------------------------
# Helpers
# -----------------------------
def normalize_percent(y_dict, keys):
    s = sum(max(y_dict.get(k, 0.0), 0.0) for k in keys)
    if s <= 0:
        return {k: 0.0 for k in keys}
    return {k: 100.0 * max(y_dict.get(k, 0.0), 0.0) / s for k in keys}


def to_dry(y_dict, keys_dry):
    # exclude H2O, normalize remaining to 100%
    return normalize_percent(y_dict, keys_dry)


def to_wet(y_dict, keys_wet):
    # include H2O, normalize to 100%
    return normalize_percent(y_dict, keys_wet)


def find_regions(x, flag01):
    """Convert 0/1 flags on x-grid into contiguous [x0,x1] regions."""
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


# -----------------------------
# Thermo: Kp from standard Gibbs (NASA polynomials inside Cantera)
# -----------------------------
def reaction_Kp(T, gas, graphite, nu_gas, nu_graphite=None):
    """
    Compute Kp for a reaction from standard-state Gibbs energies.
    Standard state: 1 bar (via Cantera standard_gibbs_RT).
    """
    gas.TP = T, ct.one_atm
    g0_RT = gas.standard_gibbs_RT

    dg0_RT = 0.0
    for sp, nu in nu_gas.items():
        if sp not in gas.species_names:
            # if missing, cannot compute
            return np.nan
        dg0_RT += nu * g0_RT[gas.species_index(sp)]

    if nu_graphite and graphite is not None:
        graphite.TP = T, ct.one_atm
        g0g_RT = graphite.standard_gibbs_RT
        for sp, nu in nu_graphite.items():
            if sp not in graphite.species_names:
                return np.nan
            dg0_RT += nu * g0g_RT[graphite.species_index(sp)]

    return np.exp(-dg0_RT)


def carbon_margin(T, P_bar, y_gas_eq, gas, graphite):
    """
    Carbon deposition margin indicator by:
