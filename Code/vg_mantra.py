
from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import networkx as nx

# ------------------ Basics ------------------
def to_flux(mag: np.ndarray) -> np.ndarray:
    return 10.0 ** (-0.4 * mag)

def mag_to_flux_err(mag: np.ndarray, magerr: np.ndarray) -> np.ndarray:
    F = to_flux(mag)
    c = 0.4 * np.log(10.0)
    Ferr = F * c * magerr
    return Ferr

def prepare_series(df_light: pd.DataFrame,
                   transient_id,
                   use_flux: bool = True,
                   min_points: int = 20,
                   winsor: float = 0.0,
                   max_points: Optional[int] = None) -> pd.DataFrame:
    """
    Prepare one light-curve for a given ID.
    Works whether 'MJD' is an index or a column in df_light.
    Returns ['MJD','x','xerr'] or empty DF if not enough points.
    """
    # filter by ID (robust types: compare as string to avoid int/str mismatch)
    s = df_light[df_light['ID'].astype(str) == str(transient_id)].copy()

    # ensure MJD column
    if 'MJD' not in s.columns:
        if s.index.name == 'MJD':
            s = s.reset_index()
        else:
            raise KeyError("MJD not found as column or index in light table.")

    s = s[['MJD', 'Mag', 'Magerr']].dropna()
    s = s.sort_values('MJD', kind="mergesort")

    # de-duplicate same MJD keeping smallest error
    s = s.loc[s.groupby('MJD')['Magerr'].idxmin()]

    if use_flux:
        x = to_flux(s['Mag'].to_numpy())
        xerr = mag_to_flux_err(s['Mag'].to_numpy(), s['Magerr'].to_numpy())
    else:
        x = (-s['Mag']).to_numpy()
        xerr = s['Magerr'].to_numpy()

    # optional winsorization
    if winsor and winsor > 0.0:
        lo = np.quantile(x, winsor)
        hi = np.quantile(x, 1.0 - winsor)
        x = np.clip(x, lo, hi)

    # optional uniform subsample in time
    if max_points is not None and len(s) > max_points:
        idx = np.linspace(0, len(s) - 1, max_points).round().astype(int)
        s = s.iloc[idx].copy()
        x = x[idx]
        xerr = xerr[idx]

    if len(s) < min_points:
        return pd.DataFrame(columns=['MJD','x','xerr'])

    s = s.assign(x=x, xerr=xerr)
    return s[['MJD','x','xerr']]

# ------------------ HVG / DHVG / W-HVG ------------------
def hvg_edges(x: np.ndarray) -> List[Tuple[int, int]]:
    """Undirected Horizontal Visibility Graph edges; O(n^2) with early checks."""
    n = len(x)
    E: List[Tuple[int, int]] = []
    for i in range(n - 1):
        E.append((i, i + 1))
        ceiling = -np.inf
        for j in range(i + 2, n):
            ceiling = max(ceiling, x[j - 1])
            if ceiling < min(x[i], x[j]):
                E.append((i, j))
            else:
                continue
    return E

def dhvg_edges(x: np.ndarray) -> List[Tuple[int, int]]:
    """Directed HVG (edges only forward in time)."""
    undirected = hvg_edges(x)
    return [(i, j) if i < j else (j, i) for (i, j) in undirected if i != j and i < j]

def whvg_edges(x: np.ndarray,
               t: Optional[np.ndarray] = None,
               xerr: Optional[np.ndarray] = None,
               scheme: str = "delta_over_err",
               use_delta_t: bool = False,
               eps: float = 1e-12):
    """Weighted HVG: build HVG topology, attach weights by chosen scheme."""
    E = hvg_edges(x)
    w: List[float] = []
    std_global = float(np.std(x)) + eps
    for (i, j) in E:
        dij = abs(float(x[j] - x[i]))
        if scheme == "delta":
            wij = dij
        elif scheme == "delta_over_err":
            if xerr is None:
                raise ValueError("scheme='delta_over_err' requires xerr array.")
            denom = math.sqrt(float(xerr[i] ** 2 + xerr[j] ** 2)) + eps
            wij = dij / denom
        elif scheme == "zscore":
            wij = dij / std_global
        else:
            raise ValueError(f"Unknown scheme '{scheme}'")
        if use_delta_t and t is not None:
            dt = abs(float(t[j] - t[i]))
            wij = wij / (1.0 + dt)
        w.append(wij)
    return E, w

# ------------------ Features ------------------
def _tail_exponential_lambda(degrees: np.ndarray, k_min: int = 3) -> float:
    """
    Estimate lambda of P(k) ~ exp(-lambda k) for the tail k >= k_min via linear fit on log-hist.
    Degrees MUST be integers for np.bincount.
    """
    if len(degrees) == 0:
        return np.nan
    deg = degrees[degrees >= k_min].astype(int)  # ensure integer degrees
    if len(deg) < 5:
        return np.nan
    hist = np.bincount(deg)
    ks = np.arange(len(hist))
    ps = hist / hist.sum()
    mask = (ps > 0) & (ks >= k_min)
    if mask.sum() < 3:
        return np.nan
    x = ks[mask].astype(float)
    y = np.log(ps[mask])
    b = np.polyfit(x, y, 1)[0]
    return float(-b)

def features_from_edges(n: int,
                        edges: List[Tuple[int, int]],
                        weights: Optional[List[float]] = None,
                        directed: bool = False,
                        add_spectral: bool = True,
                        prefix: str = "") -> Dict[str, float]:
    """Build networkx graph and compute a compact set of features."""
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    if weights is None:
        G.add_edges_from(edges)
    else:
        for (u, v), w in zip(edges, weights):
            G.add_edge(u, v, weight=float(w))

    feats: Dict[str, float] = {}

    if directed:
        kin = np.array([d for _, d in G.in_degree()], dtype=float)
        kout = np.array([d for _, d in G.out_degree()], dtype=float)
        feats[prefix + "k_in_mean"] = float(np.mean(kin))
        feats[prefix + "k_in_std"] = float(np.std(kin))
        feats[prefix + "k_out_mean"] = float(np.mean(kout))
        feats[prefix + "k_out_std"] = float(np.std(kout))
        feats[prefix + "k_imbalance"] = float(np.mean(kout - kin))
    else:
        # integer degrees for tail fit; float copy for moments
        k_int = np.array([d for _, d in G.degree()], dtype=int)
        k = k_int.astype(float)
        feats[prefix + "k_mean"] = float(np.mean(k))
        feats[prefix + "k_std"] = float(np.std(k))
        feats[prefix + "k_skew"] = float(pd.Series(k).skew()) if len(k) > 1 else 0.0
        feats[prefix + "lambda_tail"] = _tail_exponential_lambda(k_int, k_min=3)

    # undirected view for clustering/transitivity etc.
    Gu = G.to_undirected()
    feats[prefix + "clustering_mean"] = float(np.mean(list(nx.clustering(Gu).values()))) if Gu.number_of_edges() else 0.0
    feats[prefix + "transitivity"] = float(nx.transitivity(Gu)) if Gu.number_of_edges() else 0.0

    try:
        feats[prefix + "assortativity"] = float(nx.degree_assortativity_coefficient(Gu))
    except Exception:
        feats[prefix + "assortativity"] = np.nan

    try:
        tri = nx.triangles(Gu)
        feats[prefix + "triangles_mean"] = float(np.mean(list(tri.values()))) if tri else 0.0
    except Exception:
        feats[prefix + "triangles_mean"] = np.nan

    # Efficiency on largest component (avoids inf on disconnected)
    if Gu.number_of_nodes() > 0 and Gu.number_of_edges() > 0:
        comps = [Gu.subgraph(c).copy() for c in nx.connected_components(Gu)]
        if comps:
            Gc = max(comps, key=lambda g: g.number_of_nodes())
            try:
                feats[prefix + "efficiency"] = float(nx.global_efficiency(Gc))
            except Exception:
                feats[prefix + "efficiency"] = np.nan
        else:
            feats[prefix + "efficiency"] = np.nan
    else:
        feats[prefix + "efficiency"] = np.nan

    # Weighted stats
    if weights is not None and len(weights) == len(edges):
        strengths = np.zeros(n, dtype=float)
        adj: Dict[int, List[float]] = {i: [] for i in range(n)}
        for (u, v), w in zip(edges, weights):
            strengths[u] += w
            if not directed:
                strengths[v] += w
            adj[u].append(w); adj[v].append(w)
        feats[prefix + "strength_mean"] = float(np.mean(strengths))
        feats[prefix + "strength_std"] = float(np.std(strengths))
        Y = np.full(n, np.nan, dtype=float)
        for i in range(n):
            if len(adj[i]):
                s = sum(adj[i])
                Y[i] = sum(ww * ww for ww in adj[i]) / (s * s + 1e-12)
        feats[prefix + "disparity_mean"] = float(np.nanmean(Y))
        feats[prefix + "disparity_std"] = float(np.nanstd(Y))

    # Spectral (normalized Laplacian) if graph has >1 node and at least one edge
    if add_spectral and Gu.number_of_nodes() > 1 and Gu.number_of_edges() > 0:
        try:
            L = nx.normalized_laplacian_matrix(Gu).todense()
            evals = np.linalg.eigvalsh(L)
            evals = np.sort(np.real(evals))
            feats[prefix + "eig_fiedler"] = float(evals[1]) if len(evals) > 1 else np.nan
            feats[prefix + "eig_max"] = float(evals[-1])
        except Exception:
            feats[prefix + "eig_fiedler"] = np.nan
            feats[prefix + "eig_max"] = np.nan

    return feats

# ------------------ Batch builder ------------------
def build_feature_table(labels: pd.DataFrame,
                        light: pd.DataFrame,
                        min_instances: int = 40,
                        max_points: Optional[int] = 600,
                        use_flux: bool = True,
                        winsor: float = 0.0,
                        whvg_scheme: str = "delta_over_err",
                        use_delta_t_in_w: bool = False
                        ) -> pd.DataFrame:
    """
    Iterate over IDs present in `labels` (Instances >= min_instances),
    compute HVG, DHVG, W-HVG features for each LC.
    Supports labels column name 'ID' (preferred) or 'TransientID'.
    """
    labels = labels.copy()
    light = light.copy()

    # detect label ID column
    label_id_col = 'ID' if 'ID' in labels.columns else ('TransientID' if 'TransientID' in labels.columns else None)
    if label_id_col is None:
        raise KeyError("labels must contain an 'ID' column (or 'TransientID').")

    class_col = 'Classification' if 'Classification' in labels.columns else None
    if class_col is None:
        raise KeyError("labels must contain a 'Classification' column.")

    # Normalise types for join/filter (string compare is robust)
    labels[label_id_col] = labels[label_id_col].astype(str)
    light['ID'] = light['ID'].astype(str)

    # candidates by instance threshold
    ids = (labels.query('Instances >= @min_instances')[label_id_col]
           .drop_duplicates().tolist())

    rows = []
    for id_ in ids:
        s = prepare_series(light, id_, use_flux=use_flux,
                           min_points=min_instances,
                           winsor=winsor, max_points=max_points)
        if s.empty:
            continue

        x = s['x'].to_numpy(dtype=float)
        t = s['MJD'].to_numpy(dtype=float)
        xerr = s['xerr'].to_numpy(dtype=float)

        # HVG
        E_hvg = hvg_edges(x)
        feats_hvg = features_from_edges(len(x), E_hvg, directed=False, prefix="hvg_")

        # DHVG
        E_dhvg = dhvg_edges(x)
        feats_dhvg = features_from_edges(len(x), E_dhvg, directed=True, prefix="dhvg_")

        # W-HVG
        E_whvg, W_whvg = whvg_edges(x, t=t, xerr=xerr, scheme=whvg_scheme,
                                    use_delta_t=use_delta_t_in_w)
        feats_whvg = features_from_edges(len(x), E_whvg, weights=W_whvg,
                                         directed=False, prefix="whvg_")

        row = {'ID': id_}
        row.update(feats_hvg)
        row.update(feats_dhvg)
        row.update(feats_whvg)

        # attach class
        lab = labels.loc[labels[label_id_col] == id_, class_col]
        row['Classification'] = lab.iloc[0] if len(lab) else None

        rows.append(row)

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return feat_df

    cols = ['ID', 'Classification'] + [c for c in feat_df.columns if c not in ('ID', 'Classification')]
    feat_df = feat_df[cols]
    return feat_df
