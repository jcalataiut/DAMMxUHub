import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import data loaders to reuse helper functions
try:
    from .data_loaders import infer_sku_format, infer_units_per_case
except ImportError:
    from data_loaders import infer_sku_format, infer_units_per_case

class SophisticatedGraphModel:
    """
    Sophisticated Graph Model for transition cost prediction.
    Builds a SKU similarity graph and propagates observed OEE degradation and
    changeover durations to estimate transition parameters for any pair of SKUs.
    """
    def __init__(
        self,
        df_oee: pd.DataFrame,
        df_volumen: Optional[pd.DataFrame] = None,
        similarity_weights: Optional[Dict[str, float]] = None
    ):
        self.df_oee = df_oee
        self.df_volumen = df_volumen
        
        # Default similarity weights
        self.sim_weights = similarity_weights or {
            "format": 0.35,
            "brand": 0.25,
            "envase": 0.25,
            "units_per_case": 0.15
        }
        
        self.sku_properties: Dict[str, Dict] = {}
        self._extract_sku_properties()
        self.similarity_matrix: Optional[pd.DataFrame] = None
        self.similarity_graph: Optional[nx.Graph] = None
        
    def _extract_sku_properties(self):
        """Extract SKU properties from OEE history and pre-compute attributes."""
        # Clean dataframe to make sure columns exist
        df = self.df_oee.copy()
        
        # Extract properties by taking the most frequent value per SKU
        cols_to_extract = ["marca", "envase", "supramarca", "packaging_primario", "packaging_secundario"]
        existing_cols = [c for c in cols_to_extract if c in df.columns]
        
        grouped = df.groupby("sku")
        
        # Build base properties for historical SKUs
        for sku, group in grouped:
            props = {"sku": sku}
            props["format"] = infer_sku_format(sku)
            props["units_per_case"] = infer_units_per_case(sku)
            
            for col in existing_cols:
                vals = group[col].dropna()
                props[col] = vals.mode().iloc[0] if not vals.empty else "UNKNOWN"
            
            # Fill missing cols
            for col in cols_to_extract:
                if col not in props:
                    props[col] = "UNKNOWN"
            
            self.sku_properties[sku] = props

    def get_or_create_sku_properties(self, sku: str) -> Dict:
        """Get properties for a SKU, or infer them dynamically if new."""
        if sku in self.sku_properties:
            return self.sku_properties[sku]
            
        # Dynamically infer properties for a new SKU
        format_val = infer_sku_format(sku)
        units_val = infer_units_per_case(sku)
        
        # Guess brand and packaging from naming conventions if possible
        # Or search for other SKUs with similar prefixes
        brand_guess = "UNKNOWN"
        envase_guess = "UNKNOWN"
        
        for k, v in self.sku_properties.items():
            if k[:3] == sku[:3]:
                brand_guess = v.get("marca", "UNKNOWN")
                envase_guess = v.get("envase", "UNKNOWN")
                break
                
        props = {
            "sku": sku,
            "format": format_val,
            "units_per_case": units_val,
            "marca": brand_guess,
            "envase": envase_guess,
            "supramarca": "UNKNOWN",
            "packaging_primario": "UNKNOWN",
            "packaging_secundario": "UNKNOWN"
        }
        self.sku_properties[sku] = props
        return props

    def compute_sku_similarity(self, sku1: str, sku2: str) -> float:
        """Compute the weighted similarity between two SKUs based on their properties."""
        if sku1 == sku2:
            return 1.0
            
        p1 = self.get_or_create_sku_properties(sku1)
        p2 = self.get_or_create_sku_properties(sku2)
        
        sim = 0.0
        
        # Can Format (1/2, 1/3, 2/5)
        if p1.get("format") == p2.get("format") and p1.get("format") != "unknown":
            sim += self.sim_weights["format"]
            
        # Brand / Marca
        if p1.get("marca") == p2.get("marca") and p1.get("marca") != "UNKNOWN":
            sim += self.sim_weights["brand"]
            
        # Envase (packaging format)
        if p1.get("envase") == p2.get("envase") and p1.get("envase") != "UNKNOWN":
            sim += self.sim_weights["envase"]
            
        # Units per case
        if p1.get("units_per_case") == p2.get("units_per_case") and p1.get("units_per_case", 0) > 0:
            sim += self.sim_weights["units_per_case"]
            
        return sim

    def build_similarity_graph(self, skus: List[str], threshold: float = 0.25) -> nx.Graph:
        """Build the SKU similarity network G_sim."""
        G = nx.Graph()
        n = len(skus)
        
        # Add all nodes first
        for sku in skus:
            props = self.get_or_create_sku_properties(sku)
            G.add_node(sku, **props)
            
        # Compute pairwise similarities and add edges above threshold
        sim_records = []
        for i in range(n):
            u = skus[i]
            for j in range(i, n):
                v = skus[j]
                sim = self.compute_sku_similarity(u, v)
                if i != j and sim >= threshold:
                    G.add_edge(u, v, weight=sim)
                    
        self.similarity_graph = G
        return G

    def smooth_transition_matrices(
        self,
        line: str,
        skus: List[str],
        matrices: Dict,
        min_reliable_count: int = 2,
        epsilon: float = 1e-4
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies bilinear graph-regularized kernel smoothing to rebuild
        the OEE degradation and changeover duration transition matrices for a line.
        """
        # Ensure all SKUs are in our properties database
        for sku in skus:
            self.get_or_create_sku_properties(sku)
            
        # Build Similarity Graph for our SKU subset
        self.build_similarity_graph(skus, threshold=0.1)
        
        # Initialize output DataFrames
        smoothed_oee = pd.DataFrame(np.nan, index=skus, columns=skus, dtype=float)
        smoothed_co = pd.DataFrame(np.nan, index=skus, columns=skus, dtype=float)
        
        # Diagonal is always 0.0 (no cost or duration for same product)
        np.fill_diagonal(smoothed_oee.values, 0.0)
        np.fill_diagonal(smoothed_co.values, 0.0)
        
        if line not in matrices:
            # Fallback to defaults if line has no historical data
            return smoothed_oee.fillna(0.0), smoothed_co.fillna(1.5)
            
        raw_oee = matrices[line].get("oee_degradation")
        raw_co = matrices[line].get("changeover_h")
        raw_counts = matrices[line].get("count")
        
        # Build list of reliable observed transitions (source -> dest) from 2025
        observed_transitions = []
        if raw_oee is not None and raw_counts is not None:
            for u_obs in raw_oee.index:
                for v_obs in raw_oee.columns:
                    if u_obs == v_obs:
                        continue
                    cnt = raw_counts.loc[u_obs, v_obs] if u_obs in raw_counts.index and v_obs in raw_counts.columns else 0
                    oee_val = raw_oee.loc[u_obs, v_obs]
                    co_val = raw_co.loc[u_obs, v_obs] if raw_co is not None and u_obs in raw_co.index and v_obs in raw_co.columns else np.nan
                    
                    if cnt > 0 and pd.notna(oee_val):
                        observed_transitions.append({
                            "from": u_obs,
                            "to": v_obs,
                            "oee_degradation": oee_val,
                            "changeover_h": co_val if pd.notna(co_val) else 1.5,
                            "count": cnt
                        })
                        
        # Get line-wide default priors for fallback
        oee_vals = [t["oee_degradation"] for t in observed_transitions]
        co_vals = [t["changeover_h"] for t in observed_transitions if pd.notna(t["changeover_h"])]
        
        line_oee_prior = float(np.mean(oee_vals)) if oee_vals else 0.05
        line_co_prior = float(np.median(co_vals)) if co_vals else 1.5
        
        # Smooth each pair (target_u -> target_v)
        for u in skus:
            for v in skus:
                if u == v:
                    continue
                    
                # 1. If we have reliable direct history, use it
                direct_cnt = 0
                if raw_counts is not None and u in raw_counts.index and v in raw_counts.columns:
                    direct_cnt = raw_counts.loc[u, v]
                    
                if direct_cnt >= min_reliable_count:
                    smoothed_oee.loc[u, v] = float(raw_oee.loc[u, v])
                    smoothed_co.loc[u, v] = float(raw_co.loc[u, v]) if raw_co is not None else line_co_prior
                    continue
                    
                # 2. Otherwise, run bilinear smoothing over SKU similarity graph
                weighted_oee_sum = 0.0
                weighted_co_sum = 0.0
                weight_denominator = 0.0
                
                # We also compute destination-only similarity sums for first-tier fallback
                dest_only_oee_sum = 0.0
                dest_only_co_sum = 0.0
                dest_only_denominator = 0.0
                
                for t in observed_transitions:
                    # Similarity of origins and destinations
                    sim_origin = self.compute_sku_similarity(u, t["from"])
                    sim_dest = self.compute_sku_similarity(v, t["to"])
                    
                    # Bilinear kernel weight
                    bilinear_weight = sim_origin * sim_dest
                    
                    if bilinear_weight > 0:
                        # Blended weight with transition count to favor highly-observed historical records
                        effective_weight = bilinear_weight * np.log1p(t["count"])
                        weighted_oee_sum += effective_weight * t["oee_degradation"]
                        weighted_co_sum += effective_weight * t["changeover_h"]
                        weight_denominator += effective_weight
                        
                    # Destination similarity only
                    if sim_dest > 0:
                        effective_dest_weight = sim_dest * np.log1p(t["count"])
                        dest_only_oee_sum += effective_dest_weight * t["oee_degradation"]
                        dest_only_co_sum += effective_dest_weight * t["changeover_h"]
                        dest_only_denominator += effective_dest_weight
                        
                # Assign estimates based on smoothing hierarchy
                if weight_denominator > epsilon:
                    smoothed_oee.loc[u, v] = float(weighted_oee_sum / weight_denominator)
                    smoothed_co.loc[u, v] = float(weighted_co_sum / weight_denominator)
                elif dest_only_denominator > epsilon:
                    smoothed_oee.loc[u, v] = float(dest_only_oee_sum / dest_only_denominator)
                    smoothed_co.loc[u, v] = float(dest_only_co_sum / dest_only_denominator)
                else:
                    smoothed_oee.loc[u, v] = line_oee_prior
                    smoothed_co.loc[u, v] = line_co_prior
                    
        # Replace remaining NaNs (if any) with line-wide priors
        smoothed_oee = smoothed_oee.fillna(line_oee_prior)
        smoothed_co = smoothed_co.fillna(line_co_prior)
        
        # Post-processing: make sure changeover hours are positive and clamped to a minimum
        smoothed_co = smoothed_co.clip(lower=0.1)
        
        return smoothed_oee, smoothed_co

    def explain_estimated_transition(
        self,
        line: str,
        u: str,
        v: str,
        matrices: Dict,
        top_k: int = 5
    ) -> Dict:
        """
        Explain the prediction for a specific transition (u -> v).
        Returns a breakdown of similarities and top contributing historical transitions.
        """
        p_u = self.get_or_create_sku_properties(u)
        p_v = self.get_or_create_sku_properties(v)
        
        raw_counts = matrices[line].get("count") if line in matrices else None
        raw_oee = matrices[line].get("oee_degradation") if line in matrices else None
        raw_co = matrices[line].get("changeover_h") if line in matrices else None
        
        # Check if direct transition exists in history
        has_direct = False
        direct_cnt = 0
        if raw_counts is not None and u in raw_counts.index and v in raw_counts.columns:
            val = raw_counts.loc[u, v]
            if pd.notna(val):
                direct_cnt = int(val)
                if direct_cnt > 0:
                    has_direct = True
                
        # Re-build observed list
        observed_transitions = []
        if raw_oee is not None and raw_counts is not None:
            for u_obs in raw_oee.index:
                for v_obs in raw_oee.columns:
                    if u_obs == v_obs:
                        continue
                    cnt = raw_counts.loc[u_obs, v_obs] if u_obs in raw_counts.index and v_obs in raw_counts.columns else 0
                    oee_val = raw_oee.loc[u_obs, v_obs]
                    co_val = raw_co.loc[u_obs, v_obs] if raw_co is not None and u_obs in raw_co.index and v_obs in raw_co.columns else np.nan
                    
                    if cnt > 0 and pd.notna(oee_val):
                        observed_transitions.append({
                            "from": u_obs,
                            "to": v_obs,
                            "oee_degradation": float(oee_val),
                            "changeover_h": float(co_val) if pd.notna(co_val) else 1.5,
                            "count": int(cnt)
                        })
                        
        contributions = []
        for t in observed_transitions:
            sim_origin = self.compute_sku_similarity(u, t["from"])
            sim_dest = self.compute_sku_similarity(v, t["to"])
            bilinear_weight = sim_origin * sim_dest
            
            if bilinear_weight > 0:
                effective_weight = bilinear_weight * np.log1p(t["count"])
                contributions.append({
                    "historical_transition": f"{t['from']} -> {t['to']}",
                    "sim_origin": sim_origin,
                    "sim_dest": sim_dest,
                    "bilinear_weight": bilinear_weight,
                    "effective_weight": effective_weight,
                    "oee_degradation": t["oee_degradation"],
                    "changeover_h": t["changeover_h"],
                    "count": t["count"]
                })
                
        contributions = sorted(contributions, key=lambda x: x["effective_weight"], reverse=True)[:top_k]
        
        # Calculate resulting smoothed values
        smoothed_oee_val = 0.05
        smoothed_co_val = 1.5
        explanation_type = "Fallback a Prior"
        
        if u == v:
            smoothed_oee_val = 0.0
            smoothed_co_val = 0.0
            explanation_type = "Mismo SKU (Diagonal)"
        elif has_direct and direct_cnt >= 2:
            smoothed_oee_val = float(raw_oee.loc[u, v])
            smoothed_co_val = float(raw_co.loc[u, v]) if raw_co is not None else 1.5
            explanation_type = "Evidencia Directa Confiable"
        else:
            # Recompute smoothing to show exact prediction
            weight_denom = sum(c["effective_weight"] for c in contributions)
            if weight_denom > 1e-4:
                smoothed_oee_val = sum(c["effective_weight"] * c["oee_degradation"] for c in contributions) / weight_denom
                smoothed_co_val = sum(c["effective_weight"] * c["changeover_h"] for c in contributions) / weight_denom
                explanation_type = "Suavizado Bilinear sobre Grafo"
            else:
                # Try destination-only
                dest_contribs = []
                for t in observed_transitions:
                    sim_dest = self.compute_sku_similarity(v, t["to"])
                    if sim_dest > 0:
                        dest_contribs.append({
                            "historical_dest": t["to"],
                            "sim_dest": sim_dest,
                            "oee_degradation": t["oee_degradation"],
                            "changeover_h": t["changeover_h"],
                            "effective_weight": sim_dest * np.log1p(t["count"])
                        })
                dest_denom = sum(c["effective_weight"] for c in dest_contribs)
                if dest_denom > 1e-4:
                    smoothed_oee_val = sum(c["effective_weight"] * c["oee_degradation"] for c in dest_contribs) / dest_denom
                    smoothed_co_val = sum(c["effective_weight"] * c["changeover_h"] for c in dest_contribs) / dest_denom
                    explanation_type = "Suavizado por Destino Similar"
                else:
                    oee_vals = [t["oee_degradation"] for t in observed_transitions]
                    co_vals = [t["changeover_h"] for t in observed_transitions]
                    smoothed_oee_val = np.mean(oee_vals) if oee_vals else 0.05
                    smoothed_co_val = np.median(co_vals) if co_vals else 1.5
                    explanation_type = "Prior General de Línea"
                    
        return {
            "origin_sku": u,
            "origin_properties": p_u,
            "destination_sku": v,
            "destination_properties": p_v,
            "has_direct_history": has_direct,
            "direct_observations_count": direct_cnt,
            "explanation_type": explanation_type,
            "estimated_oee_degradation": smoothed_oee_val,
            "estimated_changeover_h": smoothed_co_val,
            "top_contributing_transitions": contributions
        }
