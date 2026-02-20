"""
    This file uses feature engineering to induce columns into the dataframe passed
    It also uses PCA to reduce components(number of features), by specifying variance(Default is 95%)
    Some column names have been hardcoded for a particular dataset for feature engineering purposes only 
"""
from sklearn.decomposition import PCA

# Adding Statistical Features 
def add_statistical_features(df):
    """
    Add network-specific statistical features (only if columns exist)
    
    Note: These features are designed for network traffic data (KDD dataset).
    They will be skipped if the required columns are not present. This is done so that if any other dataset is passed in future, the code will be ready to adapt to it
    """
    df = df.copy()

    # ── Normalise mixed-case / renamed UNSW-NB15 column names ─────────────────
    # UNSW-NB15.csv uses capitalised names (Sload, Dload, Spkts, Dpkts, Sjit,
    # Djit, Sintpkt, Dintpkt) and slightly different names (smeansz, res_bdy_len)
    # compared to UNSW-NB15_Dataset2.csv (sload, dload, spkts, dpkts, sjit,
    # djit, sinpkt, dinpkt, smean, response_body_len).
    # We create lowercase aliases so all downstream checks use consistent names.
    col_aliases = {
        "Sload":       "sload",
        "Dload":       "dload",
        "Spkts":       "spkts",
        "Dpkts":       "dpkts",
        "Sjit":        "sjit",
        "Djit":        "djit",
        "Sintpkt":     "sinpkt",
        "Dintpkt":     "dinpkt",
        "smeansz":     "smean",
        "dmeansz":     "dmean",
        "res_bdy_len": "response_body_len",
    }
    for src, dst in col_aliases.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    
    # ── KDD features ──────────────────────────────────────────────────────────
    # Bytes ratio (only if both src_bytes and dst_bytes exist)
    if "src_bytes" in df.columns and "dst_bytes" in df.columns:
        df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
        df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
    
    # Service ratio (only if srv_count and count exist)
    if "srv_count" in df.columns and "count" in df.columns:
        df["srv_ratio"] = df["srv_count"] / (df["count"] + 1)
    
    # Packet rate (only if count and duration exist)
    if "count" in df.columns and "duration" in df.columns:
        df["packet_rate"] = df["count"] / (df["duration"] + 1)

    # ── UNSW-NB15 features ────────────────────────────────────────────────────
    # Bytes ratio (sbytes / dbytes)
    if "sbytes" in df.columns and "dbytes" in df.columns:
        df["bytes_ratio"] = df["sbytes"] / (df["dbytes"] + 1)
        df["total_bytes"] = df["sbytes"] + df["dbytes"]

    # Packet ratio (spkts / dpkts)
    if "spkts" in df.columns and "dpkts" in df.columns:
        df["pkt_ratio"] = df["spkts"] / (df["dpkts"] + 1)
        df["total_pkts"] = df["spkts"] + df["dpkts"]

    # Bytes per packet (total bytes / total packets)
    if all(c in df.columns for c in ["sbytes", "dbytes", "spkts", "dpkts"]):
        df["bytes_per_pkt"] = (df["sbytes"] + df["dbytes"]) / (df["spkts"] + df["dpkts"] + 1)

    # Load asymmetry (sload vs dload)
    if "sload" in df.columns and "dload" in df.columns:
        df["load_asymmetry"] = abs(df["sload"] - df["dload"]) / (df["sload"] + df["dload"] + 1)

    # Jitter asymmetry (sjit vs djit)
    if "sjit" in df.columns and "djit" in df.columns:
        df["jit_asymmetry"] = abs(df["sjit"] - df["djit"]) / (df["sjit"] + df["djit"] + 1)

    return df

# Implementing additional features
def add_context_aware_features(df):
    """
    Add context-aware network traffic features (only if columns exist)
    
    Note: These features are designed for network traffic data (KDD dataset).
    They will be skipped if the required columns are not present.
    
    Args:
        df: DataFrame with existing features

    Returns:
        DataFrame with additional features (if applicable)
    """
    df = df.copy()

    # ── Normalise mixed-case / renamed UNSW-NB15 column names ─────────────────
    col_aliases = {
        "Sload":       "sload",
        "Dload":       "dload",
        "Spkts":       "spkts",
        "Dpkts":       "dpkts",
        "Sjit":        "sjit",
        "Djit":        "djit",
        "Sintpkt":     "sinpkt",
        "Dintpkt":     "dinpkt",
        "smeansz":     "smean",
        "dmeansz":     "dmean",
        "res_bdy_len": "response_body_len",
    }
    for src, dst in col_aliases.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Bytes per connection (requires: src_bytes, dst_bytes, count)
    if all(col in df.columns for col in ["src_bytes", "dst_bytes", "count"]):
        df["bytes_per_connection"] = (df["src_bytes"] + df["dst_bytes"]) / (df["count"] + 1)

    # Connection regularity score (requires: same_srv_rate, diff_srv_rate)
    if all(col in df.columns for col in ["same_srv_rate", "diff_srv_rate"]):
        df["connection_regularity"] = df["same_srv_rate"] - df["diff_srv_rate"]

    # Service diversity (requires: srv_count, count)
    if all(col in df.columns for col in ["srv_count", "count"]):
        df["service_focus"] = df["srv_count"] / (df["count"] + 1)

    # Host diversity ratio (requires: dst_host_count, count)
    if all(col in df.columns for col in ["dst_host_count", "count"]):
        df["host_diversity"] = df["dst_host_count"] / (df["count"] + 1)

    # Byte asymmetry score (requires: src_bytes, dst_bytes)
    if all(col in df.columns for col in ["src_bytes", "dst_bytes"]):
        total_bytes = df["src_bytes"] + df["dst_bytes"] + 1
        df["byte_asymmetry"] = abs(df["src_bytes"] - df["dst_bytes"]) / total_bytes

    # Total Error rate (requires: serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate)
    if all(col in df.columns for col in ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate"]):
        df["total_error_rate"] = (df["serror_rate"] + df["srv_serror_rate"] + df["rerror_rate"] + df["srv_rerror_rate"]) / 4.0

    # Duration-Normalized Packet rate (requires: count, duration)
    if all(col in df.columns for col in ["count", "duration"]):
        df["normalized_packet_rate"] = df["count"] / (df["duration"] + 1)

    # Legitimacy score (requires: logged_in, same_srv_rate, serror_rate, dst_host_same_srv_rate, land)
    if all(col in df.columns for col in ["logged_in", "same_srv_rate", "serror_rate", "dst_host_same_srv_rate", "land"]):
        # 'logged_in' status is given the most weight
        legitimacy_indicators = (
                (df["logged_in"] == 1).astype(float) * 0.3 +
                (df["same_srv_rate"] > 0.8).astype(float) * 0.2 +
                (df["serror_rate"] < 0.1).astype(float) * 0.2 + 
                (df["dst_host_same_srv_rate"] > 0.8).astype(float) * 0.15 + 
                (df["land"] == 0).astype(float) * 0.15
                                    )
        df["legitimacy_score"] = legitimacy_indicators

    # ── UNSW-NB15 context-aware features ──────────────────────────────────────
    # TCP handshake efficiency (synack + ackdat relative to tcprtt)
    if all(col in df.columns for col in ["synack", "ackdat", "tcprtt"]):
        df["handshake_efficiency"] = (df["synack"] + df["ackdat"]) / (df["tcprtt"] + 1)

    # Connection density score (how many connections to same dest/src)
    if all(col in df.columns for col in ["ct_dst_src_ltm", "ct_srv_dst"]):
        df["connection_density"] = df["ct_dst_src_ltm"] * df["ct_srv_dst"]

    # Duration-normalized byte rate (sbytes / dur)
    if "sbytes" in df.columns and "dur" in df.columns:
        df["src_byte_rate"] = df["sbytes"] / (df["dur"] + 1)

    # Response efficiency (response_body_len relative to dbytes)
    if "response_body_len" in df.columns and "dbytes" in df.columns:
        df["response_efficiency"] = df["response_body_len"] / (df["dbytes"] + 1)

    # TTL asymmetry (attacker often has different TTL than victim)
    if "sttl" in df.columns and "dttl" in df.columns:
        df["ttl_asymmetry"] = abs(df["sttl"] - df["dttl"])

    return df


# PCA
class PCATransformer:
    def __init__(self, n_components=0.95, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
    
    def fit(self, X):
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        if self.pca is None:
            raise ValueError("PCA must be fitted before transform")
        return self.pca.transform(X)

