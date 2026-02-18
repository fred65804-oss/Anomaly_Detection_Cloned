"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class NetworkTrafficInput(BaseModel):
    """
    Single network traffic sample input.
    Supports both KDD Cup 1999 and UNSW-NB15 feature sets.
    All fields are optional - send only the features your dataset has.
    The pipeline will use whichever features are present.
    """
    # ── UNSW-NB15 features ────────────────────────────────────────────────────
    # Column names vary between UNSW releases:
    #   UNSW-NB15_Dataset2.csv  →  lowercase (spkts, sload, sjit, sinpkt, smean, response_body_len)
    #   UNSW-NB15.csv (full)    →  mixed-case (Spkts, Sload, Sjit, Sintpkt) + renamed (smeansz, res_bdy_len)
    #   Both variants are accepted here.

    # Categorical (UNSW-NB15.csv only)
    proto: Optional[str] = Field(None, description="Transaction protocol (tcp, udp, arp, …)")
    state: Optional[str] = Field(None, description="Transaction state (FIN, CON, INT, …)")

    # Shared timing / volume fields
    dur: Optional[float] = Field(None, description="Record total duration")

    # Packet counts – lowercase variant (Dataset2)
    spkts: Optional[int] = Field(None, description="Source-to-destination packet count")
    dpkts: Optional[int] = Field(None, description="Destination-to-source packet count")
    # Packet counts – capitalised variant (UNSW-NB15.csv)
    Spkts: Optional[int] = Field(None, description="Source-to-destination packet count (UNSW-NB15.csv)")
    Dpkts: Optional[int] = Field(None, description="Destination-to-source packet count (UNSW-NB15.csv)")

    sbytes: Optional[int] = Field(None, description="Source-to-destination bytes")
    dbytes: Optional[int] = Field(None, description="Destination-to-source bytes")
    rate: Optional[float] = Field(None, description="Transfer rate")
    sttl: Optional[int] = Field(None, description="Source-to-destination TTL")
    dttl: Optional[int] = Field(None, description="Destination-to-source TTL")

    # Load – lowercase variant (Dataset2)
    sload: Optional[float] = Field(None, description="Source bits per second")
    dload: Optional[float] = Field(None, description="Destination bits per second")
    # Load – capitalised variant (UNSW-NB15.csv)
    Sload: Optional[float] = Field(None, description="Source bits per second (UNSW-NB15.csv)")
    Dload: Optional[float] = Field(None, description="Destination bits per second (UNSW-NB15.csv)")

    sloss: Optional[int] = Field(None, description="Source packets retransmitted/dropped")
    dloss: Optional[int] = Field(None, description="Destination packets retransmitted/dropped")

    # Inter-packet arrival – lowercase variant (Dataset2: sinpkt/dinpkt)
    sinpkt: Optional[float] = Field(None, description="Source inter-packet arrival time (ms)")
    dinpkt: Optional[float] = Field(None, description="Destination inter-packet arrival time (ms)")
    # Inter-packet arrival – capitalised variant (UNSW-NB15.csv: Sintpkt/Dintpkt)
    Sintpkt: Optional[float] = Field(None, description="Source inter-packet arrival time ms (UNSW-NB15.csv)")
    Dintpkt: Optional[float] = Field(None, description="Destination inter-packet arrival time ms (UNSW-NB15.csv)")

    # Jitter – lowercase variant (Dataset2)
    sjit: Optional[float] = Field(None, description="Source jitter (ms)")
    djit: Optional[float] = Field(None, description="Destination jitter (ms)")
    # Jitter – capitalised variant (UNSW-NB15.csv)
    Sjit: Optional[float] = Field(None, description="Source jitter ms (UNSW-NB15.csv)")
    Djit: Optional[float] = Field(None, description="Destination jitter ms (UNSW-NB15.csv)")

    swin: Optional[int] = Field(None, description="Source TCP window advertisement value")
    stcpb: Optional[int] = Field(None, description="Source TCP base sequence number")
    dtcpb: Optional[int] = Field(None, description="Destination TCP base sequence number")
    dwin: Optional[int] = Field(None, description="Destination TCP window advertisement value")
    tcprtt: Optional[float] = Field(None, description="TCP connection setup round-trip time")
    synack: Optional[float] = Field(None, description="TCP connection setup time (SYN to SYN_ACK)")
    ackdat: Optional[float] = Field(None, description="TCP connection setup time (SYN_ACK to ACK)")

    # Mean packet size – lowercase (Dataset2: smean/dmean)
    smean: Optional[int] = Field(None, description="Mean of source packet size")
    dmean: Optional[int] = Field(None, description="Mean of destination packet size")
    # Mean packet size – renamed (UNSW-NB15.csv: smeansz/dmeansz)
    smeansz: Optional[int] = Field(None, description="Mean source packet size (UNSW-NB15.csv)")
    dmeansz: Optional[int] = Field(None, description="Mean destination packet size (UNSW-NB15.csv)")

    trans_depth: Optional[int] = Field(None, description="HTTP request/response transaction depth")

    # Response body length – lowercase (Dataset2: response_body_len)
    response_body_len: Optional[int] = Field(None, description="Actual uncompressed content size of HTTP response body")
    # Response body length – renamed (UNSW-NB15.csv: res_bdy_len)
    res_bdy_len: Optional[int] = Field(None, description="HTTP response body length (UNSW-NB15.csv)")

    ct_srv_src: Optional[int] = Field(None, description="Connections to same service and source address in last 100")
    ct_state_ttl: Optional[int] = Field(None, description="Count of connections with same state and TTL")
    ct_dst_ltm: Optional[int] = Field(None, description="Connections to same destination in last 100")
    ct_src_dport_ltm: Optional[int] = Field(None, description="Connections from same source to same dest port in last 100")
    ct_dst_sport_ltm: Optional[int] = Field(None, description="Connections to same destination from same source port in last 100")
    ct_dst_src_ltm: Optional[int] = Field(None, description="Connections to same destination and source in last 100")
    is_ftp_login: Optional[int] = Field(None, description="1 if FTP login, 0 otherwise")
    ct_ftp_cmd: Optional[int] = Field(None, description="Number of FTP commands in an FTP session")
    ct_flw_http_mthd: Optional[int] = Field(None, description="Number of HTTP methods in last 100 connections")
    ct_src_ltm: Optional[int] = Field(None, description="Connections from same source in last 100")
    ct_srv_dst: Optional[int] = Field(None, description="Connections to same service and destination in last 100")
    is_sm_ips_ports: Optional[int] = Field(None, description="1 if source and destination IP and ports are equal, 0 otherwise")

    # ── KDD Cup 1999 features ─────────────────────────────────────────────────
    duration: Optional[float] = Field(None, description="Connection duration in seconds")
    protocol_type: Optional[str] = Field(None, description="Protocol type (tcp, udp, icmp)")
    service: Optional[str] = Field(None, description="Network service (http, ftp, etc.)")
    flag: Optional[str] = Field(None, description="Connection flag status")
    src_bytes: Optional[float] = Field(None, description="Source to destination bytes")
    dst_bytes: Optional[float] = Field(None, description="Destination to source bytes")
    land: Optional[int] = Field(None, description="1 if connection is from/to same host/port")
    wrong_fragment: Optional[int] = Field(None, description="Number of wrong fragments")
    urgent: Optional[int] = Field(None, description="Number of urgent packets")
    hot: Optional[int] = Field(None, description="Number of hot indicators")
    num_failed_logins: Optional[int] = Field(None, description="Number of failed login attempts")
    logged_in: Optional[int] = Field(None, description="1 if successfully logged in, 0 otherwise")
    num_compromised: Optional[int] = Field(None, description="Number of compromised conditions")
    root_shell: Optional[int] = Field(None, description="1 if root shell obtained, 0 otherwise")
    su_attempted: Optional[int] = Field(None, description="1 if su command attempted, 0 otherwise")
    num_root: Optional[int] = Field(None, description="Number of root accesses")
    num_file_creations: Optional[int] = Field(None, description="Number of file creation operations")
    num_shells: Optional[int] = Field(None, description="Number of shell prompts")
    num_access_files: Optional[int] = Field(None, description="Number of operations on access control files")
    num_outbound_cmds: Optional[int] = Field(None, description="Number of outbound commands")
    is_host_login: Optional[int] = Field(None, description="1 if login belongs to host list, 0 otherwise")
    is_guest_login: Optional[int] = Field(None, description="1 if login is guest, 0 otherwise")
    count: Optional[int] = Field(None, description="Number of connections to same host")
    srv_count: Optional[int] = Field(None, description="Number of connections to same service")
    serror_rate: Optional[float] = Field(None, description="SYN error rate")
    srv_serror_rate: Optional[float] = Field(None, description="Service SYN error rate")
    rerror_rate: Optional[float] = Field(None, description="REJ error rate")
    srv_rerror_rate: Optional[float] = Field(None, description="Service REJ error rate")
    same_srv_rate: Optional[float] = Field(None, description="Connections to same service rate")
    diff_srv_rate: Optional[float] = Field(None, description="Connections to different services rate")
    srv_diff_host_rate: Optional[float] = Field(None, description="Service connections to different hosts rate")
    dst_host_count: Optional[int] = Field(None, description="Destination host count")
    dst_host_srv_count: Optional[int] = Field(None, description="Destination host service count")
    dst_host_same_srv_rate: Optional[float] = Field(None, description="Destination host same service rate")
    dst_host_diff_srv_rate: Optional[float] = Field(None, description="Destination host different service rate")
    dst_host_same_src_port_rate: Optional[float] = Field(None, description="Destination host same source port rate")
    dst_host_srv_diff_host_rate: Optional[float] = Field(None, description="Destination host service different host rate")
    dst_host_serror_rate: Optional[float] = Field(None, description="Destination host SYN error rate")
    dst_host_srv_serror_rate: Optional[float] = Field(None, description="Destination host service SYN error rate")
    dst_host_rerror_rate: Optional[float] = Field(None, description="Destination host REJ error rate")
    dst_host_srv_rerror_rate: Optional[float] = Field(None, description="Destination host service REJ error rate")

    def to_dict(self):
        """Return only non-None fields so pipeline sees only the features that were sent"""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    class Config:
        json_schema_extra = {
            "example": {
                "dur": 0.121478,
                "spkts": 6,
                "dpkts": 4,
                "sbytes": 258,
                "dbytes": 172,
                "rate": 74.105,
                "sttl": 64,
                "dttl": 128
            }
        }




class PredictionOutput(BaseModel):
    """Single prediction output"""
    is_intrusion: bool = Field(..., description="True if intrusion detected, False if normal")
    confidence: float = Field(..., description="Confidence score (0-1)")
    intrusion_probability: float = Field(..., description="Probability of intrusion (0-1)")
    alert_level:str = Field(..., description = "Alert severity: CRITICAL, HIGH, MEDIUM, LOW or NORMAL")
    alert_message:str = Field(..., description = "Human-readable alert message")

    class Config:
        json_schema_extra = {
            "example": {
                "is_intrusion": True,
                "confidence": 0.87,
                "intrusion_probability": 0.87,
                "alert_level": "HIGH",
                "alert_message": "High confidence intrusion detected - Immediate investigation recommended"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    samples: List[NetworkTrafficInput] = Field(..., description="List of network traffic samples")
    
    class Config:
        json_schema_extra = {
            "example": {
                "samples": [
                    {
                        "duration": 0.0,
                        "protocol_type": "tcp",
                        "service": "http",
                        "flag": "SF",
                        "src_bytes": 181.0,
                        "dst_bytes": 5450.0,
                        "land": 0,
                        "wrong_fragment": 0,
                        "urgent": 0,
                        "count": 8,
                        "srv_count": 8,
                        "same_srv_rate": 1.0,
                        "diff_srv_rate": 0.0,
                        "srv_diff_host_rate": 0.0,
                        "dst_host_count": 9,
                        "dst_host_srv_count": 9,
                        "dst_host_same_srv_rate": 1.0,
                        "dst_host_diff_srv_rate": 0.0,
                        "dst_host_same_src_port_rate": 0.11,
                        "dst_host_srv_diff_host_rate": 0.0
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")
    intrusions_detected: int = Field(..., description="Number of intrusions detected")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"is_intrusion": True, "confidence": 0.87, "intrusion_probability": 0.87}
                ],
                "count": 1,
                "intrusions_detected": 1
            }
        }


class ModelInfo(BaseModel):
    """Model metadata"""
    version: str = Field(..., description="Model version")
    threshold: float = Field(..., description="Classification threshold")
    supervised_weight: float = Field(..., description="Supervised model weight in ensemble")
    ensemble_method: str = Field(..., description="Ensemble method used")
    use_autoencoder: bool = Field(..., description="Whether autoencoder is used")
    use_pca: bool = Field(..., description="Whether PCA is used")
    anomaly_detectors: List[str] = Field(..., description="List of anomaly detectors")
    num_features: int = Field(..., description="Number of features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "version": "latest",
                "threshold": 0.45,
                "supervised_weight": 0.35,
                "ensemble_method": "max",
                "use_autoencoder": True,
                "use_pca": True,
                "anomaly_detectors": ["isolation_forest", "lof"],
                "num_features": 121
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Model version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "latest"
            }
        }
