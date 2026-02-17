"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class NetworkTrafficInput(BaseModel):
    """Single network traffic sample input"""
    # Features can added or removed
    duration: float = Field(..., description="Connection duration in seconds")
    protocol_type: str = Field(..., description="Protocol type (tcp, udp, icmp)")
    service: str = Field(..., description="Network service (http, ftp, etc.)")
    flag: str = Field(..., description="Connection flag status")
    src_bytes: float = Field(..., description="Source to destination bytes")
    dst_bytes: float = Field(..., description="Destination to source bytes")
    land: int = Field(..., description="Land flag (1 if connection is from/to same host/port)")
    wrong_fragment: int = Field(..., description="Number of wrong fragments")
    urgent: int = Field(..., description="Number of urgent packets")
    hot: int = Field(0, description="Number of 'hot' indicators")
    num_failed_logins: int = Field(0, description="Number of failed login attempts")
    logged_in: int = Field(0, description="1 if successfully logged in, 0 otherwise")
    num_compromised: int = Field(0, description="Number of compromised conditions")
    root_shell: int = Field(0, description="1 if root shell obtained, 0 otherwise")
    su_attempted: int = Field(0, description="1 if su command attempted, 0 otherwise")
    num_root: int = Field(0, description="Number of root accesses")
    num_file_creations: int = Field(0, description="Number of file creation operations")
    num_shells: int = Field(0, description="Number of shell prompts")
    num_access_files: int = Field(0, description="Number of operations on access control files")
    num_outbound_cmds: int = Field(0, description="Number of outbound commands")
    is_host_login: int = Field(0, description="1 if login belongs to host list, 0 otherwise")
    is_guest_login: int = Field(0, description="1 if login is guest, 0 otherwise")
    count: int = Field(..., description="Number of connections to same host")
    srv_count: int = Field(..., description="Number of connections to same service")
    serror_rate: float = Field(0.0, description="SYN error rate")
    srv_serror_rate: float = Field(0.0, description="Service SYN error rate")
    rerror_rate: float = Field(0.0, description="REJ error rate")
    srv_rerror_rate: float = Field(0.0, description="Service REJ error rate")
    same_srv_rate: float = Field(..., description="Connections to same service rate")
    diff_srv_rate: float = Field(..., description="Connections to different services rate")
    srv_diff_host_rate: float = Field(..., description="Service connections to different hosts rate")
    dst_host_count: int = Field(..., description="Destination host count")
    dst_host_srv_count: int = Field(..., description="Destination host service count")
    dst_host_same_srv_rate: float = Field(..., description="Destination host same service rate")
    dst_host_diff_srv_rate: float = Field(..., description="Destination host different service rate")
    dst_host_same_src_port_rate: float = Field(..., description="Destination host same source port rate")
    dst_host_srv_diff_host_rate: float = Field(..., description="Destination host service different host rate")
    dst_host_serror_rate: float = Field(0.0, description="Destination host SYN error rate")
    dst_host_srv_serror_rate: float = Field(0.0, description="Destination host service SYN error rate")
    dst_host_rerror_rate: float = Field(0.0, description="Destination host REJ error rate")
    dst_host_srv_rerror_rate: float = Field(0.0, description="Destination host service REJ error rate")
    
    class Config:
        json_schema_extra = {
            "example": {
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
