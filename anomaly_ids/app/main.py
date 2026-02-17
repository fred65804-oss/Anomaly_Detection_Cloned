"""
FastAPI Application for Hybrid IDS
"""

import sys
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import (
    NetworkTrafficInput,
    PredictionOutput,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse
)
from app.dependencies import get_pipeline, reload_pipeline

# Function to detemine the alert level and its respective message
def determine_alert_level(probability: float, is_intrusion: bool) -> tuple:
    """
        Determine alert level and message based on intrusion probability.
    
        Args:
            probability: Intrusion Intrusion Probability(0-1)
            is_intrusion: Whether the prediction was an attack or not

        Returns:
            Tuple of (alert_level, alert_message)

        Alert Levels:
            CRITICAL: prob>=0.9 (Very high confidence attack)
            HIGH: prob >= 0.7 (High confidence attack)
            MEDIUM: prob >= 0.5 (Suspicious activity)
            LOW: prob >= 0.40 (Could be an attack - Monitoring required)
            NORMAL: prob < 0.40 (Normal Traffic)
    """

    if probability >= 0.9:
        return ("CRITICAL", "Critical threat detected with very high confidence - Immediate action required")

    elif probability >= 0.7:
        return ("HIGH", "Intrusion detected with high confidence rate: Proceed with caution")

    elif probability >= 0.5:
        return ("MEDIUM", "Suspicious activity detected - Action recommended")

    elif probability >= 0.40:
        return ("LOW", "Borderline Anomaly detected: Close monitoring required")

    else:
        # First case (Probability is low and 'is_intrusion' is also True)
        if is_intrusion: # If it is True
            return ("LOW", "Low confidence anomaly - May be it could be a False Anomaly")
        else:
            return ("NORMAL", "No intrusion detected - Normal Traffic")


# Create FastAPI app
app = FastAPI(
    title="Hybrid IDS API",
    description="Hybrid Intrusion Detection System combining supervised and unsupervised learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        pipeline = get_pipeline()
        print(f"✓ Pipeline loaded successfully")
        print(f"  Threshold: {pipeline.threshold_optimizer.best_threshold:.3f}")
        print(f"  Supervised weight: {pipeline.config.supervised_weight:.3f}")
    except Exception as e:
        print(f"✗ Error loading pipeline: {e}")
        print("  API will start but predictions will fail until model is loaded")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Hybrid IDS API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns status of the API and whether model is loaded
    """
    try:
        pipeline = get_pipeline()
        model_loaded = pipeline.fitted
        version = "latest"
    except Exception:
        model_loaded = False
        version = "unknown"
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=version
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(pipeline=Depends(get_pipeline)):
    """
    Get model metadata and configuration
    
    Returns information about the loaded model
    """
    try:
        return ModelInfo(
            version="latest",
            threshold=pipeline.threshold_optimizer.best_threshold,
            supervised_weight=pipeline.config.supervised_weight,
            ensemble_method=pipeline.config.ensemble_method,
            use_autoencoder=pipeline.config.use_autoencoder,
            use_pca=pipeline.config.use_pca_features,
            anomaly_detectors=pipeline.config.anomaly_detectors,
            num_features=len(pipeline.feature_names) if pipeline.feature_names else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_single(
    sample: NetworkTrafficInput,
    pipeline=Depends(get_pipeline)
):
    """
    Predict intrusion for a single network traffic sample
    
    Returns prediction with confidence score
    """
    try:
        # Convert to DataFrame
        sample_dict = sample.model_dump()
        df = pd.DataFrame([sample_dict])
        
        # Get prediction
        prob = pipeline.predict_proba(df)[0]
        is_intrusion = pipeline.predict(df)[0]

        # As soon as we capture the probability and its respective 'is_intrusion', we will generate an alert message based on them
        alert_level, alert_message = determine_alert_level(float(prob), bool(is_intrusion))

        return PredictionOutput(
            is_intrusion=bool(is_intrusion),
            # 'Confidence' is none other than the raw probability
            confidence=float(prob), 
            intrusion_probability=float(prob),
            alert_level = alert_level,
            alert_message = alert_message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    pipeline=Depends(get_pipeline)
):
    """
    Predict intrusions for multiple network traffic samples
    
    Returns list of predictions
    """
    try:
        # Convert to DataFrame
        samples_dicts = [sample.model_dump() for sample in request.samples]
        df = pd.DataFrame(samples_dicts)
        
        # Get predictions
        probs = pipeline.predict_proba(df)
        preds = pipeline.predict(df)
        
        # Create response with alert messages
        predictions = []

        for pred, prob in zip(preds, probs):
            # Capture alert_level and alert_message
            alert_level, alert_message = determine_alert_level(float(prob), bool(pred))
            predictions.append(
                PredictionOutput(
                    is_intrusion = bool(pred),
                    confidence = float(prob),
                    intrusion_probability = float(prob),
                    alert_level = alert_level,
                    alert_message = alert_message
                )
            )
        
        intrusions_detected = int(preds.sum())
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            intrusions_detected=intrusions_detected
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/model/reload", tags=["Model"])
async def reload_model(version: str = "latest"):
    """
    Reload model from disk
    
    Useful for deploying updated models without restarting the API
    """
    try:
        pipeline = reload_pipeline(version)
        return {
            "message": f"Model version '{version}' loaded successfully",
            "threshold": pipeline.threshold_optimizer.best_threshold,
            "supervised_weight": pipeline.config.supervised_weight
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid IDS API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    args = parser.parse_args()
    
    # Run the API
    print(f"Starting Hybrid IDS API on port {args.port}...")
    print(f"API will be available at: http://localhost:{args.port}")
    print(f"API documentation at: http://localhost:{args.port}/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

