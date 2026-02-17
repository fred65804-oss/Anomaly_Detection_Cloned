# Hybrid Intrusion Detection System (IDS)

A production-ready **Hybrid Intrusion Detection System** that combines supervised and unsupervised machine learning techniques to detect network intrusions, with a special focus on detecting **novel attacks** (zero-day attacks never seen during training).

## 🌟 Features

- **Hybrid Architecture**: Combines supervised (Random Forest) and unsupervised (Isolation Forest, LOF, Autoencoder) models
- **Novel Attack Detection**: Optimized to detect unseen attack patterns
- **REST API**: Production-ready FastAPI application for real-time predictions
- **Modular Design**: Clean separation of concerns with reusable components
- **Optimized Performance**: Automatic weight and threshold optimization
- **Comprehensive Evaluation**: Detailed metrics including novel attack analysis

## 📁 Project Structure

```
anomaly_ids/
├── app/                      # FastAPI Application
│   ├── main.py              # API endpoints
│   ├── schemas.py           # Request/Response models
│   └── dependencies.py      # Dependency injection
├── pipeline/                # ML Pipeline Components
│   ├── config.py           # Configuration management
│   ├── preprocessing.py    # Data preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── autoencoder.py      # Deep learning autoencoder
│   ├── anomaly_models.py   # Isolation Forest, LOF
│   ├── supervised.py       # Random Forest classifier
│   ├── normalization.py    # Score normalization
│   ├── ensemble.py         # Hybrid ensemble
│   ├── threshold.py        # Threshold optimization
│   └── pipeline.py         # Main pipeline orchestrator
├── training/               # Training Scripts
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation metrics
│   └── novel_attack_analysis.py  # Novel attack analysis
├── utils/                  # Utilities
│   ├── model_manager.py   # Model save/load
│   └── logger.py          # Logging configuration
├── artifacts/              # Saved models (created during training)
├── tests/                  # Unit tests
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd anomaly_ids

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model

```bash
# Ensure KDDTrain.csv and KDDTest.csv are in the training/ directory
cd training

# Train the hybrid IDS pipeline
python train.py
```

This will:
- Load and preprocess the KDD dataset
- Train all components (autoencoder, anomaly detectors, supervised model)
- Optimize ensemble weights and classification threshold
- Save the trained pipeline to `artifacts/latest/`

### 3. Evaluate the Model

```bash
# Evaluate on test set
python evaluate.py

# Analyze novel attack performance
python novel_attack_analysis.py
```

### 4. Run the API

```bash
# From the anomaly_ids directory
python -m app.main
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Info

```bash
curl http://localhost:8000/model/info
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 🧠 How It Works

### 1. Preprocessing
- Removes constant and highly correlated features
- One-hot encodes categorical variables (protocol, service, flag)
- Applies robust scaling (resistant to outliers)

### 2. Feature Engineering
- Creates statistical features:
  - `bytes_ratio`: src_bytes / dst_bytes
  - `total_bytes`: src_bytes + dst_bytes
  - `srv_ratio`: srv_count / count
  - `packet_rate`: count / duration
- Applies PCA for dimensionality reduction

### 3. Unsupervised Learning (Trained on Normal Traffic Only)
- **Autoencoder**: Deep learning model learns normal patterns, reconstruction error indicates anomalies
- **Isolation Forest**: Isolates outliers using random partitioning
- **Local Outlier Factor (LOF)**: Detects local density anomalies

### 4. Supervised Learning
- **Random Forest**: Trained on labeled data (normal vs intrusion)
- Uses autoencoder-encoded features for better generalization

### 5. Hybrid Ensemble
- Combines supervised and unsupervised scores
- **MAX strategy**: If ANY detector flags it, it's suspicious
- Optimized weights: ~30% supervised, ~70% unsupervised (optimized on validation)

### 6. Threshold Optimization
- Classification threshold optimized for **recall** (minimize false negatives)
- Validated on held-out dataset

## 📊 Performance

The hybrid model significantly outperforms supervised-only approaches, especially on **novel attacks**:

- **Novel Attack Recall**: ~15-20% improvement over supervised-only
- **Overall F1 Score**: Balanced precision and recall
- **False Negative Reduction**: Detects more attacks with minimal false positive increase

## ⚙️ Configuration

Modify `pipeline/config.py` or pass a config dictionary to customize:

```python
from pipeline import HybridIDSPipeline, IDSConfig

config = IDSConfig({
    'use_autoencoder': True,
    'use_pca_features': True,
    'anomaly_detectors': ['isolation_forest', 'lof'],
    'supervised_weight': 0.3,
    'ensemble_method': 'max',
    'optimize_weights': True,
    
    # Autoencoder settings
    'ae_encoding_dim': 32,
    'ae_epochs': 20,
    'ae_batch_size': 256,
    
    # Anomaly detector settings
    'iso_n_estimators': 300,
    'iso_contamination': 0.30,
    'lof_n_neighbors': 10,
    
    # Supervised model settings
    'rf_n_estimators': 100,
    'rf_max_depth': 20,
})

pipeline = HybridIDSPipeline(config)
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_pipeline.py
```

## 📦 Model Versioning

Models are saved with versions in the `artifacts/` directory:

```python
from utils import ModelManager

manager = ModelManager("artifacts")

# Save pipeline with version
manager.save_pipeline(pipeline, version="v1.0")

# Load specific version
pipeline = manager.load_pipeline(version="v1.0")

# List all versions
versions = manager.list_versions()
```

## 🔧 Development

### Adding New Anomaly Detectors

1. Create detector class in `pipeline/anomaly_models.py`
2. Add detector name to config
3. Update pipeline to initialize and use the detector

### Customizing Ensemble Strategy

Modify `pipeline/ensemble.py` to implement new combination strategies (e.g., weighted average, voting).

## 📝 License

This project is part of research on hybrid intrusion detection systems.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Built with**: Python • FastAPI • Scikit-learn • TensorFlow • Keras
