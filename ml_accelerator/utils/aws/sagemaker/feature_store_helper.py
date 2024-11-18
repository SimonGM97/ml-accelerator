from ml_accelerator.config.env import Env
from sagemaker.feature_store.feature_group import FeatureGroup


"""
Components of SageMaker Feature Store:
- Feature Group:
    A collection of features that describe records.
    Each feature group is mapped to a specific schema and can store both historical and real-time data.
- Online Store:
    Provides low-latency access to features for real-time model inference.
    Designed for quick lookups during inference.
- Offline Store:
    Stores a historical record of feature values.
    Useful for model training, batch inference, and analysis.
"""

