import mlflow
import mlflow.pytorch
import mlflow.sklearn
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


class MLflowTracker:
    """
    Unified MLflow tracker for multimodal emotion recognition models.
    Supports tracking of audio, text, and video models.
    """
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "multimodal_emotion"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking URI (default: local ./mlruns folder)
            experiment_name: Name of the experiment
        """
        # Default to local storage if no URI provided
        if tracking_uri is None:
            tracking_uri = os.path.join(os.path.dirname(__file__), "..", "mlruns")
            os.makedirs(tracking_uri, exist_ok=True)
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Create experiment if it doesn't exist."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"Warning: Could not set up MLflow experiment: {e}")
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name of the run
            tags: Dictionary of tags for the run
        """
        mlflow.start_run(run_name=run_name)
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def log_model_metadata(self, model_name: str, modality: str, architecture: str, params: Dict[str, Any]):
        """
        Log model metadata and hyperparameters.
        
        Args:
            model_name: Name of the model
            modality: Modality type (audio, text, or video)
            architecture: Model architecture description
            params: Hyperparameters dictionary
        """
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("modality", modality)
        mlflow.set_tag("architecture", architecture)
        mlflow.log_params(params)
        mlflow.log_param("timestamp", datetime.now().isoformat())
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log training/evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step/epoch number
        """
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact (file or directory).
        
        Args:
            local_path: Path to the file or directory
            artifact_path: Path in MLflow to store the artifact
        """
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
        else:
            print(f"Warning: Artifact path does not exist: {local_path}")
    
    def log_model(self, model, model_type: str, artifact_path: str = "model"):
        """
        Log a model to MLflow.
        
        Args:
            model: The model object to log
            model_type: Type of model ('pytorch', 'sklearn', or 'custom')
            artifact_path: Artifact path to store the model
        """
        try:
            if model_type.lower() == 'pytorch':
                mlflow.pytorch.log_model(model, artifact_path)
            elif model_type.lower() == 'sklearn':
                mlflow.sklearn.log_model(model, artifact_path)
            else:
                # For custom models, log as generic artifact
                mlflow.log_artifact(model, artifact_path)
        except Exception as e:
            print(f"Error logging model: {e}")
    
    def log_confusion_matrix(self, y_true: list, y_pred: list, labels: list, artifact_name: str = "confusion_matrix"):
        """
        Log confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
            artifact_name: Name for the artifact
        """
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(10, 10))
            disp.plot(ax=ax)
            
            artifact_path = f"/tmp/{artifact_name}.png"
            os.makedirs("/tmp", exist_ok=True)
            plt.savefig(artifact_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(artifact_path)
        except Exception as e:
            print(f"Error logging confusion matrix: {e}")
    
    def load_model(self, model_uri: str):
        """
        Load a model from MLflow.
        
        Args:
            model_uri: URI of the model in MLflow format
            
        Returns:
            Loaded model
        """
        return mlflow.pyfunc.load_model(model_uri)


class AudioModelTracker:
    """Tracker specifically for audio emotion recognition models."""
    
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.tracker = mlflow_tracker
        self.modality = "audio"
    
    def track_training(self, model, params: Dict[str, Any], metrics: Dict[str, float], 
                      model_path: Optional[str] = None):
        """
        Track audio model training.
        
        Args:
            model: The trained model
            params: Training parameters
            metrics: Evaluation metrics
            model_path: Path to save the model
        """
        self.tracker.start_run(run_name=f"audio_emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.tracker.log_model_metadata(
            model_name="audio_emotion_model",
            modality=self.modality,
            architecture="CNN/RNN based audio classifier",
            params=params
        )
        
        self.tracker.log_metrics(metrics)
        
        if model_path and os.path.exists(model_path):
            self.tracker.log_artifact(model_path, artifact_path="audio_model")
        
        self.tracker.end_run()


class TextModelTracker:
    """Tracker specifically for text emotion recognition models."""
    
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.tracker = mlflow_tracker
        self.modality = "text"
    
    def track_training(self, model, params: Dict[str, Any], metrics: Dict[str, float], 
                      model_path: Optional[str] = None):
        """
        Track text model training.
        
        Args:
            model: The trained model
            params: Training parameters
            metrics: Evaluation metrics
            model_path: Path to save the model
        """
        self.tracker.start_run(run_name=f"text_emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.tracker.log_model_metadata(
            model_name="text_emotion_model",
            modality=self.modality,
            architecture="Transformer-based text classifier",
            params=params
        )
        
        self.tracker.log_metrics(metrics)
        
        if model_path and os.path.exists(model_path):
            self.tracker.log_artifact(model_path, artifact_path="text_model")
        
        self.tracker.end_run()


class VideoModelTracker:
    """Tracker specifically for video emotion recognition models."""
    
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.tracker = mlflow_tracker
        self.modality = "video"
    
    def track_training(self, model, params: Dict[str, Any], metrics: Dict[str, float], 
                      model_path: Optional[str] = None):
        """
        Track video model training.
        
        Args:
            model: The trained model
            params: Training parameters
            metrics: Evaluation metrics
            model_path: Path to save the model
        """
        self.tracker.start_run(run_name=f"video_emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.tracker.log_model_metadata(
            model_name="video_emotion_model",
            modality=self.modality,
            architecture="3D CNN for facial landmark/action unit analysis",
            params=params
        )
        
        self.tracker.log_metrics(metrics)
        
        if model_path and os.path.exists(model_path):
            self.tracker.log_artifact(model_path, artifact_path="video_model")
        
        self.tracker.end_run()


class MultimodalModelTracker:
    """Tracker for multimodal fusion model combining all three modalities."""
    
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.tracker = mlflow_tracker
        self.modality = "multimodal"
    
    def track_training(self, model, params: Dict[str, Any], metrics: Dict[str, float],
                      fusion_strategy: str = "late", model_path: Optional[str] = None):
        """
        Track multimodal fusion model training.
        
        Args:
            model: The fused model
            params: Training parameters
            metrics: Evaluation metrics
            fusion_strategy: Type of fusion (late, early, hybrid)
            model_path: Path to save the model
        """
        self.tracker.start_run(run_name=f"multimodal_emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.tracker.log_model_metadata(
            model_name="multimodal_emotion_model",
            modality=self.modality,
            architecture=f"{fusion_strategy} fusion of audio, text, and video",
            params=params
        )
        
        self.tracker.log_metric("fusion_strategy", fusion_strategy)
        self.tracker.log_metrics(metrics)
        
        if model_path and os.path.exists(model_path):
            self.tracker.log_artifact(model_path, artifact_path="multimodal_model")
        
        self.tracker.end_run()


# Utility function for easy access
def get_mlflow_tracker(tracking_uri: str = None, 
                       experiment_name: str = "multimodal_emotion") -> MLflowTracker:
    """
    Get or create MLflow tracker.
    
    Args:
        tracking_uri: MLflow tracking URI (default: local ./mlruns folder)
        experiment_name: Name of the experiment
        
    Returns:
        MLflowTracker instance
    """
    return MLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)


if __name__ == "__main__":
    # Example usage
    tracker = get_mlflow_tracker()
    
    # Example: Track audio model
    audio_tracker = AudioModelTracker(tracker)
    audio_params = {"batch_size": 32, "learning_rate": 0.001, "epochs": 50}
    audio_metrics = {"accuracy": 0.92, "f1_score": 0.91, "loss": 0.18}
    audio_tracker.track_training(None, audio_params, audio_metrics)
    
    # Example: Track text model
    text_tracker = TextModelTracker(tracker)
    text_params = {"batch_size": 16, "learning_rate": 0.0001, "epochs": 30}
    text_metrics = {"accuracy": 0.88, "f1_score": 0.87, "loss": 0.25}
    text_tracker.track_training(None, text_params, text_metrics)
    
    # Example: Track video model
    video_tracker = VideoModelTracker(tracker)
    video_params = {"batch_size": 8, "learning_rate": 0.0005, "epochs": 40}
    video_metrics = {"accuracy": 0.90, "f1_score": 0.89, "loss": 0.20}
    video_tracker.track_training(None, video_params, video_metrics)
    
    # Example: Track multimodal model
    multimodal_tracker = MultimodalModelTracker(tracker)
    fusion_params = {"fusion_method": "attention", "learning_rate": 0.0001, "epochs": 50}
    fusion_metrics = {"accuracy": 0.95, "f1_score": 0.94, "loss": 0.12}
    multimodal_tracker.track_training(None, fusion_params, fusion_metrics, fusion_strategy="hybrid")
