import mlflow
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path


class ModelRegistry:
    """
    Model registry for managing and versioning multimodal emotion recognition models.
    Integrates with MLflow for model tracking and management.
    """
    
    def __init__(self, registry_uri: str = None):
        """
        Initialize model registry.
        
        Args:
            registry_uri: URI for model registry (default: local mlruns)
        """
        if registry_uri is None:
            registry_uri = os.path.join(os.path.dirname(__file__), "..", "mlruns")
        
        self.registry_uri = registry_uri
        mlflow.set_tracking_uri(registry_uri)
    
    def register_model(self, model_uri: str, model_name: str, description: str = "") -> str:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model (e.g., "runs:/run_id/model")
            model_name: Name for the registered model
            description: Description of the model
            
        Returns:
            Model version URI
        """
        try:
            result = mlflow.register_model(model_uri, model_name)
            print(f"✓ Model registered: {model_name} v{result.version}")
            return result.model_uri
        except Exception as e:
            print(f"✗ Error registering model: {e}")
            return None
    
    def get_model_version(self, model_name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Get model version information.
        
        Args:
            model_name: Name of the registered model
            version: Specific version (None for latest)
            
        Returns:
            Model version details
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            if version is None:
                # Get latest version
                versions = client.get_latest_versions(model_name)
                if versions:
                    version = versions[0].version
                else:
                    return None
            
            model_version = client.get_model_version(model_name, version)
            return {
                "name": model_version.name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "uri": model_version.source,
                "status": model_version.status,
                "created_timestamp": model_version.creation_timestamp,
                "last_updated": model_version.last_updated_timestamp
            }
        except Exception as e:
            print(f"Error getting model version: {e}")
            return None
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a registered model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            List of model version details
        """
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=None)
            
            result = []
            for version in versions:
                result.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "uri": version.source,
                    "status": version.status,
                    "created": datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat()
                })
            return result
        except Exception as e:
            print(f"Error listing model versions: {e}")
            return []
    
    def transition_model_stage(self, model_name: str, version: int, new_stage: str) -> bool:
        """
        Transition model to a new stage (Staging, Production, Archived).
        
        Args:
            model_name: Name of the registered model
            version: Version number
            new_stage: Target stage (Staging, Production, Archived)
            
        Returns:
            Success status
        """
        valid_stages = ["Staging", "Production", "Archived"]
        if new_stage not in valid_stages:
            print(f"Invalid stage. Must be one of: {valid_stages}")
            return False
        
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(model_name, version, new_stage)
            print(f"✓ Model {model_name} v{version} moved to {new_stage}")
            return True
        except Exception as e:
            print(f"Error transitioning model stage: {e}")
            return False
    
    def load_production_model(self, model_name: str):
        """
        Load the production version of a model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Loaded model
        """
        try:
            model_uri = f"models:/{model_name}/production"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✓ Loaded production model: {model_name}")
            return model
        except Exception as e:
            print(f"Error loading production model: {e}")
            return None
    
    def load_staging_model(self, model_name: str):
        """
        Load the staging version of a model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Loaded model
        """
        try:
            model_uri = f"models:/{model_name}/staging"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✓ Loaded staging model: {model_name}")
            return model
        except Exception as e:
            print(f"Error loading staging model: {e}")
            return None
    
    def get_all_registered_models(self) -> List[str]:
        """
        Get list of all registered models.
        
        Returns:
            List of model names
        """
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.list_registered_models()
            return [model.name for model in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


class AudioModelRegistry:
    """Registry for audio emotion recognition models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.model_name = "audio_emotion_model"
    
    def register_version(self, model_uri: str, version_description: str = "") -> str:
        """Register a new audio model version."""
        return self.registry.register_model(
            model_uri,
            self.model_name,
            f"Audio emotion recognition model. {version_description}"
        )
    
    def get_latest_version(self) -> Dict[str, Any]:
        """Get latest audio model version."""
        return self.registry.get_model_version(self.model_name)
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all audio model versions."""
        return self.registry.list_model_versions(self.model_name)
    
    def to_production(self, version: int) -> bool:
        """Move audio model to production."""
        return self.registry.transition_model_stage(self.model_name, version, "Production")
    
    def to_staging(self, version: int) -> bool:
        """Move audio model to staging."""
        return self.registry.transition_model_stage(self.model_name, version, "Staging")
    
    def load_production(self):
        """Load production audio model."""
        return self.registry.load_production_model(self.model_name)


class TextModelRegistry:
    """Registry for text emotion recognition models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.model_name = "text_emotion_model"
    
    def register_version(self, model_uri: str, version_description: str = "") -> str:
        """Register a new text model version."""
        return self.registry.register_model(
            model_uri,
            self.model_name,
            f"Text emotion recognition model. {version_description}"
        )
    
    def get_latest_version(self) -> Dict[str, Any]:
        """Get latest text model version."""
        return self.registry.get_model_version(self.model_name)
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all text model versions."""
        return self.registry.list_model_versions(self.model_name)
    
    def to_production(self, version: int) -> bool:
        """Move text model to production."""
        return self.registry.transition_model_stage(self.model_name, version, "Production")
    
    def to_staging(self, version: int) -> bool:
        """Move text model to staging."""
        return self.registry.transition_model_stage(self.model_name, version, "Staging")
    
    def load_production(self):
        """Load production text model."""
        return self.registry.load_production_model(self.model_name)


class VideoModelRegistry:
    """Registry for video emotion recognition models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.model_name = "video_emotion_model"
    
    def register_version(self, model_uri: str, version_description: str = "") -> str:
        """Register a new video model version."""
        return self.registry.register_model(
            model_uri,
            self.model_name,
            f"Video emotion recognition model. {version_description}"
        )
    
    def get_latest_version(self) -> Dict[str, Any]:
        """Get latest video model version."""
        return self.registry.get_model_version(self.model_name)
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all video model versions."""
        return self.registry.list_model_versions(self.model_name)
    
    def to_production(self, version: int) -> bool:
        """Move video model to production."""
        return self.registry.transition_model_stage(self.model_name, version, "Production")
    
    def to_staging(self, version: int) -> bool:
        """Move video model to staging."""
        return self.registry.transition_model_stage(self.model_name, version, "Staging")
    
    def load_production(self):
        """Load production video model."""
        return self.registry.load_production_model(self.model_name)


class MultimodalModelRegistry:
    """Registry for multimodal fusion models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.model_name = "multimodal_emotion_model"
    
    def register_version(self, model_uri: str, fusion_strategy: str = "", version_description: str = "") -> str:
        """Register a new multimodal model version."""
        description = f"Multimodal emotion model ({fusion_strategy} fusion). {version_description}"
        return self.registry.register_model(model_uri, self.model_name, description)
    
    def get_latest_version(self) -> Dict[str, Any]:
        """Get latest multimodal model version."""
        return self.registry.get_model_version(self.model_name)
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all multimodal model versions."""
        return self.registry.list_model_versions(self.model_name)
    
    def to_production(self, version: int) -> bool:
        """Move multimodal model to production."""
        return self.registry.transition_model_stage(self.model_name, version, "Production")
    
    def to_staging(self, version: int) -> bool:
        """Move multimodal model to staging."""
        return self.registry.transition_model_stage(self.model_name, version, "Staging")
    
    def load_production(self):
        """Load production multimodal model."""
        return self.registry.load_production_model(self.model_name)


# Utility function for easy access
def get_model_registry(registry_uri: str = None) -> ModelRegistry:
    """
    Get or create model registry.
    
    Args:
        registry_uri: Registry URI (default: local mlruns)
        
    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(registry_uri=registry_uri)


if __name__ == "__main__":
    # Example usage
    registry = get_model_registry()
    
    # Initialize registries for each modality
    audio_registry = AudioModelRegistry(registry)
    text_registry = TextModelRegistry(registry)
    video_registry = VideoModelRegistry(registry)
    multimodal_registry = MultimodalModelRegistry(registry)
    
    # Example: List all registered models
    print("Registered models:", registry.get_all_registered_models())
    
    # Example: Get audio model latest version
    audio_version = audio_registry.get_latest_version()
    if audio_version:
        print(f"Audio model latest version: {audio_version['version']} - {audio_version['stage']}")
    
    # Example: List all text model versions
    print("Text model versions:", text_registry.list_versions())
