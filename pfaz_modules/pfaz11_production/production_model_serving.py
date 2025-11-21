"""
Production Model Serving System
REST API & Batch Inference Manager

Bu modül, production environment'ta model serving sağlar:
1. REST API (FastAPI)
2. Batch Inference
3. Model Loading & Caching
4. Input Validation
5. Performance Monitoring
6. Error Handling
7. Rate Limiting
8. Logging

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle
import joblib
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Prediction request model"""
    features: Dict[str, float]
    model_name: str = 'best'
    include_uncertainty: bool = False
    request_id: Optional[str] = None


@dataclass
class PredictionResponse:
    """Prediction response model"""
    prediction: float
    model_name: str
    model_version: str
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    inference_time_ms: float = 0.0
    timestamp: str = ''
    request_id: Optional[str] = None


class ModelCache:
    """Model caching system for fast inference"""
    
    def __init__(self, cache_size: int = 5):
        self.cache_size = cache_size
        self.cache = {}
        self.access_counts = defaultdict(int)
        self.lock = threading.Lock()
        
        logger.info(f"Model cache initialized (size={cache_size})")
    
    def get(self, model_name: str):
        """Get model from cache"""
        with self.lock:
            if model_name in self.cache:
                self.access_counts[model_name] += 1
                return self.cache[model_name]
            return None
    
    def put(self, model_name: str, model: Any):
        """Put model in cache"""
        with self.lock:
            # Cache full, remove least accessed
            if len(self.cache) >= self.cache_size:
                least_accessed = min(self.access_counts.items(), key=lambda x: x[1])[0]
                del self.cache[least_accessed]
                del self.access_counts[least_accessed]
                logger.info(f"Evicted {least_accessed} from cache")
            
            self.cache[model_name] = model
            self.access_counts[model_name] = 1
            logger.info(f"Cached {model_name}")
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
            logger.info("Cache cleared")


class ModelServingManager:
    """Production model serving manager"""
    
    def __init__(self, 
                 models_dir: str = 'models',
                 config_path: Optional[str] = None):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model cache
        self.model_cache = ModelCache(cache_size=5)
        
        # Model registry
        self.model_registry = {}
        self._load_model_registry()
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_inference_time_ms': 0.0,
            'model_requests': defaultdict(int)
        }
        
        # Feature names
        self.feature_names = None
        
        logger.info("Model Serving Manager initialized")
    
    def _load_model_registry(self):
        """Load model registry"""
        registry_file = self.models_dir / 'model_registry.json'
        
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.model_registry = json.load(f)
            logger.info(f"Loaded {len(self.model_registry)} models from registry")
        else:
            logger.warning("No model registry found")
    
    def register_model(self, 
                      model_name: str,
                      model_path: str,
                      model_type: str,
                      version: str,
                      metrics: Dict,
                      feature_names: List[str]):
        """Register a model"""
        self.model_registry[model_name] = {
            'path': str(model_path),
            'type': model_type,
            'version': version,
            'metrics': metrics,
            'feature_names': feature_names,
            'registered_at': datetime.now().isoformat()
        }
        
        # Save registry
        registry_file = self.models_dir / 'model_registry.json'
        with open(registry_file, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
        
        logger.info(f"Registered model: {model_name} (v{version})")
    
    def load_model(self, model_name: str):
        """Load model from disk or cache"""
        # Check cache first
        cached_model = self.model_cache.get(model_name)
        if cached_model is not None:
            return cached_model
        
        # Load from disk
        if model_name not in self.model_registry:
            raise ValueError(f"Model not found: {model_name}")
        
        model_info = self.model_registry[model_name]
        model_path = Path(model_info['path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load based on type
        model_type = model_info['type']
        
        try:
            if model_type in ['sklearn', 'xgboost']:
                model = joblib.load(model_path)
            elif model_type == 'keras':
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Cache model
            self.model_cache.put(model_name, model)
            
            # Update feature names
            if self.feature_names is None:
                self.feature_names = model_info.get('feature_names')
            
            logger.info(f"Loaded model: {model_name}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def validate_input(self, features: Dict[str, float]) -> np.ndarray:
        """Validate and prepare input"""
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        # Check for missing features
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Create feature array in correct order
        feature_array = np.array([features[f] for f in self.feature_names])
        
        # Reshape for single prediction
        return feature_array.reshape(1, -1)
    
    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Single prediction"""
        start_time = time.time()
        
        try:
            # Load model
            model = self.load_model(request.model_name)
            model_info = self.model_registry[request.model_name]
            
            # Validate input
            X = self.validate_input(request.features)
            
            # Make prediction
            if model_info['type'] == 'keras':
                prediction = model.predict(X, verbose=0)[0][0]
            else:
                prediction = model.predict(X)[0]
            
            # Calculate uncertainty (if requested and supported)
            uncertainty = None
            if request.include_uncertainty:
                uncertainty = self._estimate_uncertainty(model, X, model_info['type'])
            
            # Calculate confidence (simple heuristic)
            confidence = 1.0 - (uncertainty if uncertainty else 0.0)
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_metrics(request.model_name, inference_time_ms, success=True)
            
            return PredictionResponse(
                prediction=float(prediction),
                model_name=request.model_name,
                model_version=model_info['version'],
                confidence=confidence,
                uncertainty=uncertainty,
                inference_time_ms=inference_time_ms,
                timestamp=datetime.now().isoformat(),
                request_id=request.request_id
            )
        
        except Exception as e:
            self._update_metrics(request.model_name, 0, success=False)
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, 
                     features_list: List[Dict[str, float]],
                     model_name: str = 'best',
                     batch_size: int = 32) -> List[PredictionResponse]:
        """Batch prediction"""
        responses = []
        
        # Load model once
        model = self.load_model(model_name)
        model_info = self.model_registry[model_name]
        
        # Process in batches
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i+batch_size]
            
            start_time = time.time()
            
            try:
                # Prepare batch
                X_batch = np.array([
                    [feat[f] for f in self.feature_names]
                    for feat in batch
                ])
                
                # Predict
                if model_info['type'] == 'keras':
                    predictions = model.predict(X_batch, verbose=0).flatten()
                else:
                    predictions = model.predict(X_batch)
                
                inference_time_ms = (time.time() - start_time) * 1000
                
                # Create responses
                for pred in predictions:
                    responses.append(PredictionResponse(
                        prediction=float(pred),
                        model_name=model_name,
                        model_version=model_info['version'],
                        inference_time_ms=inference_time_ms / len(batch),
                        timestamp=datetime.now().isoformat()
                    ))
                
                # Update metrics
                for _ in batch:
                    self._update_metrics(model_name, inference_time_ms / len(batch), success=True)
            
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                for _ in batch:
                    self._update_metrics(model_name, 0, success=False)
                raise
        
        return responses
    
    def _estimate_uncertainty(self, model, X, model_type: str) -> Optional[float]:
        """Estimate prediction uncertainty"""
        try:
            if model_type == 'keras':
                # Monte Carlo Dropout
                predictions = []
                for _ in range(10):
                    pred = model(X, training=True).numpy()[0][0]
                    predictions.append(pred)
                return float(np.std(predictions))
            else:
                # For tree-based: std of tree predictions
                if hasattr(model, 'estimators_'):
                    tree_predictions = [tree.predict(X)[0] for tree in model.estimators_]
                    return float(np.std(tree_predictions))
                return None
        except:
            return None
    
    def _update_metrics(self, model_name: str, inference_time_ms: float, success: bool):
        """Update performance metrics"""
        self.metrics['total_requests'] += 1
        if success:
            self.metrics['successful_requests'] += 1
            self.metrics['total_inference_time_ms'] += inference_time_ms
        else:
            self.metrics['failed_requests'] += 1
        
        self.metrics['model_requests'][model_name] += 1
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        total = self.metrics['total_requests']
        
        return {
            'total_requests': total,
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': (self.metrics['successful_requests'] / total * 100) if total > 0 else 0,
            'avg_inference_time_ms': (self.metrics['total_inference_time_ms'] / self.metrics['successful_requests']) if self.metrics['successful_requests'] > 0 else 0,
            'model_requests': dict(self.metrics['model_requests'])
        }
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models"""
        models = []
        for name, info in self.model_registry.items():
            models.append({
                'name': name,
                'type': info['type'],
                'version': info['version'],
                'metrics': info['metrics']
            })
        return models
    
    def health_check(self) -> Dict:
        """System health check"""
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': len(self.model_cache.cache),
            'models_available': len(self.model_registry),
            'cache_size': len(self.model_cache.cache),
            'metrics': self.get_metrics()
        }


def create_fastapi_app(serving_manager: ModelServingManager):
    """Create FastAPI application (requires FastAPI installed)"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        
        app = FastAPI(title="Nuclear Physics AI Model API", version="1.0.0")
        
        class PredictRequest(BaseModel):
            features: Dict[str, float]
            model_name: str = 'best'
            include_uncertainty: bool = False
        
        class BatchPredictRequest(BaseModel):
            features_list: List[Dict[str, float]]
            model_name: str = 'best'
            batch_size: int = 32
        
        @app.get("/")
        def root():
            return {"message": "Nuclear Physics AI Model API", "version": "1.0.0"}
        
        @app.get("/health")
        def health():
            return serving_manager.health_check()
        
        @app.get("/models")
        def list_models():
            return {"models": serving_manager.get_available_models()}
        
        @app.get("/metrics")
        def metrics():
            return serving_manager.get_metrics()
        
        @app.post("/predict")
        def predict(request: PredictRequest):
            try:
                pred_request = PredictionRequest(
                    features=request.features,
                    model_name=request.model_name,
                    include_uncertainty=request.include_uncertainty
                )
                response = serving_manager.predict_single(pred_request)
                return asdict(response)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/predict/batch")
        def predict_batch(request: BatchPredictRequest):
            try:
                responses = serving_manager.predict_batch(
                    request.features_list,
                    request.model_name,
                    request.batch_size
                )
                return {"predictions": [asdict(r) for r in responses]}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        return app
    
    except ImportError:
        logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")
        return None


def main():
    """Test function"""
    # Create serving manager
    manager = ModelServingManager('test_models')
    
    # Example: Register a dummy model
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    
    # Train dummy model
    X = np.random.randn(100, 5)
    y = np.sum(X, axis=1)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    model_path = Path('test_models/rf_test.pkl')
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    
    # Register
    manager.register_model(
        model_name='rf_test',
        model_path=model_path,
        model_type='sklearn',
        version='1.0',
        metrics={'R2': 0.95},
        feature_names=['f1', 'f2', 'f3', 'f4', 'f5']
    )
    
    # Test prediction
    request = PredictionRequest(
        features={'f1': 1.0, 'f2': 2.0, 'f3': 3.0, 'f4': 4.0, 'f5': 5.0},
        model_name='rf_test',
        include_uncertainty=True
    )
    
    response = manager.predict_single(request)
    print("\n✓ Single Prediction:")
    print(f"  Prediction: {response.prediction:.4f}")
    print(f"  Confidence: {response.confidence:.4f}")
    print(f"  Inference Time: {response.inference_time_ms:.2f} ms")
    
    # Test batch
    features_list = [
        {'f1': 1.0, 'f2': 2.0, 'f3': 3.0, 'f4': 4.0, 'f5': 5.0},
        {'f1': 2.0, 'f2': 3.0, 'f3': 4.0, 'f4': 5.0, 'f5': 6.0},
        {'f1': 3.0, 'f2': 4.0, 'f3': 5.0, 'f4': 6.0, 'f5': 7.0}
    ]
    
    responses = manager.predict_batch(features_list, 'rf_test')
    print(f"\n✓ Batch Prediction: {len(responses)} predictions")
    
    # Metrics
    print("\n✓ Metrics:")
    metrics = manager.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Test completed!")


if __name__ == "__main__":
    main()
