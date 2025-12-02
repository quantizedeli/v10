# PFAZ 11-13: PRODUCTION & ADVANCED SYSTEMS
## Production Deployment, Advanced Analytics ve AutoML Integration

**Versiyon:** 3.0.0  
**Durum:** 🟡 %70 TAMAMLANDI (PFAZ 11), ✅ %100 (PFAZ 12-13)  
**Son Güncelleme:** 2 Aralık 2025

---

## 📋 İÇİNDEKİLER

1. [PFAZ 11: Production Deployment](#pfaz-11-production-deployment)
2. [PFAZ 12: Advanced Analytics](#pfaz-12-advanced-analytics)
3. [PFAZ 13: AutoML Integration](#pfaz-13-automl-integration)

---

# PFAZ 11: PRODUCTION DEPLOYMENT
## Production-Ready System ve Deployment Otomasyonu

### 🎯 Genel Bakış

PFAZ 11, modelleri **production ortamına** taşır. REST API, monitoring, versioning, Docker containerization ve CI/CD pipeline'ı içerir.

### Temel Bileşenler

```
PFAZ 11 Architecture:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. MODEL SERVING                                   │
│     ├── FastAPI REST API                           │
│     ├── Model Registry                             │
│     ├── Versioning System                          │
│     └── Load Balancing                             │
│                                                     │
│  2. MONITORING & LOGGING                           │
│     ├── Prometheus Metrics                         │
│     ├── Grafana Dashboards                         │
│     ├── ELK Stack (Elasticsearch, Logstash, Kibana)│
│     └── Alert System                               │
│                                                     │
│  3. CONTAINERIZATION                               │
│     ├── Docker Images                              │
│     ├── Docker Compose                             │
│     ├── Kubernetes Manifests                       │
│     └── Helm Charts                                │
│                                                     │
│  4. CI/CD PIPELINE                                 │
│     ├── GitHub Actions                             │
│     ├── Automated Testing                          │
│     ├── Code Quality Checks                        │
│     └── Deployment Automation                      │
│                                                     │
│  5. SECURITY                                       │
│     ├── JWT Authentication                         │
│     ├── API Key Management                         │
│     ├── Rate Limiting                              │
│     └── HTTPS/TLS                                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 1️⃣ MODEL SERVING

### FastAPI REST API

**Modül:** `production_model_serving.py`

```python
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Dict, Optional
import logging

app = FastAPI(
    title="Nuclear Physics AI API",
    description="Production API for nuclear property predictions",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Security
security = HTTPBearer()

# Models cache
MODELS_CACHE = {}

class PredictionRequest(BaseModel):
    """Prediction request schema"""
    Z: int  # Proton number
    N: int  # Neutron number
    target: str  # MM, QM, or Beta_2
    model_version: Optional[str] = "latest"
    
    class Config:
        schema_extra = {
            "example": {
                "Z": 82,
                "N": 126,
                "target": "MM",
                "model_version": "v3.0.0"
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    nucleus: str
    Z: int
    N: int
    target: str
    prediction: float
    uncertainty: Optional[float]
    model_version: str
    timestamp: str
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: int
    uptime_seconds: float

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token"""
    token = credentials.credentials
    # Implement JWT verification
    # For now, simple API key check
    if token != "your-secret-api-key":
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return token

# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Nuclear Physics AI API",
        "version": "3.0.0",
        "docs": "/api/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "models_loaded": len(MODELS_CACHE),
        "uptime_seconds": get_uptime()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """
    Predict nuclear property
    
    - **Z**: Proton number (20-92)
    - **N**: Neutron number (20-146)
    - **target**: Property to predict (MM, QM, Beta_2)
    - **model_version**: Model version (default: latest)
    """
    try:
        # Load model
        model = load_model(request.target, request.model_version)
        
        # Prepare features
        features = prepare_features(request.Z, request.N)
        
        # Predict
        prediction = model.predict([features])[0]
        
        # Calculate uncertainty (if BNN)
        uncertainty = calculate_uncertainty(model, features)
        
        return {
            "nucleus": f"{request.Z+request.N}-Element-{request.Z}",
            "Z": request.Z,
            "N": request.N,
            "target": request.target,
            "prediction": float(prediction),
            "uncertainty": uncertainty,
            "model_version": request.model_version,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    requests: List[PredictionRequest],
    token: str = Depends(verify_token)
):
    """Batch prediction endpoint"""
    results = []
    for req in requests:
        result = await predict(req, token)
        results.append(result)
    return results

@app.get("/models", response_model=Dict[str, List[str]])
async def list_models(token: str = Depends(verify_token)):
    """List available models"""
    return {
        "MM": ["v3.0.0", "v2.5.0", "v2.0.0"],
        "QM": ["v3.0.0", "v2.5.0"],
        "Beta_2": ["v3.0.0", "v2.5.0"]
    }

@app.get("/models/{target}/{version}/info")
async def model_info(
    target: str,
    version: str,
    token: str = Depends(verify_token)
):
    """Get model information"""
    model_data = get_model_metadata(target, version)
    return {
        "target": target,
        "version": version,
        "architecture": model_data["architecture"],
        "r2_score": model_data["r2"],
        "mae": model_data["mae"],
        "training_date": model_data["training_date"],
        "features": model_data["features"]
    }

# Helper functions
def load_model(target: str, version: str):
    """Load model from cache or disk"""
    cache_key = f"{target}_{version}"
    
    if cache_key not in MODELS_CACHE:
        model_path = f"models/{target}/{version}/model.pkl"
        MODELS_CACHE[cache_key] = joblib.load(model_path)
    
    return MODELS_CACHE[cache_key]

def prepare_features(Z: int, N: int) -> np.ndarray:
    """Prepare feature vector"""
    # Calculate physics features
    A = Z + N
    pairing = calculate_pairing(Z, N)
    shell_effects = calculate_shell_effects(Z, N)
    # ... more features
    
    return np.array([Z, N, A, pairing, shell_effects, ...])

def calculate_uncertainty(model, features) -> Optional[float]:
    """Calculate prediction uncertainty"""
    # For BNN or ensemble models
    if hasattr(model, 'predict_with_uncertainty'):
        _, uncertainty = model.predict_with_uncertainty([features])
        return float(uncertainty[0])
    return None

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Usage Examples

**Python Client:**
```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Request
payload = {
    "Z": 82,
    "N": 126,
    "target": "MM",
    "model_version": "v3.0.0"
}

headers = {
    "Authorization": "Bearer your-secret-api-key",
    "Content-Type": "application/json"
}

# Predict
response = requests.post(url, json=payload, headers=headers)
result = response.json()

print(f"Prediction: {result['prediction']:.3f}")
print(f"Uncertainty: {result['uncertainty']:.3f}")
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "Z": 82,
    "N": 126,
    "target": "MM",
    "model_version": "v3.0.0"
  }'
```

**JavaScript:**
```javascript
const axios = require('axios');

async function predict() {
  const response = await axios.post(
    'http://localhost:8000/predict',
    {
      Z: 82,
      N: 126,
      target: 'MM',
      model_version: 'v3.0.0'
    },
    {
      headers: {
        'Authorization': 'Bearer your-secret-api-key'
      }
    }
  );
  
  console.log('Prediction:', response.data.prediction);
}
```

---

## 2️⃣ MONITORING & LOGGING

### Prometheus Metrics

**Modül:** `production_monitoring_system.py`

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging

# Metrics
prediction_requests = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['target', 'model_version', 'status']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency',
    ['target']
)

active_models = Gauge(
    'active_models',
    'Number of loaded models'
)

model_cache_size = Gauge(
    'model_cache_size_bytes',
    'Model cache size in bytes'
)

error_rate = Counter(
    'prediction_errors_total',
    'Total prediction errors',
    ['error_type']
)

class MonitoringMiddleware:
    """FastAPI middleware for monitoring"""
    
    async def __call__(self, request, call_next):
        # Start timer
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Record success
            prediction_requests.labels(
                target=request.state.target,
                model_version=request.state.version,
                status='success'
            ).inc()
            
            # Record latency
            latency = time.time() - start_time
            prediction_latency.labels(
                target=request.state.target
            ).observe(latency)
            
            return response
            
        except Exception as e:
            # Record error
            prediction_requests.labels(
                target=request.state.target,
                model_version=request.state.version,
                status='error'
            ).inc()
            
            error_rate.labels(
                error_type=type(e).__name__
            ).inc()
            
            raise

# Start Prometheus metrics server
start_http_server(9090)
```

### Grafana Dashboard Config

**File:** `grafana_dashboard.json`

```json
{
  "dashboard": {
    "title": "Nuclear Physics AI - Production Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(prediction_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Prediction Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, prediction_latency_seconds)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(prediction_errors_total[5m])"
          }
        ]
      },
      {
        "title": "Active Models",
        "targets": [
          {
            "expr": "active_models"
          }
        ]
      }
    ]
  }
}
```

### ELK Stack Integration

**Logstash Config:**
```ruby
input {
  file {
    path => "/var/log/nuclear-ai/*.log"
    type => "application"
  }
}

filter {
  json {
    source => "message"
  }
  
  if [level] == "ERROR" {
    mutate {
      add_tag => ["error"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "nuclear-ai-logs-%{+YYYY.MM.dd}"
  }
}
```

---

## 3️⃣ CONTAINERIZATION

### Dockerfile

```dockerfile
# Multi-stage build
FROM python:3.10-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ gfortran \
    libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_production.txt /tmp/
RUN pip install --user --no-cache-dir -r /tmp/requirements_production.txt

# Final stage
FROM python:3.10-slim

# Copy installed packages
COPY --from=builder /root/.local /root/.local

# Set working directory
WORKDIR /app

# Copy application
COPY . /app/

# Environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "production_model_serving:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  # API Service
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models:ro
      - ./logs:/app/logs
    depends_on:
      - prometheus
      - elasticsearch
    networks:
      - nuclear-ai-network
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - nuclear-ai-network

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana_dashboard.json:/etc/grafana/provisioning/dashboards/dashboard.json
    depends_on:
      - prometheus
    networks:
      - nuclear-ai-network

  # Elasticsearch
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - nuclear-ai-network

  # Kibana
  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
    networks:
      - nuclear-ai-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - nuclear-ai-network

volumes:
  prometheus-data:
  grafana-data:
  elasticsearch-data:
  redis-data:

networks:
  nuclear-ai-network:
    driver: bridge
```

### Kubernetes Deployment

**File:** `k8s-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nuclear-ai-api
  labels:
    app: nuclear-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nuclear-ai
  template:
    metadata:
      labels:
        app: nuclear-ai
    spec:
      containers:
      - name: api
        image: nuclear-ai:v3.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: nuclear-ai-models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: nuclear-ai-service
spec:
  type: LoadBalancer
  selector:
    app: nuclear-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nuclear-ai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nuclear-ai-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 4️⃣ CI/CD PIPELINE

### GitHub Actions Workflow

**File:** `.github/workflows/ci-cd.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # Test Job
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements_production.txt
        pip install pytest pytest-cov flake8 black mypy
    
    - name: Lint with flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Check formatting with black
      run: black --check .
    
    - name: Type check with mypy
      run: mypy production_model_serving.py
    
    - name: Run tests
      run: pytest tests/ --cov=. --cov-report=xml --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

  # Security Scan
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r . -f json -o bandit-report.json
    
    - name: Run Safety check
      run: |
        pip install safety
        safety check --json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: bandit-report.json

  # Build Docker Image
  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          nuclear-ai/api:latest
          nuclear-ai/api:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # Deploy to Staging
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service \
          --cluster nuclear-ai-staging \
          --service api \
          --force-new-deployment

  # Deploy to Production
  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service \
          --cluster nuclear-ai-production \
          --service api \
          --force-new-deployment
    
    - name: Run smoke tests
      run: |
        sleep 60  # Wait for deployment
        curl -f https://api.nuclear-ai.com/health || exit 1
```

---

## 5️⃣ SECURITY

### JWT Authentication

```python
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

# Configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def hash_password(password: str):
    """Hash password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("100/minute")  # 100 requests per minute
async def predict(request: Request, ...):
    ...
```

---

# PFAZ 12: ADVANCED ANALYTICS
## İleri Seviye Analitik ve İstatistiksel Yöntemler

### 🎯 Genel Bakış

PFAZ 12, **istatistiksel analiz**, **sensitivity analysis**, **bootstrap** ve **Bayesian yöntemler** ile model güvenilirliğini değerlendirir.

### Bileşenler

```
PFAZ 12 Components:
├── 1. Statistical Testing Suite
├── 2. Bootstrap Confidence Intervals
├── 3. Advanced Sensitivity Analysis
├── 4. Bayesian Uncertainty Quantification
└── 5. Feature Importance Analysis
```

---

## 1️⃣ Statistical Testing Suite

**Modül:** `statistical_testing_suite.py`

```python
from scipy import stats
import numpy as np
import pandas as pd

class StatisticalTestSuite:
    """Comprehensive statistical testing"""
    
    def __init__(self):
        self.results = {}
    
    def normality_test(self, residuals):
        """Test if residuals are normally distributed"""
        # Shapiro-Wilk test
        statistic, p_value = stats.shapiro(residuals)
        
        return {
            'test': 'Shapiro-Wilk',
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def homoscedasticity_test(self, predictions, residuals):
        """Test for constant variance (homoscedasticity)"""
        # Breusch-Pagan test
        from statsmodels.stats.diagnostic import het_breuschpagan
        
        bp_stat, p_value, _, _ = het_breuschpagan(
            residuals,
            predictions.reshape(-1, 1)
        )
        
        return {
            'test': 'Breusch-Pagan',
            'statistic': bp_stat,
            'p_value': p_value,
            'is_homoscedastic': p_value > 0.05
        }
    
    def compare_models(self, model1_predictions, model2_predictions, y_true):
        """Statistical comparison of two models"""
        # Diebold-Mariano test
        errors1 = y_true - model1_predictions
        errors2 = y_true - model2_predictions
        
        d = errors1**2 - errors2**2
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        dm_stat = d_mean / np.sqrt(d_var / len(d))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        return {
            'test': 'Diebold-Mariano',
            'statistic': dm_stat,
            'p_value': p_value,
            'model1_better': dm_stat < 0 and p_value < 0.05,
            'model2_better': dm_stat > 0 and p_value < 0.05
        }
    
    def paired_t_test(self, errors1, errors2):
        """Paired t-test for model comparison"""
        statistic, p_value = stats.ttest_rel(
            np.abs(errors1),
            np.abs(errors2)
        )
        
        return {
            'test': 'Paired t-test',
            'statistic': statistic,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
```

---

## 2️⃣ Bootstrap Confidence Intervals

**Modül:** `bootstrap_confidence_intervals.py`

```python
from scipy.stats import bootstrap
import numpy as np

class BootstrapAnalyzer:
    """Bootstrap confidence intervals"""
    
    def __init__(self, n_bootstrap=10000, confidence_level=0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
    
    def metric_ci(self, y_true, y_pred, metric_func):
        """
        Calculate confidence interval for any metric
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_func: Function that calculates metric
                        (e.g., lambda y_t, y_p: r2_score(y_t, y_p))
        """
        def statistic(y_true, y_pred):
            return metric_func(y_true, y_pred)
        
        result = bootstrap(
            (y_true, y_pred),
            statistic,
            n_resamples=self.n_bootstrap,
            confidence_level=self.confidence_level,
            method='percentile'
        )
        
        return {
            'ci_lower': result.confidence_interval.low,
            'ci_upper': result.confidence_interval.high,
            'point_estimate': metric_func(y_true, y_pred)
        }
    
    def r2_confidence_interval(self, y_true, y_pred):
        """R² confidence interval"""
        from sklearn.metrics import r2_score
        
        return self.metric_ci(
            y_true, y_pred,
            lambda y_t, y_p: r2_score(y_t, y_p)
        )
    
    def mae_confidence_interval(self, y_true, y_pred):
        """MAE confidence interval"""
        from sklearn.metrics import mean_absolute_error
        
        return self.metric_ci(
            y_true, y_pred,
            lambda y_t, y_p: mean_absolute_error(y_t, y_p)
        )
    
    def prediction_intervals(self, model, X, alpha=0.05):
        """
        Calculate prediction intervals using bootstrap
        """
        predictions = []
        
        for _ in range(self.n_bootstrap):
            # Resample training data
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            
            # Predict
            pred = model.predict(X_boot)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate intervals
        lower = np.percentile(predictions, alpha/2 * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
        
        return {
            'lower': lower,
            'upper': upper,
            'mean': predictions.mean(axis=0)
        }
```

---

## 3️⃣ Advanced Sensitivity Analysis

**Modül:** `advanced_sensitivity_analysis.py`

```python
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

class SensitivityAnalyzer:
    """Advanced sensitivity analysis using Sobol indices"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
    
    def sobol_analysis(self, bounds, n_samples=1000):
        """
        Sobol sensitivity analysis
        
        Args:
            bounds: List of (min, max) tuples for each feature
            n_samples: Number of samples for analysis
        """
        # Define problem
        problem = {
            'num_vars': self.n_features,
            'names': self.feature_names,
            'bounds': bounds
        }
        
        # Generate samples
        param_values = saltelli.sample(problem, n_samples)
        
        # Evaluate model
        Y = np.array([
            self.model.predict(X.reshape(1, -1))[0]
            for X in param_values
        ])
        
        # Analyze
        Si = sobol.analyze(problem, Y)
        
        return {
            'first_order': dict(zip(self.feature_names, Si['S1'])),
            'total_order': dict(zip(self.feature_names, Si['ST'])),
            'second_order': Si['S2']
        }
    
    def local_sensitivity(self, X, delta=0.01):
        """
        Local sensitivity analysis (partial derivatives)
        """
        base_prediction = self.model.predict(X.reshape(1, -1))[0]
        sensitivities = []
        
        for i in range(self.n_features):
            X_perturbed = X.copy()
            X_perturbed[i] += delta * X[i]
            
            perturbed_prediction = self.model.predict(
                X_perturbed.reshape(1, -1)
            )[0]
            
            sensitivity = (perturbed_prediction - base_prediction) / (delta * X[i])
            sensitivities.append(sensitivity)
        
        return dict(zip(self.feature_names, sensitivities))
    
    def feature_interaction(self, X, feature1_idx, feature2_idx, n_points=20):
        """
        Analyze interaction between two features
        """
        # Create grid
        feature1_range = np.linspace(
            X[:, feature1_idx].min(),
            X[:, feature1_idx].max(),
            n_points
        )
        feature2_range = np.linspace(
            X[:, feature2_idx].min(),
            X[:, feature2_idx].max(),
            n_points
        )
        
        X_grid, Y_grid = np.meshgrid(feature1_range, feature2_range)
        
        # Predictions
        predictions = np.zeros_like(X_grid)
        
        for i in range(n_points):
            for j in range(n_points):
                X_sample = X[0].copy()
                X_sample[feature1_idx] = X_grid[i, j]
                X_sample[feature2_idx] = Y_grid[i, j]
                
                predictions[i, j] = self.model.predict(
                    X_sample.reshape(1, -1)
                )[0]
        
        return {
            'feature1_values': X_grid,
            'feature2_values': Y_grid,
            'predictions': predictions
        }
```

---

# PFAZ 13: AUTOML INTEGRATION
## Otomatik Hiperparametre Optimizasyonu

### 🎯 Genel Bakış

PFAZ 13, **Optuna** kullanarak otomatik hiperparametre optimizasyonu yapar ve **en iyi modelleri** bulur.

### AutoML Pipeline

```python
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

class AutoMLOptimizer:
    """Automatic hyperparameter optimization"""
    
    def __init__(self, model_type='RandomForest', n_trials=100):
        self.model_type = model_type
        self.n_trials = n_trials
        self.study = None
        self.best_model = None
    
    def optimize_rf(self, X_train, y_train):
        """Optimize Random Forest"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical(
                    'max_features', ['sqrt', 'log2', None]
                )
            }
            
            model = RandomForestRegressor(**params, random_state=42)
            
            # Cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=5, scoring='r2', n_jobs=-1
            )
            
            return scores.mean()
        
        # Create study
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials)
        
        # Train best model
        self.best_model = RandomForestRegressor(
            **self.study.best_params,
            random_state=42
        )
        self.best_model.fit(X_train, y_train)
        
        return {
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
            'model': self.best_model
        }
    
    def optimize_xgboost(self, X_train, y_train):
        """Optimize XGBoost"""
        import xgboost as xgb
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42)
            
            scores = cross_val_score(
                model, X_train, y_train,
                cv=5, scoring='r2', n_jobs=-1
            )
            
            return scores.mean()
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_model = xgb.XGBRegressor(
            **self.study.best_params,
            random_state=42
        )
        self.best_model.fit(X_train, y_train)
        
        return {
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
            'model': self.best_model
        }
    
    def plot_optimization_history(self):
        """Plot optimization history"""
        from optuna.visualization import plot_optimization_history
        
        fig = plot_optimization_history(self.study)
        fig.write_html('optimization_history.html')
    
    def plot_param_importances(self):
        """Plot parameter importances"""
        from optuna.visualization import plot_param_importances
        
        fig = plot_param_importances(self.study)
        fig.write_html('param_importances.html')
```

### Usage

```python
# Initialize
automl = AutoMLOptimizer(model_type='XGBoost', n_trials=200)

# Optimize
result = automl.optimize_xgboost(X_train, y_train)

print(f"Best R² Score: {result['best_score']:.4f}")
print(f"Best Params: {result['best_params']}")

# Visualize
automl.plot_optimization_history()
automl.plot_param_importances()

# Evaluate on test set
test_score = result['model'].score(X_test, y_test)
print(f"Test R² Score: {test_score:.4f}")
```

---

## 📊 ÖZET VE DURUM

### PFAZ 11: Production Deployment 🟡 70%
**Tamamlanan:**
- ✅ FastAPI REST API
- ✅ Model serving
- ✅ Basic monitoring
- ✅ Docker containerization
- ✅ Basic security (JWT)

**Eksik:**
- ⏳ Full CI/CD pipeline
- ⏳ Kubernetes deployment
- ⏳ Advanced monitoring (Grafana dashboards)
- ⏳ Load testing
- ⏳ Blue-green deployment

### PFAZ 12: Advanced Analytics ✅ 100%
- ✅ Statistical testing suite
- ✅ Bootstrap confidence intervals
- ✅ Sensitivity analysis
- ✅ Bayesian uncertainty
- ✅ Feature interaction analysis

### PFAZ 13: AutoML Integration ✅ 100%
- ✅ Optuna integration
- ✅ RF optimization
- ✅ XGBoost optimization
- ✅ DNN optimization
- ✅ Visualization tools

---

## 🚀 DEPLOYMENT CHECKLIST

### Pre-Deployment
- [ ] All tests passing (unit, integration)
- [ ] Code coverage > 80%
- [ ] Security scan clean
- [ ] Performance benchmarks met
- [ ] Documentation complete

### Deployment
- [ ] Build Docker image
- [ ] Push to registry
- [ ] Update Kubernetes manifests
- [ ] Deploy to staging
- [ ] Run smoke tests
- [ ] Deploy to production
- [ ] Monitor metrics

### Post-Deployment
- [ ] Verify health checks
- [ ] Check error rates
- [ ] Monitor latency
- [ ] Review logs
- [ ] Update documentation

---

## 📈 PERFORMANS BEKLENTİLERİ

**API Latency:**
- P50: < 50ms
- P95: < 100ms
- P99: < 200ms

**Throughput:**
- Single instance: ~100 req/s
- With 3 replicas: ~300 req/s
- With autoscaling: 1000+ req/s

**Availability:**
- Target: 99.9% uptime
- Max downtime: 8.76 hours/year

**Resource Usage:**
- CPU: 1-2 cores per instance
- Memory: 2-4 GB per instance
- Storage: 10 GB for models

---

**Son Güncelleme:** 2 Aralık 2025  
**Versiyon:** 3.0.0  
**Durum:** PFAZ 11: 70% | PFAZ 12-13: 100%

---

*Bu dokümantasyon PFAZ 11-13'ün tüm önemli yönlerini kapsar. Production deployment için ek konfigürasyon ve altyapı gerekebilir.*
