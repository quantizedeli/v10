"""
Production Monitoring & Alerting System
Real-time Model Performance Monitoring

Bu modül, production'da model performansını izler:
1. Real-time Performance Metrics
2. Data Drift Detection
3. Model Degradation Detection
4. Alerting System
5. Performance Dashboard
6. Logging & Audit Trail
7. Anomaly Detection
8. Auto-retraining Triggers

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import threading
import time
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    throughput_rps: float  # requests per second


@dataclass
class DataDriftMetrics:
    """Data drift metrics"""
    timestamp: str
    feature_drifts: Dict[str, float]
    overall_drift_score: float
    drift_detected: bool
    drift_threshold: float = 0.1


@dataclass
class Alert:
    """Alert model"""
    timestamp: str
    severity: str  # 'critical', 'warning', 'info'
    alert_type: str  # 'performance', 'drift', 'error', 'latency'
    message: str
    details: Dict


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Metrics buffers
        self.latencies = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        
        # Aggregated metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Performance Monitor initialized (window={window_size})")
    
    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request"""
        with self.lock:
            self.latencies.append(latency_ms)
            self.timestamps.append(time.time())
            self.errors.append(0 if success else 1)
            
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self.lock:
            if len(self.latencies) == 0:
                return PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    total_requests=0,
                    successful_requests=0,
                    failed_requests=0,
                    avg_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    error_rate=0.0,
                    throughput_rps=0.0
                )
            
            latencies_array = np.array(self.latencies)
            errors_array = np.array(self.errors)
            
            # Calculate throughput
            if len(self.timestamps) > 1:
                time_span = self.timestamps[-1] - self.timestamps[0]
                throughput = len(self.timestamps) / time_span if time_span > 0 else 0
            else:
                throughput = 0
            
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                total_requests=self.total_requests,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                avg_latency_ms=float(np.mean(latencies_array)),
                p95_latency_ms=float(np.percentile(latencies_array, 95)),
                p99_latency_ms=float(np.percentile(latencies_array, 99)),
                error_rate=float(np.mean(errors_array)),
                throughput_rps=float(throughput)
            )
    
    def reset(self):
        """Reset metrics"""
        with self.lock:
            self.latencies.clear()
            self.timestamps.clear()
            self.errors.clear()
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0


class DataDriftDetector:
    """Data drift detection using statistical methods"""
    
    def __init__(self, reference_data: np.ndarray, feature_names: List[str]):
        self.reference_data = reference_data
        self.feature_names = feature_names
        
        # Calculate reference statistics
        self.reference_mean = reference_data.mean(axis=0)
        self.reference_std = reference_data.std(axis=0)
        
        # History
        self.drift_history = []
        
        logger.info(f"Data Drift Detector initialized ({len(feature_names)} features)")
    
    def detect_drift(self, 
                    current_data: np.ndarray,
                    threshold: float = 0.1) -> DataDriftMetrics:
        """
        Detect data drift using KL divergence approximation
        
        Args:
            current_data: Current data samples
            threshold: Drift threshold
        """
        current_mean = current_data.mean(axis=0)
        current_std = current_data.std(axis=0)
        
        # Calculate drift score for each feature (normalized difference)
        feature_drifts = {}
        drift_scores = []
        
        for i, fname in enumerate(self.feature_names):
            # Normalized difference in means
            mean_diff = abs(current_mean[i] - self.reference_mean[i])
            normalized_diff = mean_diff / (self.reference_std[i] + 1e-10)
            
            feature_drifts[fname] = float(normalized_diff)
            drift_scores.append(normalized_diff)
        
        # Overall drift score (average)
        overall_drift = float(np.mean(drift_scores))
        drift_detected = overall_drift > threshold
        
        metrics = DataDriftMetrics(
            timestamp=datetime.now().isoformat(),
            feature_drifts=feature_drifts,
            overall_drift_score=overall_drift,
            drift_detected=drift_detected,
            drift_threshold=threshold
        )
        
        # Store in history
        self.drift_history.append(metrics)
        
        if drift_detected:
            logger.warning(f"[WARNING] Data drift detected! Score: {overall_drift:.4f}")
        
        return metrics
    
    def get_top_drifted_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top drifted features"""
        if not self.drift_history:
            return []
        
        latest = self.drift_history[-1]
        sorted_features = sorted(
            latest.feature_drifts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:n]


class ModelDegradationDetector:
    """Detect model performance degradation over time"""
    
    def __init__(self, baseline_metrics: Dict, degradation_threshold: float = 0.1):
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold
        
        # History
        self.performance_history = []
        
        logger.info("Model Degradation Detector initialized")
    
    def check_degradation(self, current_metrics: Dict) -> Tuple[bool, Dict]:
        """
        Check for model degradation
        
        Returns:
            (degraded, degradation_details)
        """
        degradation = {}
        degraded = False
        
        # Compare key metrics
        for metric in ['R2', 'MAE', 'RMSE']:
            if metric in self.baseline_metrics and metric in current_metrics:
                baseline = self.baseline_metrics[metric]
                current = current_metrics[metric]
                
                if metric == 'R2':
                    # R2: lower is worse
                    change = (baseline - current) / baseline if baseline != 0 else 0
                else:
                    # MAE, RMSE: higher is worse
                    change = (current - baseline) / baseline if baseline != 0 else 0
                
                degradation[metric] = {
                    'baseline': baseline,
                    'current': current,
                    'change_percent': change * 100
                }
                
                if abs(change) > self.degradation_threshold:
                    degraded = True
        
        # Store in history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics,
            'degraded': degraded
        })
        
        if degraded:
            logger.warning(f"[WARNING] Model degradation detected!")
        
        return degraded, degradation


class AlertManager:
    """Alert management system"""
    
    def __init__(self, alert_file: str = 'alerts.json'):
        self.alert_file = Path(alert_file)
        self.alerts = []
        
        # Alert thresholds
        self.thresholds = {
            'error_rate': 0.05,  # 5%
            'latency_p95': 1000,  # 1000ms
            'drift_score': 0.1,
            'degradation': 0.1
        }
        
        logger.info("Alert Manager initialized")
    
    def create_alert(self, 
                    severity: str,
                    alert_type: str,
                    message: str,
                    details: Dict) -> Alert:
        """Create an alert"""
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            severity=severity,
            alert_type=alert_type,
            message=message,
            details=details
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_fn = logger.critical if severity == 'critical' else logger.warning
        log_fn(f"🚨 {severity.upper()} ALERT: {message}")
        
        # Save to file
        self._save_alerts()
        
        return alert
    
    def check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check performance metrics for alerts"""
        # Error rate alert
        if metrics.error_rate > self.thresholds['error_rate']:
            self.create_alert(
                severity='critical',
                alert_type='error',
                message=f"High error rate: {metrics.error_rate*100:.2f}%",
                details=asdict(metrics)
            )
        
        # Latency alert
        if metrics.p95_latency_ms > self.thresholds['latency_p95']:
            self.create_alert(
                severity='warning',
                alert_type='latency',
                message=f"High P95 latency: {metrics.p95_latency_ms:.2f}ms",
                details=asdict(metrics)
            )
    
    def check_drift_alerts(self, drift_metrics: DataDriftMetrics):
        """Check drift metrics for alerts"""
        if drift_metrics.drift_detected:
            self.create_alert(
                severity='warning',
                alert_type='drift',
                message=f"Data drift detected: {drift_metrics.overall_drift_score:.4f}",
                details=asdict(drift_metrics)
            )
    
    def check_degradation_alerts(self, degraded: bool, degradation_details: Dict):
        """Check degradation for alerts"""
        if degraded:
            self.create_alert(
                severity='critical',
                alert_type='performance',
                message="Model performance degradation detected",
                details=degradation_details
            )
    
    def get_recent_alerts(self, hours: int = 24, severity: Optional[str] = None) -> List[Alert]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert.timestamp) > cutoff
        ]
        
        if severity:
            recent = [a for a in recent if a.severity == severity]
        
        return recent
    
    def _save_alerts(self):
        """Save alerts to file"""
        with open(self.alert_file, 'w') as f:
            json.dump([asdict(a) for a in self.alerts], f, indent=2)


class ProductionMonitoringSystem:
    """Comprehensive production monitoring system"""
    
    def __init__(self,
                 reference_data: np.ndarray,
                 feature_names: List[str],
                 baseline_metrics: Dict,
                 output_dir: str = 'monitoring'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(window_size=1000)
        self.drift_detector = DataDriftDetector(reference_data, feature_names)
        self.degradation_detector = ModelDegradationDetector(baseline_metrics)
        self.alert_manager = AlertManager(str(self.output_dir / 'alerts.json'))
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
        # Metrics history
        self.metrics_history = []
        
        logger.info("Production Monitoring System initialized")
    
    def start_monitoring(self, check_interval: int = 60):
        """Start monitoring loop"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Monitoring started (interval={check_interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self, check_interval: int):
        """Monitoring loop"""
        while self.monitoring:
            try:
                # Get current metrics
                perf_metrics = self.performance_monitor.get_current_metrics()
                
                # Check for performance alerts
                self.alert_manager.check_performance_alerts(perf_metrics)
                
                # Store metrics
                self.metrics_history.append(perf_metrics)
                
                # Save dashboard
                self._save_dashboard()
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(check_interval)
    
    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request (wrapper)"""
        self.performance_monitor.record_request(latency_ms, success)
    
    def check_drift(self, current_data: np.ndarray):
        """Check for data drift"""
        drift_metrics = self.drift_detector.detect_drift(current_data)
        self.alert_manager.check_drift_alerts(drift_metrics)
        return drift_metrics
    
    def check_degradation(self, current_metrics: Dict):
        """Check for model degradation"""
        degraded, details = self.degradation_detector.check_degradation(current_metrics)
        self.alert_manager.check_degradation_alerts(degraded, details)
        return degraded, details
    
    def get_dashboard_data(self) -> Dict:
        """Get dashboard data"""
        current_metrics = self.performance_monitor.get_current_metrics()
        recent_alerts = self.alert_manager.get_recent_alerts(hours=24)
        
        return {
            'current_metrics': asdict(current_metrics),
            'recent_alerts': [asdict(a) for a in recent_alerts],
            'alert_summary': {
                'total': len(recent_alerts),
                'critical': len([a for a in recent_alerts if a.severity == 'critical']),
                'warning': len([a for a in recent_alerts if a.severity == 'warning'])
            },
            'top_drifted_features': self.drift_detector.get_top_drifted_features(5)
        }
    
    def _save_dashboard(self):
        """Save dashboard data"""
        dashboard_file = self.output_dir / 'dashboard.json'
        with open(dashboard_file, 'w') as f:
            json.dump(self.get_dashboard_data(), f, indent=2)
    
    def export_metrics_history(self, output_file: str = 'metrics_history.csv'):
        """Export metrics history to CSV"""
        if not self.metrics_history:
            logger.warning("No metrics history to export")
            return
        
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        output_path = self.output_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"[OK] Metrics history exported: {output_path}")


def main():
    """Test function"""
    # Create dummy reference data
    np.random.seed(42)
    reference_data = np.random.randn(1000, 5)
    feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
    baseline_metrics = {'R2': 0.95, 'MAE': 0.1, 'RMSE': 0.15}
    
    # Initialize monitoring
    monitoring = ProductionMonitoringSystem(
        reference_data, feature_names, baseline_metrics, 'test_monitoring'
    )
    
    # Simulate requests
    print("\n[OK] Simulating requests...")
    for i in range(100):
        latency = np.random.uniform(10, 100)
        success = np.random.rand() > 0.05  # 95% success rate
        monitoring.record_request(latency, success)
    
    # Get current metrics
    metrics = monitoring.performance_monitor.get_current_metrics()
    print(f"\n[OK] Current Metrics:")
    print(f"  Total Requests: {metrics.total_requests}")
    print(f"  Avg Latency: {metrics.avg_latency_ms:.2f} ms")
    print(f"  P95 Latency: {metrics.p95_latency_ms:.2f} ms")
    print(f"  Error Rate: {metrics.error_rate*100:.2f}%")
    
    # Test drift detection
    print("\n[OK] Testing drift detection...")
    current_data = np.random.randn(100, 5) + 0.5  # Shifted data
    drift_metrics = monitoring.check_drift(current_data)
    print(f"  Overall Drift: {drift_metrics.overall_drift_score:.4f}")
    print(f"  Drift Detected: {drift_metrics.drift_detected}")
    
    # Dashboard
    print("\n[OK] Dashboard data:")
    dashboard = monitoring.get_dashboard_data()
    print(f"  Total Alerts: {dashboard['alert_summary']['total']}")
    
    print("\n[OK] Test completed!")


if __name__ == "__main__":
    main()
