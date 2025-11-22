"""
İlerleme Takip Sistemi
Progress Tracking System with Time Estimation
"""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from tqdm import tqdm
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Gelişmiş ilerleme takip sistemi
    - İlerleme çubuğu (tqdm)
    - Tahmini süre (ETA)
    - Geçen süre
    - Detaylı zaman logları
    """
    
    def __init__(self, total_tasks, task_name="Tasks", log_file=None):
        """
        Args:
            total_tasks: Toplam görev sayısı
            task_name: Görev ismi (görüntü için)
            log_file: Zaman loglarını kaydetmek için dosya yolu
        """
        self.total_tasks = total_tasks
        self.task_name = task_name
        self.log_file = Path(log_file) if log_file else None
        
        # Zaman takibi
        self.start_time = None
        self.task_times = []  # Her görevin süresi
        self.task_logs = []   # Detaylı loglar
        
        # Progress bar
        self.pbar = None
        self.current_task = 0
    
    def start(self):
        """İlerlemeyi başlat"""
        self.start_time = time.time()
        self.pbar = tqdm(
            total=self.total_tasks,
            desc=f"{self.task_name}",
            unit="task",
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
            file=sys.stdout
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"[START] {self.task_name.upper()} BAŞLIYOR")
        logger.info(f"{'='*70}")
        logger.info(f"Toplam görev: {self.total_tasks}")
        logger.info(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def update(self, task_description="", elapsed_task_time=None):
        """
        İlerlemeyi güncelle
        
        Args:
            task_description: Görev açıklaması
            elapsed_task_time: Görevin süresi (saniye)
        """
        self.current_task += 1
        
        # Görev süresini kaydet
        if elapsed_task_time is not None:
            self.task_times.append(elapsed_task_time)
        
        # Log kaydet
        task_log = {
            'task_number': self.current_task,
            'description': task_description,
            'elapsed_time': elapsed_task_time,
            'timestamp': datetime.now().isoformat()
        }
        self.task_logs.append(task_log)
        
        # ETA hesapla
        if len(self.task_times) > 0:
            avg_time = sum(self.task_times) / len(self.task_times)
            remaining_tasks = self.total_tasks - self.current_task
            eta_seconds = avg_time * remaining_tasks
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "N/A"
        
        # Geçen süre
        elapsed_total = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_total)))
        
        # Progress bar güncelle
        postfix = {
            'Current': task_description[:30] if task_description else "...",
            'Elapsed': elapsed_str,
            'ETA': eta_str
        }
        
        if elapsed_task_time:
            postfix['Last'] = f"{elapsed_task_time:.1f}s"
        
        self.pbar.set_postfix(postfix)
        self.pbar.update(1)
    
    def finish(self):
        """İlerlemeyi bitir"""
        total_time = time.time() - self.start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        
        self.pbar.close()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"[SUCCESS] {self.task_name.upper()} TAMAMLANDI")
        logger.info(f"{'='*70}")
        logger.info(f"Toplam görev: {self.current_task}/{self.total_tasks}")
        logger.info(f"Toplam süre: {total_time_str}")
        
        if len(self.task_times) > 0:
            avg_time = sum(self.task_times) / len(self.task_times)
            min_time = min(self.task_times)
            max_time = max(self.task_times)
            
            logger.info(f"Görev başına ortalama: {avg_time:.2f}s")
            logger.info(f"En hızlı görev: {min_time:.2f}s")
            logger.info(f"En yavaş görev: {max_time:.2f}s")
        
        logger.info(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*70}\n")
        
        # Log dosyasına kaydet
        if self.log_file:
            self._save_log()
    
    def _save_log(self):
        """Detaylı logları JSON olarak kaydet"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_data = {
            'task_name': self.task_name,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.current_task,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': time.time() - self.start_time,
            'task_logs': self.task_logs,
            'statistics': {
                'avg_task_time': sum(self.task_times) / len(self.task_times) if self.task_times else 0,
                'min_task_time': min(self.task_times) if self.task_times else 0,
                'max_task_time': max(self.task_times) if self.task_times else 0,
                'total_logged_tasks': len(self.task_times)
            }
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Detaylı log kaydedildi: {self.log_file}")


class ModelTrainingTracker(ProgressTracker):
    """Model eğitimi için özelleştirilmiş tracker"""
    
    def __init__(self, total_datasets, model_name, log_dir='logs'):
        log_file = Path(log_dir) / f'{model_name}_training_log.json'
        super().__init__(
            total_tasks=total_datasets,
            task_name=f"{model_name} Training",
            log_file=log_file
        )
        self.model_name = model_name
        self.model_metrics = []
    
    def update_with_metrics(self, dataset_name, metrics, training_time):
        """
        Model metrikleriyle güncelle
        
        Args:
            dataset_name: Dataset adı
            metrics: dict (r2, rmse, mae, vb.)
            training_time: Eğitim süresi (saniye)
        """
        # Metrikleri kaydet
        self.model_metrics.append({
            'dataset': dataset_name,
            'metrics': metrics,
            'training_time': training_time
        })
        
        # Progress bar güncelle
        desc = f"{dataset_name[:40]} | R²={metrics.get('r2', 0):.4f}"
        self.update(task_description=desc, elapsed_task_time=training_time)
    
    def get_performance_summary(self):
        """Model performans özetini al"""
        if not self.model_metrics:
            return None
        
        r2_scores = [m['metrics'].get('r2', 0) for m in self.model_metrics]
        rmse_scores = [m['metrics'].get('rmse', 0) for m in self.model_metrics]
        times = [m['training_time'] for m in self.model_metrics]
        
        return {
            'model': self.model_name,
            'total_datasets': len(self.model_metrics),
            'r2_mean': sum(r2_scores) / len(r2_scores),
            'r2_std': (sum((x - sum(r2_scores)/len(r2_scores))**2 for x in r2_scores) / len(r2_scores))**0.5,
            'r2_min': min(r2_scores),
            'r2_max': max(r2_scores),
            'rmse_mean': sum(rmse_scores) / len(rmse_scores),
            'total_training_time': sum(times),
            'avg_training_time': sum(times) / len(times)
        }


class DatasetGenerationTracker(ProgressTracker):
    """Dataset oluşturma için özelleştirilmiş tracker"""
    
    def __init__(self, total_datasets, log_dir='logs'):
        log_file = Path(log_dir) / 'dataset_generation_log.json'
        super().__init__(
            total_tasks=total_datasets,
            task_name="Dataset Generation",
            log_file=log_file
        )
        self.dataset_info = []
    
    def update_with_dataset(self, dataset_name, n_samples, generation_time):
        """
        Dataset bilgileriyle güncelle
        
        Args:
            dataset_name: Dataset adı
            n_samples: Örnek sayısı
            generation_time: Oluşturma süresi (saniye)
        """
        self.dataset_info.append({
            'name': dataset_name,
            'samples': n_samples,
            'time': generation_time
        })
        
        desc = f"{dataset_name[:40]} | n={n_samples}"
        self.update(task_description=desc, elapsed_task_time=generation_time)


class MultiStageTracker:
    """
    Çoklu aşamalı projeler için tracker
    Örnek: Phase 1 -> Phase 2 -> Phase 3
    """
    
    def __init__(self, stages, log_dir='logs'):
        """
        Args:
            stages: dict {stage_name: total_tasks}
        """
        self.stages = stages
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_stage = None
        self.current_tracker = None
        self.stage_start_times = {}
        self.stage_end_times = {}
        
        self.project_start_time = None
    
    def start_project(self, project_name="Multi-Stage Project"):
        """Projeyi başlat"""
        self.project_start_time = time.time()
        self.project_name = project_name
        
        logger.info("\n" + "="*70)
        logger.info(f"[TARGET] {project_name.upper()}")
        logger.info("="*70)
        logger.info(f"Toplam aşama: {len(self.stages)}")
        for stage, tasks in self.stages.items():
            logger.info(f"  - {stage}: {tasks} görev")
        logger.info(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70 + "\n")
    
    def start_stage(self, stage_name):
        """Yeni aşama başlat"""
        if stage_name not in self.stages:
            raise ValueError(f"Stage '{stage_name}' not found in stages")
        
        # Önceki stage'i bitir
        if self.current_tracker:
            self.current_tracker.finish()
            self.stage_end_times[self.current_stage] = time.time()
        
        # Yeni stage başlat
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.time()
        
        total_tasks = self.stages[stage_name]
        self.current_tracker = ProgressTracker(
            total_tasks=total_tasks,
            task_name=stage_name,
            log_file=self.log_dir / f'{stage_name.lower().replace(" ", "_")}_log.json'
        )
        self.current_tracker.start()
    
    def update(self, task_description="", elapsed_task_time=None):
        """Mevcut aşamayı güncelle"""
        if self.current_tracker is None:
            raise RuntimeError("No active stage. Call start_stage() first.")
        
        self.current_tracker.update(task_description, elapsed_task_time)
    
    def finish_project(self):
        """Projeyi bitir"""
        # Son stage'i bitir
        if self.current_tracker:
            self.current_tracker.finish()
            self.stage_end_times[self.current_stage] = time.time()
        
        # Proje özeti
        total_time = time.time() - self.project_start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        
        logger.info("\n" + "="*70)
        logger.info(f"[BEST] {self.project_name.upper()} TAMAMLANDI")
        logger.info("="*70)
        logger.info(f"Toplam süre: {total_time_str}")
        logger.info("\nAşama süreleri:")
        
        for stage in self.stages.keys():
            if stage in self.stage_start_times and stage in self.stage_end_times:
                stage_time = self.stage_end_times[stage] - self.stage_start_times[stage]
                stage_time_str = str(timedelta(seconds=int(stage_time)))
                pct = (stage_time / total_time) * 100
                logger.info(f"  {stage}: {stage_time_str} ({pct:.1f}%)")
        
        logger.info(f"\nBitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70 + "\n")
        
        # Genel log kaydet
        self._save_project_log(total_time)
    
    def _save_project_log(self, total_time):
        """Proje logunu kaydet"""
        log_file = self.log_dir / 'project_summary.json'
        
        stage_durations = {}
        for stage in self.stages.keys():
            if stage in self.stage_start_times and stage in self.stage_end_times:
                duration = self.stage_end_times[stage] - self.stage_start_times[stage]
                stage_durations[stage] = {
                    'duration_seconds': duration,
                    'duration_str': str(timedelta(seconds=int(duration))),
                    'percentage': (duration / total_time) * 100
                }
        
        log_data = {
            'project_name': self.project_name,
            'start_time': datetime.fromtimestamp(self.project_start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': total_time,
            'total_duration_str': str(timedelta(seconds=int(total_time))),
            'stages': self.stages,
            'stage_durations': stage_durations
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Proje özeti kaydedildi: {log_file}")


def main():
    """Test fonksiyonu"""
    import random
    
    # Basit tracker testi
    print("\n=== BASIT TRACKER TESTİ ===")
    tracker = ProgressTracker(10, "Test Tasks", log_file="logs/test_log.json")
    tracker.start()
    
    for i in range(10):
        task_time = random.uniform(0.5, 2.0)
        time.sleep(task_time)
        tracker.update(f"Task {i+1}", task_time)
    
    tracker.finish()
    
    # Model training tracker testi
    print("\n=== MODEL TRAINING TRACKER TESTİ ===")
    model_tracker = ModelTrainingTracker(5, "RandomForest", log_dir="logs")
    model_tracker.start()
    
    for i in range(5):
        metrics = {
            'r2': random.uniform(0.7, 0.95),
            'rmse': random.uniform(0.1, 0.5),
            'mae': random.uniform(0.05, 0.3)
        }
        train_time = random.uniform(1.0, 3.0)
        time.sleep(train_time)
        model_tracker.update_with_metrics(f"Dataset_{i+1}", metrics, train_time)
    
    model_tracker.finish()
    summary = model_tracker.get_performance_summary()
    print(f"\nPerformans özeti: R² mean={summary['r2_mean']:.4f}")
    
    # Multi-stage tracker testi
    print("\n=== MULTI-STAGE TRACKER TESTİ ===")
    stages = {
        'Data Preparation': 5,
        'Model Training': 10,
        'Evaluation': 3
    }
    
    multi_tracker = MultiStageTracker(stages, log_dir="logs")
    multi_tracker.start_project("Test Project")
    
    for stage, tasks in stages.items():
        multi_tracker.start_stage(stage)
        for i in range(tasks):
            task_time = random.uniform(0.3, 1.0)
            time.sleep(task_time)
            multi_tracker.update(f"Task {i+1}", task_time)
    
    multi_tracker.finish_project()
    
    print("\n[SUCCESS] Tüm testler tamamlandı")


if __name__ == "__main__":
    main()