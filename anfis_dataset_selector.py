"""
ANFIS Dataset Selector
ANFIS için Dataset Seçim Stratejisi

İki Metot:
1. Method 1 - Layered Selection:
   - Top performers (R² > 0.90): 20 datasets
   - Mid performers (0.80 < R² < 0.90): 15 datasets
   - Low performers (R² < 0.80): 15 datasets
   
2. Method 2 - Balanced Selection:
   - Feature set balance
   - Scenario balance (S70/S80)
   - Anomaly mode balance
   - Top 50 by composite score

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import shutil
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ANFIS DATASET SELECTOR
# ============================================================================

class ANFISDatasetSelector:
    """
    ANFIS için optimal dataset seçimi
    
    AI model training sonuçlarını analiz eder ve
    ANFIS eğitimi için en uygun 50 dataset seçer.
    """
    
    def __init__(self, ai_results_dir='trained_models', output_dir='ANFIS_Selected_Datasets'):
        self.ai_results_dir = Path(ai_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_df = None
        self.targets = ['MM', 'QM', 'MM_QM', 'Beta_2']
        
        logger.info("ANFIS Dataset Selector başlatıldı")
    
    def load_ai_results(self, summary_file=None):
        """
        AI training sonuçlarını yükle
        
        Args:
            summary_file: Training summary Excel/CSV file
        """
        
        logger.info("AI training sonuçları yükleniyor...")
        
        if summary_file is None:
            # Default: training_summary.xlsx
            summary_file = self.ai_results_dir / 'training_summary.xlsx'
        
        summary_file = Path(summary_file)
        
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file bulunamadı: {summary_file}")
        
        # Load
        if summary_file.suffix == '.xlsx':
            self.results_df = pd.read_excel(summary_file)
        elif summary_file.suffix == '.csv':
            self.results_df = pd.read_csv(summary_file)
        else:
            raise ValueError(f"Desteklenmeyen format: {summary_file.suffix}")
        
        logger.info(f"✓ {len(self.results_df)} sonuç yüklendi")
        
        return self.results_df
    
    def select_method_1_layered(self, target, n_datasets=50):
        """
        Method 1: Layered Selection
        
        Performance katmanlarına göre seçim:
        - Top: R² > 0.90 (20 datasets)
        - Mid: 0.80 < R² < 0.90 (15 datasets)
        - Low: R² < 0.80 (15 datasets)
        
        Args:
            target: Target variable (MM, QM, MM_QM, Beta_2)
            n_datasets: Total datasets to select
        """
        
        logger.info(f"\nMethod 1 (Layered) - Target: {target}")
        
        # Filter by target
        target_df = self.results_df[self.results_df['Target'] == target].copy()
        
        if len(target_df) == 0:
            logger.warning(f"⚠ {target} için sonuç bulunamadı!")
            return pd.DataFrame()
        
        # Layers
        top_threshold = 0.90
        mid_threshold = 0.80
        
        top_df = target_df[target_df['R2_test'] >= top_threshold]
        mid_df = target_df[(target_df['R2_test'] >= mid_threshold) & 
                           (target_df['R2_test'] < top_threshold)]
        low_df = target_df[target_df['R2_test'] < mid_threshold]
        
        logger.info(f"  Top layer (R²≥{top_threshold}): {len(top_df)} datasets")
        logger.info(f"  Mid layer ({mid_threshold}≤R²<{top_threshold}): {len(mid_df)} datasets")
        logger.info(f"  Low layer (R²<{mid_threshold}): {len(low_df)} datasets")
        
        # Selection counts
        n_top = min(20, len(top_df))
        n_mid = min(15, len(mid_df))
        n_low = min(15, len(low_df))
        
        # Adjust if total < 50
        total_available = n_top + n_mid + n_low
        if total_available < n_datasets:
            logger.warning(f"  ⚠ Sadece {total_available} dataset mevcut (hedef: {n_datasets})")
            n_datasets = total_available
        
        # Select randomly from each layer (or all if not enough)
        selected_top = top_df.sample(n=n_top, random_state=42) if len(top_df) > n_top else top_df
        selected_mid = mid_df.sample(n=n_mid, random_state=42) if len(mid_df) > n_mid else mid_df
        selected_low = low_df.sample(n=n_low, random_state=42) if len(low_df) > n_low else low_df
        
        # Combine
        selected = pd.concat([selected_top, selected_mid, selected_low], ignore_index=True)
        
        # Add selection method column
        selected['Selection_Method'] = 'Layered'
        selected['Layer'] = (['Top'] * len(selected_top) + 
                             ['Mid'] * len(selected_mid) + 
                             ['Low'] * len(selected_low))
        
        logger.info(f"  ✓ {len(selected)} datasets seçildi")
        
        return selected
    
    def select_method_2_balanced(self, target, n_datasets=50):
        """
        Method 2: Balanced Selection
        
        Top performers içinden balanced seçim:
        - Feature set balance
        - Scenario balance (S70/S80)
        - Anomaly mode balance
        
        Args:
            target: Target variable
            n_datasets: Total datasets to select
        """
        
        logger.info(f"\nMethod 2 (Balanced) - Target: {target}")
        
        # Filter by target
        target_df = self.results_df[self.results_df['Target'] == target].copy()
        
        if len(target_df) == 0:
            logger.warning(f"⚠ {target} için sonuç bulunamadı!")
            return pd.DataFrame()
        
        # Sort by Composite Score (or R² if not available)
        if 'Composite_Score' in target_df.columns:
            target_df = target_df.sort_values('Composite_Score', ascending=False)
        else:
            target_df = target_df.sort_values('R2_test', ascending=False)
        
        # Extract metadata from dataset names
        target_df = self._parse_dataset_metadata(target_df)
        
        # Stratified sampling
        selected = self._stratified_sample(target_df, n_datasets)
        
        # Add selection method
        selected['Selection_Method'] = 'Balanced'
        
        logger.info(f"  ✓ {len(selected)} datasets seçildi")
        
        return selected
    
    def _parse_dataset_metadata(self, df):
        """Dataset isimlerinden metadata çıkar"""
        
        # Example format: MM_100_S70_Anomaly_StandardScaler_Random
        
        def parse_name(name):
            parts = name.split('_')
            return {
                'nucleus_count': parts[1] if len(parts) > 1 else 'Unknown',
                'scenario': parts[2] if len(parts) > 2 else 'Unknown',
                'anomaly_mode': 'WithAnomaly' if 'Anomaly' in name else 'NoAnomaly',
                'scaling': 'Standard' if 'Standard' in name else ('Robust' if 'Robust' in name else 'None'),
                'sampling': 'Random' if 'Random' in name else 'Stratified'
            }
        
        metadata = df['Dataset_Name'].apply(parse_name)
        metadata_df = pd.DataFrame(metadata.tolist())
        
        return pd.concat([df.reset_index(drop=True), metadata_df], axis=1)
    
    def _stratified_sample(self, df, n_datasets):
        """Stratified sampling for balance"""
        
        # Define strata
        strata_cols = ['scenario', 'anomaly_mode', 'sampling']
        
        # Group by strata
        grouped = df.groupby(strata_cols)
        
        # Calculate samples per stratum
        n_strata = len(grouped)
        samples_per_stratum = n_datasets // n_strata
        remainder = n_datasets % n_strata
        
        selected_dfs = []
        
        for i, (name, group) in enumerate(grouped):
            # Add 1 extra to first 'remainder' groups
            n_samples = samples_per_stratum + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(group))
            
            # Sample
            sampled = group.head(n_samples)  # Already sorted by composite score
            selected_dfs.append(sampled)
        
        selected = pd.concat(selected_dfs, ignore_index=True)
        
        # Log balance
        logger.info("  Balance:")
        for col in strata_cols:
            counts = selected[col].value_counts()
            logger.info(f"    {col}: {dict(counts)}")
        
        return selected
    
    def select_both_methods(self, targets=None, n_datasets=50):
        """
        Her target için her iki metodla seçim yap
        
        Args:
            targets: List of targets (None = all)
            n_datasets: Datasets per target per method
        
        Returns:
            dict: {target: {'method1': df, 'method2': df}}
        """
        
        if targets is None:
            targets = self.targets
        
        logger.info("\n" + "="*80)
        logger.info("ANFIS DATASET SELECTION - BOTH METHODS")
        logger.info("="*80)
        
        all_selections = {}
        
        for target in targets:
            logger.info(f"\n{'='*80}")
            logger.info(f"TARGET: {target}")
            logger.info(f"{'='*80}")
            
            # Method 1
            method1_df = self.select_method_1_layered(target, n_datasets)
            
            # Method 2
            method2_df = self.select_method_2_balanced(target, n_datasets)
            
            all_selections[target] = {
                'method1': method1_df,
                'method2': method2_df
            }
            
            # Save
            self._save_selection(target, method1_df, method2_df)
        
        # Summary report
        self._create_summary_report(all_selections)
        
        logger.info("\n" + "="*80)
        logger.info("✓ DATASET SELECTION TAMAMLANDI")
        logger.info("="*80)
        
        return all_selections
    
    def _save_selection(self, target, method1_df, method2_df):
        """Save selection results"""
        
        # Method 1
        if not method1_df.empty:
            save_dir1 = self.output_dir / 'Method_1_Layered' / target
            save_dir1.mkdir(parents=True, exist_ok=True)
            
            method1_df.to_csv(save_dir1 / 'selected_datasets.csv', index=False)
            
            # Copy dataset files
            self._copy_dataset_files(method1_df, save_dir1)
        
        # Method 2
        if not method2_df.empty:
            save_dir2 = self.output_dir / 'Method_2_Balanced' / target
            save_dir2.mkdir(parents=True, exist_ok=True)
            
            method2_df.to_csv(save_dir2 / 'selected_datasets.csv', index=False)
            
            # Copy dataset files
            self._copy_dataset_files(method2_df, save_dir2)
        
        logger.info(f"  ✓ {target} seçimleri kaydedildi")
    
    def _copy_dataset_files(self, df, dest_dir):
        """Copy dataset files to destination"""
        
        datasets_dir = dest_dir / 'datasets'
        datasets_dir.mkdir(exist_ok=True)
        
        for idx, row in df.iterrows():
            dataset_name = row['Dataset_Name']
            
            # Find source dataset
            source_path = self._find_dataset_path(dataset_name)
            
            if source_path and source_path.exists():
                # Copy entire dataset folder
                dest_path = datasets_dir / dataset_name
                
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                
                shutil.copytree(source_path, dest_path)
            else:
                logger.warning(f"    ⚠ Dataset bulunamadı: {dataset_name}")
    
    def _find_dataset_path(self, dataset_name):
        """Find dataset path"""
        
        # Search in common locations
        search_paths = [
            Path('datasets') / dataset_name,
            Path('output/datasets') / dataset_name,
            self.ai_results_dir.parent / 'datasets' / dataset_name
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _create_summary_report(self, all_selections):
        """Create summary report"""
        
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'targets': {},
            'totals': {
                'method1': 0,
                'method2': 0
            }
        }
        
        for target, selections in all_selections.items():
            method1_count = len(selections['method1'])
            method2_count = len(selections['method2'])
            
            summary['targets'][target] = {
                'method1_count': method1_count,
                'method2_count': method2_count
            }
            
            summary['totals']['method1'] += method1_count
            summary['totals']['method2'] += method2_count
            
            # Method 1 stats
            if not selections['method1'].empty:
                m1_df = selections['method1']
                summary['targets'][target]['method1_stats'] = {
                    'r2_mean': float(m1_df['R2_test'].mean()),
                    'r2_std': float(m1_df['R2_test'].std()),
                    'r2_min': float(m1_df['R2_test'].min()),
                    'r2_max': float(m1_df['R2_test'].max())
                }
            
            # Method 2 stats
            if not selections['method2'].empty:
                m2_df = selections['method2']
                summary['targets'][target]['method2_stats'] = {
                    'r2_mean': float(m2_df['R2_test'].mean()),
                    'r2_std': float(m2_df['R2_test'].std()),
                    'r2_min': float(m2_df['R2_test'].min()),
                    'r2_max': float(m2_df['R2_test'].max())
                }
        
        # Save summary
        with open(self.output_dir / 'selection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✓ Summary report: {self.output_dir / 'selection_summary.json'}")
        
        # Print summary
        print("\n" + "="*80)
        print("SELECTION SUMMARY")
        print("="*80)
        print(f"\nTotal Datasets Selected:")
        print(f"  Method 1 (Layered): {summary['totals']['method1']}")
        print(f"  Method 2 (Balanced): {summary['totals']['method2']}")
        print(f"\nPer Target:")
        for target, stats in summary['targets'].items():
            print(f"\n  {target}:")
            print(f"    Method 1: {stats['method1_count']} datasets")
            if 'method1_stats' in stats:
                print(f"      R² range: [{stats['method1_stats']['r2_min']:.4f}, {stats['method1_stats']['r2_max']:.4f}]")
            print(f"    Method 2: {stats['method2_count']} datasets")
            if 'method2_stats' in stats:
                print(f"      R² range: [{stats['method2_stats']['r2_min']:.4f}, {stats['method2_stats']['r2_max']:.4f}]")


# ============================================================================
# ANFIS TRAINING QUEUE GENERATOR
# ============================================================================

class ANFISTrainingQueueGenerator:
    """
    ANFIS training queue oluştur
    
    Seçilen datasetler için eğitim sırası ve konfigürasyonları hazırla
    """
    
    def __init__(self, selected_datasets_dir='ANFIS_Selected_Datasets'):
        self.selected_datasets_dir = Path(selected_datasets_dir)
        self.queue = []
        
        logger.info("ANFIS Training Queue Generator başlatıldı")
    
    def generate_queue(self):
        """Generate training queue"""
        
        logger.info("\nANFIS training queue oluşturuluyor...")
        
        # Iterate through methods and targets
        for method_dir in self.selected_datasets_dir.glob('Method_*'):
            method_name = method_dir.name
            
            for target_dir in method_dir.glob('*'):
                if not target_dir.is_dir():
                    continue
                
                target = target_dir.name
                
                # Load selected datasets
                csv_file = target_dir / 'selected_datasets.csv'
                if not csv_file.exists():
                    continue
                
                df = pd.read_csv(csv_file)
                
                # Add to queue
                for idx, row in df.iterrows():
                    dataset_name = row['Dataset_Name']
                    
                    queue_item = {
                        'method': method_name,
                        'target': target,
                        'dataset_name': dataset_name,
                        'dataset_path': target_dir / 'datasets' / dataset_name,
                        'ai_r2': float(row['R2_test']),
                        'priority': self._calculate_priority(row)
                    }
                    
                    self.queue.append(queue_item)
        
        # Sort by priority (high to low)
        self.queue.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"✓ {len(self.queue)} ANFIS training task oluşturuldu")
        
        # Save queue
        self._save_queue()
        
        return self.queue
    
    def _calculate_priority(self, row):
        """Calculate training priority"""
        
        # Higher priority for:
        # 1. Better AI performance (R²)
        # 2. Balanced datasets (Method 2)
        # 3. Top layer (Method 1)
        
        priority = float(row['R2_test'])
        
        if row.get('Selection_Method') == 'Balanced':
            priority += 0.1
        
        if row.get('Layer') == 'Top':
            priority += 0.05
        
        return priority
    
    def _save_queue(self):
        """Save training queue"""
        
        queue_df = pd.DataFrame(self.queue)
        
        save_path = self.selected_datasets_dir / 'anfis_training_queue.csv'
        queue_df.to_csv(save_path, index=False)
        
        logger.info(f"✓ Training queue saved: {save_path}")
        
        # JSON version
        json_path = self.selected_datasets_dir / 'anfis_training_queue.json'
        with open(json_path, 'w') as f:
            json.dump(self.queue, f, indent=2, default=str)
        
        logger.info(f"✓ Training queue (JSON): {json_path}")


# ============================================================================
# MAIN TEST
# ============================================================================

def test_anfis_dataset_selector():
    """Test ANFIS dataset selector"""
    
    print("\n" + "="*80)
    print("ANFIS DATASET SELECTOR TEST")
    print("="*80)
    
    # Create dummy AI results
    np.random.seed(42)
    n_results = 200
    
    targets = ['MM', 'QM', 'MM_QM', 'Beta_2']
    scenarios = ['S70', 'S80']
    anomaly_modes = ['Anomaly', 'NoAnomaly']
    scalings = ['StandardScaler', 'RobustScaler', 'None']
    samplings = ['Random', 'Stratified']
    
    dummy_results = []
    
    for i in range(n_results):
        target = np.random.choice(targets)
        scenario = np.random.choice(scenarios)
        anomaly = np.random.choice(anomaly_modes)
        scaling = np.random.choice(scalings)
        sampling = np.random.choice(samplings)
        
        dataset_name = f"{target}_100_{scenario}_{anomaly}_{scaling}_{sampling}"
        
        # Random R² (biased towards higher values)
        r2 = np.random.beta(5, 2)  # Beta distribution
        
        dummy_results.append({
            'Dataset_Name': dataset_name,
            'Target': target,
            'R2_test': r2,
            'RMSE_test': np.random.uniform(0.05, 0.3),
            'MAE_test': np.random.uniform(0.03, 0.2),
            'Composite_Score': r2 * 0.8 + np.random.uniform(0, 0.2)
        })
    
    results_df = pd.DataFrame(dummy_results)
    
    # Save dummy results
    results_df.to_csv('test_ai_results.csv', index=False)
    
    # Test selector
    selector = ANFISDatasetSelector(output_dir='test_anfis_selection')
    selector.results_df = results_df
    
    # Select datasets
    selections = selector.select_both_methods(targets=['MM', 'QM'], n_datasets=50)
    
    # Generate queue
    queue_gen = ANFISTrainingQueueGenerator('test_anfis_selection')
    queue = queue_gen.generate_queue()
    
    print("\n" + "="*80)
    print(f"✓ Test tamamlandı!")
    print(f"  Total queue items: {len(queue)}")
    print(f"  Top 5 priority tasks:")
    for i, task in enumerate(queue[:5]):
        print(f"    {i+1}. {task['dataset_name']} (Priority: {task['priority']:.4f})")
    print("="*80)


if __name__ == "__main__":
    test_anfis_dataset_selector()