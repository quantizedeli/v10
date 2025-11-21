"""
ANFIS Configuration Manager
8 different FIS configurations like ooo.py
"""

from typing import Dict, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FISConfig:
    """FIS Configuration structure"""
    id: str
    name: str
    method: str  # 'grid' or 'subclust'
    mfs: int = None  # Number of membership functions (for grid)
    mf_type: str = None  # 'gaussmf', 'gbellmf', 'trimf', 'trapmf' (for grid)
    radii: float = None  # Cluster radii (for subclust)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'method': self.method,
            'mfs': self.mfs,
            'mf_type': self.mf_type,
            'radii': self.radii
        }
    
    def generate_matlab_code(self, data_var: str = 'tr') -> str:
        """
        Generate MATLAB code for FIS initialization
        
        Args:
            data_var: Name of data variable in MATLAB
            
        Returns:
            MATLAB code string
        """
        if self.method == 'grid':
            # Grid partition
            return f"fis = genfis1({data_var}, {self.mfs}, '{self.mf_type}');"
        
        elif self.method == 'subclust':
            # Subtractive clustering
            return f"fis = genfis2({data_var}(:,1:end-1), {data_var}(:,end), {self.radii});"
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def __str__(self) -> str:
        if self.method == 'grid':
            return f"{self.name} (Grid: {self.mfs} MFs, {self.mf_type})"
        else:
            return f"{self.name} (SubClust: radii={self.radii})"


class ANFISConfigManager:
    """
    Manages all ANFIS configurations
    Based on ooo.py's 8 configurations
    """
    
    def __init__(self):
        self.configs = self._initialize_configs()
        logger.info(f"Initialized {len(self.configs)} ANFIS configurations")
    
    def _initialize_configs(self) -> List[FISConfig]:
        """
        Initialize 8 FIS configurations matching ooo.py
        
        Configurations:
        1. Grid_2MF_Gauss - Grid partition, 2 MFs, Gaussian
        2. Grid_2MF_Bell - Grid partition, 2 MFs, Generalized Bell
        3. Grid_2MF_Tri - Grid partition, 2 MFs, Triangular
        4. Grid_2MF_Trap - Grid partition, 2 MFs, Trapezoidal
        5. Grid_3MF_Gauss - Grid partition, 3 MFs, Gaussian
        6. SubClust_R03 - Subtractive clustering, radii=0.3
        7. SubClust_R05 - Subtractive clustering, radii=0.5
        8. SubClust_R07 - Subtractive clustering, radii=0.7
        """
        return [
            # Grid Partition - Different MF types (2 MFs each)
            FISConfig(
                id='CFG001',
                name='Grid_2MF_Gauss',
                method='grid',
                mfs=2,
                mf_type='gaussmf'
            ),
            FISConfig(
                id='CFG002',
                name='Grid_2MF_Bell',
                method='grid',
                mfs=2,
                mf_type='gbellmf'
            ),
            FISConfig(
                id='CFG003',
                name='Grid_2MF_Tri',
                method='grid',
                mfs=2,
                mf_type='trimf'
            ),
            FISConfig(
                id='CFG004',
                name='Grid_2MF_Trap',
                method='grid',
                mfs=2,
                mf_type='trapmf'
            ),
            
            # Grid Partition - 3 MFs
            FISConfig(
                id='CFG005',
                name='Grid_3MF_Gauss',
                method='grid',
                mfs=3,
                mf_type='gaussmf'
            ),
            
            # Subtractive Clustering - Different radii
            FISConfig(
                id='CFG006',
                name='SubClust_R03',
                method='subclust',
                radii=0.3
            ),
            FISConfig(
                id='CFG007',
                name='SubClust_R05',
                method='subclust',
                radii=0.5
            ),
            FISConfig(
                id='CFG008',
                name='SubClust_R07',
                method='subclust',
                radii=0.7
            ),
        ]
    
    def get_config(self, config_id: str) -> FISConfig:
        """Get configuration by ID"""
        for cfg in self.configs:
            if cfg.id == config_id:
                return cfg
        raise ValueError(f"Config not found: {config_id}")
    
    def get_config_by_name(self, name: str) -> FISConfig:
        """Get configuration by name"""
        for cfg in self.configs:
            if cfg.name == name:
                return cfg
        raise ValueError(f"Config not found: {name}")
    
    def get_all_configs(self) -> List[FISConfig]:
        """Get all configurations"""
        return self.configs
    
    def get_grid_configs(self) -> List[FISConfig]:
        """Get only grid partition configs"""
        return [cfg for cfg in self.configs if cfg.method == 'grid']
    
    def get_subclust_configs(self) -> List[FISConfig]:
        """Get only subtractive clustering configs"""
        return [cfg for cfg in self.configs if cfg.method == 'subclust']
    
    def print_all_configs(self):
        """Print all configurations"""
        logger.info("\n" + "="*60)
        logger.info("ANFIS CONFIGURATIONS")
        logger.info("="*60)
        
        for cfg in self.configs:
            logger.info(f"{cfg.id}: {cfg}")
        
        logger.info(f"\nTotal: {len(self.configs)} configurations")
        logger.info(f"  Grid: {len(self.get_grid_configs())}")
        logger.info(f"  SubClust: {len(self.get_subclust_configs())}")
        logger.info("="*60)
    
    def generate_matlab_training_script(self,
                                       config_id: str,
                                       train_file: str,
                                       check_file: str,
                                       test_file: str,
                                       epochs: int = 100,
                                       output_dir: str = '.') -> str:
        """
        Generate complete MATLAB training script for a configuration
        
        Args:
            config_id: Configuration ID
            train_file, check_file, test_file: Data file paths
            epochs: Number of training epochs
            output_dir: Output directory for results
            
        Returns:
            MATLAB script as string
        """
        cfg = self.get_config(config_id)
        
        script = f"""
% ANFIS Training Script
% Configuration: {cfg.name}
% Generated automatically

try
    % Load data
    tr = readmatrix('{train_file}');
    ch = readmatrix('{check_file}');
    te = readmatrix('{test_file}');
    
    fprintf('Data loaded successfully\\n');
    fprintf('  Train: %d samples\\n', size(tr, 1));
    fprintf('  Check: %d samples\\n', size(ch, 1));
    fprintf('  Test: %d samples\\n', size(te, 1));
    
    % Generate FIS
    fprintf('Generating FIS: {cfg.name}\\n');
    {cfg.generate_matlab_code('tr')}
    
    % ANFIS training options
    opt = anfisOptions('InitialFIS', fis, ...
                       'EpochNumber', {epochs}, ...
                       'ErrorGoal', 0.001, ...
                       'ValidationData', ch, ...
                       'DisplayANFISInformation', 0, ...
                       'DisplayErrorValues', 0, ...
                       'DisplayStepSize', 0);
    
    % Train
    fprintf('Training ANFIS...\\n');
    [trained_fis, train_error, step_size, chk_fis, chk_error] = anfis(tr, opt);
    
    % Evaluate on test set
    fprintf('Evaluating on test set...\\n');
    y_pred = evalfis(chk_fis, te(:, 1:end-1));
    y_true = te(:, end);
    
    % Calculate metrics
    errors = y_true - y_pred;
    rmse = sqrt(mean(errors.^2));
    mae = mean(abs(errors));
    r2 = 1 - sum(errors.^2) / sum((y_true - mean(y_true)).^2);
    mape = mean(abs(errors ./ (y_true + 1e-10))) * 100;
    
    % Outlier detection on training set
    y_train_pred = evalfis(chk_fis, tr(:, 1:end-1));
    train_errors = tr(:, end) - y_train_pred;
    z_scores = abs((train_errors - mean(train_errors)) / std(train_errors));
    outliers = find(z_scores > 2.0);
    
    % Save FIS
    fis_file = fullfile('{output_dir}', 'trained_fis_{cfg.id}.fis');
    writefis(chk_fis, fis_file);
    fprintf('FIS saved: %s\\n', fis_file);
    
    % Save outliers
    if ~isempty(outliers)
        outlier_file = fullfile('{output_dir}', 'outliers_{cfg.id}.csv');
        writematrix(outliers, outlier_file);
        fprintf('Outliers saved: %s\\n', outlier_file);
    end
    
    % Save metrics
    metrics = struct();
    metrics.config_id = '{cfg.id}';
    metrics.config_name = '{cfg.name}';
    metrics.test_r2 = r2;
    metrics.test_rmse = rmse;
    metrics.test_mae = mae;
    metrics.test_mape = mape;
    metrics.n_outliers = length(outliers);
    metrics.n_rules = length(chk_fis.Rules);
    
    metrics_file = fullfile('{output_dir}', 'metrics_{cfg.id}.json');
    fid = fopen(metrics_file, 'w');
    fprintf(fid, '%s', jsonencode(metrics));
    fclose(fid);
    
    fprintf('\\nTraining completed successfully!\\n');
    fprintf('  R²: %.4f\\n', r2);
    fprintf('  RMSE: %.4f\\n', rmse);
    fprintf('  MAE: %.4f\\n', mae);
    fprintf('  Rules: %d\\n', length(chk_fis.Rules));
    fprintf('  Outliers: %d\\n', length(outliers));
    
catch ME
    fprintf('ERROR: %s\\n', ME.message);
    fprintf('Stack trace:\\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\\n', ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end
"""
        
        return script
    
    def recommend_config(self, 
                        n_features: int,
                        n_samples: int,
                        complexity: str = 'auto') -> FISConfig:
        """
        Recommend best configuration based on data characteristics
        
        Args:
            n_features: Number of input features
            n_samples: Number of training samples
            complexity: 'simple', 'medium', 'complex', or 'auto'
            
        Returns:
            Recommended FISConfig
        """
        # Auto-determine complexity
        if complexity == 'auto':
            if n_features <= 3 and n_samples < 500:
                complexity = 'simple'
            elif n_features <= 5 and n_samples < 2000:
                complexity = 'medium'
            else:
                complexity = 'complex'
        
        logger.info(f"Data complexity: {complexity} ({n_features} features, {n_samples} samples)")
        
        # Recommendations
        if complexity == 'simple':
            # Simple: Grid with 2 MFs, Gaussian
            recommended = self.get_config('CFG001')
            logger.info(f"Recommended: {recommended.name} (fast convergence)")
            
        elif complexity == 'medium':
            # Medium: Grid with 3 MFs or SubClust R05
            recommended = self.get_config('CFG005')
            logger.info(f"Recommended: {recommended.name} (balanced performance)")
            
        else:  # complex
            # Complex: SubClust with smaller radius
            recommended = self.get_config('CFG006')
            logger.info(f"Recommended: {recommended.name} (handles complexity)")
        
        return recommended


if __name__ == "__main__":
    # Test config manager
    manager = ANFISConfigManager()
    
    # Print all configs
    manager.print_all_configs()
    
    # Get specific config
    print("\n=== Testing Config Retrieval ===")
    cfg = manager.get_config('CFG002')
    print(f"Config: {cfg}")
    print(f"MATLAB code: {cfg.generate_matlab_code()}")
    
    # Generate training script
    print("\n=== Testing Script Generation ===")
    script = manager.generate_matlab_training_script(
        config_id='CFG001',
        train_file='data/train.csv',
        check_file='data/check.csv',
        test_file='data/test.csv',
        epochs=100,
        output_dir='outputs'
    )
    print("Script generated successfully!")
    print(f"Length: {len(script)} characters")
    
    # Test recommendation
    print("\n=== Testing Config Recommendation ===")
    rec1 = manager.recommend_config(n_features=3, n_samples=300)
    rec2 = manager.recommend_config(n_features=5, n_samples=1500)
    rec3 = manager.recommend_config(n_features=7, n_samples=5000)
