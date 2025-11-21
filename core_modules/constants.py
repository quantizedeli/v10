"""
Nuclear Physics Constants and Configuration
Nükleer Fizik Sabitleri ve Konfigürasyon

Bu modül projedeki tüm sabit değerleri, fiziksel parametreleri ve konfigürasyonları içerir.
"""

import numpy as np

# ============================================================================
# MAGİK SAYILAR (Magic Numbers)
# ============================================================================

MAGIC_NUMBERS_Z = [2, 8, 20, 28, 50, 82, 114, 126]  # Proton için
MAGIC_NUMBERS_N = [2, 8, 20, 28, 50, 82, 126, 184]  # Nötron için

# ============================================================================
# SEMF (Semi-Empirical Mass Formula) PARAMETRELERİ
# ============================================================================

SEMF_PARAMS = {
    'a_v': 15.75,    # Hacim terimi (MeV)
    'a_s': 17.8,     # Yüzey terimi (MeV)
    'a_c': 0.711,    # Coulomb terimi (MeV)
    'a_a': 23.7,     # Asimetri terimi (MeV)
    'a_p': 11.18     # Pairing terimi (MeV)
}

# ============================================================================
# SCHMIDT MODELİ PARAMETRELERİ
# ============================================================================

SCHMIDT_PARAMS = {
    'proton': {
        'g_l': 1.0,  # Yörüngesel g-faktörü
        'g_s': 5.586 # Spin g-faktörü
    },
    'neutron': {
        'g_l': 0.0,
        'g_s': -3.826
    }
}

# ============================================================================
# DEFORMASYON PARAMETRELERİ
# ============================================================================

# Küresel nükleus eşik değeri
SPHERICAL_THRESHOLD = 0.05  # |β₂| < 0.05 → küresel

# Bölge tanımlamaları (β₂ değerlerine göre)
DEFORMATION_REGIONS = {
    'spherical': (-0.05, 0.05),
    'weakly_prolate': (0.05, 0.15),
    'prolate': (0.15, 0.35),
    'strongly_prolate': (0.35, 1.0),
    'weakly_oblate': (-0.15, -0.05),
    'oblate': (-0.35, -0.15),
    'strongly_oblate': (-1.0, -0.35)
}

# ============================================================================
# KÜTLE GRUBU SINIFLARI
# ============================================================================

MASS_GROUPS = {
    'light': (0, 40),
    'medium_light': (40, 90),
    'medium': (90, 140),
    'medium_heavy': (140, 200),
    'heavy': (200, 300)
}

# ============================================================================
# WEISSKOPF BİRİMLERİ (E2 geçişleri için)
# ============================================================================

def weisskopf_unit(A):
    """Weisskopf birimi hesapla (e²·fm⁴)"""
    return 1.2**4 * A**(4/3)

# ============================================================================
# NÜKLEUS TİP TANIMLARI
# ============================================================================

NUCLEUS_TYPES = {
    'even-even': (lambda Z, N: Z % 2 == 0 and N % 2 == 0),
    'even-odd': (lambda Z, N: Z % 2 == 0 and N % 2 == 1),
    'odd-even': (lambda Z, N: Z % 2 == 1 and N % 2 == 0),
    'odd-odd': (lambda Z, N: Z % 2 == 1 and N % 2 == 1)
}

# ============================================================================
# ANOMALİ TESPİT EŞIK DEĞERLERİ
# ============================================================================

ANOMALY_THRESHOLDS = {
    'z_score': 3.5,              # İstatistiksel z-score
    'schmidt_deviation': 3.0,    # μN cinsinden
    'magic_beta2': 0.15,         # Magik nükleuslar için β₂
    'shell_gap_low': 2.0,        # MeV (düşük kabuk açıklığı)
    'mm_jump': 0.5,              # μN (izotop zincirinde sıçrama)
    'q_jump': 0.3,               # barn (izotop zincirinde sıçrama)
    'isolation_forest_contamination': 0.08
}

# ============================================================================
# VERİ SETİ OLUŞTURMA PARAMETRELERİ
# ============================================================================

# ✅ FAZ 1 - DÜZELTİLDİ: 120 ve 250 kaldırıldı
NUCLEUS_COUNTS = [75, 100, 150, 200, 'ALL']

SCENARIOS = {
    'S70': (0.70, 0.15, 0.15),  # (train, check, test)
    'S80': (0.80, 0.10, 0.10)
}

ANOMALY_MODES = ['anomalili', 'anomalisiz']

SCALING_METHODS = ['none', 'standard', 'robust']

SAMPLING_METHODS = ['random', 'stratified']

# ============================================================================
# HEDEF DEĞİŞKENLER
# ============================================================================

TARGETS = {
    'MM': ['MM'],
    'QM': ['Q'],
    'MM_QM': ['MM', 'Q'],
    'Beta_2': ['Beta_2']
}

# ============================================================================
# ÖZELLİK SETLERİ / FEATURE SETS
# ============================================================================

STANDARD_FEATURE_SETS = {
    'AZN': ['A', 'Z', 'N'],
    'AZNS': ['A', 'Z', 'N', 'SPIN'],
    'AZNP': ['A', 'Z', 'N', 'PARITY'],
    'AZNSP': ['A', 'Z', 'N', 'SPIN', 'PARITY'],
    'AZN_beta': ['A', 'Z', 'N', 'beta_2'],
    'AZN_p': ['A', 'Z', 'N', 'p_factor'],
    'AZN_beta_p': ['A', 'Z', 'N', 'beta_2', 'p_factor'],
    'AZNSP_beta_p': ['A', 'Z', 'N', 'SPIN', 'PARITY', 'beta_2', 'p_factor'],
    'ADVANCED': ['A', 'Z', 'N', 'SPIN', 'PARITY', 'beta_2', 'p_factor',
                 'BE_per_A', 'Z_magic_dist', 'N_magic_dist']
}

BETA2_FEATURE_SETS = {
    # 2 input
    'Beta2_AZ': ['A', 'Z'],
    'Beta2_AN': ['A', 'N'],
    'Beta2_ZN': ['Z', 'N'],
    
    # 3 input
    'Beta2_AZN': ['A', 'Z', 'N'],
    'Beta2_AZP': ['A', 'Z', 'p_factor'],
    'Beta2_AZMagic': ['A', 'Z', 'Z_magic_dist'],
    
    # 4 input
    'Beta2_AZNP': ['A', 'Z', 'N', 'p_factor'],
    'Beta2_AZNMagic': ['A', 'Z', 'N', 'magic_character'],
    
    # Gelişmiş
    'Beta2_Physics': ['A', 'Z', 'N', 'p_factor', 'Z_magic_dist', 
                      'N_magic_dist', 'asymmetry', 'BE_per_A']
}

# ============================================================================
# ANFİS KONFİGÜRASYONLARI
# ============================================================================

# ✅ FAZ 2 - GÜNCELLENDİ: Config isimleri değiştirildi
ANFIS_CONFIGS = {
    # Grid Partition (genfis1) - 5 config
    'GAU2MF': {
        'id': 'CFG001',
        'method': 'grid',
        'num_mfs': 2,
        'mf_type': 'gaussmf',
        'epochs': 100,
        'display': False,
        'description': 'Grid partition - 2 MF, Gaussian'
    },
    'GEN2MF': {
        'id': 'CFG002',
        'method': 'grid',
        'num_mfs': 2,
        'mf_type': 'gbellmf',
        'epochs': 100,
        'display': False,
        'description': 'Grid partition - 2 MF, Generalized Bell'
    },
    'TRI2MF': {
        'id': 'CFG003',
        'method': 'grid',
        'num_mfs': 2,
        'mf_type': 'trimf',
        'epochs': 100,
        'display': False,
        'description': 'Grid partition - 2 MF, Triangular'
    },
    'TRA2MF': {
        'id': 'CFG004',
        'method': 'grid',
        'num_mfs': 2,
        'mf_type': 'trapmf',
        'epochs': 100,
        'display': False,
        'description': 'Grid partition - 2 MF, Trapezoidal'
    },
    'GAU3MF': {
        'id': 'CFG005',
        'method': 'grid',
        'num_mfs': 3,
        'mf_type': 'gaussmf',
        'epochs': 100,
        'display': False,
        'description': 'Grid partition - 3 MF, Gaussian'
    },
    
    # Subtractive Clustering (genfis2/3) - 3 config
    'SUBR03': {
        'id': 'CFG006',
        'method': 'subtractive',
        'radii': 0.3,
        'epochs': 100,
        'display': False,
        'description': 'Subtractive clustering - radii=0.3'
    },
    'SUBR05': {
        'id': 'CFG007',
        'method': 'subtractive',
        'radii': 0.5,
        'epochs': 100,
        'display': False,
        'description': 'Subtractive clustering - radii=0.5'
    },
    'SUBR07': {
        'id': 'CFG008',
        'method': 'subtractive',
        'radii': 0.7,
        'epochs': 100,
        'display': False,
        'description': 'Subtractive clustering - radii=0.7'
    }
}

# Config ID'den isim mapping
ANFIS_CONFIG_ID_TO_NAME = {
    'CFG001': 'GAU2MF',
    'CFG002': 'GEN2MF',
    'CFG003': 'TRI2MF',
    'CFG004': 'TRA2MF',
    'CFG005': 'GAU3MF',
    'CFG006': 'SUBR03',
    'CFG007': 'SUBR05',
    'CFG008': 'SUBR07'
}

# Config isimden ID mapping
ANFIS_CONFIG_NAME_TO_ID = {v: k for k, v in ANFIS_CONFIG_ID_TO_NAME.items()}

# ============================================================================
# AI MODEL PARAMETRELERİ
# ============================================================================

AI_MODELS = {
    'RandomForest': {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'n_jobs': -1
    },
    'GradientBoosting': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8
    },
    'XGBoost': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'DNN': {
        'hidden_layers': [128, 64, 32],
        'activation': 'relu',
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 32
    },
    'BNN': {
        'hidden_layers': [64, 32],
        'num_samples': 1000,
        'learning_rate': 0.001,
        'epochs': 100
    },
    'PINN': {
        'hidden_layers': [64, 64, 32],
        'physics_weight': 0.5,
        'learning_rate': 0.001,
        'epochs': 150
    }
}

# ============================================================================
# GÖRSELLEŞTİRME PARAMETRELERİ
# ============================================================================

PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_scheme': 'viridis',
    'font_size': 12,
    'save_formats': ['png', 'html']
}

# ============================================================================
# DOSYA YOLLARI
# ============================================================================

PATHS = {
    'data': 'data/',
    'output': 'output/',
    'datasets': 'ANFIS_Datasets/',
    'models': 'trained_models/',
    'visualizations': 'visualizations/',
    'reports': 'reports/',
    'logs': 'logs/'
}

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def get_nucleus_type(Z, N):
    """Nükleus tipini belirle"""
    for name, condition in NUCLEUS_TYPES.items():
        if condition(Z, N):
            return name
    return 'unknown'

def get_mass_group(A):
    """Kütle grubunu belirle"""
    for group_name, (min_a, max_a) in MASS_GROUPS.items():
        if min_a <= A < max_a:
            return group_name
    return 'unknown'

def get_deformation_type(beta_2):
    """Deformasyon tipini belirle"""
    if np.isnan(beta_2):
        return 'unknown'
    
    for def_type, (min_beta, max_beta) in DEFORMATION_REGIONS.items():
        if min_beta <= beta_2 < max_beta:
            return def_type
    return 'unknown'

def nearest_magic_number(n, magic_numbers):
    """En yakın magik sayıyı bul"""
    return min(magic_numbers, key=lambda x: abs(x - n))

def distance_to_magic(n, magic_numbers):
    """Magik sayıya uzaklık"""
    return abs(n - nearest_magic_number(n, magic_numbers))

def get_anfis_config_name(config_id):
    """Config ID'den isim al"""
    return ANFIS_CONFIG_ID_TO_NAME.get(config_id, config_id)

def get_anfis_config_id(config_name):
    """Config isminden ID al"""
    return ANFIS_CONFIG_NAME_TO_ID.get(config_name, config_name)

# ============================================================================
# VERSİYON BİLGİSİ
# ============================================================================

VERSION = "1.0.2"
AUTHOR = "Nuclear Physics AI Project"
DATE = "2025-10-14"
CHANGELOG = """
v1.0.2 (2025-10-14):
  - FAZ 2: ANFIS config isimleri güncellendi (GAU2MF, GEN2MF, TRI2MF, TRA2MF, GAU3MF, SUBR03, SUBR05, SUBR07)
  - Config ID <-> Name mapping eklendi

v1.0.1 (2025-10-14):
  - FAZ 1: NUCLEUS_COUNTS düzeltildi (120 ve 250 kaldırıldı)
  - ANFIS_CONFIGS eklenmiş fakat isimler henüz güncellenmedi
"""

if __name__ == "__main__":
    print(f"Nükleer Fizik Sabitleri Modülü v{VERSION}")
    print(f"Magik Sayılar Z: {MAGIC_NUMBERS_Z}")
    print(f"Magik Sayılar N: {MAGIC_NUMBERS_N}")
    print(f"NUCLEUS_COUNTS: {NUCLEUS_COUNTS}")
    print(f"\nANFIS Configurations:")
    for name, cfg in ANFIS_CONFIGS.items():
        print(f"  {name}: {cfg['description']}")
