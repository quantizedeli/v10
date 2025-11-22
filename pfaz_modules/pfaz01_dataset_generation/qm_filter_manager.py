"""
QM Filter Manager
Target Bazlı QM Filtreleme Sistemi

Bu modül, hedef değişkenlere göre QM (Quadrupole Moment) filtrele sistemi sağlar:
- MM target: QM olmayan çekirdekler kullanılabilir [OK]
- QM target: QM olmayan çekirdekler kullanılamaz [FAIL]
- MM_QM target: QM olmayan çekirdekler kullanılamaz [FAIL]
- Beta_2 target: Q'ya bağlı feature varsa QM gerekli [FAIL] (yoksa kullanılabilir [OK])
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QMFilterManager:
    """QM filtreleme yöneticisi"""
    
    def __init__(self):
        self.qm_columns = ['Q', 'QUADRUPOLE MOMENT [Q]']  # Olası QM sütun isimleri
        
    def filter_by_target(self, df, target_name, target_cols, features=None):
        """
        Target'a göre QM filtrele
        
        Args:
            df: DataFrame - Veri seti
            target_name: str - Hedef ismi ('MM', 'QM', 'MM_QM', 'Beta_2')
            target_cols: list - Hedef sütunları
            features: list - Kullanılan özellikler (Beta_2 için gerekli)
            
        Returns:
            DataFrame: Filtrelenmiş veri
            dict: Filtreleme raporu
        """
        # QM sütununu bul
        qm_col = self._find_qm_column(df)
        if qm_col is None:
            logger.warning("[WARNING] QM sütunu bulunamadı, filtreleme yapılamıyor")
            return df, {'status': 'no_qm_column', 'removed': 0}
        
        df_filtered = df.copy()
        initial_count = len(df_filtered)
        removed_nuclei = []
        
        # Filtreleme mantığı
        if target_name == 'MM':
            # MM için QM yoksa sorun değil [OK]
            filter_result = 'no_filtering_needed'
            logger.info(f"[OK] Target: MM -> QM filtresi UYGULANMADI (QM olmayan çekirdekler kullanılabilir)")
            
        elif target_name == 'QM':
            # QM hedefi, QM olmayan çıkar [FAIL]
            df_filtered, removed = self._remove_missing_qm(df_filtered, qm_col)
            removed_nuclei = removed
            filter_result = 'qm_required'
            logger.info(f"[OK] Target: QM -> QM filtresi UYGULAN DI (QM olmayan {len(removed)} çekirdek çıkarıldı)")
            
        elif target_name == 'MM_QM':
            # MM_QM hedefi, QM olmayan çıkar [FAIL]
            df_filtered, removed = self._remove_missing_qm(df_filtered, qm_col)
            removed_nuclei = removed
            filter_result = 'qm_required'
            logger.info(f"[OK] Target: MM_QM -> QM filtresi UYGULANDI (QM olmayan {len(removed)} çekirdek çıkarıldı)")
            
        elif target_name == 'Beta_2':
            # Beta_2 için Q'ya bağlı feature kontrolü
            if features and self._has_q_dependent_features(features):
                # Q'ya bağlı feature varsa QM gerekli [FAIL]
                df_filtered, removed = self._remove_missing_qm(df_filtered, qm_col)
                removed_nuclei = removed
                filter_result = 'qm_required_for_features'
                logger.info(f"[OK] Target: Beta_2 + Q-bağlı features -> QM filtresi UYGULANDI ({len(removed)} çekirdek çıkarıldı)")
            else:
                # Q'ya bağlı feature yoksa QM gerekmez [OK]
                filter_result = 'no_filtering_needed'
                logger.info(f"[OK] Target: Beta_2 -> QM filtresi UYGULANMADI (Q-bağlı feature yok)")
        
        else:
            logger.warning(f"[WARNING] Bilinmeyen target: {target_name}, filtreleme yapılmadı")
            filter_result = 'unknown_target'
        
        final_count = len(df_filtered)
        removed_count = initial_count - final_count
        
        report = {
            'status': filter_result,
            'initial_count': initial_count,
            'final_count': final_count,
            'removed': removed_count,
            'removed_nuclei': removed_nuclei
        }
        
        return df_filtered, report
    
    def _find_qm_column(self, df):
        """QM sütununu bul"""
        for col in self.qm_columns:
            if col in df.columns:
                return col
        return None
    
    def _remove_missing_qm(self, df, qm_col):
        """QM olmayan çekirdekleri çıkar"""
        missing_qm = df[df[qm_col].isna() | (df[qm_col] == 0)]
        removed_nuclei = missing_qm['NUCLEUS'].tolist() if 'NUCLEUS' in df.columns else []
        
        df_filtered = df[df[qm_col].notna() & (df[qm_col] != 0)].copy()
        
        return df_filtered, removed_nuclei
    
    def _has_q_dependent_features(self, features):
        """
        Feature listesinde Q'ya bağlı özellik var mı kontrol et
        
        Q-bağlı features:
        - 'Q' (direkt)
        - 'q_schmidt_ratio'
        - 'q_theoretical'
        - 'q_*' ile başlayanlar
        """
        q_related = ['Q', 'q_schmidt_ratio', 'q_theoretical', 'quadrupole']
        
        for feature in features:
            feature_lower = feature.lower()
            # Direkt Q veya q_ ile başlayanlar
            if feature_lower in ['q'] or feature_lower.startswith('q_') or 'quadrupole' in feature_lower:
                return True
        
        return False
    
    def create_filter_report(self, reports, output_path='reports/qm_filter_report.xlsx'):
        """
        Tüm filtreleme raporlarını birleştir ve Excel'e kaydet
        
        Args:
            reports: list of dict - Her dataset için filtreleme raporu
            output_path: str - Excel dosya yolu
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Özet rapor
        summary_data = []
        for report in reports:
            summary_data.append({
                'Target': report.get('target', 'N/A'),
                'Filter_Status': report.get('status', 'N/A'),
                'Initial_Count': report.get('initial_count', 0),
                'Final_Count': report.get('final_count', 0),
                'Removed_Count': report.get('removed', 0),
                'Removal_Rate': f"{report.get('removed', 0) / report.get('initial_count', 1) * 100:.1f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Excel'e kaydet
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Her target için detay
            for target in summary_df['Target'].unique():
                target_reports = [r for r in reports if r.get('target') == target]
                if target_reports:
                    removed_nuclei_all = []
                    for r in target_reports:
                        removed_nuclei_all.extend(r.get('removed_nuclei', []))
                    
                    if removed_nuclei_all:
                        removed_df = pd.DataFrame({
                            'Nucleus': removed_nuclei_all,
                            'Reason': 'QM missing or zero'
                        })
                        sheet_name = f'Removed_{target}'[:31]
                        removed_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"[OK] QM filtreleme raporu kaydedildi: {output_path}")


def main():
    """Test fonksiyonu"""
    # Test verisi oluştur
    test_data = {
        'NUCLEUS': ['53 I 126', '49 In 126', '85 At 208', '87 Fr 210'],
        'A': [126, 126, 208, 210],
        'Z': [53, 49, 85, 87],
        'N': [73, 77, 123, 123],
        'MM': [2.8, 2.5, -0.044, -0.052],
        'Q': [0.7, None, -0.4, 0.19],  # In-126'da QM yok
        'Beta_2': [0.2, 0.15, None, 0.1]
    }
    df_test = pd.DataFrame(test_data)
    
    # Filter manager oluştur
    qm_filter = QMFilterManager()
    
    # Test 1: MM target (QM gerekmez)
    print("\n" + "="*80)
    print("TEST 1: MM Target")
    print("="*80)
    df_mm, report_mm = qm_filter.filter_by_target(df_test, 'MM', ['MM'])
    print(f"Başlangıç: {report_mm['initial_count']}, Son: {report_mm['final_count']}, Çıkarılan: {report_mm['removed']}")
    
    # Test 2: QM target (QM gerekli)
    print("\n" + "="*80)
    print("TEST 2: QM Target")
    print("="*80)
    df_qm, report_qm = qm_filter.filter_by_target(df_test, 'QM', ['Q'])
    print(f"Başlangıç: {report_qm['initial_count']}, Son: {report_qm['final_count']}, Çıkarılan: {report_qm['removed']}")
    print(f"Çıkarılan çekirdekler: {report_qm['removed_nuclei']}")
    
    # Test 3: Beta_2 + Q-bağlı feature
    print("\n" + "="*80)
    print("TEST 3: Beta_2 + Q-bağlı feature")
    print("="*80)
    df_beta2_q, report_beta2_q = qm_filter.filter_by_target(df_test, 'Beta_2', ['Beta_2'], features=['A', 'Z', 'Q'])
    print(f"Başlangıç: {report_beta2_q['initial_count']}, Son: {report_beta2_q['final_count']}, Çıkarılan: {report_beta2_q['removed']}")
    
    # Test 4: Beta_2 + Q-bağlı olmayan feature
    print("\n" + "="*80)
    print("TEST 4: Beta_2 + Q-bağlı olmayan feature")
    print("="*80)
    df_beta2_noq, report_beta2_noq = qm_filter.filter_by_target(df_test, 'Beta_2', ['Beta_2'], features=['A', 'Z', 'N'])
    print(f"Başlangıç: {report_beta2_noq['initial_count']}, Son: {report_beta2_noq['final_count']}, Çıkarılan: {report_beta2_noq['removed']}")


if __name__ == "__main__":
    main()
