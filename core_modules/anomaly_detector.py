"""
Anomali Tespit Modülü - AÇIKLAMALI VERSYON
Anomaly Detection Module - WITH EXPLANATIONS

YENİ ÖZELLİK:
- Anomali açıklamaları (hangi özellik, neden)
- Detaylı anomali skoru
- İnsan okuyabilir açıklamalar
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomali tespit edici - açıklamalı"""
    
    def __init__(self, contamination=0.1, output_dir='output'):
        """
        Args:
            contamination: Beklenen anomali oranı (0.1 = %10)
            output_dir: Çıktı klasörü
        """
        self.contamination = contamination
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_stats = {}
    
    def detect_anomalies(self, df, feature_cols):
        """
        Anomali tespiti yap ve AÇIKLA
        
        Args:
            df: DataFrame
            feature_cols: Anomali tespiti için kullanılacak sütunlar
            
        Returns:
            df: Anomali sütunları eklenmiş DataFrame
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"ANOMALİ TESPİTİ BAŞLIYOR")
        logger.info(f"{'='*70}")
        logger.info(f"Kullanılacak özellikler: {len(feature_cols)}")
        logger.info(f"Beklenen anomali oranı: {self.contamination*100:.0f}%")
        
        # Özellikleri hazırla
        X = df[feature_cols].copy()
        self.feature_names = feature_cols
        
        # NaN'ları doldur (median ile)
        for col in feature_cols:
            if X[col].isna().sum() > 0:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
        
        # Özellik istatistiklerini kaydet (açıklama için)
        for col in feature_cols:
            self.feature_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'q25': X[col].quantile(0.25),
                'q75': X[col].quantile(0.75)
            }
        
        # Normalize et
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Tahmin: -1 = anomali, 1 = normal
        predictions = self.model.fit_predict(X_scaled)
        
        # Anomali skorları (ne kadar negatif o kadar anormal)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # DataFrame'e ekle
        df['anomaly_label'] = predictions
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = predictions == -1
        
        # Her anomali için AÇIKLAMA oluştur
        logger.info("\nAnomaliler açıklanıyor...")
        df['anomaly_reason'] = df.apply(
            lambda row: self._explain_anomaly(row, X.loc[row.name], X_scaled[row.name]) 
            if row['is_anomaly'] else '',
            axis=1
        )
        
        # İstatistikler
        n_anomalies = df['is_anomaly'].sum()
        logger.info(f"\n[OK] Tespit edilen anomali: {n_anomalies} ({n_anomalies/len(df)*100:.1f}%)")
        
        # Rapor kaydet
        self._save_anomaly_report(df)
        
        return df
    
    def _explain_anomaly(self, row, feature_values, scaled_values):
        """
        Anomalinin nedenini açıkla
        
        Returns:
            str: İnsan okuyabilir açıklama
        """
        # En aykırı özellikleri bul (z-score bazlı)
        z_scores = np.abs(scaled_values)
        
        # En yüksek 3 z-score
        top_indices = np.argsort(z_scores)[-3:][::-1]
        
        explanations = []
        
        for idx in top_indices:
            feat_name = self.feature_names[idx]
            feat_value = feature_values[feat_name]
            feat_stats = self.feature_stats[feat_name]
            z_score = z_scores[idx]
            
            # Açıklama oluştur
            if z_score > 2.0:  # Anlamlı sapma
                # Değer hangi bölgede?
                if feat_value < feat_stats['q25']:
                    position = "çok düşük"
                elif feat_value > feat_stats['q75']:
                    position = "çok yüksek"
                else:
                    position = "aykırı"
                
                explanation = (
                    f"{feat_name}={feat_value:.3f} ({position}, "
                    f"beklenen: {feat_stats['mean']:.3f}±{feat_stats['std']:.3f}, "
                    f"z={z_score:.2f})"
                )
                explanations.append(explanation)
        
        if not explanations:
            return "Genel veri dağılımından sapma"
        
        return " | ".join(explanations)
    
    def _save_anomaly_report(self, df):
        """Anomali raporunu kaydet"""
        output_file = self.output_dir / 'Anomaly_Detection.xlsx'
        
        # Sadece anomalileri seç
        df_anomalies = df[df['is_anomaly']].copy()
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Tüm anomaliler
            df_anomalies[['NUCLEUS', 'Z', 'N', 'A', 'SPIN', 'PARITY', 
                         'anomaly_score', 'anomaly_reason']].to_excel(
                writer, sheet_name='Anomalies', index=False
            )
            
            # Sheet 2: Anomali özeti
            summary = pd.DataFrame([{
                'Total_Nuclei': len(df),
                'Anomalies_Detected': len(df_anomalies),
                'Anomaly_Rate_%': (len(df_anomalies) / len(df)) * 100,
                'Contamination_Set': self.contamination * 100,
                'Method': 'Isolation Forest',
                'N_Features': len(self.feature_names)
            }])
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 3: Özellik bazlı anomali sayıları
            # Her özellik için kaç anomaliye katkı yaptığını say
            feature_contributions = []
            for feat in self.feature_names:
                # Bu özellik içeren açıklamaları say
                count = df_anomalies['anomaly_reason'].str.contains(feat, na=False).sum()
                feature_contributions.append({
                    'Feature': feat,
                    'Anomaly_Count': count,
                    'Contribution_%': (count / len(df_anomalies)) * 100 if len(df_anomalies) > 0 else 0
                })
            
            df_contrib = pd.DataFrame(feature_contributions)
            df_contrib = df_contrib.sort_values('Anomaly_Count', ascending=False)
            df_contrib.to_excel(writer, sheet_name='Feature_Contributions', index=False)
            
            # Sheet 4: Z değerlerine göre anomaliler
            df_z_dist = df_anomalies.groupby('Z').size().reset_index(name='Count')
            df_z_dist = df_z_dist.sort_values('Count', ascending=False)
            df_z_dist.to_excel(writer, sheet_name='By_Proton_Number', index=False)
        
        logger.info(f"[OK] Anomali raporu kaydedildi: {output_file}")
        
        # Görselleştirme
        self._plot_anomalies(df)
    
    def _plot_anomalies(self, df):
        """Anomali görselleştirmeleri"""
        
        # 1. Anomali skoru dağılımı
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Anomaly score histogram
        axes[0, 0].hist(df['anomaly_score'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=df[df['is_anomaly']]['anomaly_score'].max(), 
                          color='r', linestyle='--', linewidth=2, label='Anomali Eşiği')
        axes[0, 0].set_xlabel('Anomali Skoru', fontsize=12)
        axes[0, 0].set_ylabel('Frekans', fontsize=12)
        axes[0, 0].set_title('Anomali Skoru Dağılımı', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Z-N scatter (anomaliler vurgulanmış)
        normal = df[~df['is_anomaly']]
        anomalies = df[df['is_anomaly']]
        
        axes[0, 1].scatter(normal['N'], normal['Z'], alpha=0.5, s=30, label='Normal', c='blue')
        axes[0, 1].scatter(anomalies['N'], anomalies['Z'], alpha=0.8, s=80, 
                          label='Anomali', c='red', edgecolors='black', linewidth=1)
        axes[0, 1].set_xlabel('Nötron Sayısı (N)', fontsize=12)
        axes[0, 1].set_ylabel('Proton Sayısı (Z)', fontsize=12)
        axes[0, 1].set_title('Z-N Haritası (Anomaliler Vurgulanmış)', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spin dağılımı
        spin_counts = df.groupby(['SPIN', 'is_anomaly']).size().unstack(fill_value=0)
        spin_counts.plot(kind='bar', ax=axes[1, 0], color=['blue', 'red'])
        axes[1, 0].set_xlabel('Spin', fontsize=12)
        axes[1, 0].set_ylabel('Sayı', fontsize=12)
        axes[1, 0].set_title('Spin Dağılımı', fontsize=14, fontweight='bold')
        axes[1, 0].legend(['Normal', 'Anomali'])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Anomali oranı Z'ye göre
        z_anomaly_rate = df.groupby('Z')['is_anomaly'].mean() * 100
        z_anomaly_rate.plot(kind='bar', ax=axes[1, 1], color='coral')
        axes[1, 1].set_xlabel('Proton Sayısı (Z)', fontsize=12)
        axes[1, 1].set_ylabel('Anomali Oranı (%)', fontsize=12)
        axes[1, 1].set_title('Z\'ye Göre Anomali Oranı', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Anomali görselleştirmeleri kaydedildi")


def main():
    """Test fonksiyonu"""
    # Örnek veri oluştur
    np.random.seed(42)
    
    data = {
        'NUCLEUS': [f'Nucleus_{i}' for i in range(100)],
        'Z': np.random.randint(20, 90, 100),
        'N': np.random.randint(20, 120, 100),
        'A': np.random.randint(40, 200, 100),
        'SPIN': np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5], 100),
        'PARITY': np.random.choice([-1, 1], 100),
        'MM': np.random.normal(2.0, 1.0, 100),
        'Q': np.random.normal(0.5, 0.3, 100)
    }
    
    # Birkaç anomali ekle
    data['MM'][95:] = [10.0, 11.0, -8.0, 12.0, -9.0]  # Aşırı değerler
    
    df = pd.DataFrame(data)
    
    # Anomali tespiti
    detector = AnomalyDetector(contamination=0.1, output_dir='test_anomaly')
    df_result = detector.detect_anomalies(df, ['Z', 'N', 'SPIN', 'MM', 'Q'])
    
    print("\n[OK] Anomaly detection test completed")
    print(f"  Detected anomalies: {df_result['is_anomaly'].sum()}")
    print(f"  Example explanation: {df_result[df_result['is_anomaly']]['anomaly_reason'].iloc[0]}")


if __name__ == "__main__":
    main()