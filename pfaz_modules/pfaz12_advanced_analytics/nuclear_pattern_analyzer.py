"""
Nuclear Pattern Analyzer
========================
PFAZ 12 — Nükleer Veri Desen Analizi

Hedefler:
  MM  (Manyetik Moment  µ)
  QM  (Kuadrupol Moment Q)
  Beta_2  (Deformasyon)

Analizler:
  1. Ortalama küme analizi  — hedef değeri ortalamaya yakın çekirdekler; özellik dağılımı
  2. İzotop zinciri analizi — sabit Z, değişen N: ani hedef değişimlerini tespit et
  3. İzotone zinciri       — sabit N, değişen Z
  4. İzobar zinciri        — sabit A, değişen Z
  5. Sihirli sayı analizi  — shell kapanmalarında hedef dağılımı (KS testi)
  6. Sıçrama özellik analizi — sıçrayan çekirdeklerde hangi özellikler ortak?

Çıktı:
  Excel   — ExcelStandardizer ile standart biçimlendirilmiş çok sayfalı rapor
  Grafikler — matplotlib: zincir çizgileri, violin, heatmap, dağılım

Kullanım:
    from pfaz_modules.pfaz12_advanced_analytics.nuclear_pattern_analyzer import NuclearPatternAnalyzer
    npa = NuclearPatternAnalyzer(data_path='aaa2.txt', output_dir='outputs/nuclear_patterns')
    npa.run_all()
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sihirli sayılar (nükleer shell kapanmaları)
# ---------------------------------------------------------------------------
MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]
NEAR_MAGIC_DIST = 3      # ±3 birim "yakın sihirli" sayılır

# ---------------------------------------------------------------------------
# Matplotlib opsiyonel
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT = True
except ImportError:
    _PLOT = False
    matplotlib = None
    plt = None
    sns = None

# ---------------------------------------------------------------------------
# ExcelStandardizer opsiyonel
# ---------------------------------------------------------------------------
try:
    from pfaz_modules.pfaz06_final_reporting.excel_standardizer import ExcelStandardizer
    _ES = True
except ImportError:
    _ES = False
    ExcelStandardizer = None


# ===========================================================================
class NuclearPatternAnalyzer:
    """
    Nükleer veri desen analizi.

    Parameters
    ----------
    data_path : str
        aaa2.txt veya AAA2_enriched.csv yolu
    output_dir : str
        Çıktı klasörü (Excel + grafikler)
    jump_sigma : float
        Sıçrama eşiği: zincir içi ΔTarget'ın kaç standart sapması (varsayılan 2.0)
    min_chain_len : int
        Analiz için minimum zincir uzunluğu (varsayılan 3)
    """

    TARGET_MAP = {
        "MM":     ["MAGNETIC MOMENT [µ]", "MM", "magnetic moment"],
        "QM":     ["QUADRUPOLE MOMENT [Q]", "QM", "quadrupole moment"],
        "Beta_2": ["Beta_2", "BETA_2", "beta2", "Beta2"],
    }

    FEATURE_COLS = [
        "SPIN", "PARITY", "P-factor", "Nn", "Np",
        "magic_character", "BE_per_A", "Beta_2_estimated",
        "Z_magic_dist", "N_magic_dist", "BE_asymmetry",
        "Z_valence", "N_valence", "Z_shell_gap", "N_shell_gap",
        "BE_pairing", "spherical_index", "Q0_intrinsic",
        "BE_total", "S_n_approx", "S_p_approx",
    ]

    def __init__(
        self,
        data_path: str = "aaa2.txt",
        output_dir: str = "outputs/nuclear_patterns",
        jump_sigma: float = 2.0,
        min_chain_len: int = 3,
    ):
        self.data_path     = Path(data_path)
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jump_sigma    = jump_sigma
        self.min_chain_len = min_chain_len
        self.df: Optional[pd.DataFrame] = None
        self._results: Dict = {}

    # -----------------------------------------------------------------------
    # Veri yükleme
    # -----------------------------------------------------------------------
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        aaa2.txt veya enriched CSV/XLSX yükle; sütunları normalize et.
        Döndürür: temizlenmiş DataFrame (A, Z, N, MM, QM, Beta_2, özellikler…)
        """
        path = Path(data_path) if data_path else self.data_path
        logger.info(f"[NuclearPatternAnalyzer] Veri yükleniyor: {path}")

        # --- Okuma ---
        if path.suffix == ".txt" or path.suffix == ".csv":
            try:
                from utils.file_io_utils import read_nuclear_data
                df = read_nuclear_data(str(path))
            except Exception:
                df = pd.read_csv(str(path), sep="\t", encoding="utf-8", on_bad_lines="skip")
        elif path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(str(path))
        else:
            raise ValueError(f"Desteklenmeyen format: {path.suffix}")

        # --- Sütun adı normaliz. ---
        df.columns = [c.strip() for c in df.columns]

        # Hedef sütunlarını bul + yeniden adlandır
        for canonical, aliases in self.TARGET_MAP.items():
            for alias in aliases:
                matches = [c for c in df.columns if c.lower() == alias.lower()]
                if matches and canonical not in df.columns:
                    df = df.rename(columns={matches[0]: canonical})
                    break

        # Sayısal temizlik: virgül → nokta, to_numeric (korumalı)
        TEXT_COLS = {"NUCLEUS", "ELEMENT", "SYMBOL"}
        for col in df.columns:
            if df[col].dtype == object and col not in TEXT_COLS:
                cleaned = (
                    df[col].astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.strip()
                )
                numeric_try = pd.to_numeric(cleaned, errors="coerce")
                # Dönüşüm yeterince başarılıysa kullan (>%30 non-NaN)
                n_valid_orig = (df[col].astype(str).str.strip() != "").sum()
                n_valid_conv = numeric_try.notna().sum()
                if n_valid_orig > 0 and (n_valid_conv / n_valid_orig) >= 0.3:
                    df[col] = numeric_try

        # Zorunlu sütunlar
        for req in ("A", "Z", "N"):
            if req not in df.columns:
                raise KeyError(f"Zorunlu sütun eksik: {req}")

        df["A"] = pd.to_numeric(df["A"], errors="coerce")
        df["Z"] = pd.to_numeric(df["Z"], errors="coerce")
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
        df = df.dropna(subset=["A", "Z", "N"])

        # Sihirli sayı mesafesi ekle (eğer yoksa)
        for col, base in [("Z_magic_dist_calc", "Z"), ("N_magic_dist_calc", "N")]:
            df[col] = df[base].apply(
                lambda v: min(abs(v - m) for m in MAGIC_NUMBERS)
            )

        # is_magic bayrakları
        df["Z_is_magic"] = df["Z"].isin(MAGIC_NUMBERS).astype(int)
        df["N_is_magic"] = df["N"].isin(MAGIC_NUMBERS).astype(int)
        df["ZN_both_magic"] = ((df["Z_is_magic"] == 1) & (df["N_is_magic"] == 1)).astype(int)

        logger.info(f"  -> {len(df)} çekirdek yüklendi, {len(df.columns)} sütun")
        self.df = df
        return df

    # -----------------------------------------------------------------------
    # Ana giriş: tüm analizler
    # -----------------------------------------------------------------------
    def run_all(self, data_path: Optional[str] = None) -> Dict:
        """Tüm analizleri çalıştır; Excel + grafikler üret."""
        if self.df is None or data_path:
            self.load_data(data_path)

        logger.info("\n" + "="*70)
        logger.info("[NuclearPatternAnalyzer] TÜM ANALİZLER BAŞLIYOR")
        logger.info("="*70)

        available_targets = [t for t in ("MM", "QM") if t in self.df.columns]
        if not available_targets:
            logger.warning("  [!] Hiç hedef sütunu bulunamadı (MM, QM)")
            return {}

        for target in available_targets:
            logger.info(f"\n--- Target: {target} ---")
            try:
                self._results[target] = self._analyze_target(target)
            except Exception as e:
                logger.warning(f"  [WARNING] {target} analizi başarısız: {e}")

        logger.info("\n[NuclearPatternAnalyzer] -> Excel raporu üretiliyor...")
        excel_path = self.generate_excel_report()

        logger.info("[NuclearPatternAnalyzer] -> Grafikler üretiliyor...")
        plot_paths = self.generate_plots() if _PLOT else []

        logger.info("\n" + "="*70)
        logger.info(f"[OK] NuclearPatternAnalyzer tamamlandı")
        logger.info(f"     Excel : {excel_path}")
        logger.info(f"     Grafik: {len(plot_paths)} dosya")
        logger.info("="*70)
        return {
            "results":    self._results,
            "excel_path": str(excel_path) if excel_path else None,
            "plot_paths": plot_paths,
        }

    # -----------------------------------------------------------------------
    # Tek hedef analizi
    # -----------------------------------------------------------------------
    def _analyze_target(self, target: str) -> Dict:
        df = self.df.copy()
        valid = df[["A", "Z", "N", target]].dropna()
        y    = valid[target].astype(float)

        return {
            "mean_cluster":    self._mean_cluster_analysis(valid, target, y),
            "isotope_chains":  self._chain_analysis(valid, target, groupby="Z", varyby="N"),
            "isotone_chains":  self._chain_analysis(valid, target, groupby="N", varyby="Z"),
            "isobar_chains":   self._chain_analysis(valid, target, groupby="A", varyby="Z"),
            "magic_analysis":  self._magic_number_analysis(valid, target, y),
            "jump_features":   self._jump_feature_analysis(valid, target),
        }

    # -----------------------------------------------------------------------
    # 1. Ortalama küme analizi
    # -----------------------------------------------------------------------
    def _mean_cluster_analysis(self, df: pd.DataFrame, target: str, y: pd.Series) -> Dict:
        """Hedef değerine göre çekirdekleri tipik/yüksek/düşük olarak grupla."""
        mu, sigma = y.mean(), y.std()
        bands = {
            "Tipik (±1σ)":      df[(y >= mu - sigma)   & (y <= mu + sigma)],
            "Yüksek (>1σ)":     df[y > mu + sigma],
            "Çok Yüksek (>2σ)": df[y > mu + 2 * sigma],
            "Düşük (<-1σ)":     df[y < mu - sigma],
            "Çok Düşük (<-2σ)": df[y < mu - 2 * sigma],
        }
        summary_rows = []
        for band_name, sub in bands.items():
            if len(sub) == 0:
                continue
            row = {"Grup": band_name, "N": len(sub)}
            for col in ("Z", "N", "A"):
                if col in sub.columns:
                    row[f"{col}_ort"] = round(sub[col].mean(), 2)
                    row[f"{col}_med"] = round(sub[col].median(), 2)
            row[f"{target}_ort"] = round(sub[target].mean(), 4)
            row["Z_is_magic_%"]  = round(sub.get("Z_is_magic", pd.Series(dtype=float)).mean() * 100, 1) if "Z_is_magic" in sub else None
            row["N_is_magic_%"]  = round(sub.get("N_is_magic", pd.Series(dtype=float)).mean() * 100, 1) if "N_is_magic" in sub else None
            # SPIN dağılımı
            if "SPIN" in sub.columns:
                spin_mode = sub["SPIN"].mode()
                row["SPIN_mod"] = float(spin_mode.iloc[0]) if len(spin_mode) > 0 else None
            summary_rows.append(row)
        logger.info(f"  Küme analizi: {len(summary_rows)} grup, µ={mu:.4f} +/-{sigma:.4f}")
        return {"summary": pd.DataFrame(summary_rows), "mean": float(mu), "std": float(sigma)}

    # -----------------------------------------------------------------------
    # 2/3/4. Zincir analizi (izotop / izotone / izobar)
    # -----------------------------------------------------------------------
    def _chain_analysis(
        self,
        df: pd.DataFrame,
        target: str,
        groupby: str,
        varyby: str,
    ) -> Dict:
        """
        Sabit `groupby` değeri için `varyby` boyunca hedef değişimini analiz et.
        Sıçrama noktaları: |ΔTarget| > jump_sigma × zincir_std
        """
        chains     = df.groupby(groupby)
        jump_rows  = []
        chain_stats = []

        for gval, chain_df in chains:
            chain_df = chain_df.sort_values(varyby)
            vals = chain_df[target].astype(float).values
            vary = chain_df[varyby].values

            if len(chain_df) < self.min_chain_len:
                continue

            delta   = np.diff(vals)
            delta_std  = delta.std() if len(delta) > 1 else 0.0
            delta_mean = delta.mean()
            threshold  = max(self.jump_sigma * delta_std, 0.01)

            # Zincir özet
            chain_stats.append({
                groupby:         gval,
                "uzunluk":       len(chain_df),
                f"{target}_ort": round(float(np.mean(vals)), 4),
                f"{target}_std": round(float(np.std(vals)), 4),
                "delta_ort":     round(float(delta_mean), 4),
                "delta_std":     round(float(delta_std), 4),
                "max_sıçrama":   round(float(np.abs(delta).max()), 4) if len(delta) > 0 else 0,
            })

            # Sıçrama noktaları
            jump_idx = np.where(np.abs(delta) > threshold)[0]
            for ji in jump_idx:
                row_before = chain_df.iloc[ji]
                row_after  = chain_df.iloc[ji + 1]
                jump_rows.append({
                    groupby:              gval,
                    f"{varyby}_öncesi":   float(vary[ji]),
                    f"{varyby}_sonrası":  float(vary[ji + 1]),
                    f"{target}_öncesi":   round(float(vals[ji]), 4),
                    f"{target}_sonrası":  round(float(vals[ji + 1]), 4),
                    "delta":              round(float(delta[ji]), 4),
                    "delta_sigma":        round(float(delta[ji] / delta_std) if delta_std > 0 else 0, 2),
                    "yön":                "↑" if delta[ji] > 0 else "↓",
                    "Z_öncesi":           float(row_before.get("Z", np.nan)),
                    "N_öncesi":           float(row_before.get("N", np.nan)),
                    "A_öncesi":           float(row_before.get("A", np.nan)),
                    "SPIN_öncesi":        float(row_before.get("SPIN", np.nan)) if "SPIN" in row_before.index else None,
                    "PARITY_öncesi":      float(row_before.get("PARITY", np.nan)) if "PARITY" in row_before.index else None,
                    "Z_magic_öncesi":     int(row_before.get("Z_is_magic", 0)),
                    "N_magic_öncesi":     int(row_before.get("N_is_magic", 0)),
                })

        jump_df  = pd.DataFrame(jump_rows)
        chain_df_out = pd.DataFrame(chain_stats).sort_values("max_sıçrama", ascending=False)

        n_jumps = len(jump_df)
        magic_jumps = (
            ((jump_df["N_magic_öncesi"] == 1) | (jump_df.get("Z_magic_öncesi", 0) == 1)).sum()
            if n_jumps > 0 else 0
        )
        logger.info(
            f"  {groupby} zinciri ({varyby} boyunca): "
            f"{len(chain_stats)} zincir, {n_jumps} sıçrama, "
            f"{magic_jumps} magic yakını"
        )
        return {
            "jump_df":     jump_df,
            "chain_stats": chain_df_out,
            "n_jumps":     n_jumps,
            "magic_jumps": magic_jumps,
        }

    # -----------------------------------------------------------------------
    # 5. Sihirli sayı analizi
    # -----------------------------------------------------------------------
    def _magic_number_analysis(
        self, df: pd.DataFrame, target: str, y: pd.Series
    ) -> Dict:
        """Shell kapanmalarındaki hedef değeri dağılımını non-magic ile karşılaştır."""
        rows = []
        for nucleon in ("Z", "N"):
            if nucleon not in df.columns:
                continue
            for magic in MAGIC_NUMBERS:
                near_mask = df[nucleon].between(magic - NEAR_MAGIC_DIST, magic + NEAR_MAGIC_DIST)
                far_mask  = ~near_mask
                near_vals = y[near_mask].dropna()
                far_vals  = y[far_mask].dropna()
                if len(near_vals) < 3 or len(far_vals) < 3:
                    continue
                ks_stat, ks_p = stats.ks_2samp(near_vals, far_vals)
                mw_stat, mw_p = stats.mannwhitneyu(near_vals, far_vals, alternative="two-sided")
                rows.append({
                    "Nükleon":         nucleon,
                    "Magic_N":         magic,
                    "Yakın_sayı":      len(near_vals),
                    "Uzak_sayı":       len(far_vals),
                    f"Yakın_{target}_ort":  round(float(near_vals.mean()), 4),
                    f"Uzak_{target}_ort":   round(float(far_vals.mean()), 4),
                    "Delta_ort":       round(float(near_vals.mean() - far_vals.mean()), 4),
                    "KS_stat":         round(float(ks_stat), 4),
                    "KS_p":            round(float(ks_p), 5),
                    "MW_p":            round(float(mw_p), 5),
                    "Sig_p05": "✓" if ks_p < 0.05 else "",
                })
        result_df = pd.DataFrame(rows)
        sig = result_df[result_df["Sig_p05"] == "✓"] if len(result_df) > 0 else result_df
        logger.info(f"  Sihirli sayı analizi: {len(rows)} karşılaştırma, {len(sig)} anlamlı (p<0.05)")
        return {"stats": result_df}

    # -----------------------------------------------------------------------
    # 6. Sıçrama özellik analizi
    # -----------------------------------------------------------------------
    def _jump_feature_analysis(self, df: pd.DataFrame, target: str) -> Dict:
        """
        Tüm zincirlerdeki sıçrama noktalarını topla;
        bu çekirdeklerin hangi özellikleri farklı/ortak olduğunu analiz et.
        """
        iso_jumps = self._chain_analysis(df, target, "Z", "N")["jump_df"]
        ison_jumps = self._chain_analysis(df, target, "N", "Z")["jump_df"]

        # Sıçrayan çekirdekleri ana DataFrame'den çek
        jump_nuclei_idx = []
        for jump_df, groupcol, varycol in [
            (iso_jumps,  "Z", "N_öncesi"),
            (ison_jumps, "N", "Z_öncesi"),
        ]:
            if len(jump_df) == 0:
                continue
            for _, row in jump_df.iterrows():
                g = row.get(groupcol)
                v = row.get(varycol)
                if pd.isna(g) or pd.isna(v):
                    continue
                mask = (df[groupcol] == g) & (df["N" if groupcol == "Z" else "Z"] == v)
                jump_nuclei_idx.extend(df[mask].index.tolist())

        jump_nuclei_idx = list(set(jump_nuclei_idx))
        if not jump_nuclei_idx:
            return {"comparison": pd.DataFrame()}

        normal_df = df.drop(index=jump_nuclei_idx)
        jump_df_sub = df.loc[jump_nuclei_idx]

        # Özellik karşılaştırması: sayısal sütunlar
        feat_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        # ek mevcut sayısal sütunlar
        feat_cols += [
            c for c in df.select_dtypes("number").columns
            if c not in feat_cols and c not in ("A", "Z", "N", target)
        ]
        feat_cols = list(dict.fromkeys(feat_cols))  # deduplicate

        rows = []
        for col in feat_cols:
            jv = jump_df_sub[col].dropna().astype(float)
            nv = normal_df[col].dropna().astype(float)
            if len(jv) < 2 or len(nv) < 2:
                continue
            t_stat, t_p = stats.ttest_ind(jv, nv, equal_var=False)
            rows.append({
                "Özellik":          col,
                "Sıçrama_ort":      round(float(jv.mean()), 4),
                "Normal_ort":       round(float(nv.mean()), 4),
                "Fark":             round(float(jv.mean() - nv.mean()), 4),
                "Sıçrama_std":      round(float(jv.std()), 4),
                "Normal_std":       round(float(nv.std()), 4),
                "t_stat":           round(float(t_stat), 4),
                "t_p":              round(float(t_p), 5),
                "Sig_p05": "✓" if t_p < 0.05 else "",
                "N_sıçrama":        len(jv),
                "N_normal":         len(nv),
            })

        comp_df = pd.DataFrame(rows)
        if len(comp_df) > 0 and "t_p" in comp_df.columns:
            comp_df = comp_df.sort_values("t_p")
        logger.info(
            f"  Sıçrama özellik analizi: {len(jump_nuclei_idx)} sıçrama çekirdeği, "
            f"{len(comp_df)} özellik, "
            f"{(comp_df.get('Sig_p05', pd.Series()) == '✓').sum()} anlamli"
        )
        return {
            "comparison":     comp_df,
            "jump_nuclei":    jump_df_sub[["NUCLEUS", "A", "Z", "N"] + [target]].reset_index(drop=True)
                              if "NUCLEUS" in jump_df_sub.columns else jump_df_sub[["A", "Z", "N", target]].reset_index(drop=True),
            "n_jump_nuclei":  len(jump_nuclei_idx),
        }

    # -----------------------------------------------------------------------
    # Excel raporu
    # -----------------------------------------------------------------------
    def generate_excel_report(self) -> Optional[Path]:
        """Tüm analiz sonuçlarını standart biçimlendirilmiş Excel'e yaz."""
        if not self._results:
            logger.warning("  [!] Sonuç yok -- Excel üretilemiyor")
            return None

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        excel_path = self.output_dir / f"nuclear_pattern_analysis_{timestamp}.xlsx"

        sheets: Dict[str, pd.DataFrame] = {}
        r2_map: Dict[str, List[str]] = {}

        # Genel özet
        overview_rows = []
        for target, res in self._results.items():
            mc = res.get("mean_cluster", {})
            iso = res.get("isotope_chains", {})
            ison = res.get("isotone_chains", {})
            isobar = res.get("isobar_chains", {})
            magic = res.get("magic_analysis", {})
            jf = res.get("jump_features", {})
            overview_rows.append({
                "Hedef":               target,
                "Ortalama":            round(mc.get("mean", 0), 4),
                "Std":                 round(mc.get("std", 0), 4),
                "İzotop_Sıçrama":      iso.get("n_jumps", 0),
                "İzotop_Magic_Sıç":    iso.get("magic_jumps", 0),
                "İzotone_Sıçrama":     ison.get("n_jumps", 0),
                "İzobar_Sıçrama":      isobar.get("n_jumps", 0),
                "Magic_Significant":       (magic.get("stats", pd.DataFrame()).get("Sig_p05", pd.Series()) == "✓").sum(),
                "Sıçrama_Çekirdek_N":  jf.get("n_jump_nuclei", 0),
                "Feature_Sig_N":   (jf.get("comparison", pd.DataFrame()).get("Sig_p05", pd.Series()) == "✓").sum(),
            })
        sheets["Genel_Özet"] = pd.DataFrame(overview_rows)

        for target, res in self._results.items():
            # Küme özeti
            mc = res.get("mean_cluster", {})
            if isinstance(mc.get("summary"), pd.DataFrame) and len(mc["summary"]) > 0:
                sheets[f"{target}_Küme"] = mc["summary"]

            # Zincir sıçramaları
            for chain_key, label in [
                ("isotope_chains",  f"{target}_İzotop_Sıçrama"),
                ("isotone_chains",  f"{target}_İzotone_Sıçrama"),
                ("isobar_chains",   f"{target}_İzobar_Sıçrama"),
            ]:
                ch = res.get(chain_key, {})
                jdf = ch.get("jump_df", pd.DataFrame())
                if len(jdf) > 0:
                    sheets[label[:31]] = jdf.sort_values("delta", key=abs, ascending=False)

            # Zincir istatistikleri
            for chain_key, label in [
                ("isotope_chains",  f"{target}_İzotop_ZincirStat"),
                ("isotone_chains",  f"{target}_İzotone_ZincirStat"),
                ("isobar_chains",   f"{target}_İzobar_ZincirStat"),
            ]:
                ch = res.get(chain_key, {})
                cdf = ch.get("chain_stats", pd.DataFrame())
                if len(cdf) > 0:
                    sheets[label[:31]] = cdf.head(50)

            # Sihirli sayı istatistikleri
            magic = res.get("magic_analysis", {})
            mdf = magic.get("stats", pd.DataFrame())
            if len(mdf) > 0:
                sheets[f"{target}_Magic_Analiz"[:31]] = mdf

            # Özellik karşılaştırması
            jf = res.get("jump_features", {})
            comp = jf.get("comparison", pd.DataFrame())
            if len(comp) > 0:
                sheets[f"{target}_Sıçrama_Özellik"[:31]] = comp
            jn = jf.get("jump_nuclei", pd.DataFrame())
            if len(jn) > 0:
                sheets[f"{target}_Sıçrama_Çekirdek"[:31]] = jn

        if not sheets:
            logger.warning("  [!] Sayfa verisi yok -- Excel üretilemiyor")
            return None

        if _ES:
            try:
                with ExcelStandardizer(excel_path) as es:
                    for sname, sdf in sheets.items():
                        # p-değeri sütunlarını koşullu biçimlendir
                        cond_cols = [c for c in sdf.columns if "p" in c.lower() and "_" in c]
                        es.write_sheet(sname[:31], sdf, conditional_cols=cond_cols)
                logger.info(f"  [OK] Excel (ExcelStandardizer): {excel_path.name}")
            except Exception as e:
                logger.warning(f"  ExcelStandardizer hatası, düz pandas: {e}")
                self._write_plain_excel(excel_path, sheets)
        else:
            self._write_plain_excel(excel_path, sheets)

        return excel_path

    def _write_plain_excel(self, excel_path: Path, sheets: Dict) -> None:
        with pd.ExcelWriter(str(excel_path), engine="openpyxl") as w:
            for sname, sdf in sheets.items():
                sdf.to_excel(w, sheet_name=sname[:31], index=False)
        logger.info(f"  [OK] Excel (düz): {excel_path.name}")

    # -----------------------------------------------------------------------
    # Grafikler
    # -----------------------------------------------------------------------
    def generate_plots(self) -> List[str]:
        """Tüm grafikler: zincir çizgileri, violin, heatmap."""
        if not _PLOT:
            logger.info("  [INFO] matplotlib yok -- grafikler atlanıyor")
            return []

        out_dir = self.output_dir / "plots"
        out_dir.mkdir(exist_ok=True)
        paths = []

        plt.style.use("seaborn-v0_8-whitegrid")

        for target, res in self._results.items():
            # 1. İzotop zinciri grafikleri — en çok sıçrayan top-8 Z
            paths += self._plot_chains(res, target, "isotope_chains", "Z", "N", out_dir)
            # 2. İzotone zinciri grafikleri — top-8 N
            paths += self._plot_chains(res, target, "isotone_chains", "N", "Z", out_dir)
            # 3. Sihirli sayı violin
            paths += self._plot_magic_violin(res, target, out_dir)
            # 4. Özellik fark bar chart
            paths += self._plot_feature_diff(res, target, out_dir)
            # 5. Küme dağılım barplot
            paths += self._plot_cluster_bar(res, target, out_dir)

        logger.info(f"  [OK] {len(paths)} grafik üretildi -> {out_dir}")
        return [str(p) for p in paths]

    def _plot_chains(
        self, res: Dict, target: str, chain_key: str,
        groupby: str, varyby: str, out_dir: Path
    ) -> List[Path]:
        """Zincir çizgisi grafiği — top-8 en değişken zincir."""
        paths = []
        ch = res.get(chain_key, {})
        cstats = ch.get("chain_stats", pd.DataFrame())
        jdf    = ch.get("jump_df",   pd.DataFrame())
        if len(cstats) == 0:
            return paths

        top_chains = cstats.nlargest(8, "max_sıçrama")[groupby].tolist()
        df = self.df.copy()
        valid = df[["A", "Z", "N", target]].dropna()

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(
            f"{target} — {groupby} Sabit Zincirleri (En Yüksek Sıçrama)",
            fontsize=14, fontweight="bold"
        )
        axes = axes.flatten()

        for ax_i, gval in enumerate(top_chains):
            ax = axes[ax_i]
            chain = valid[valid[groupby] == gval].sort_values(varyby)
            if len(chain) < 2:
                ax.set_visible(False)
                continue
            ax.plot(chain[varyby], chain[target], marker="o", linewidth=1.8,
                    markersize=5, color="steelblue", label=f"{groupby}={gval}")
            # Sıçrama noktalarını kırmızı ile işaretle
            if len(jdf) > 0 and groupby in jdf.columns:
                g_jumps = jdf[jdf[groupby] == gval]
                for _, jr in g_jumps.iterrows():
                    xv = jr.get(f"{varyby}_öncesi")
                    yv = jr.get(f"{target}_öncesi")
                    if pd.notna(xv) and pd.notna(yv):
                        ax.axvline(x=xv, color="red", alpha=0.5, linewidth=1.2, linestyle="--")
                        ax.scatter([xv], [yv], color="red", zorder=5, s=60)
            # Sihirli sayı çizgileri
            for mn in MAGIC_NUMBERS:
                ax.axvline(x=mn, color="gray", alpha=0.3, linewidth=0.8, linestyle=":")
            ax.set_xlabel(varyby, fontsize=8)
            ax.set_ylabel(target, fontsize=8)
            ax.set_title(f"{groupby} = {gval}", fontsize=9)
            ax.tick_params(labelsize=7)

        for ax in axes[len(top_chains):]:
            ax.set_visible(False)

        plt.tight_layout()
        label = "izotop" if chain_key == "isotope_chains" else ("izotone" if "isotone" in chain_key else "izobar")
        fname = out_dir / f"{target}_{label}_zincir.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fname)
        return paths

    def _plot_magic_violin(self, res: Dict, target: str, out_dir: Path) -> List[Path]:
        """Sihirli sayı yakını vs. uzağı violin plot."""
        paths = []
        df = self.df.copy()
        valid = df[["Z", "N", target, "Z_is_magic", "N_is_magic"]].dropna()
        if len(valid) < 10:
            return paths

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{target} — Sihirli Sayı Etkisi", fontsize=13, fontweight="bold")

        for ax, nucleon, col in [
            (axes[0], "Z", "Z_is_magic"),
            (axes[1], "N", "N_is_magic"),
        ]:
            plot_df = valid[[target, col]].copy()
            plot_df["Grup"] = plot_df[col].map({1: "Magic", 0: "Non-magic"})
            try:
                sns.violinplot(data=plot_df, x="Grup", y=target, ax=ax, palette=["#4472C4", "#ED7D31"])
                sns.stripplot(data=plot_df, x="Grup", y=target, ax=ax, color="k", alpha=0.25, size=3)
            except Exception:
                plot_df.boxplot(column=target, by="Grup", ax=ax)
            ax.set_title(f"{nucleon} bazında", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel(target, fontsize=9)

        plt.tight_layout()
        fname = out_dir / f"{target}_magic_violin.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fname)
        return paths

    def _plot_feature_diff(self, res: Dict, target: str, out_dir: Path) -> List[Path]:
        """Sıçrama çekirdekleri ile normal çekirdekler arasındaki özellik farklarının bar grafiği."""
        paths = []
        jf = res.get("jump_features", {})
        comp = jf.get("comparison", pd.DataFrame())
        if len(comp) < 3:
            return paths

        top_sig = comp[comp["Sig_p05"] == "✓"].head(12)
        if len(top_sig) < 2:
            top_sig = comp.head(12)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#C6EFCE" if v >= 0 else "#FFC7CE" for v in top_sig["Fark"]]
        bars = ax.barh(top_sig["Özellik"], top_sig["Fark"], color=colors, edgecolor="gray", linewidth=0.5)
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlabel("Ortalama Fark (Sıçrama − Normal)", fontsize=9)
        ax.set_title(f"{target} — Sıçrama Çekirdeklerinde Özellik Farkları", fontsize=11, fontweight="bold")
        _sig_col = "Sig_p05"
        for bar, (_, row) in zip(bars, top_sig.iterrows()):
            sig = "✓" if row.get(_sig_col, "") == "✓" else ""
            t_p_val = row.get("t_p", float("nan"))
            ax.text(
                bar.get_width() + 0.001 if bar.get_width() >= 0 else bar.get_width() - 0.001,
                bar.get_y() + bar.get_height() / 2,
                f" p={t_p_val:.3f} {sig}", va="center", fontsize=7,
                ha="left" if bar.get_width() >= 0 else "right"
            )
        plt.tight_layout()
        fname = out_dir / f"{target}_sicrama_ozellik_fark.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fname)
        return paths

    def _plot_cluster_bar(self, res: Dict, target: str, out_dir: Path) -> List[Path]:
        """Küme grupları: ortalama ± std bar grafiği."""
        paths = []
        mc  = res.get("mean_cluster", {})
        sdf = mc.get("summary", pd.DataFrame())
        if len(sdf) < 2:
            return paths

        fig, ax = plt.subplots(figsize=(9, 5))
        groups    = sdf["Grup"].tolist()
        means     = sdf[f"{target}_ort"].tolist()
        colors    = ["#4472C4", "#70AD47", "#FF0000", "#FFC000", "#7030A0"]
        x_pos     = range(len(groups))
        ax.bar(x_pos, means, color=colors[:len(groups)], alpha=0.8, edgecolor="gray")
        ax.axhline(y=res.get("mean_cluster", {}).get("mean", 0), color="navy",
                   linestyle="--", linewidth=1.2, label="Genel ort.")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(f"{target} Ortalaması", fontsize=9)
        ax.set_title(f"{target} — Hedef Değer Kümelerine Göre Dağılım", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = out_dir / f"{target}_kume_bar.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fname)
        return paths


# ===========================================================================
# Standalone çalıştırma
# ===========================================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = sys.argv[1] if len(sys.argv) > 1 else "aaa2.txt"
    npa = NuclearPatternAnalyzer(data_path=data, output_dir="outputs/nuclear_patterns")
    npa.run_all()
