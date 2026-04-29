"""
Nuclear Moment Band Analyzer
============================
PFAZ 12 -- Ek Modul: Deger Bandi ve Oruntu Analizi

Temel sorular:
  1. BANT ANALIZI  -- Ornegin MM~4 olan tum cekirdekler: ortak ozellikleri ne?
                      Hafif (A<50) vs orta (50<=A<=100) vs agir (A>100) ayni bantta neden?
  2. KOMSU SICRAMA -- Izotop/izoton zincirinde ani MM/QM degisimi: hangi ozellik degisiyor?
  3. CORR IN BAND  -- Bant uyeligi ile hangi ozellikler en cok korelasyon gosteriyor?
  4. FIZIKSEL YORUM -- Valans nokleon konfigurasyonu, spin, kabuk dolulugu

Cikti (Excel cok sayfalı):
  Bant_Ozeti      -- her bant icin istatistik + ozellik profili
  Sicrama_Analizi -- sican cekirdek ciftleri + degisen ozellikler
  Capraz_Kutle    -- hafif vs agir karsilastirmasi ayni bantta
  Korelasyon      -- bant uyeligiyle korelasyon siralamasi
  Cekirdek_Detay  -- her cekirdek icin bant atamasi + z-skor
  Aciklama        -- fiziksel yorum ozeti

Not: Veri setinde sihirli cekirdek (magic nucleus) yok;
     tum analizler bu varsayimla yapilir.
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
# Magic number sinirlari (mesafe hesabi icin; ama veri setinde magic yok)
# ---------------------------------------------------------------------------
MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]

# Kutle bolgeleri
MASS_LIGHT  = (0,   50)    # A < 50
MASS_MEDIUM = (50, 100)    # 50 <= A < 100
MASS_HEAVY  = (100, 300)   # A >= 100

# Otomatik bant sayisi (her hedef icin)
N_BANDS = 6                # ceyrekler + alt/ust uclar

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT = True
except ImportError:
    _PLOT = False

try:
    from pfaz_modules.pfaz06_final_reporting.excel_standardizer import ExcelStandardizer
    _ES = True
except ImportError:
    _ES = False


# ===========================================================================
class NuclearMomentBandAnalyzer:
    """
    Moment degeri bant analizi ve oruntu tespiti.

    Kullanim:
        analyzer = NuclearMomentBandAnalyzer(data_path='data/aaa2.txt',
                                              output_dir='outputs/band_analysis')
        analyzer.run_all()
    """

    TARGET_COLS = {
        "MM":     ["MAGNETIC MOMENT [µ]", "MAGNETIC MOMENT [μ]", "MM", "magnetic_moment"],
        "QM":     ["QUADRUPOLE MOMENT [Q]", "QM", "quadrupole_moment"],
        "Beta_2": ["Beta_2", "BETA_2", "beta2", "Beta2"],
    }

    # Analiz edilecek ozellikler -- fiziksel anlami olan sutunlar
    PHYSICS_FEATURES = [
        "SPIN", "PARITY",
        "Z", "N", "A",
        "P-factor", "Nn", "Np",
        "BE_per_A", "BE_total",
        "Z_magic_dist", "N_magic_dist",
        "Z_valence", "N_valence",
        "Z_shell_gap", "N_shell_gap",
        "BE_pairing", "BE_asymmetry",
        "spherical_index", "Q0_intrinsic",
        "S_n_approx", "S_p_approx",
        "Beta_2_estimated", "magic_character",
        # Hesaplanacak (yoksa atlanir)
        "A_mod_2",       # tek/cift kutle
        "Z_mod_2",       # tek/cift proton
        "N_mod_2",       # tek/cift notron
        "Z_N_ratio",     # Z/N orani
        "valence_total", # Z_valence + N_valence
    ]

    def __init__(
        self,
        data_path: str = "data/aaa2.txt",
        output_dir: str = "outputs/band_analysis",
        jump_sigma: float = 2.0,    # sicrama esigi: kac std sapma
        n_bands: int = N_BANDS,
        min_band_size: int = 3,     # bant analizi icin minimum cekirdek sayisi
    ):
        self.data_path     = Path(data_path)
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jump_sigma    = jump_sigma
        self.n_bands       = n_bands
        self.min_band_size = min_band_size
        self.df: Optional[pd.DataFrame] = None
        self._results: Dict = {}

    # -----------------------------------------------------------------------
    # Veri yukleme ve hazırlık
    # -----------------------------------------------------------------------
    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        p = Path(path) if path else self.data_path
        logger.info(f"[BandAnalyzer] Veri yukleniyor: {p}")

        if p.suffix in (".txt", ".csv"):
            try:
                from utils.file_io_utils import read_nuclear_data
                df = read_nuclear_data(str(p))
            except Exception:
                df = pd.read_csv(str(p), sep="\t", encoding="utf-8", on_bad_lines="skip")
        elif p.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(str(p))
        else:
            raise ValueError(f"Desteklenmeyen dosya formati: {p.suffix}")

        df.columns = [c.strip() for c in df.columns]

        # Hedef sutun eslestirme
        for canonical, aliases in self.TARGET_COLS.items():
            for alias in aliases:
                matched = [c for c in df.columns if c.lower() == alias.lower()]
                if matched and canonical not in df.columns:
                    df = df.rename(columns={matched[0]: canonical})
                    break

        # Sayisal donusum
        text_skip = {"NUCLEUS", "ELEMENT", "SYMBOL"}
        for col in df.columns:
            if col in text_skip or df[col].dtype != object:
                continue
            cleaned = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
            num = pd.to_numeric(cleaned, errors="coerce")
            if num.notna().sum() / max(len(df), 1) >= 0.30:
                df[col] = num

        for req in ("A", "Z", "N"):
            if req not in df.columns:
                raise KeyError(f"Zorunlu sutun eksik: {req}")
            df[req] = pd.to_numeric(df[req], errors="coerce")
        df = df.dropna(subset=["A", "Z", "N"])
        df["A"] = df["A"].astype(int)
        df["Z"] = df["Z"].astype(int)
        df["N"] = df["N"].astype(int)

        # Turetilen ozellikler
        df["A_mod_2"]       = df["A"] % 2
        df["Z_mod_2"]       = df["Z"] % 2
        df["N_mod_2"]       = df["N"] % 2
        df["Z_N_ratio"]     = df["Z"] / df["N"].replace(0, np.nan)
        df["valence_total"] = df.get("Z_valence", pd.Series(np.nan, index=df.index)).fillna(0) + \
                              df.get("N_valence", pd.Series(np.nan, index=df.index)).fillna(0)

        # Magic distance (her zaman yeniden hesapla)
        df["Z_magic_dist_c"] = df["Z"].apply(lambda v: min(abs(v - m) for m in MAGIC_NUMBERS))
        df["N_magic_dist_c"] = df["N"].apply(lambda v: min(abs(v - m) for m in MAGIC_NUMBERS))

        # Kutle bolge etiketi
        df["mass_region"] = pd.cut(
            df["A"],
            bins=[0, 50, 100, 999],
            labels=["Hafif(A<50)", "Orta(50-100)", "Agir(A>100)"],
        )

        logger.info(f"  -> {len(df)} cekirdek, {len(df.columns)} sutun yüklendi")
        self.df = df
        return df

    # -----------------------------------------------------------------------
    # Ana calisma
    # -----------------------------------------------------------------------
    def run_all(self, data_path: Optional[str] = None) -> Dict:
        if self.df is None or data_path:
            self.load_data(data_path)

        logger.info("\n" + "=" * 70)
        logger.info("[BandAnalyzer] BANT + ORUNTU ANALİZİ BASLIYOR")
        logger.info("=" * 70)

        targets = [t for t in ("MM", "QM") if t in self.df.columns]
        if not targets:
            logger.warning("[BandAnalyzer] Hic hedef sutun bulunamadi")
            return {}

        all_band_summaries   = []
        all_jump_records     = []
        all_cross_mass       = []
        all_corr_records     = []
        all_nucleus_detail   = []
        explanations         = []
        self._accuracy_rows    = []
        self._accuracy_summary = []

        for target in targets:
            logger.info(f"\n--- Hedef: {target} ---")
            df_t = self.df.dropna(subset=[target]).copy()
            df_t[target] = df_t[target].astype(float)

            bands = self._define_bands(df_t[target], self.n_bands)
            df_t  = self._assign_bands(df_t, target, bands)

            # 1. Bant ozeti + ozellik profili
            band_sum = self._band_summary(df_t, target, bands)
            for row in band_sum:
                row["target"] = target
            all_band_summaries.extend(band_sum)

            # 2. Sicrama analizi (izotop + izoton)
            jumps = self._jump_analysis(df_t, target)
            for row in jumps:
                row["target"] = target
            all_jump_records.extend(jumps)

            # 3. Capraz kutle bolgesi analizi
            cross = self._cross_mass_analysis(df_t, target, bands)
            for row in cross:
                row["target"] = target
            all_cross_mass.extend(cross)

            # 4. Korelasyon (bant uyeligiyle)
            corr = self._band_correlation(df_t, target, bands)
            for row in corr:
                row["target"] = target
            all_corr_records.extend(corr)

            # 5. Cekirdek detay tablosu
            detail = self._nucleus_detail(df_t, target)
            all_nucleus_detail.extend(detail)

            # 6. Aciklama/yorum
            expl = self._generate_explanation(df_t, target, bands, jumps, cross, corr)
            explanations.append({"target": target, "yorum": expl})

            self._results[target] = {
                "bands":      bands,
                "band_sum":   band_sum,
                "jumps":      jumps,
                "cross_mass": cross,
                "corr":       corr,
            }

        # 7. Tahmin dogrulugu analizi — tum 267 cekirdek (PFAZ4 ciktisiyla kesisim)
        acc_rows = self._prediction_accuracy_analysis(all_jump_records, all_nucleus_detail)
        if _PLOT and acc_rows:
            self._plot_jump_accuracy(acc_rows)

        # 8. Pivot tablo — bant x model tipi hata ozeti
        pivot_rows = self._build_pivot_summary(acc_rows)

        # 9. Harici Excel korelasyonu (dis referans dosyasi varsa)
        ext_corr_rows = self._external_excel_correlation(all_nucleus_detail)

        # Grafik
        if _PLOT:
            self._make_plots(targets)

        # Excel
        excel_path = self._save_excel(
            all_band_summaries, all_jump_records, all_cross_mass,
            all_corr_records, all_nucleus_detail, explanations,
            acc_rows, self._accuracy_summary,
            pivot_rows, ext_corr_rows
        )

        logger.info(f"\n[OK] BandAnalyzer tamamlandi → {excel_path}")
        return {
            "results":    self._results,
            "excel_path": str(excel_path) if excel_path else None,
        }

    # -----------------------------------------------------------------------
    # 1. Band tanimlama
    # -----------------------------------------------------------------------
    def _define_bands(self, series: pd.Series, n: int) -> List[Tuple[float, float, str]]:
        """
        Degere gore n bant tanimla.
        Donurus: [(alt, ust, etiket), ...]
        """
        clean = series.dropna()
        if len(clean) < 4:
            return [(float(clean.min()), float(clean.max()), "Tum")]

        percentiles = np.linspace(0, 100, n + 1)
        edges = np.percentile(clean, percentiles)
        edges = np.unique(np.round(edges, 4))

        bands = []
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            # Son bantta ust sinir dahil
            if i == len(edges) - 2:
                hi = float(clean.max()) + 1e-9
            mid = (lo + hi) / 2
            label = f"B{i+1}[{lo:.2f},{hi:.2f})"
            bands.append((lo, hi, label))
        return bands

    def _assign_bands(self, df: pd.DataFrame, target: str, bands) -> pd.DataFrame:
        df = df.copy()
        df["band_label"] = "disi"
        df["band_idx"]   = -1
        for idx, (lo, hi, lbl) in enumerate(bands):
            mask = (df[target] >= lo) & (df[target] < hi)
            df.loc[mask, "band_label"] = lbl
            df.loc[mask, "band_idx"]   = idx
        return df

    # -----------------------------------------------------------------------
    # 2. Bant ozeti
    # -----------------------------------------------------------------------
    def _band_summary(self, df: pd.DataFrame, target: str, bands) -> List[Dict]:
        rows = []
        feats = self._available_features(df)

        for idx, (lo, hi, lbl) in enumerate(bands):
            sub = df[df["band_idx"] == idx]
            if len(sub) < self.min_band_size:
                continue

            row = {
                "bant":          lbl,
                "n_cekirdek":    len(sub),
                "deger_min":     float(sub[target].min()),
                "deger_max":     float(sub[target].max()),
                "deger_ort":     float(sub[target].mean()),
                "deger_std":     float(sub[target].std()),
                "A_ort":         float(sub["A"].mean()),
                "Z_ort":         float(sub["Z"].mean()),
                "N_ort":         float(sub["N"].mean()),
                "hafif_sayi":    int((sub["A"] < 50).sum()),
                "orta_sayi":     int(((sub["A"] >= 50) & (sub["A"] < 100)).sum()),
                "agir_sayi":     int((sub["A"] >= 100).sum()),
                "tek_A":         int((sub["A_mod_2"] == 1).sum()),
                "cift_A":        int((sub["A_mod_2"] == 0).sum()),
            }

            # Her ozellik icin: bant ici ort ve tam veri ile karsilastirma (z-skor)
            full_feat_stats = {}
            for f in feats:
                full_vals = pd.to_numeric(df[f], errors="coerce").dropna()
                band_vals = pd.to_numeric(sub[f], errors="coerce").dropna()
                if len(band_vals) < 2 or len(full_vals) < 2:
                    continue
                full_mu  = full_vals.mean()
                full_sig = full_vals.std() + 1e-10
                band_mu  = band_vals.mean()
                z_score  = (band_mu - full_mu) / full_sig
                row[f"ozellik_{f}_ort"]    = round(float(band_mu), 4)
                row[f"ozellik_{f}_zscore"] = round(float(z_score), 3)
                full_feat_stats[f] = z_score

            # En ayirt edici 5 ozellik (mutlak z-skor en yuksek)
            top5 = sorted(full_feat_stats.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            row["en_ayirt_edici_ozellikler"] = " | ".join(
                f"{f}(z={z:.2f})" for f, z in top5
            )

            # Cekirdek listesi (kisa)
            if "NUCLEUS" in sub.columns:
                nuclei_list = sub["NUCLEUS"].astype(str).tolist()
            else:
                nuclei_list = [f"Z{r.Z}N{r.N}" for _, r in sub.iterrows()]
            row["cekirdekler"] = ", ".join(nuclei_list[:30])
            if len(nuclei_list) > 30:
                row["cekirdekler"] += f" ... (+{len(nuclei_list)-30})"

            rows.append(row)
        return rows

    # -----------------------------------------------------------------------
    # 3. Komsu sicrama analizi
    # -----------------------------------------------------------------------
    def _jump_analysis(self, df: pd.DataFrame, target: str) -> List[Dict]:
        """
        Izotop zincirleri (sabit Z, artan N) ve izoton zincirleri (sabit N, artan Z)'nda
        komsu cekirdek arasındaki delta > jump_sigma * zincir_std ise SICRAMA olarak isaretler.
        Sicrama icin: hangi ozellikler de degisti? (delta_ozellik)
        """
        records = []
        feats   = self._available_features(df)

        def _chain_jumps(chain_key: str, step_key: str, label: str):
            for chain_val, grp in df.groupby(chain_key):
                grp = grp.sort_values(step_key).reset_index(drop=True)
                if len(grp) < 3:
                    continue
                vals = grp[target].astype(float).values
                deltas = np.abs(np.diff(vals))
                mu, sig = deltas.mean(), deltas.std() + 1e-10
                threshold = mu + self.jump_sigma * sig

                for i, delta in enumerate(deltas):
                    if delta < threshold:
                        continue
                    r0, r1 = grp.iloc[i], grp.iloc[i + 1]
                    nuc0 = r0.get("NUCLEUS", f"Z{r0.Z}N{r0.N}") if hasattr(r0, 'get') else \
                           r0["NUCLEUS"] if "NUCLEUS" in grp.columns else f"Z{r0['Z']}N{r0['N']}"
                    nuc1 = r1.get("NUCLEUS", f"Z{r1.Z}N{r1.N}") if hasattr(r1, 'get') else \
                           r1["NUCLEUS"] if "NUCLEUS" in grp.columns else f"Z{r1['Z']}N{r1['N']}"

                    # Hangi ozellikler de buyuk degisim gosterdi?
                    changing = []
                    for f in feats:
                        v0 = pd.to_numeric(r0.get(f, np.nan) if hasattr(r0, 'get') else r0[f], errors="coerce")
                        v1 = pd.to_numeric(r1.get(f, np.nan) if hasattr(r1, 'get') else r1[f], errors="coerce")
                        if pd.isna(v0) or pd.isna(v1):
                            continue
                        d_feat = abs(float(v1) - float(v0))
                        full_std = pd.to_numeric(df[f], errors="coerce").std()
                        if full_std > 1e-10 and d_feat > 0.5 * full_std:
                            changing.append(f"{f}(d={d_feat:.3f})")

                    records.append({
                        "zincir_turu":       label,
                        "zincir_degeri":     int(chain_val),
                        "cekirdek_A":        str(nuc0),
                        "cekirdek_B":        str(nuc1),
                        f"{step_key}_A":     int(r0[step_key]),
                        f"{step_key}_B":     int(r1[step_key]),
                        f"{target}_A":       round(float(r0[target]), 4),
                        f"{target}_B":       round(float(r1[target]), 4),
                        f"delta_{target}":   round(float(r1[target] - r0[target]), 4),
                        "abs_delta":         round(float(delta), 4),
                        "esik":              round(float(threshold), 4),
                        "sigma_kati":        round(float(delta / (sig + 1e-10)), 2),
                        "band_A":            r0.get("band_label", "?") if hasattr(r0, 'get') else r0.get("band_label", "?"),
                        "band_B":            r1.get("band_label", "?") if hasattr(r1, 'get') else r1.get("band_label", "?"),
                        "A_A":               int(r0["A"]),
                        "A_B":               int(r1["A"]),
                        "degisen_ozellikler": " | ".join(changing[:8]) if changing else "—",
                    })

        _chain_jumps("Z", "N", "Izotop(sabit_Z)")
        _chain_jumps("N", "Z", "Izoton(sabit_N)")

        # A'ya gore de sirala (izobar icin) -- opsiyonel
        for a_val, grp in df.groupby("A"):
            grp = grp.sort_values("Z").reset_index(drop=True)
            if len(grp) < 2:
                continue
            vals   = grp[target].astype(float).values
            deltas = np.abs(np.diff(vals))
            mu, sig = deltas.mean(), deltas.std() + 1e-10
            threshold = mu + self.jump_sigma * sig
            for i, delta in enumerate(deltas):
                if delta < threshold:
                    continue
                r0, r1 = grp.iloc[i], grp.iloc[i + 1]
                nuc0 = r0["NUCLEUS"] if "NUCLEUS" in grp.columns else f"Z{r0['Z']}N{r0['N']}"
                nuc1 = r1["NUCLEUS"] if "NUCLEUS" in grp.columns else f"Z{r1['Z']}N{r1['N']}"
                changing = []
                for f in feats:
                    v0 = pd.to_numeric(r0[f] if f in r0.index else np.nan, errors="coerce")
                    v1 = pd.to_numeric(r1[f] if f in r1.index else np.nan, errors="coerce")
                    if pd.isna(v0) or pd.isna(v1):
                        continue
                    d_feat = abs(float(v1) - float(v0))
                    fs = pd.to_numeric(df[f], errors="coerce").std()
                    if fs > 1e-10 and d_feat > 0.5 * fs:
                        changing.append(f"{f}(d={d_feat:.3f})")
                records.append({
                    "zincir_turu":       "Izobar(sabit_A)",
                    "zincir_degeri":     int(a_val),
                    "cekirdek_A":        str(nuc0),
                    "cekirdek_B":        str(nuc1),
                    "Z_A":               int(r0["Z"]),
                    "Z_B":               int(r1["Z"]),
                    f"{target}_A":       round(float(r0[target]), 4),
                    f"{target}_B":       round(float(r1[target]), 4),
                    f"delta_{target}":   round(float(r1[target] - r0[target]), 4),
                    "abs_delta":         round(float(delta), 4),
                    "esik":              round(float(threshold), 4),
                    "sigma_kati":        round(float(delta / (sig + 1e-10)), 2),
                    "band_A":            r0.get("band_label", "?"),
                    "band_B":            r1.get("band_label", "?"),
                    "A_A":               int(r0["A"]),
                    "A_B":               int(r1["A"]),
                    "degisen_ozellikler": " | ".join(changing[:8]) if changing else "—",
                })
        return records

    # -----------------------------------------------------------------------
    # 4. Capraz kutle bolge analizi
    # -----------------------------------------------------------------------
    def _cross_mass_analysis(self, df: pd.DataFrame, target: str, bands) -> List[Dict]:
        """
        Ayni bantta farkli kutle bolgelerindeki cekirdekleri karsilastir.
        Hafif(A<50) vs Agir(A>=100) ayni MM bandinda: ortak ozellikler neler?
        """
        records = []
        feats   = self._available_features(df)
        regions = [("Hafif(A<50)", df["A"] < 50),
                   ("Orta(50-100)", (df["A"] >= 50) & (df["A"] < 100)),
                   ("Agir(A>=100)", df["A"] >= 100)]

        for idx, (lo, hi, lbl) in enumerate(bands):
            sub = df[df["band_idx"] == idx]
            if len(sub) < self.min_band_size:
                continue

            region_stats = {}
            for reg_name, reg_mask in regions:
                reg_sub = sub[reg_mask]
                if len(reg_sub) < 2:
                    continue
                feat_means = {}
                for f in feats:
                    vals = pd.to_numeric(reg_sub[f], errors="coerce").dropna()
                    if len(vals) >= 2:
                        feat_means[f] = float(vals.mean())
                region_stats[reg_name] = {"n": len(reg_sub), "feats": feat_means}

            if len(region_stats) < 2:
                continue

            reg_names = list(region_stats.keys())

            # Capraz karsilastirma: hangi ozellikler tum bolgelerde benzer?
            shared_feats = []
            diff_feats   = []
            for f in feats:
                f_vals = [region_stats[r]["feats"].get(f) for r in reg_names
                          if f in region_stats[r]["feats"]]
                if len(f_vals) < 2:
                    continue
                spread = max(f_vals) - min(f_vals)
                full_std = pd.to_numeric(df[f], errors="coerce").std() + 1e-10
                norm_spread = spread / full_std
                if norm_spread < 0.3:   # benzer bolgede
                    shared_feats.append(f"{f}(yayilma={norm_spread:.2f})")
                elif norm_spread > 1.0:  # farkli
                    diff_feats.append(f"{f}(yayilma={norm_spread:.2f})")

            row = {
                "bant":              lbl,
                "toplam_n":          len(sub),
                "ortak_ozellikler":  " | ".join(shared_feats[:6]) if shared_feats else "—",
                "farkli_ozellikler": " | ".join(diff_feats[:6])   if diff_feats  else "—",
            }
            for reg_name, rstat in region_stats.items():
                row[f"{reg_name}_n"] = rstat["n"]
                # SPIN, Z_valence, N_valence gibi temel fiziksel ozellikler
                for pf in ["SPIN", "Z_valence", "N_valence", "BE_per_A", "Z_magic_dist_c", "N_magic_dist_c"]:
                    if pf in rstat["feats"]:
                        row[f"{reg_name}_{pf}"] = round(rstat["feats"][pf], 3)

            # Fiziksel yorum: neden bu kadar farkli bolgeden cekirdek ayni bantta?
            if shared_feats:
                row["fiziksel_yorum"] = (
                    f"Paylaşılan ozellikler ({len(shared_feats)} adet) "
                    f"farkli A bolgelerini ayni {target} bandina birlestiriyor. "
                    f"Muhtemel neden: benzer valans nokleon konfigurasyonu veya deformasyon."
                )
            else:
                row["fiziksel_yorum"] = (
                    f"Bolgeler arasi ortak ozellik az → "
                    f"farkli fiziksel mekanizmalar ayni {target} degerini uretebilir."
                )

            records.append(row)
        return records

    # -----------------------------------------------------------------------
    # 5. Korelasyon analizi
    # -----------------------------------------------------------------------
    def _band_correlation(self, df: pd.DataFrame, target: str, bands) -> List[Dict]:
        """
        Her bant icin: ozellikler ile bant uyeliginin korelasyonu.
        Point-biserial + Spearman kullanilir.
        """
        records = []
        feats = self._available_features(df)
        target_vals = pd.to_numeric(df[target], errors="coerce")

        for f in feats:
            feat_vals = pd.to_numeric(df[f], errors="coerce")
            valid_mask = target_vals.notna() & feat_vals.notna()
            if valid_mask.sum() < 10:
                continue

            tv = target_vals[valid_mask].values
            fv = feat_vals[valid_mask].values

            try:
                spearman_r, spearman_p = stats.spearmanr(tv, fv)
                pearson_r,  pearson_p  = stats.pearsonr(tv, fv)
            except Exception:
                continue

            records.append({
                "ozellik":     f,
                "spearman_r":  round(float(spearman_r), 4),
                "spearman_p":  round(float(spearman_p), 6),
                "pearson_r":   round(float(pearson_r), 4),
                "pearson_p":   round(float(pearson_p), 6),
                "abs_spearman":abs(float(spearman_r)),
                "yorum":       self._corr_label(abs(float(spearman_r))),
            })

        # Spearman mutlak degerine gore siralama
        records.sort(key=lambda x: x["abs_spearman"], reverse=True)
        for i, r in enumerate(records):
            r["sira"] = i + 1
        return records

    @staticmethod
    def _corr_label(abs_r: float) -> str:
        if abs_r >= 0.70:  return "Guclu korelasyon"
        if abs_r >= 0.50:  return "Orta korelasyon"
        if abs_r >= 0.30:  return "Zayif korelasyon"
        return "Cok zayif / yok"

    # -----------------------------------------------------------------------
    # 6. Cekirdek detay tablosu
    # -----------------------------------------------------------------------
    def _nucleus_detail(self, df: pd.DataFrame, target: str) -> List[Dict]:
        rows = []
        target_mu  = df[target].mean()
        target_sig = df[target].std() + 1e-10
        for _, r in df.iterrows():
            val = pd.to_numeric(r.get(target, np.nan), errors="coerce")
            if pd.isna(val):
                continue
            rows.append({
                "target":       target,
                "cekirdek":     r["NUCLEUS"] if "NUCLEUS" in df.columns else f"Z{r['Z']}N{r['N']}",
                "A":            int(r["A"]),
                "Z":            int(r["Z"]),
                "N":            int(r["N"]),
                target:         round(float(val), 4),
                "z_skor":       round(float((val - target_mu) / target_sig), 3),
                "bant":         str(r.get("band_label", "?")),
                "kutle_bolgesi":str(r.get("mass_region", "?")),
                "SPIN":         r.get("SPIN", "?"),
                "Z_valence":    round(float(r["Z_valence"]), 2) if "Z_valence" in df.columns and pd.notna(r.get("Z_valence")) else "?",
                "N_valence":    round(float(r["N_valence"]), 2) if "N_valence" in df.columns and pd.notna(r.get("N_valence")) else "?",
                "Z_magic_dist": int(r["Z_magic_dist_c"]) if "Z_magic_dist_c" in df.columns else "?",
                "N_magic_dist": int(r["N_magic_dist_c"]) if "N_magic_dist_c" in df.columns else "?",
            })
        return rows

    # -----------------------------------------------------------------------
    # 7. Aciklama/yorum uretimi
    # -----------------------------------------------------------------------
    def _generate_explanation(self, df, target, bands, jumps, cross, corr) -> str:
        lines = []
        n_nuclei = len(df)
        n_bands  = len(bands)
        n_jumps  = len(jumps)

        lines.append(f"=== {target} BANT VE ORUNTU ANALIZI OZETI ===")
        lines.append(f"Toplam cekirdek: {n_nuclei} | Bant sayisi: {n_bands} | Sican nokta: {n_jumps}")
        lines.append("")

        # Korelasyon liderleri
        if corr:
            top3 = corr[:3]
            lines.append("EN GUCLU KORELASYONLAR:")
            for r in top3:
                lines.append(f"  {r['ozellik']}: Spearman_r={r['spearman_r']:.3f} ({r['yorum']})")
        lines.append("")

        # Sicrama ozeti
        if jumps:
            jump_df = pd.DataFrame(jumps)
            lines.append("SICRAMA ANALIZI OZETI:")
            for chain_type in jump_df["zincir_turu"].unique():
                sub = jump_df[jump_df["zincir_turu"] == chain_type]
                lines.append(f"  {chain_type}: {len(sub)} sicrama tespit edildi")
                if "degisen_ozellikler" in sub.columns:
                    # En sik degisen ozellikler
                    all_changing = []
                    for _, r in sub.iterrows():
                        if r["degisen_ozellikler"] != "—":
                            all_changing.extend(r["degisen_ozellikler"].split(" | "))
                    feat_counts = pd.Series(all_changing).value_counts().head(3)
                    if not feat_counts.empty:
                        lines.append(f"  En sik degisen: {', '.join(feat_counts.index.tolist())}")
        lines.append("")

        # Capraz kutle ozeti
        if cross:
            lines.append("CAPRAZ KUTLE BOLGE OZETI:")
            multi_region_bands = [r for r in cross
                                   if r.get("Hafif(A<50)_n", 0) > 0 and r.get("Agir(A>=100)_n", 0) > 0]
            lines.append(f"  {len(multi_region_bands)} bantta hem hafif hem agir cekirdek mevcut")
            if multi_region_bands:
                lines.append("  Ortak ozellik ornegi:")
                for r in multi_region_bands[:2]:
                    if r.get("ortak_ozellikler") != "—":
                        lines.append(f"    {r['bant']}: {r['ortak_ozellikler'][:80]}")
        lines.append("")

        lines.append("NOT: Veri setinde sihirli (magic) cekirdek bulunmamaktadir.")
        lines.append("     Tum analizler bu varsayi altinda yapilmistir.")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Yardimci
    # -----------------------------------------------------------------------
    def _available_features(self, df: pd.DataFrame) -> List[str]:
        """Mevcut ve sayisal olan PHYSICS_FEATURES listesini dondur."""
        available = []
        for f in self.PHYSICS_FEATURES:
            if f in df.columns:
                vals = pd.to_numeric(df[f], errors="coerce")
                if vals.notna().sum() >= 5:
                    available.append(f)
        # Ek hesaplanmis sutunlar
        for extra in ["Z_magic_dist_c", "N_magic_dist_c", "A_mod_2", "Z_mod_2",
                      "N_mod_2", "Z_N_ratio", "valence_total"]:
            if extra in df.columns and extra not in available:
                vals = pd.to_numeric(df[extra], errors="coerce")
                if vals.notna().sum() >= 5:
                    available.append(extra)
        return available

    # -----------------------------------------------------------------------
    # Grafik uretimi
    # -----------------------------------------------------------------------
    def _make_plots(self, targets: List[str]):
        try:
            fig_dir = self.output_dir / "band_plots"
            fig_dir.mkdir(exist_ok=True)

            for target in targets:
                if target not in self._results:
                    continue
                df_t = self.df.dropna(subset=[target]).copy()
                df_t[target] = df_t[target].astype(float)
                bands = self._results[target]["bands"]
                df_t  = self._assign_bands(df_t, target, bands)

                # 1. Bant dagılım grafigi (violin)
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                ax = axes[0]
                bp_data = []
                bp_labels = []
                for idx, (lo, hi, lbl) in enumerate(bands):
                    sub = df_t[df_t["band_idx"] == idx][target].dropna().values
                    if len(sub) >= 2:
                        bp_data.append(sub)
                        bp_labels.append(f"B{idx+1}")
                if bp_data:
                    ax.violinplot(bp_data, showmedians=True)
                    ax.set_xticks(range(1, len(bp_labels) + 1))
                    ax.set_xticklabels(bp_labels, fontsize=8)
                    ax.set_title(f"{target} Bant Dagilimi", fontsize=10)
                    ax.set_ylabel(target)
                    ax.grid(True, alpha=0.3)

                # 2. Z-N dagilimi (renk = bant)
                ax2 = axes[1]
                cmap = plt.cm.get_cmap("tab10", len(bands))
                for idx, (lo, hi, lbl) in enumerate(bands):
                    sub = df_t[df_t["band_idx"] == idx]
                    ax2.scatter(sub["N"], sub["Z"], c=[cmap(idx)]*len(sub),
                                label=f"B{idx+1}", alpha=0.7, s=40, edgecolors="k", linewidths=0.3)
                ax2.set_xlabel("N (Notron sayisi)")
                ax2.set_ylabel("Z (Proton sayisi)")
                ax2.set_title(f"{target} — Z-N Duzlemi Bant Haritasi", fontsize=10)
                ax2.legend(fontsize=7, ncol=2)
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                fig_path = fig_dir / f"{target}_band_analysis.png"
                plt.savefig(str(fig_path), dpi=120, bbox_inches="tight")
                plt.close()
                logger.info(f"  [Grafik] {fig_path.name}")

                # 3. Komsu sicrama grafigi
                jumps = self._results[target].get("jumps", [])
                if jumps:
                    jdf = pd.DataFrame(jumps)
                    fig2, ax3 = plt.subplots(figsize=(10, 5))
                    for chain_type, sub in jdf.groupby("zincir_turu"):
                        ax3.scatter(sub["A_A"], sub[f"abs_delta"],
                                    label=chain_type, alpha=0.7, s=50)
                    ax3.axhline(jdf["esik"].mean(), color="red", linestyle="--",
                                label=f"Ort. Esik ({jdf['esik'].mean():.3f})")
                    ax3.set_xlabel("A (Kutle Sayisi)")
                    ax3.set_ylabel(f"|Delta {target}|")
                    ax3.set_title(f"{target} — Komsu Sicrama Buyuklugu vs A", fontsize=10)
                    ax3.legend(fontsize=8)
                    ax3.grid(True, alpha=0.3)
                    plt.tight_layout()
                    j_path = fig_dir / f"{target}_jumps.png"
                    plt.savefig(str(j_path), dpi=120, bbox_inches="tight")
                    plt.close()

        except Exception as e:
            logger.warning(f"[BandAnalyzer] Grafik hatasi: {e}")

    # -----------------------------------------------------------------------
    # Excel kayit
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # 7. Tahmin Dogrulugu Analizi — PFAZ4 ciktisiyla kesisim
    # -----------------------------------------------------------------------
    def _prediction_accuracy_analysis(
        self,
        all_jump_records: List[Dict],
        all_nucleus_detail: List[Dict],
    ) -> List[Dict]:
        """
        PFAZ4'ten AAA2_Original_vs_Predictions.xlsx'i yukle.
        Her hedef icin: sicrama nukleusu vs normal nukleus tahmin hatasi karsilastir.
        Bant bazinda da ortalama abs_error hesapla.

        Donurus: her nukleusa/banta ait accuracy kayitlari (sheet: Tahmin_Dogrulugu)
        """
        rows = []

        # PFAZ4 ciktisi icin olasilikli yollar
        search_roots = [
            self.output_dir.parent.parent,              # outputs/
            self.output_dir.parent.parent.parent,       # proje koku
        ]
        excel_candidates = []
        for root in search_roots:
            excel_candidates += list(root.rglob("AAA2_Original_vs_Predictions.xlsx"))

        if not excel_candidates:
            logger.info("[BandAnalyzer] PFAZ4 AAA2 karsilastirma Excel bulunamadi — tahmin dogrulugu atlandı")
            return rows

        pfaz4_excel = excel_candidates[0]
        logger.info(f"[BandAnalyzer] PFAZ4 tahmin dosyasi bulundu: {pfaz4_excel}")

        # Jump nukleuslarini set olarak topla (target bazli)
        jump_set: Dict[str, set] = {}
        for jr in all_jump_records:
            tgt = jr.get("target", "")
            n1  = str(jr.get("nucleus_1", jr.get("nucleus_a", "")))
            n2  = str(jr.get("nucleus_2", jr.get("nucleus_b", "")))
            jump_set.setdefault(tgt, set()).update([n1, n2])

        # Bant atamasini al (cekirdek_detay'dan)
        band_lookup: Dict[str, Dict] = {}
        for d in all_nucleus_detail:
            key = (str(d.get("NUCLEUS", "")), str(d.get("target", "")))
            band_lookup[key] = {
                "band_label": d.get("band_label", "?"),
                "band_idx":   d.get("band_idx",   -1),
            }

        try:
            xl4 = pd.ExcelFile(pfaz4_excel)
        except Exception as e:
            logger.warning(f"[BandAnalyzer] PFAZ4 Excel acilamadi: {e}")
            return rows

        target_sheet_map = {"MM": "MM", "QM": "QM", "Beta_2": "Beta_2"}

        for target, sheet_name in target_sheet_map.items():
            if sheet_name not in xl4.sheet_names:
                continue
            try:
                df_pred = pd.read_excel(xl4, sheet_name=sheet_name)
            except Exception as e:
                logger.warning(f"[BandAnalyzer] {sheet_name} sheet okunamadi: {e}")
                continue

            if df_pred.empty:
                continue

            # Sutu tespiti
            nuc_col  = next((c for c in df_pred.columns if "NUCLEUS" in c.upper()), None)
            orig_col = next((c for c in df_pred.columns if "ORIGINAL" in c.upper() or "TRUE" in c.upper()), None)
            err_cols = [c for c in df_pred.columns if "abs_error" in c.lower() or "abserror" in c.lower()]
            pred_cols = [c for c in df_pred.columns if "y_pred" in c.lower() or "_pred_" in c.lower() or "_pred" in c.lower()]

            if nuc_col is None:
                logger.warning(f"[BandAnalyzer] {sheet_name}: NUCLEUS sutunu yok")
                continue

            j_set = jump_set.get(target, set())

            for _, row in df_pred.iterrows():
                nuc = str(row.get(nuc_col, ""))
                is_jump = nuc in j_set
                orig_val = float(row[orig_col]) if orig_col and pd.notna(row.get(orig_col)) else None
                band_info = band_lookup.get((nuc, target), {"band_label": "?", "band_idx": -1})

                # Ortalama abs hata (tum model tiplerine ait)
                abs_errs = [float(row[c]) for c in err_cols if pd.notna(row.get(c))]
                mean_abs_err = float(np.mean(abs_errs)) if abs_errs else None

                # En iyi model tahmini (en dusuk abs_err)
                best_model = None
                best_pred  = None
                best_err   = None
                if err_cols:
                    valid_err = {c: float(row[c]) for c in err_cols if pd.notna(row.get(c))}
                    if valid_err:
                        best_model = min(valid_err, key=valid_err.get).replace("abs_error_mean", "").replace("_abs_error", "").strip("_")
                        best_err   = min(valid_err.values())
                if pred_cols:
                    valid_pred = {c: float(row[c]) for c in pred_cols if pd.notna(row.get(c))}
                    if valid_pred and best_err is not None:
                        best_pred_col = min(
                            {c: abs(v - (orig_val or 0)) for c, v in valid_pred.items()},
                            key=lambda x: abs(valid_pred[x] - (orig_val or 0))
                        )
                        best_pred = valid_pred.get(best_pred_col)

                rec = {
                    "Target":       target,
                    "NUCLEUS":      nuc,
                    "Band":         band_info["band_label"],
                    "Band_Idx":     band_info["band_idx"],
                    "Is_Jump_Nucleus": int(is_jump),
                    "Sinif":        "Sicrama Nukleusu" if is_jump else "Normal Nukleus",
                    "Gercek_Deger": orig_val,
                    "En_Iyi_Tahmin": best_pred,
                    "En_Iyi_Model": best_model,
                    "En_Iyi_Abs_Hata": best_err,
                    "Ort_Abs_Hata": mean_abs_err,
                }
                rows.append(rec)

        # Ozet: sicrama vs normal karsilastirma
        summary_rows = []
        if rows:
            df_acc = pd.DataFrame(rows)
            for tgt in df_acc["Target"].unique():
                sub = df_acc[df_acc["Target"] == tgt]
                for sinif in ["Sicrama Nukleusu", "Normal Nukleus"]:
                    s2 = sub[sub["Sinif"] == sinif]["Ort_Abs_Hata"].dropna()
                    summary_rows.append({
                        "Target":  tgt,
                        "Sinif":   sinif,
                        "N":       len(s2),
                        "Ort_Abs_Hata": round(float(s2.mean()), 4) if len(s2) else None,
                        "Maks_Abs_Hata": round(float(s2.max()),  4) if len(s2) else None,
                        "Min_Abs_Hata":  round(float(s2.min()),  4) if len(s2) else None,
                    })

            # Bant bazli ozet
            for tgt in df_acc["Target"].unique():
                sub = df_acc[df_acc["Target"] == tgt]
                for band in sorted(sub["Band"].unique()):
                    s2 = sub[sub["Band"] == band]["Ort_Abs_Hata"].dropna()
                    n_jump = int((sub[sub["Band"] == band]["Is_Jump_Nucleus"]).sum())
                    summary_rows.append({
                        "Target":  tgt,
                        "Sinif":   f"Bant:{band}",
                        "N":       len(s2),
                        "Ort_Abs_Hata": round(float(s2.mean()), 4) if len(s2) else None,
                        "Maks_Abs_Hata": round(float(s2.max()),  4) if len(s2) else None,
                        "Min_Abs_Hata":  round(float(s2.min()),  4) if len(s2) else None,
                    })

        self._accuracy_rows    = rows
        self._accuracy_summary = summary_rows
        logger.info(f"[BandAnalyzer] Tahmin dogrulugu: {len(rows)} kayit, {len(summary_rows)} ozet satir")
        return rows

    def _build_pivot_summary(self, acc_rows: List[Dict]) -> List[Dict]:
        """
        Bant x Sinif pivot tablosu.
        Hedef | Bant | Sinif | N | Ort_Abs_Hata | Maks | Min
        """
        if not acc_rows:
            return []
        df = pd.DataFrame(acc_rows)
        err_col  = next((c for c in df.columns if 'ort_abs' in c.lower()), None)
        band_col = next((c for c in df.columns if 'band' in c.lower() and 'idx' not in c.lower()), None)
        tgt_col  = next((c for c in df.columns if 'target' in c.lower()), None)
        sinif_col = next((c for c in df.columns if 'sinif' in c.lower()), None)
        if not all([err_col, band_col, tgt_col, sinif_col]):
            return []
        rows_out = []
        for tgt in df[tgt_col].dropna().unique():
            sub_t = df[df[tgt_col] == tgt]
            # Bant x Sinif pivot
            for band in sorted(sub_t[band_col].dropna().unique()):
                sub_b = sub_t[sub_t[band_col] == band]
                for sinif in sorted(sub_b[sinif_col].dropna().unique()):
                    sub_s = sub_b[sub_b[sinif_col] == sinif][err_col].dropna()
                    rows_out.append({
                        "Target":        tgt,
                        "Bant":          band,
                        "Sinif":         sinif,
                        "N_Cekirdek":    len(sub_s),
                        "Ort_Abs_Hata":  round(float(sub_s.mean()), 4) if len(sub_s) else None,
                        "Maks_Abs_Hata": round(float(sub_s.max()),  4) if len(sub_s) else None,
                        "Min_Abs_Hata":  round(float(sub_s.min()),  4) if len(sub_s) else None,
                        "Std_Abs_Hata":  round(float(sub_s.std()),  4) if len(sub_s) > 1 else 0.0,
                    })
            # Toplam (bant bazinda)
            for band in sorted(sub_t[band_col].dropna().unique()):
                sub_b = sub_t[sub_t[band_col] == band][err_col].dropna()
                rows_out.append({
                    "Target": tgt,
                    "Bant":   band,
                    "Sinif":  "TOPLAM",
                    "N_Cekirdek":    len(sub_b),
                    "Ort_Abs_Hata":  round(float(sub_b.mean()), 4) if len(sub_b) else None,
                    "Maks_Abs_Hata": round(float(sub_b.max()),  4) if len(sub_b) else None,
                    "Min_Abs_Hata":  round(float(sub_b.min()),  4) if len(sub_b) else None,
                    "Std_Abs_Hata":  round(float(sub_b.std()),  4) if len(sub_b) > 1 else 0.0,
                })
        logger.info(f"[BandAnalyzer] Pivot ozeti: {len(rows_out)} satir")
        return rows_out

    def _external_excel_correlation(self, all_nucleus_detail: List[Dict]) -> List[Dict]:
        """
        Harici Excel/CSV referans dosyasindan (aaa2.txt veya baska nuklear hesaplama Excel)
        bant atamasina ek korelasyon analizi.
        Bulunan harici sutunlar ile bant_idx arasindaki Spearman r hesaplanir.
        """
        rows = []

        # Kaynak: self.df (zaten yuklu olan aaa2.txt) kullanilir
        if self.df is None:
            return rows

        # Bant atamasini birlestirecek kolon: NUCLEUS + target
        detail_df = pd.DataFrame(all_nucleus_detail) if all_nucleus_detail else pd.DataFrame()
        if detail_df.empty:
            return rows

        nuc_col = next((c for c in self.df.columns if 'NUCLEUS' in c.upper()), None)
        if nuc_col is None:
            return rows

        # NUMERIC sutunlar — aaa2.txt'nin tum sayisal kolonlari
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        tgt_cols_to_skip = list(self.TARGET_COLS.keys())  # MM, QM, Beta_2 zaten analiz edildi

        # Her target icin bant_idx ile aaa2 ozellikleri arasinda Spearman
        for tgt in detail_df["target"].dropna().unique() if "target" in detail_df.columns else []:
            sub_det = detail_df[detail_df["target"] == tgt][["NUCLEUS", "band_idx"]].dropna()
            if sub_det.empty or nuc_col not in self.df.columns:
                continue
            merged = self.df.merge(sub_det, left_on=nuc_col, right_on="NUCLEUS", how="inner")
            if merged.empty:
                continue
            for col in num_cols:
                if col in tgt_cols_to_skip:
                    continue
                valid = merged[[col, "band_idx"]].dropna()
                if len(valid) < 5:
                    continue
                try:
                    spear_r, spear_p = stats.spearmanr(valid[col], valid["band_idx"])
                    pear_r,  pear_p  = stats.pearsonr(valid[col],  valid["band_idx"])
                    rows.append({
                        "Target":     tgt,
                        "Ozellik":    col,
                        "N":          len(valid),
                        "Spearman_r": round(float(spear_r), 4),
                        "Spearman_p": round(float(spear_p), 4),
                        "Pearson_r":  round(float(pear_r),  4),
                        "Pearson_p":  round(float(pear_p),  4),
                        "Abs_Spearman": round(abs(float(spear_r)), 4),
                    })
                except Exception:
                    continue

        # Target bazinda |Spearman_r| sirali
        if rows:
            rows.sort(key=lambda x: (-x.get("Abs_Spearman", 0), x.get("Target", "")))
        logger.info(f"[BandAnalyzer] Dis Excel korelasyon: {len(rows)} satir")
        return rows

    def _plot_jump_accuracy(self, rows: List[Dict]):
        """
        Sicrama nukleusu vs normal nukleus ortalama abs_error karsilastirma grafigi.
        """
        if not rows or not _PLOT:
            return

        df = pd.DataFrame(rows)
        if "Ort_Abs_Hata" not in df.columns or df["Ort_Abs_Hata"].isna().all():
            return

        targets = df["Target"].unique()
        fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), squeeze=False)

        for ax, tgt in zip(axes[0], targets):
            sub = df[df["Target"] == tgt]
            grp = sub.groupby("Sinif")["Ort_Abs_Hata"].agg(["mean", "std"]).reset_index()
            grp.columns = ["Sinif", "mean", "std"]

            colors = ["#EF5350" if "Sicrama" in s else "#42A5F5" for s in grp["Sinif"]]
            x = np.arange(len(grp))
            ax.bar(x, grp["mean"], color=colors, alpha=0.85,
                   yerr=grp["std"].fillna(0), capsize=5, edgecolor="k", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(grp["Sinif"], rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Ort. Abs. Hata")
            ax.set_title(f"{tgt} — Sicrama vs Normal Tahmin Hatasi", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("AI/ANFIS: Ani Sicrama Nukleuslarinda Tahmin Dogrulugu", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = self.output_dir / "jump_prediction_accuracy.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[BandAnalyzer] Grafik: {out_path.name}")

    def _save_excel(self, band_sum, jumps, cross, corr, detail, explanations,
                    acc_rows=None, acc_summary=None,
                    pivot_rows=None, ext_corr_rows=None) -> Optional[Path]:
        xlsx_path = self.output_dir / "nuclear_band_analysis.xlsx"

        def _safe_df(lst) -> pd.DataFrame:
            return pd.DataFrame(lst) if lst else pd.DataFrame({"bilgi": ["Veri yok"]})

        df_band    = _safe_df(band_sum)
        df_jump    = _safe_df(jumps)
        df_cross   = _safe_df(cross)
        df_corr    = _safe_df(corr)
        df_det     = _safe_df(detail)
        df_expl    = pd.DataFrame(explanations) if explanations else pd.DataFrame()
        df_acc     = _safe_df(acc_rows)
        df_acc_s   = _safe_df(acc_summary)
        df_pivot   = _safe_df(pivot_rows)
        df_ext_cor = _safe_df(ext_corr_rows)

        sheets = [
            ("Bant_Ozeti",        df_band),
            ("Sicrama_Analizi",   df_jump),
            ("Capraz_Kutle",      df_cross),
            ("Korelasyon",        df_corr),
            ("Dis_Excel_Korel",   df_ext_cor),   # aaa2 ozellikleri x bant_idx
            ("Pivot_Bant_Sinif",  df_pivot),      # bant x sinif ozet
            ("Cekirdek_Detay",    df_det),
            ("Tahmin_Dogrulugu",  df_acc),        # tum 267 cekirdek, sicrama/normal
            ("Tahmin_Ozeti",      df_acc_s),      # bant/sinif bazli hata ozeti
        ]
        if not df_expl.empty:
            sheets.append(("Aciklama", df_expl))

        try:
            if _ES:
                with ExcelStandardizer(xlsx_path) as es:
                    for sheet_name, df in sheets:
                        es.write_sheet(sheet_name, df, freeze_header=True)
            else:
                with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as w:
                    for sheet_name, df in sheets:
                        df.to_excel(w, sheet_name=sheet_name[:31], index=False)

            logger.info(f"[BandAnalyzer] Excel kaydedildi ({len(sheets)} sayfa): {xlsx_path}")
            return xlsx_path
        except Exception as e:
            logger.error(f"[BandAnalyzer] Excel yazma hatasi: {e}")
            return None
