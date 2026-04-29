"""
Pipeline Warning Tracker
========================
Tüm pipeline çalışması boyunca WARNING ve ERROR düzeyindeki log
mesajlarını yapılandırılmış biçimde yakalar.

Kullanım (main.py içinde):
    from utils.warning_tracker import WarningTracker
    tracker = WarningTracker('outputs/pipeline_warnings.json')
    tracker.attach()          # root logger'a ekle
    ...
    tracker.save_report()     # Excel özeti üret

Ayrıca her modülden doğrudan kayıt da yapılabilir:
    from utils.warning_tracker import get_tracker
    get_tracker().warn('PFAZ3', 'ANFISRobustnessTester', exc)
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    pd = None
    _PANDAS_OK = False

# ---------------------------------------------------------------------------
# Singleton erişimi
# ---------------------------------------------------------------------------
_global_tracker: Optional["WarningTracker"] = None


def get_tracker() -> "WarningTracker":
    """Global tracker'ı döndür; yoksa varsayılan yolda oluştur."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = WarningTracker()
    return _global_tracker


# ---------------------------------------------------------------------------
# Özel log handler
# ---------------------------------------------------------------------------
class _WarningHandler(logging.Handler):
    """WARNING ve üstü mesajları yakalar, WarningTracker'a iletir."""

    def __init__(self, tracker: "WarningTracker"):
        super().__init__(level=logging.WARNING)
        self._tracker = tracker

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "timestamp":   datetime.fromtimestamp(record.created).isoformat(),
                "level":       record.levelname,
                "logger":      record.name,
                "module":      record.module,
                "function":    record.funcName,
                "line":        record.lineno,
                "message":     record.getMessage(),
                "pfaz":        self._infer_pfaz(record),
                "traceback":   None,
            }
            if record.exc_info and record.exc_info[0]:
                entry["traceback"] = "".join(
                    traceback.format_exception(*record.exc_info)
                )
            self._tracker._add(entry)
        except Exception:
            pass  # handler içinde exception fırlatma

    @staticmethod
    def _infer_pfaz(record: logging.LogRecord) -> str:
        """Logger veya modül adından PFAZ numarasını tahmin et."""
        for src in (record.name, record.module, record.pathname):
            if not src:
                continue
            s = src.lower()
            for i in range(13, 0, -1):
                tag = f"pfaz{i:02d}"
                if tag in s or f"pfaz_{i}" in s or f"pfaz {i}" in s:
                    return f"PFAZ{i:02d}"
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------
class WarningTracker:
    """
    Pipeline boyunca WARNING / ERROR mesajlarını toplar ve raporlar.

    Attributes:
        json_path: Çalışma sırasında yazılan ham JSON log dosyası
        excel_path: Çalışma sonunda üretilen Excel rapor dosyası
    """

    def __init__(
        self,
        json_path: str = "outputs/pipeline_warnings.json",
        excel_path: str = "outputs/pipeline_warnings_report.xlsx",
    ):
        self.json_path  = Path(json_path)
        self.excel_path = Path(excel_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list = []
        self._handler: Optional[_WarningHandler] = None
        # Önceki oturumdan kalan kayıtları yükle (resume desteği)
        self._load_existing()

    # -----------------------------------------------------------------------
    # Logging entegrasyonu
    # -----------------------------------------------------------------------
    def attach(self, logger: Optional[logging.Logger] = None) -> None:
        """Root (veya verilen) logger'a handler ekle."""
        global _global_tracker
        _global_tracker = self
        target = logger or logging.getLogger()
        self._handler = _WarningHandler(self)
        target.addHandler(self._handler)

    def detach(self) -> None:
        """Handler'ı kaldır."""
        if self._handler:
            logging.getLogger().removeHandler(self._handler)
            self._handler = None

    # -----------------------------------------------------------------------
    # Manuel kayıt (try/except bloklarından çağrılabilir)
    # -----------------------------------------------------------------------
    def warn(
        self,
        pfaz: str,
        component: str,
        message: str,
        exc: Optional[Exception] = None,
    ) -> None:
        """
        try/except bloğundan doğrudan çağır:
            tracker.warn('PFAZ3', 'ANFISRobustnessTester', str(e), e)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level":     "WARNING",
            "logger":    component,
            "module":    component,
            "function":  "—",
            "line":      0,
            "message":   message,
            "pfaz":      pfaz,
            "traceback": traceback.format_exc() if exc else None,
        }
        self._add(entry)

    # -----------------------------------------------------------------------
    # İç işlemler
    # -----------------------------------------------------------------------
    def _add(self, entry: dict) -> None:
        self._entries.append(entry)
        # Anlık JSON yazımı (resume için kritik)
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_existing(self) -> None:
        """Önceki çalışmadan kalan JSON'ı yükle."""
        if self.json_path.exists():
            try:
                with open(self.json_path, encoding="utf-8") as f:
                    self._entries = json.load(f)
            except Exception:
                self._entries = []

    # -----------------------------------------------------------------------
    # Rapor üretimi
    # -----------------------------------------------------------------------
    def save_report(self) -> Optional[Path]:
        """
        Toplanan uyarıları Excel'e yaz.
        Sayfalar:
          1. Tüm_Uyarılar  — her satır bir uyarı
          2. PFAZ_Özeti    — PFAZ'a göre hata sayısı
          3. Seviye_Özeti  — WARNING vs ERROR sayısı
        Döndürür:
            Path: Excel dosyası yolu (kayıt yoksa None)
        """
        if not self._entries:
            return None

        if not _PANDAS_OK:
            logging.getLogger(__name__).warning(
                "[WarningTracker] pandas yuklu degil — Excel raporu atlandi"
            )
            return None

        self.excel_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._entries)

        # Traceback'i kısalt (Excel'de okunabilir olsun)
        if "traceback" in df.columns:
            df["traceback"] = df["traceback"].fillna("").apply(
                lambda t: t[-800:] if len(str(t)) > 800 else t
            )

        # --- PFAZ özeti ---
        pfaz_summary = (
            df.groupby(["pfaz", "level"])
            .size()
            .reset_index(name="count")
            .sort_values(["pfaz", "level"])
        )

        # --- Seviye özeti ---
        level_summary = (
            df.groupby("level")
            .agg(
                Sayı=("message", "count"),
                İlk_Zaman=("timestamp", "min"),
                Son_Zaman=("timestamp", "max"),
            )
            .reset_index()
        )

        try:
            from pfaz_modules.pfaz06_final_reporting.excel_standardizer import ExcelStandardizer
            with ExcelStandardizer(self.excel_path) as es:
                es.write_sheet("Tüm_Uyarılar",  df,            freeze_header=True)
                es.write_sheet("PFAZ_Özeti",     pfaz_summary,  freeze_header=True)
                es.write_sheet("Seviye_Özeti",   level_summary, freeze_header=True)
        except ImportError:
            # ExcelStandardizer kurulu değilse düz pandas
            with pd.ExcelWriter(str(self.excel_path), engine="openpyxl") as w:
                df.to_excel(w,            sheet_name="Tüm_Uyarılar", index=False)
                pfaz_summary.to_excel(w,  sheet_name="PFAZ_Özeti",   index=False)
                level_summary.to_excel(w, sheet_name="Seviye_Özeti", index=False)

        logging.getLogger(__name__).info(
            f"[WarningTracker] Rapor kaydedildi: {self.excel_path.name} "
            f"({len(df)} uyarı)"
        )
        return self.excel_path

    # -----------------------------------------------------------------------
    # Özet yazdırma (pipeline sonu için)
    # -----------------------------------------------------------------------
    def print_summary(self) -> None:
        if not self._entries:
            print("[WarningTracker] Hic uyari/hata kaydedilmedi.")
            return
        if not _PANDAS_OK:
            warnings = sum(1 for e in self._entries if e.get("level") == "WARNING")
            errors   = sum(1 for e in self._entries if e.get("level") == "ERROR")
            print(f"\n[WarningTracker] OZET: {warnings} WARNING, {errors} ERROR")
            return
        df = pd.DataFrame(self._entries)
        errors   = (df["level"] == "ERROR").sum()
        warnings = (df["level"] == "WARNING").sum()
        print("\n" + "="*70)
        print(f"[WarningTracker] OZET: {warnings} WARNING, {errors} ERROR")
        print("="*70)
        by_pfaz = df.groupby("pfaz")["level"].value_counts()
        print(by_pfaz.to_string())
        print(f"\nDetay için: {self.excel_path}")
        print("="*70)

    # -----------------------------------------------------------------------
    # Özellikler
    # -----------------------------------------------------------------------
    @property
    def n_warnings(self) -> int:
        return sum(1 for e in self._entries if e.get("level") == "WARNING")

    @property
    def n_errors(self) -> int:
        return sum(1 for e in self._entries if e.get("level") == "ERROR")

    @property
    def has_errors(self) -> bool:
        return self.n_errors > 0
