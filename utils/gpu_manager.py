"""
GPU Manager - Merkezi GPU algilama ve yapilandirma
==================================================
Tum PFAZ fazlari tarafindan kullanilan ortak GPU yonetim modulu.

Desteklenen backend'ler:
  - TensorFlow (DNN, PFAZ 2)
  - PyTorch (ANFIS L-BFGS-B, PFAZ 3; AutoML, PFAZ 13)
  - XGBoost CUDA (PFAZ 2, 13)

Kullanim:
    from utils.gpu_manager import GPUManager
    gm = GPUManager()
    gm.configure_tf()          # TF bellek buyumesini etkinlestir
    device = gm.torch_device   # 'cuda' veya 'cpu'
    n_workers = gm.optimal_workers()
"""

import logging
import multiprocessing
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend kontrolleri (import basarisizsa False)
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    _TF_OK = True
except Exception:
    tf = None
    _TF_OK = False

try:
    import torch
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False


class GPUManager:
    """
    Tek noktadan GPU algilama, yapilandirma ve worker sayisi hesaplama.

    Ozellikler:
        available   : bool - herhangi bir GPU bulundu mu
        torch_device: str  - 'cuda' veya 'cpu' (PyTorch icin)
        vram_mb     : int  - GPU bellek (MB), bilinmiyorsa 0
    """

    # GTX 1650 icin guvenli TF bellek siniri (4096 MB dedicated VRAM)
    _TF_MEMORY_LIMIT_MB: int = 3800   # 4 GB - 200 MB guvenlik payi

    def __init__(self):
        self.available: bool = False
        self.torch_device: str = 'cpu'
        self.vram_mb: int = 0
        self._tf_configured: bool = False
        self._detect()

    # ------------------------------------------------------------------
    def _detect(self) -> None:
        """GPU varligini TF ve PyTorch uzerinden sirasyla dene."""
        # 1) TensorFlow
        if _TF_OK:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    self.available = True
                    logger.info(f"[GPU] TF: {len(gpus)} GPU bulundu: {[g.name for g in gpus]}")
            except Exception as e:
                logger.debug(f"[GPU] TF algilama hatasi: {e}")

        # 2) PyTorch
        if _TORCH_OK:
            try:
                if torch.cuda.is_available():
                    self.available = True
                    dev_name = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    self.vram_mb = props.total_memory // (1024 * 1024)
                    self.torch_device = 'cuda'
                    logger.info(f"[GPU] PyTorch CUDA: {dev_name}, {self.vram_mb} MB")
            except Exception as e:
                logger.debug(f"[GPU] PyTorch algilama hatasi: {e}")

        if not self.available:
            logger.info("[GPU] GPU bulunamadi -- CPU modu")

    # ------------------------------------------------------------------
    def configure_tf(self) -> bool:
        """
        TensorFlow GPU bellek buyumesini etkinlestir.
        Sadece bir kez calistirilir (idempotent).

        Returns:
            True: yapilandirma basarili
        """
        if self._tf_configured or not _TF_OK or not self.available:
            return self._tf_configured

        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # GTX 1650: dedicated 4 GB, paylasilan bellek icin limit koy
            if gpus and self.vram_mb > 0:
                limit = min(self._TF_MEMORY_LIMIT_MB, self.vram_mb - 200)
                try:
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=limit)]
                    )
                    logger.info(f"[GPU] TF bellek siniri: {limit} MB")
                except Exception:
                    pass  # memory_growth yeterli
            self._tf_configured = True
            logger.info("[GPU] TF yapilandirmasi tamamlandi (memory_growth=True)")
        except Exception as e:
            logger.warning(f"[GPU] TF yapilandirma hatasi: {e}")

        return self._tf_configured

    # ------------------------------------------------------------------
    def optimal_workers(self, mode: str = 'ai') -> int:
        """
        Donaniima gore optimal ThreadPoolExecutor worker sayisi.

        Args:
            mode: 'ai'    - PFAZ 2 AI egitimi (her RF ic olarak cok cekirdek kullanir)
                  'anfis' - PFAZ 3 ANFIS (modeller kucuk, cok worker iyi)
                  'mc'    - PFAZ 9 Monte Carlo (inference agirlikli)
                  'auto'  - genel amac

        Returns:
            int: onerilen worker sayisi
        """
        n_cpu = multiprocessing.cpu_count()  # i7-13700 => 24 mantiksal islemci

        if mode == 'ai':
            # RF/GBM her model icin tum cekirdekleri kullanir -> az dis worker
            # GPU varsa DNN/XGB GPU'ya gider, RF/GBM CPU'da kucuk gruplar halinde calisir
            # 24 thread: 8 dis worker, her RF ~3 cekirdek paylasiyor
            return max(4, min(8, n_cpu // 3))

        elif mode == 'anfis':
            # Modeller kucuk (267 cekirdek), cok worker = paralel dataset egitimi
            # 24 thread: 22 worker
            return max(4, n_cpu - 2)

        elif mode == 'mc':
            # Inference agirlikli: modeli yukle + predict
            # GPU DNN inference paralel calisir, IO'yu threadler yonetir
            return max(4, min(16, n_cpu - 2))

        else:  # auto
            return max(4, n_cpu - 2)

    # ------------------------------------------------------------------
    def get_xgb_params(self) -> dict:
        """
        XGBoost GPU parametrelerini dondur. Surume gore uyarlanir.
        Returns: bos dict eger GPU yoksa.
        """
        if not self.available:
            return {}
        try:
            import xgboost as xgb
            ver = tuple(int(x) for x in xgb.__version__.split('.')[:2])
            # Hizli dogrulama
            test = xgb.XGBRegressor(n_estimators=2, verbosity=0,
                                     **({'device': 'cuda'} if ver >= (2, 0)
                                        else {'tree_method': 'gpu_hist'}))
            import numpy as _np
            test.fit(_np.array([[1.0], [2.0]]), _np.array([1.0, 2.0]))
            if ver >= (2, 0):
                logger.info("[GPU] XGBoost >= 2.0: device='cuda'")
                return {'device': 'cuda'}
            else:
                logger.info("[GPU] XGBoost < 2.0: tree_method='gpu_hist'")
                return {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    def get_torch_lbfgs_device(self) -> Optional[object]:
        """
        PyTorch cihazini dondur (ANFIS L-BFGS-B GPU icin).
        Returns: torch.device veya None
        """
        if not _TORCH_OK or not self.available:
            return None
        try:
            return torch.device('cuda')
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_gm_instance: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Global GPUManager singletonunu dondur."""
    global _gm_instance
    if _gm_instance is None:
        _gm_instance = GPUManager()
    return _gm_instance
