"""
Feature Combination Manager - SHAP-Based Physics Feature Sets
=============================================================

SHAP (SHapley Additive exPlanations) analizinden elde edilen feature önem
sıralamasına göre tasarlanmış, fizik-tabanlı feature kombinasyonları sistemi.

Dataset klasör adlarında kullanılan kısaltmalar bu dosyada tanımlıdır.

Abbreviation Map (FEATURE_ABBREV):
    A    = A                  (mass number)
    Z    = Z                  (proton number)
    N    = N                  (neutron number)
    S    = SPIN               (nuclear spin)
    PAR  = PARITY             (parity +1/-1)
    MC   = magic_character    (shell magic score 0-1)
    BEPA = BE_per_A           (binding energy per nucleon)
    B2E  = Beta_2_estimated   (estimated deformation param)
    ZMD  = Z_magic_dist       (Z distance from nearest magic)
    NMD  = N_magic_dist       (N distance from nearest magic)
    BEA  = BE_asymmetry       (asymmetry binding energy)
    ZV   = Z_valence          (valence proton count)
    NV   = N_valence          (valence neutron count)
    ZSG  = Z_shell_gap        (proton shell gap energy MeV)
    NSG  = N_shell_gap        (neutron shell gap energy MeV)
    BEP  = BE_pairing         (pairing energy)
    SPHI = spherical_index    (sphericity index 0-1)
    CP   = Q0_intrinsic       (collective/intrinsic quadrupole)
    PF   = P_FACTOR           (P-factor, parity factor)
    BET  = BE_total           (total binding energy)
    SN   = S_n_approx         (neutron separation energy approx)
    SP   = S_p_approx         (proton separation energy approx)
    NN   = Nn                 (valence neutron count - raw aaa2.txt column)
    NP   = Np                 (valence proton count - raw aaa2.txt column)

Target SHAP Rankings:
    MM:    A(19.2%) > Z(17.5%) > S(12.8%) > MC(9.7%) > BEPA(8.3%) >
           B2E(7.1%) > ZMD(5.4%) > N(4.9%) > BEP(4.2%) > NMD(3.1%) >
           ZSG(2.7%) > SPHI(2.4%) > PAR(2.1%) > PF(1.8%)

    QM:    Z(21.5%) > B2E(18.3%) > A(15.7%) > MC(10.2%) > S(8.9%) >
           BEA(6.4%) > ZV(5.1%) > NV(4.8%) > SPHI(4.3%) > CP(3.7%) >
           ZMD(3.2%) > N(2.9%) > PAR(2.3%) > BEPA(2.1%)

    B2:    MC(22.1%) > ZMD(18.7%) > NMD(17.3%) > A(12.9%) > ZV(8.4%) >
           NV(7.8%) > BEA(5.6%) > Z(3.9%) > N(3.4%) > CP(2.8%) >
           S(2.3%) > BEPA(1.9%)

Author: Nuclear Physics AI Project
Version: 2.0.0 - SHAP-based redesign
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureCombinationManager:
    """
    SHAP-tabanlı feature kombinasyonlarini yöneten sınıf.

    Feature set adı = kısaltma kodu (örn. 'AZS', 'AZMC', 'MCZMNM').
    Klasör isimleri bu kodlari kullanir:
        MM_75_S70_AZS_NoScaling_Random
        MM_75_S70_AZSMC_NoScaling_Random
    """

    # =========================================================================
    # FEATURE ABBREVIATION MAP
    # Kisaltma -> Gercek sutun adi eslestirmesi
    # =========================================================================
    FEATURE_ABBREV = {
        'A':    'A',
        'Z':    'Z',
        'N':    'N',
        'S':    'SPIN',
        'PAR':  'PARITY',
        'MC':   'magic_character',
        'BEPA': 'BE_per_A',
        'B2E':  'Beta_2_estimated',
        'ZMD':  'Z_magic_dist',
        'NMD':  'N_magic_dist',
        'BEA':  'BE_asymmetry',
        'ZV':   'Z_valence',
        'NV':   'N_valence',
        'ZSG':  'Z_shell_gap',
        'NSG':  'N_shell_gap',
        'BEP':  'BE_pairing',
        'SPHI': 'spherical_index',
        'CP':   'Q0_intrinsic',
        'PF':   'P_FACTOR',
        'BET':  'BE_total',
        'SN':   'S_n_approx',
        'SP':   'S_p_approx',
        # Raw data columns (from aaa2.txt directly)
        'NN':   'Nn',   # valence neutron count (aaa2.txt raw column)
        'NP':   'Np',   # valence proton count (aaa2.txt raw column)
    }

    # =========================================================================
    # SHAP-BASED FEATURE SET DEFINITIONS
    # Her set: kısaltma listesi + açıklama + hangi hedefler için uygun
    # =========================================================================
    FEATURE_SETS = {

        # =====================================================================
        # ORTAK SETLER (multiple targets)
        # =====================================================================

        'AZN': {
            'abbrevs': ['A', 'Z', 'N'],
            'category': 'Common_3in',
            'target_affinity': ['MM', 'QM', 'Beta_2', 'MM_QM'],
            'description': 'A, Z, N - temel nukler uclu',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AZS': {
            'abbrevs': ['A', 'Z', 'S'],
            'category': 'Common_3in',
            'target_affinity': ['MM', 'QM', 'MM_QM'],
            'description': 'A, Z, SPIN - cekirdek spini ile temel',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AZMC': {
            'abbrevs': ['A', 'Z', 'MC'],
            'category': 'Common_3in',
            'target_affinity': ['MM', 'QM', 'MM_QM'],
            'description': 'A, Z, magic_character - kabuk etkisi ile',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AZBEPA': {
            'abbrevs': ['A', 'Z', 'BEPA'],
            'category': 'Common_3in',
            'target_affinity': ['MM', 'QM'],
            'description': 'A, Z, BE/A - baglanma enerjisi ile',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AZB2E': {
            'abbrevs': ['A', 'Z', 'B2E'],
            'category': 'Common_3in',
            'target_affinity': ['QM', 'MM_QM'],
            'description': 'A, Z, Beta_2_est - deformasyon ile (QM icin kritik)',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AZNS': {
            'abbrevs': ['A', 'Z', 'N', 'S'],
            'category': 'Common_4in',
            'target_affinity': ['MM', 'QM', 'MM_QM'],
            'description': 'A, Z, N, SPIN - dort temel ozellik',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        # =====================================================================
        # MM SETLERI (Magnetic Moment)
        # SHAP: A > Z > S > MC > BEPA > B2E > ZMD > N > BEP > NMD
        # =====================================================================

        'ASMC': {
            'abbrevs': ['A', 'S', 'MC'],
            'category': 'MM_3in',
            'target_affinity': ['MM'],
            'description': 'A, SPIN, magic_char - MM icin spin+kabuk kombinasyonu',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AMCBEPA': {
            'abbrevs': ['A', 'MC', 'BEPA'],
            'category': 'MM_3in',
            'target_affinity': ['MM'],
            'description': 'A, magic_char, BE/A - kabuk+baglanma enerjisi',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AZSMC': {
            'abbrevs': ['A', 'Z', 'S', 'MC'],
            'category': 'MM_4in',
            'target_affinity': ['MM', 'MM_QM'],
            'description': 'A, Z, SPIN, magic_char - MM icin top-4 SHAP',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'AZSBEPA': {
            'abbrevs': ['A', 'Z', 'S', 'BEPA'],
            'category': 'MM_4in',
            'target_affinity': ['MM'],
            'description': 'A, Z, SPIN, BE/A - cekirdek ozellikleri + baglanma',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'AZMCBEPA': {
            'abbrevs': ['A', 'Z', 'MC', 'BEPA'],
            'category': 'MM_4in',
            'target_affinity': ['MM'],
            'description': 'A, Z, magic_char, BE/A - kabuk+baglanma (spin yok)',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'AZSB2E': {
            'abbrevs': ['A', 'Z', 'S', 'B2E'],
            'category': 'MM_4in',
            'target_affinity': ['MM', 'QM', 'MM_QM'],
            'description': 'A, Z, SPIN, Beta_2_est - deformasyon dahil',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'AZSMCBEPA': {
            'abbrevs': ['A', 'Z', 'S', 'MC', 'BEPA'],
            'category': 'MM_5in',
            'target_affinity': ['MM'],
            'description': 'A, Z, SPIN, MC, BE/A - MM top-5 SHAP kombinasyonu',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        'AZSMCB2E': {
            'abbrevs': ['A', 'Z', 'S', 'MC', 'B2E'],
            'category': 'MM_5in',
            'target_affinity': ['MM', 'MM_QM'],
            'description': 'A, Z, SPIN, MC, Beta_2_est - MM+deformasyon',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        # =====================================================================
        # QM SETLERI (Quadrupole Moment)
        # SHAP: Z > B2E > A > MC > S > BEA > ZV > NV > SPHI > CP > ZMD > N
        # =====================================================================

        'ZB2EMC': {
            'abbrevs': ['Z', 'B2E', 'MC'],
            'category': 'QM_3in',
            'target_affinity': ['QM'],
            'description': 'Z, Beta_2_est, magic_char - QM icin top-3',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'B2EMCBEA': {
            'abbrevs': ['B2E', 'MC', 'BEA'],
            'category': 'QM_3in',
            'target_affinity': ['QM'],
            'description': 'Beta_2_est, MC, BE_asym - deformasyon+kabuk+asimetri',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AZB2EMC': {
            'abbrevs': ['A', 'Z', 'B2E', 'MC'],
            'category': 'QM_4in',
            'target_affinity': ['QM', 'MM_QM'],
            'description': 'A, Z, Beta_2_est, MC - QM icin 4-giris kombinasyonu',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'ZB2EMCS': {
            'abbrevs': ['Z', 'B2E', 'MC', 'S'],
            'category': 'QM_4in',
            'target_affinity': ['QM'],
            'description': 'Z, Beta_2_est, MC, SPIN - QM top-4+spin',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'AZB2EBEA': {
            'abbrevs': ['A', 'Z', 'B2E', 'BEA'],
            'category': 'QM_4in',
            'target_affinity': ['QM'],
            'description': 'A, Z, Beta_2_est, BE_asym - elektrik+deformasyon+asimetri',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'AZB2EMCS': {
            'abbrevs': ['A', 'Z', 'B2E', 'MC', 'S'],
            'category': 'QM_5in',
            'target_affinity': ['QM'],
            'description': 'A, Z, B2E, MC, S - QM top-5 SHAP kombinasyonu',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        'AZB2EMCBEA': {
            'abbrevs': ['A', 'Z', 'B2E', 'MC', 'BEA'],
            'category': 'QM_5in',
            'target_affinity': ['QM', 'MM_QM'],
            'description': 'A, Z, B2E, MC, BE_asym - QM kapsamli 5-giris',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        # =====================================================================
        # BETA_2 SETLERI (Deformation Parameter)
        # SHAP: MC > ZMD > NMD > A > ZV > NV > BEA > Z > N > CP > S
        # =====================================================================

        'MCZMNM': {
            'abbrevs': ['MC', 'ZMD', 'NMD'],
            'category': 'B2_3in',
            'target_affinity': ['Beta_2'],
            'description': 'MC, Z_magic_dist, N_magic_dist - B2 top-3 (magic uzakligi)',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'AZVNV': {
            'abbrevs': ['A', 'ZV', 'NV'],
            'category': 'B2_3in',
            'target_affinity': ['Beta_2'],
            'description': 'A, Z_valence, N_valence - A + valans nukleonlar',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'ZMNMBEA': {
            'abbrevs': ['ZMD', 'NMD', 'BEA'],
            'category': 'B2_3in',
            'target_affinity': ['Beta_2'],
            'description': 'Z_magic_dist, N_magic_dist, BE_asym - magic uzakligi + asimetri',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'MCZMNMZV': {
            'abbrevs': ['MC', 'ZMD', 'NMD', 'ZV'],
            'category': 'B2_4in',
            'target_affinity': ['Beta_2'],
            'description': 'MC, ZMD, NMD, Z_val - B2 top-4 SHAP',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'MCZMNMBEA': {
            'abbrevs': ['MC', 'ZMD', 'NMD', 'BEA'],
            'category': 'B2_4in',
            'target_affinity': ['Beta_2'],
            'description': 'MC, ZMD, NMD, BE_asym - magic+asimetri',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'AMCZMNM': {
            'abbrevs': ['A', 'MC', 'ZMD', 'NMD'],
            'category': 'B2_4in',
            'target_affinity': ['Beta_2'],
            'description': 'A, MC, ZMD, NMD - kutle + magic ozellikleri',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'ZVNVZMNM': {
            'abbrevs': ['ZV', 'NV', 'ZMD', 'NMD'],
            'category': 'B2_4in',
            'target_affinity': ['Beta_2'],
            'description': 'Z_val, N_val, ZMD, NMD - valans + magic uzakligi',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'MCZMNMZVNV': {
            'abbrevs': ['MC', 'ZMD', 'NMD', 'ZV', 'NV'],
            'category': 'B2_5in',
            'target_affinity': ['Beta_2'],
            'description': 'MC, ZMD, NMD, ZV, NV - B2 top-5 SHAP kombinasyonu',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        'AMCZMNMBEA': {
            'abbrevs': ['A', 'MC', 'ZMD', 'NMD', 'BEA'],
            'category': 'B2_5in',
            'target_affinity': ['Beta_2'],
            'description': 'A, MC, ZMD, NMD, BE_asym - B2 kapsamli 5-giris',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        # =====================================================================
        # Nn/Np SETLERI (aaa2.txt ham sutunlari - valans nukleon sayilari)
        # Nn ve Np: aaa2.txt'den dogrudan gelen sutunlar (ham veri)
        # Fiziksel anlam: nukleonlarin kabuk doluluk/bosluklarini gosteriyor olabilir
        # =====================================================================

        'AZNNP': {
            'abbrevs': ['A', 'Z', 'NN', 'NP'],
            'category': 'NnNp_4in',
            'target_affinity': ['MM', 'QM', 'MM_QM'],
            'description': 'A, Z, Nn, Np - temel + ham valans sayilari',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'NNPMC': {
            'abbrevs': ['NN', 'NP', 'MC'],
            'category': 'NnNp_3in',
            'target_affinity': ['MM', 'QM', 'Beta_2', 'MM_QM'],
            'description': 'Nn, Np, magic_character - valans + kabuk yapisi',
            'n_inputs': 3,
            'anfis_feasible': True,
            'anfis_rules_3mf': 27,
        },

        'ZNNPMC': {
            'abbrevs': ['Z', 'NN', 'NP', 'MC'],
            'category': 'NnNp_4in',
            'target_affinity': ['QM', 'Beta_2'],
            'description': 'Z, Nn, Np, MC - proton sayisi + valans + kabuk',
            'n_inputs': 4,
            'anfis_feasible': True,
            'anfis_rules_3mf': 81,
        },

        'AZNNPMC': {
            'abbrevs': ['A', 'Z', 'NN', 'NP', 'MC'],
            'category': 'NnNp_5in',
            'target_affinity': ['MM', 'QM', 'Beta_2', 'MM_QM'],
            'description': 'A, Z, Nn, Np, MC - kapsamli valans+kabuk seti',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        'AZSNNNP': {
            'abbrevs': ['A', 'Z', 'S', 'NN', 'NP'],
            'category': 'NnNp_5in',
            'target_affinity': ['MM', 'QM', 'MM_QM'],
            'description': 'A, Z, SPIN, Nn, Np - spin + valans konfigurasyonu (MM icin)',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        # =====================================================================
        # LEGACY SETLER (geri uyumluluk icin - eski isimler)
        # =====================================================================

        'Basic': {
            'abbrevs': ['A', 'Z', 'N', 'S', 'PAR'],
            'category': 'Legacy',
            'target_affinity': ['MM', 'QM', 'Beta_2', 'MM_QM'],
            'description': '[LEGACY] Temel cekirdek ozellikleri (A,Z,N,SPIN,PARITY)',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        'Extended': {
            'abbrevs': ['A', 'Z', 'N', 'S', 'PAR', 'BEPA', 'BEA', 'BEP', 'MC', 'ZMD', 'NMD', 'SPHI'],
            'category': 'Legacy',
            'target_affinity': ['MM', 'QM', 'Beta_2', 'MM_QM'],
            'description': '[LEGACY] Genisletilmis fizik ozellikleri (12 giris)',
            'n_inputs': 12,
            'anfis_feasible': False,
        },

        'Full': {
            'abbrevs': 'ALL',
            'category': 'Legacy',
            'target_affinity': ['MM', 'QM', 'Beta_2', 'MM_QM'],
            'description': '[LEGACY] Tum mevcut ozellikler (~40+ giris)',
            'n_inputs': None,
            'anfis_feasible': False,
        },

        'ANFIS_Compact': {
            'abbrevs': ['A', 'Z', 'S', 'MC', 'BEPA'],
            'category': 'Legacy_ANFIS',
            'target_affinity': ['MM'],
            'description': '[LEGACY] ANFIS icin optimize (5 giris, 32 kural 2MF)',
            'n_inputs': 5,
            'anfis_feasible': True,
            'anfis_rules_2mf': 32,
        },

        'ANFIS_Standard': {
            'abbrevs': ['A', 'Z', 'N', 'S', 'PAR', 'MC', 'BEPA', 'ZSG'],
            'category': 'Legacy_ANFIS',
            'target_affinity': ['MM', 'QM'],
            'description': '[LEGACY] ANFIS standart (8 giris, 256 kural 2MF)',
            'n_inputs': 8,
            'anfis_feasible': True,
            'anfis_rules_2mf': 256,
        },
    }

    # =========================================================================
    # TARGET-SPECIFIC RECOMMENDED FEATURE SETS
    # Hedef bazinda onerilen feature set listeleri (kucukten buyuge giris sayisi)
    # =========================================================================
    TARGET_RECOMMENDED_SETS = {
        'MM': [
            # 3-input (ANFIS 3MF: 27 kural)
            'AZS', 'AZMC', 'AZBEPA', 'ASMC', 'AMCBEPA', 'NNPMC',
            # 4-input (ANFIS 3MF: 81 kural)
            'AZSMC', 'AZSBEPA', 'AZMCBEPA', 'AZSB2E', 'AZNNP',
            # 5-input (ANFIS 2MF: 32 kural)
            'AZSMCBEPA', 'AZSMCB2E', 'AZSNNNP',
        ],
        'QM': [
            # 3-input
            'AZS', 'AZB2E', 'AZMC', 'ZB2EMC', 'B2EMCBEA', 'NNPMC',
            # 4-input
            'AZB2EMC', 'ZB2EMCS', 'AZSB2E', 'AZB2EBEA', 'AZNNP', 'ZNNPMC',
            # 5-input
            'AZB2EMCS', 'AZB2EMCBEA', 'AZNNPMC',
        ],
        'Beta_2': [
            # 3-input
            'AZN', 'MCZMNM', 'AZVNV', 'ZMNMBEA', 'NNPMC',
            # 4-input
            'MCZMNMZV', 'MCZMNMBEA', 'AMCZMNM', 'ZVNVZMNM', 'ZNNPMC',
            # 5-input
            'MCZMNMZVNV', 'AMCZMNMBEA', 'AZNNPMC',
        ],
        'MM_QM': [
            # 3-input
            'AZS', 'AZB2E', 'AZMC', 'NNPMC',
            # 4-input
            'AZSMC', 'AZB2EMC', 'AZSB2E', 'AZNNP',
            # 5-input
            'AZSMCB2E', 'AZB2EMCBEA', 'AZNNPMC',
        ],
    }

    def __init__(self):
        """Initialize Feature Combination Manager"""
        logger.info("Feature Combination Manager initialized")
        n_shap = sum(1 for s in self.FEATURE_SETS.values() if 'Legacy' not in s['category'])
        n_legacy = len(self.FEATURE_SETS) - n_shap
        logger.info(f"SHAP-based sets: {n_shap}, Legacy sets: {n_legacy}")

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def get_feature_set(self,
                        set_name: str,
                        available_columns: List[str],
                        target_cols: List[str]) -> List[str]:
        """
        Belirtilen feature set icin gercek sutun adlarini dondur.

        Args:
            set_name: Feature set adi (AZS, AZMC, Basic, ...)
            available_columns: DataFrame'de mevcut sutunlar
            target_cols: Hedef sutunlari (dis bicaklanacak)

        Returns:
            Gercek sutun adlari listesi
        """
        if set_name not in self.FEATURE_SETS:
            raise ValueError(f"Unknown feature set: '{set_name}'. "
                             f"Available: {sorted(self.FEATURE_SETS.keys())}")

        feature_def = self.FEATURE_SETS[set_name]
        abbrevs = feature_def['abbrevs']

        # 'Full' / 'ALL' -> tum mevcut sutunlar
        if abbrevs == 'ALL':
            features = [col for col in available_columns
                        if col not in target_cols and col != 'NUCLEUS'
                        and col not in ('deformation_type',)]  # kategorik disla
            logger.info(f"Feature set '{set_name}': {len(features)} features (ALL available)")
            return features

        # Abbreviation -> gercek sutun adi
        requested = []
        for abbr in abbrevs:
            col = self.FEATURE_ABBREV.get(abbr, abbr)  # bilinmiyorsa abbr'yi aynen kullan
            requested.append(col)

        # Mevcut sutunlari filtrele
        available = [f for f in requested if f in available_columns]
        missing = [f for f in requested if f not in available_columns]

        if missing:
            logger.warning(f"Feature set '{set_name}': {len(missing)} feature(s) not in DataFrame: {missing}")

        if not available:
            raise ValueError(f"Feature set '{set_name}': No features found in DataFrame! "
                             f"Requested: {requested}")

        logger.info(f"Feature set '{set_name}': {len(available)}/{len(requested)} features available")
        return available

    def resolve_abbrevs_to_columns(self, abbrev_list: List[str]) -> List[str]:
        """Kisaltma listesini gercek sutun adlarina cevir."""
        return [self.FEATURE_ABBREV.get(a, a) for a in abbrev_list]

    def get_feature_set_info(self, set_name: str) -> Dict:
        """Feature set hakkinda bilgi dondur."""
        if set_name not in self.FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {set_name}")
        return self.FEATURE_SETS[set_name].copy()

    def get_target_feature_sets(self, target: str) -> List[str]:
        """
        Hedefe gore onerilen feature setlerini dondur.

        Args:
            target: MM, QM, Beta_2, MM_QM

        Returns:
            Feature set isimleri listesi (kucukten buyuge giris sayisi)
        """
        return self.TARGET_RECOMMENDED_SETS.get(target, list(self.FEATURE_SETS.keys()))

    def list_feature_sets(self, category: Optional[str] = None) -> List[str]:
        """Mevcut feature setlerini listele."""
        if category is None:
            return list(self.FEATURE_SETS.keys())
        return [name for name, info in self.FEATURE_SETS.items()
                if info['category'] == category]

    def get_sets_by_n_inputs(self, n_inputs: int) -> List[str]:
        """Belirtilen giris sayisina gore feature setleri dondur."""
        return [name for name, info in self.FEATURE_SETS.items()
                if info.get('n_inputs') == n_inputs]

    def get_anfis_feasible_sets(self, max_inputs: int = 5) -> List[str]:
        """ANFIS icin uygun feature setlerini dondur."""
        return [name for name, info in self.FEATURE_SETS.items()
                if info.get('anfis_feasible', False)
                and info.get('n_inputs', 999) <= max_inputs
                and 'Legacy' not in info['category']]

    def save_feature_combinations_json(self, output_path: str):
        """Feature kombinasyonlarini JSON dosyasina kaydet."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            'feature_abbrev_map': self.FEATURE_ABBREV,
            'feature_sets': {},
            'target_recommended_sets': self.TARGET_RECOMMENDED_SETS,
            'metadata': {
                'version': '2.0.0',
                'total_sets': len(self.FEATURE_SETS),
                'shap_based_sets': sum(
                    1 for s in self.FEATURE_SETS.values()
                    if 'Legacy' not in s['category']
                ),
            }
        }

        for name, info in self.FEATURE_SETS.items():
            abbrevs = info['abbrevs']
            export_data['feature_sets'][name] = {
                'abbrevs': abbrevs if abbrevs != 'ALL' else 'ALL_AVAILABLE',
                'columns': (
                    self.resolve_abbrevs_to_columns(abbrevs)
                    if abbrevs != 'ALL' else 'ALL_AVAILABLE'
                ),
                'category': info['category'],
                'description': info['description'],
                'n_inputs': info.get('n_inputs'),
                'anfis_feasible': info.get('anfis_feasible', False),
                'target_affinity': info.get('target_affinity', []),
            }
            if 'anfis_rules_3mf' in info:
                export_data['feature_sets'][name]['anfis_rules_3mf'] = info['anfis_rules_3mf']
            if 'anfis_rules_2mf' in info:
                export_data['feature_sets'][name]['anfis_rules_2mf'] = info['anfis_rules_2mf']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] Feature combinations saved to: {output_path}")

    def create_feature_set_summary_table(self) -> pd.DataFrame:
        """Feature setlerinin ozet tablosunu olustur."""
        data = []
        for name, info in self.FEATURE_SETS.items():
            abbrevs = info['abbrevs']
            cols = (self.resolve_abbrevs_to_columns(abbrevs)
                    if abbrevs != 'ALL' else ['ALL'])
            row = {
                'Feature_Set_Code': name,
                'N_Inputs': info.get('n_inputs', 'ALL'),
                'Category': info['category'],
                'ANFIS_Feasible': info.get('anfis_feasible', False),
                'ANFIS_Rules_3MF': info.get('anfis_rules_3mf', '-'),
                'ANFIS_Rules_2MF': info.get('anfis_rules_2mf', '-'),
                'Target_Affinity': ', '.join(info.get('target_affinity', [])),
                'Abbrev_Codes': ', '.join(abbrevs) if abbrevs != 'ALL' else 'ALL',
                'Column_Names': ', '.join(cols),
                'Description': info['description'],
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df.sort_values(['Category', 'N_Inputs'])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_default_feature_sets() -> List[str]:
    """
    Varsayilan feature setlerini dondur.
    Artik 'Basic' degil SHAP-bazli setler kullaniliyor.
    """
    return ['AZS', 'AZMC', 'AZBEPA', 'AZSMC', 'AZSMCBEPA']


def get_target_default_feature_sets(target: str) -> List[str]:
    """
    Hedef icin varsayilan SHAP-bazli feature setlerini dondur.
    main.py'nin feature_sets=None oldugunda kullanmasi icin.
    """
    mgr = FeatureCombinationManager()
    return mgr.get_target_feature_sets(target)


def get_anfis_feature_sets() -> List[str]:
    """ANFIS icin optimize edilmis feature setleri (<=4 giris, ANFIS 3MF)."""
    mgr = FeatureCombinationManager()
    return mgr.get_anfis_feasible_sets(max_inputs=4)


def get_all_shap_feature_sets() -> List[str]:
    """Tum SHAP-bazli (legacy olmayan) feature setleri."""
    return [name for name, info in FeatureCombinationManager.FEATURE_SETS.items()
            if 'Legacy' not in info['category']]
