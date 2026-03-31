#!/usr/bin/env python3
# Created: 2026-03-31
# Purpose: Manley Massive Passive + API Vision 550 + Pultec MEQ-5 프로파일링
# Usage: python scripts/profile_mp_api_meq5.py

import json
import logging
import time
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "audioman" / "src"))

from audioman.core.plugin_analysis import (
    measure_eq_response, measure_eq_parameter_sweep,
    measure_eq_nonlinearity, EQResponseResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


# =============================================================================
# Manley Massive Passive
#
# boost-only 패시브 EQ (0~20dB), 밴드 활성화: ch1XXenable = "BOOST" or "CUT"
# 4밴드: lo (shelf/bell), lo-mid (shelf/bell), hi-mid (shelf/bell), hi (shelf/bell)
# bw: 1~3 (bandwidth), gain: 0~20
# =============================================================================

MP_PATH = "/Library/Audio/Plug-Ins/VST3/uaudio_manley_massive_passive.vst3"

# bypass: 모든 밴드 OUT
MP_BYPASS = {
    "ch1loenable": "OUT",
    "ch1lomidenable": "OUT",
    "ch1himidenable": "OUT",
    "ch1hienable": "OUT",
}

# 기본값: 다른 밴드 모두 OFF
_MP_ALL_OFF = {
    "ch1loenable": "OUT", "ch1logain": 0.0,
    "ch1lomidenable": "OUT", "ch1lomidgain": 0.0,
    "ch1himidenable": "OUT", "ch1himidgain": 0.0,
    "ch1hienable": "OUT", "ch1higain": 0.0,
}

MP_SWEEPS = {
    # --- Lo band (shelf) ---
    "lo_boost_gain": {
        "param": "ch1logain",
        "values": [0, 4, 8, 12, 16, 20],
        "fixed": {**_MP_ALL_OFF,
                  "ch1loenable": "BOOST", "ch1loshape": "SHELF",
                  "ch1lofreq": 150.0, "ch1lobw": 2.0},
    },
    "lo_cut_gain": {
        "param": "ch1logain",
        "values": [0, 4, 8, 12, 16, 20],
        "fixed": {**_MP_ALL_OFF,
                  "ch1loenable": "CUT", "ch1loshape": "SHELF",
                  "ch1lofreq": 150.0, "ch1lobw": 2.0},
    },
    "lo_freq": {
        "param": "ch1lofreq",
        "values": [22, 47, 100, 220, 470, 1000],
        "fixed": {**_MP_ALL_OFF,
                  "ch1loenable": "BOOST", "ch1loshape": "SHELF",
                  "ch1logain": 10.0, "ch1lobw": 2.0},
    },
    "lo_bw": {
        "param": "ch1lobw",
        "values": [1.0, 1.5, 2.0, 2.5, 3.0],
        "fixed": {**_MP_ALL_OFF,
                  "ch1loenable": "BOOST", "ch1loshape": "SHELF",
                  "ch1logain": 10.0, "ch1lofreq": 150.0},
    },
    # --- Lo-mid band (bell) ---
    "lomid_boost_gain": {
        "param": "ch1lomidgain",
        "values": [0, 4, 8, 12, 16, 20],
        "fixed": {**_MP_ALL_OFF,
                  "ch1lomidenable": "BOOST", "ch1lomidshape": "BELL",
                  "ch1lomidfreq": 560.0, "ch1lomidbw": 2.0},
    },
    "lomid_freq": {
        "param": "ch1lomidfreq",
        "values": [82, 180, 390, 820, 1800, 3900],
        "fixed": {**_MP_ALL_OFF,
                  "ch1lomidenable": "BOOST", "ch1lomidshape": "BELL",
                  "ch1lomidgain": 10.0, "ch1lomidbw": 2.0},
    },
    # --- Hi-mid band (bell) ---
    "himid_boost_gain": {
        "param": "ch1himidgain",
        "values": [0, 4, 8, 12, 16, 20],
        "fixed": {**_MP_ALL_OFF,
                  "ch1himidenable": "BOOST", "ch1himidshape": "BELL",
                  "ch1himidfreq": 1500.0, "ch1himidbw": 2.0},
    },
    "himid_freq": {
        "param": "ch1himidfreq",
        "values": [220, 470, 1000, 2200, 4700, 10000],
        "fixed": {**_MP_ALL_OFF,
                  "ch1himidenable": "BOOST", "ch1himidshape": "BELL",
                  "ch1himidgain": 10.0, "ch1himidbw": 2.0},
    },
    # --- Hi band (shelf) ---
    "hi_boost_gain": {
        "param": "ch1higain",
        "values": [0, 4, 8, 12, 16, 20],
        "fixed": {**_MP_ALL_OFF,
                  "ch1hienable": "BOOST", "ch1hishape": "SHELF",
                  "ch1hifreq": 3900.0, "ch1hibw": 2.0},
    },
    "hi_cut_gain": {
        "param": "ch1higain",
        "values": [0, 4, 8, 12, 16, 20],
        "fixed": {**_MP_ALL_OFF,
                  "ch1hienable": "CUT", "ch1hishape": "SHELF",
                  "ch1hifreq": 3900.0, "ch1hibw": 2.0},
    },
    "hi_freq": {
        "param": "ch1hifreq",
        "values": [560, 1200, 2700, 5600, 12000, 27000],
        "fixed": {**_MP_ALL_OFF,
                  "ch1hienable": "BOOST", "ch1hishape": "SHELF",
                  "ch1higain": 10.0, "ch1hibw": 2.0},
    },
}


# =============================================================================
# API Vision 550L EQ Section
#
# 4밴드 파라메트릭: lf, lmf, hmf, hf (gain: -12~+12)
# eq_on=True, eq_type="550L" 설정 필수
# lf_filter: "Shelf"/"Peak", hf_filter: "Shelf"/"Peak"
# =============================================================================

API_PATH = "/Library/Audio/Plug-Ins/VST3/uaudio_api_vision_channel_strip.vst3"

# EQ만 활성화, 다이내믹스 비활성
_API_BASE = {
    "eq_on": True,
    "eq_type": "550L",
    "215_on": False,  # filter off
    "235_on": False,  # gate off
    "225_on": False,  # comp off
}

API_BYPASS = {
    **_API_BASE,
    "550_lf_gain": 0.0,
    "550_lmf_gain": 0.0,
    "550_hmf_gain": 0.0,
    "550_hf_gain": 0.0,
}

_API_GAINS_ZERO = {
    "550_lf_gain": 0.0,
    "550_lmf_gain": 0.0,
    "550_hmf_gain": 0.0,
    "550_hf_gain": 0.0,
}

API_SWEEPS = {
    # LF gain 스윕
    "lf_gain": {
        "param": "550_lf_gain",
        "values": [-12, -8, -4, 0, 4, 8, 12],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_lf_freq": 300.0, "550_lf_filter": "Shelf"},
    },
    # LF freq 스윕
    "lf_freq": {
        "param": "550_lf_freq",
        "values": [30, 60, 100, 200, 300, 400],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_lf_gain": 8.0, "550_lf_filter": "Shelf"},
    },
    # LMF gain 스윕
    "lmf_gain": {
        "param": "550_lmf_gain",
        "values": [-12, -8, -4, 0, 4, 8, 12],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_lmf_freq": 700.0},
    },
    # LMF freq 스윕
    "lmf_freq": {
        "param": "550_lmf_freq",
        "values": [75, 150, 350, 700, 1000],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_lmf_gain": 8.0},
    },
    # HMF gain 스윕
    "hmf_gain": {
        "param": "550_hmf_gain",
        "values": [-12, -8, -4, 0, 4, 8, 12],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_hmf_freq": 3000.0},
    },
    # HMF freq 스윕
    "hmf_freq": {
        "param": "550_hmf_freq",
        "values": [800, 1500, 3000, 5000, 8000, 12500],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_hmf_gain": 8.0},
    },
    # HF gain 스윕
    "hf_gain": {
        "param": "550_hf_gain",
        "values": [-12, -8, -4, 0, 4, 8, 12],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_hf_freq": 15000.0, "550_hf_filter": "Shelf"},
    },
    # HF freq 스윕
    "hf_freq": {
        "param": "550_hf_freq",
        "values": [2500, 5000, 8000, 12000, 15000, 20000],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_hf_gain": 8.0, "550_hf_filter": "Shelf"},
    },
    # LF peak vs shelf 비교
    "lf_peak_gain": {
        "param": "550_lf_gain",
        "values": [-8, 0, 8],
        "fixed": {**_API_BASE, **_API_GAINS_ZERO,
                  "550_lf_freq": 200.0, "550_lf_filter": "Peak"},
    },
}


# =============================================================================
# Pultec MEQ-5
#
# 3밴드: lm_peak (boost), mid_dip (cut), hm_peak (boost)
# lm_freq: enum ("200 CPS", "300 CPS", "500 CPS", "700 CPS", "1000 CPS")
# mid_freq: enum ("200 CPS", "300 CPS", "500 CPS", "700 CPS", "1 KCS", "1.5 KCS", "2 KCS", "3 KCS", "4 KCS", "5 KCS", "7 KCS")
# hm_freq: enum ("1.5 KCS", "2 KCS", "3 KCS", "4 KCS", "5 KCS")
# =============================================================================

MEQ5_PATH = "/Library/Audio/Plug-Ins/VST3/uaudio_pultec_meq-5.vst3"

MEQ5_BYPASS = {
    "lm_peak": 0.0,
    "mid_dip": 0.0,
    "hm_peak": 0.0,
}

MEQ5_SWEEPS = {
    # Lo-mid peak 게인 스윕
    "lm_peak_gain": {
        "param": "lm_peak",
        "values": [0, 2, 4, 6, 8, 10],
        "fixed": {"lm_freq": "700 CPS", "mid_dip": 0, "hm_peak": 0},
    },
    # Lo-mid peak 주파수 스윕
    "lm_freq": {
        "param": "lm_freq",
        "values": ["200 CPS", "300 CPS", "500 CPS", "700 CPS", "1000 CPS"],
        "fixed": {"lm_peak": 6, "mid_dip": 0, "hm_peak": 0},
    },
    # Mid dip 게인 스윕
    "mid_dip_gain": {
        "param": "mid_dip",
        "values": [0, 2, 4, 6, 8, 10],
        "fixed": {"mid_freq": "1 KCS", "lm_peak": 0, "hm_peak": 0},
    },
    # Mid dip 주파수 스윕
    "mid_freq": {
        "param": "mid_freq",
        "values": ["200 CPS", "500 CPS", "700 CPS", "1 KCS", "2 KCS", "3 KCS", "5 KCS", "7 KCS"],
        "fixed": {"mid_dip": 6, "lm_peak": 0, "hm_peak": 0},
    },
    # Hi-mid peak 게인 스윕
    "hm_peak_gain": {
        "param": "hm_peak",
        "values": [0, 2, 4, 6, 8, 10],
        "fixed": {"hm_freq": "3 KCS", "lm_peak": 0, "mid_dip": 0},
    },
    # Hi-mid peak 주파수 스윕
    "hm_freq": {
        "param": "hm_freq",
        "values": ["1.5 KCS", "2 KCS", "3 KCS", "4 KCS", "5 KCS"],
        "fixed": {"hm_peak": 6, "lm_peak": 0, "mid_dip": 0},
    },
    # 조합: lm_peak + mid_dip (보컬 EQ 패턴)
    "lm_plus_dip": {
        "param": "lm_peak",
        "values": [4, 6, 8],
        "fixed": {"lm_freq": "700 CPS", "mid_dip": 4, "mid_freq": "1 KCS", "hm_peak": 0},
    },
    # 조합: mid_dip + hm_peak (presence)
    "dip_plus_hm": {
        "param": "hm_peak",
        "values": [4, 6, 8],
        "fixed": {"hm_freq": "3 KCS", "mid_dip": 4, "mid_freq": "1 KCS", "lm_peak": 0},
    },
}


# =============================================================================
# 프로파일링 공통 함수 (profile_nseq2_pultec.py에서 재사용)
# =============================================================================

def profile_plugin(name: str, plugin_path: str, sweep_config: dict,
                   bypass_params: dict):
    plugin_dir = DATA_DIR / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_measurements = sum(len(s["values"]) for s in sweep_config.values())
    logger.info(f"{name}: {len(sweep_config)}개 스윕, {total_measurements}개 설정")

    # 1. 파라미터 스윕
    logger.info(f"[1/3] 파라미터 스윕 측정 중...")
    all_results: list[EQResponseResult] = []

    try:
        results = measure_eq_parameter_sweep(
            plugin_path, sweep_config, bypass_params,
            sample_rate=44100, fft_size=32768, level_db=-12.0,
        )
        all_results.extend(results)
        logger.info(f"  스윕 완료: {len(results)}개 측정")
    except Exception as e:
        logger.error(f"  스윕 실패: {e}")

    # 2. 비선형성 측정
    logger.info(f"[2/3] 비선형성 측정 중...")
    first_sweep = list(sweep_config.values())[0]
    mid_idx = len(first_sweep["values"]) // 2
    test_params = dict(bypass_params)
    test_params[first_sweep["param"]] = first_sweep["values"][mid_idx]
    test_params.update(first_sweep.get("fixed", {}))

    nl_results = []
    try:
        nl_results = measure_eq_nonlinearity(
            plugin_path, test_params, bypass_params,
            levels_db=[-36, -24, -18, -12, -6, -3, 0],
            sample_rate=44100, fft_size=32768,
        )
        logger.info(f"  비선형성 완료: {len(nl_results)}개 레벨")
    except Exception as e:
        logger.error(f"  비선형성 실패: {e}")

    # 3. 저장
    logger.info(f"[3/3] 결과 저장 중...")

    is_level_dependent = False
    max_deviation = 0.0
    if len(nl_results) >= 2:
        ref = np.array(nl_results[0].magnitude_db)
        for r in nl_results[1:]:
            diff = np.max(np.abs(np.array(r.magnitude_db) - ref))
            max_deviation = max(max_deviation, diff)
        is_level_dependent = max_deviation > 0.5

    profile = {
        "plugin": name,
        "path": plugin_path,
        "plugin_type": "eq",
        "n_measurements": len(all_results),
        "measurements": [],
        "nonlinearity": {
            "is_level_dependent": is_level_dependent,
            "max_response_deviation_db": round(max_deviation, 2),
            "thd_per_level": [round(r.thd_at_1k, 4) for r in nl_results],
            "levels_db": [r.params.get("_input_level_db", 0) for r in nl_results],
        },
    }

    for r in all_results:
        freq_arr = np.array(r.frequencies)
        mag_arr = np.array(r.magnitude_db)
        mask = (freq_arr >= 20) & (freq_arr <= 20000)
        mag_range = mag_arr[mask]
        profile["measurements"].append({
            "params": r.params,
            "magnitude_range_db": [round(float(np.min(mag_range)), 2),
                                   round(float(np.max(mag_range)), 2)],
            "is_minimum_phase": r.is_minimum_phase,
            "thd_at_1k": r.thd_at_1k,
        })

    with open(plugin_dir / "profile.json", "w") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False, default=str)

    if all_results:
        freq_curves = np.array([r.magnitude_db for r in all_results], dtype=np.float32)
        phase_curves = np.array([r.phase_deg for r in all_results], dtype=np.float32)
        delay_curves = np.array([r.group_delay_ms for r in all_results], dtype=np.float32)
        freq_axis = np.array(all_results[0].frequencies, dtype=np.float32)

        np.save(str(plugin_dir / "frequency_response_curves.npy"), freq_curves)
        np.save(str(plugin_dir / "phase_response_curves.npy"), phase_curves)
        np.save(str(plugin_dir / "group_delay_curves.npy"), delay_curves)
        np.save(str(plugin_dir / "frequency_axis.npy"), freq_axis)

        labels = [r.params for r in all_results]
        with open(plugin_dir / "settings_labels.json", "w") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"  곡선: {freq_curves.shape[0]} settings × {freq_curves.shape[1]} bins")

    elapsed = time.time() - t0
    logger.info(f"{name} 완료: {len(all_results)} 측정, {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    return len(all_results)


def main():
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Manley Massive Passive")
    print(f"{'='*60}")
    n1 = profile_plugin("massive_passive", MP_PATH, MP_SWEEPS, MP_BYPASS)

    print(f"\n{'='*60}")
    print(f"  API Vision 550L")
    print(f"{'='*60}")
    n2 = profile_plugin("api_550", API_PATH, API_SWEEPS, API_BYPASS)

    print(f"\n{'='*60}")
    print(f"  Pultec MEQ-5")
    print(f"{'='*60}")
    n3 = profile_plugin("pultec_meq5", MEQ5_PATH, MEQ5_SWEEPS, MEQ5_BYPASS)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  완료:")
    print(f"    Massive Passive: {n1} 측정")
    print(f"    API 550L:        {n2} 측정")
    print(f"    Pultec MEQ-5:    {n3} 측정")
    print(f"  총 소요: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    print(f"  데이터: data/massive_passive/, data/api_550/, data/pultec_meq5/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
