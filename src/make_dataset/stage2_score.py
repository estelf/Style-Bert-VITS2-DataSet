"""Stage 2: 発話スコアリング. speaker_sim / rolloff / jitter / shimmer / F0 / snr / lufs を manifest に積み上げる.

ドラフト段階の注記:
- speaker_sim は MFCC 統計埋め込みの cosine (プレースホルダ)。WavLM 系・ERes2Net/TitaNet への差し替えが今後の作業。
- overlap_flag (pyannote overlap) は未実装 (None 記録)。
- reverb_score は尾引きエネルギー比の簡易プロキシ (DRR 代替)。
"""

from __future__ import annotations

import numpy as np

from .manifest import Record

_ENCODER = None


def _encoder():
    """話者埋め込みエンコーダ。ドラフトでは MFCC 統計のプレースホルダ (WavLM系/ERes2Net/TitaNet へ差し替え予定)."""
    global _ENCODER
    if _ENCODER is None:
        import torchaudio

        _ENCODER = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80}
        )
        _ENCODER.eval()
    return _ENCODER


def _embed(audio16k_batch: list[np.ndarray]) -> np.ndarray:
    import torch

    with torch.inference_mode():
        embs = []
        for a in audio16k_batch:
            f = _encoder()(torch.from_numpy(a).unsqueeze(0)).squeeze(0)  # (40, T)
            stats = torch.cat([f.mean(dim=1), f.std(dim=1)])
            norm = torch.linalg.norm(stats)
            if not torch.isfinite(norm) or norm == 0:  # 極短セグメントで std が NaN になり得る
                stats = torch.nan_to_num(f.mean(dim=1))
            embs.append(torch.nan_to_num(stats))
        return torch.stack(embs).numpy()


def score_segments(segments) -> None:
    """各 Segment.record.scores にスコアを追記する (除外可否は Stage 6 の閾値で決める)."""
    import pyloudnorm as pyln
    import scipy.signal

    embs = _embed([s.audio16k for s in segments])
    centroid = np.median(embs, axis=0)
    centroid /= np.linalg.norm(centroid)
    meter = pyln.Meter(44100)
    import parselmouth

    for s, emb in zip(segments, embs):
        x, sr = s.audio441, 44100
        sim = float((emb @ centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-9))
        # 帯域ロールオフ (99% エネルギー) と高域エネルギー比: 16k/24k アップコンの「見せかけ高品質」検出は
        # ロールオフ単体だと有声音量の低い文で誤爆するので hf_energy_ratio の方をゲートに使う
        freqs, psd = scipy.signal.welch(x, fs=sr, nperseg=2048)
        cum = np.cumsum(psd)
        rolloff = float(freqs[min(int(np.searchsorted(cum, 0.99 * cum[-1])), len(freqs) - 1)])
        hf_ratio = float(psd[freqs > 7000].sum() / (psd.sum() + 1e-12))
        # フレームエネルギーから voiced/unvoiced を雑に分離し、SNR (発話/ノイズ床) と perturbation プロキシを作る
        hop = sr // 100  # 10ms
        frames = x[: len(x) // hop * hop].reshape(-1, hop)
        fe = np.sqrt((frames**2).mean(axis=1)) if len(frames) else np.zeros(1)
        vmask = fe > np.percentile(fe, 50)
        snr = 10 * np.log10(fe[vmask].mean() / (fe[~vmask].mean() + 1e-9)) if len(frames) else 0.0
        lufs = float(meter.integrated_loudness(x)) if len(x) > sr else -120.0
        seg = parselmouth.Sound(s.audio16k.astype(np.float32), sampling_frequency=16000)
        try:
            st = seg.to_pitch_ac(time_step=0.01)
            f0 = st.selected_array["frequency"]
            f0 = f0[~np.isnan(f0) & (f0 > 0)]  # 無声区 (NaN/0) は除外
            # jitter/shimmer は Praat PointProcess 相当の簡易プロキシ (周期・振幅の摂動比)。praat-parselmouth API へ差し替え予定
            jitter = float(np.mean(np.abs(np.diff(f0))) / np.median(f0)) if len(f0) > 2 else -1.0
            shimmer = jitter  # TODO: アムピチュード摂動版。ドラフトでは F0 摂動をそのまま流用
        except Exception:
            f0, jitter, shimmer = np.array([]), -1.0, -1.0
        s.record.scores.update(
            {
                "speaker_sim": round(sim, 3),
                "overlap_flag": None,  # pyannote overlap はドラフトでは未実装
                "reverb_score": round(_tail_ratio(x), 3),
                "rolloff_hz": round(rolloff),
                "hf_energy_ratio": round(hf_ratio, 5),
                "jitter_local": round(float(jitter), 4),
                "shimmer_local": round(float(shimmer), 4),
                "f0_median_hz": round(float(np.median(f0)), 1) if len(f0) else None,
                "snr": round(float(snr), 1),
                "lufs": round(lufs, 1),
                "voiced_ratio": round(float(vmask.mean()), 2),
            }
        )


def _tail_ratio(x: np.ndarray) -> float:
    """残響プロキシ: 発話ピーク区間に対する末尾減衰エネルギー比."""
    env = np.abs(x)
    peak = np.max(np.lib.stride_tricks.sliding_window_view(env, 4410).mean(axis=1)) if len(env) > 4410 else env.max()
    tail = env[-4410:].mean() if len(env) >= 4410 else env.mean()
    return float(tail / (peak + 1e-9))
