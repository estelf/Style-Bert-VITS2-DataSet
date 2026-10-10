"""Stage 2: 発話スコアリング. speaker_sim / rolloff / jitter / shimmer / F0 / snr / lufs を manifest に積み上げる.

- C1: 埋め込み統計はコーパス全体で次元ごとに std へスケーリング (CMVN の分散側) してから cosine を取る。
  c0 (エネルギー) が cosine を支配して speaker_sim が「音量類似度」になるのを防ぐ。正規化後のベクトルを
  scores["emb"] に保存するので、rescore は再計算なしに同じ指紋で比較できる。
- C2: voiced/unvoiced は絶対的な騒音床で分離する (中央値分割だと voiced_ratio が常に ≒0.5 で無情報)。
- C3: jitter/shimmer は有声区間 (連続した pitch 区間) の中に限り、Praat の local 定義に倣う
  (無声フレームを落とした列に np.diff を当てて境界を跨がない。shimmer は振幅摂動で別に計算)。
- C4/A1: LUFS ゲートはメータのブロック長 (0.4s) ベース。計算不能・非有限なら None を記録し、
  x * -inf で nan を撒き散らさない。
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .config import Config
from .manifest import Record


def _voiced_runs(valid: np.ndarray) -> list[slice]:
    """pitch track の連続有声区間 (NaN/0 で切ったスライス列). 境界を跨いだ差分を混ぜないための分割 (C3)."""
    runs, start = [], None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 3:
                runs.append(slice(start, i))
            start = None
    if start is not None and len(valid) - start >= 3:
        runs.append(slice(start, len(valid)))
    return runs


def score_segments(records: list[Record], cfg: Config, cache_dir) -> None:
    """各 Record.scores にスコアを追記する (除外可否は Stage 6 の閾値で決める)."""
    import pyloudnorm as pyln
    import scipy.signal

    from .stage0_ingest import load_cached, resample_to

    meter = pyln.Meter(44100)
    encoder = _encoder()
    raw_embs: list[np.ndarray] = []
    emb_rows: list[Record] = []

    for r in tqdm(records, desc="stage2 score", unit="utt"):
        try:  # B2: 1 本の異常セグメントで数時間のランを全損させない
            x = load_cached(r, cache_dir, 44100)
            x16 = resample_to(x, 44100, 16000)
            sr = 44100

            # D2: クリッピング / デジタル無音 / DC オフセット (学習後に修復不可能なので投入時点で弾く)
            clip_ratio = float(np.mean(np.abs(x) >= 0.999))
            dc_offset = float(abs(np.mean(x)))
            digital_silence = float(np.mean(x == 0.0))

            # 帯域ロールオフ / 高域エネルギー比: 16k/24k アップコンの「見せかけ高品質」検出
            freqs, psd = scipy.signal.welch(x, fs=sr, nperseg=min(2048, max(len(x), 2)))
            cum = np.cumsum(psd)
            rolloff = float(freqs[min(int(np.searchsorted(cum, 0.99 * cum[-1])), len(freqs) - 1)])
            hf_ratio = float(psd[freqs > 7000].sum() / (psd.sum() + 1e-12))

            # C2: フレームエネルギーを絶対閾値 (騒音床の推定) で分割して SNR と voiced_ratio を作る。
            # 床は max(セグメント内 p20×3, -55 dBFS) ではなく min を取る: 無音を含む収録では p20×3 が
            # 実騒音床になるが、無音ゼロのホットな収録では p20×3 が「静かな発話」になって SNR が
            # ダイナミックレンジ化するため、絶対床で頭打ちする (voiced_ratio は床超過率のまま)。
            hop = sr // 100  # 10ms
            fe = _frame_rms(x, sr, hop)
            floor = min(float(np.percentile(fe, 20)) * 3.0, 10 ** (-55 / 20))
            vmask = fe > floor
            if (~vmask).any() and vmask.any():
                snr = float(
                    10 * np.log10((fe[vmask] ** 2).mean() / ((fe[~vmask] ** 2).mean() + 1e-12))
                )
            elif vmask.any():
                # 床を下回るフレームが1本もない (無音なしのホットな収録) → 絶対床 (-55 dBFS) 比で算出
                abs_floor = 10 ** (-55 / 20)
                snr = float(10 * np.log10((fe[vmask] ** 2).mean() / (abs_floor**2)))
            else:
                snr = None  # ほぼ無音のセグメント (検証不能としてゲート側で不合格扱い)
            voiced_ratio = float(vmask.mean())

            # C4/A1: LUFS はメータのブロック長以上ある時だけ計算。不能/非有限は None (偽値 -120 を撒かない)
            if len(x) >= meter.block_size * sr:
                v = meter.integrated_loudness(x)
                lufs = float(v) if np.isfinite(v) else None
            else:
                lufs = None

            # C3: 有声区間ごとに周期・振幅摂動を計算 (境界跨ぎ差分を混ぜない、shimmer は jitter のコピーではない)
            import parselmouth

            seg = parselmouth.Sound(
                np.ascontiguousarray(x16, dtype=np.float32), sampling_frequency=16000
            )
            st = seg.to_pitch_ac(time_step=0.01)
            f = st.selected_array["frequency"]
            valid = ~np.isnan(f) & (f > 0)
            jitter = shimmer = None
            if valid.any():
                num, den = 0.0, 0.0
                for sl in _voiced_runs(valid):  # 有声区間ごとに周期摂動 (境界跨ぎの差分を混ぜない)
                    t = 1.0 / f[sl]
                    num += float(np.abs(np.diff(t)).mean())
                    den += float(np.mean(t)) * (sl.stop - sl.start)
                jitter = num / den if den > 0 else None
                # shimmer_local 相当: 有声グリッド上の振幅の摂動比 (振幅は音声から直接)
                af = 160  # 16k で 10ms
                amps = np.full(len(f), np.nan)
                t0, dt = float(st.start_time), float(st.time_step)  # pitch は等間隔グリッド
                for k in range(len(f)):
                    c = int((t0 + k * dt) * 16000)
                    if af // 2 <= c < len(x16) - af // 2:
                        amps[k] = np.sqrt(np.mean(x16[c - af // 2 : c + af // 2] ** 2))
                an, ad = 0.0, 0.0
                for sl in _voiced_runs(valid):
                    a = amps[sl]
                    if len(a) >= 3 and np.isfinite(a).all():
                        an += float(np.abs(np.diff(a)).mean())
                        ad += float(np.mean(a))
                shimmer = an / ad if ad > 0 else None
            f0_median = float(np.median(f[valid])) if valid.any() else None

            r.scores.update(
                {
                    "overlap_flag": None,  # D7: pyannote 系 diarization が必要。ドラフトでは未接続 (None 記録)
                    "reverb_score": round(_tail_ratio(x), 3),
                    "rolloff_hz": round(rolloff),
                    "hf_energy_ratio": round(hf_ratio, 5),
                    "jitter_local": round(float(jitter), 4) if jitter is not None else None,
                    "shimmer_local": round(float(shimmer), 4) if shimmer is not None else None,
                    "f0_median_hz": round(f0_median, 1) if f0_median is not None else None,
                    "snr": round(snr, 1) if snr is not None else None,
                    "lufs": round(lufs, 1) if lufs is not None else None,
                    "voiced_ratio": round(voiced_ratio, 2),
                    "clip_ratio": round(clip_ratio, 5),
                    "dc_offset_abs": round(dc_offset, 5),
                    "digital_silence_ratio": round(digital_silence, 4),
                }
            )
            if not np.all(np.isfinite(x)):
                r.scores.setdefault("errors", []).append("stage2: non-finite audio samples")
            raw_embs.append(_embed_one(encoder, x16))
            try:  # 指紋計算の失敗で本体スコアを捨てない (dedup が弱まるだけ)
                r.scores["fp_sig"] = _content_fingerprint(
                    encoder, x16
                )  # D8: 内容指紋 (話者ベクトルではない)
            except Exception as e:  # noqa: BLE001
                r.scores["fp_sig"] = []
                r.scores.setdefault("fp_error", []).append(repr(e))
            emb_rows.append(r)
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("errors", []).append(f"stage2: {e!r}")

    _cmvn_store(raw_embs, emb_rows)


def _frame_rms(audio: np.ndarray, sr: int, hop: int) -> np.ndarray:
    n = len(audio) // hop * hop
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    frames = audio[:n].reshape(-1, hop)
    return np.sqrt((frames**2).mean(axis=1))


def _tail_ratio(x: np.ndarray) -> float:
    """残響プロキシ: 発話ピーク区間に対する末尾減衰エネルギー比 (F4: O(N) の移動平均)."""
    import scipy.ndimage

    env = np.abs(x)
    if len(env) < 4410:
        return float(env[-1] / (env.max() + 1e-9))
    smooth = scipy.ndimage.uniform_filter1d(
        env, size=4410, mode="nearest"
    )  # F4: sliding_window_view(4410).mean の O(N×W) 回避
    peak = float(smooth.max())
    tail = float(smooth[-1])  # 末尾 100ms 移動平均 = 元の env[-4410:].mean() と同値を O(N) で
    return float(tail / (peak + 1e-9))


_ENCODER = None


def _encoder():
    """話者埋め込みエンコーダ。ドラフトでは MFCC 統計のプレースホルダ (WavLM系/ERes2Net/TitaNet へ差し替え予定)."""
    global _ENCODER
    if _ENCODER is None:
        import torchaudio

        _ENCODER = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
        _ENCODER.eval()
    return _ENCODER


def _embed_one(encoder, audio16k: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        f = encoder(torch.from_numpy(np.ascontiguousarray(audio16k)).unsqueeze(0)).squeeze(
            0
        )  # (40, T)
        # B1: std が非有限/0 でも次元数を 80 のまま保つ (フォールバックで 40 次元になり torch.stack が落ちるのを防ぐ)
        return torch.cat([torch.nan_to_num(f.mean(dim=1)), torch.nan_to_num(f.std(dim=1))]).numpy()


def _content_fingerprint(encoder, audio16k: np.ndarray) -> list[int]:
    """D8: 内容指紋 — MFCC フレームの2値化コード → 3フレーム gram → MinHash 署名.

    DESIGN.md:77 の注意 (話者埋め込みは話者ベクトルなので内容照合に使わない) に従い、
    時系列順を持つフレーム列から作る。ズレ切り出し重複で high、別発話で low になる。
    署名は manifest に保存するので rescore は再計算なしで dedup を再実行できる。
    """
    import torch

    from .stage5_dedup import minhash_signature

    with torch.inference_mode():
        f = encoder(torch.from_numpy(np.ascontiguousarray(audio16k)).unsqueeze(0)).squeeze(
            0
        )  # (40,T)
    if f.shape[1] < 3:
        return []
    codes = (
        (f > f.mean(dim=1, keepdim=True)).T.numpy().astype(np.uint8)
    )  # (T,40) 発話平均に対する2値化
    frames = [codes[i].tobytes() for i in range(len(codes))]
    grams = {frames[i] + frames[i + 1] + frames[i + 2] for i in range(len(frames) - 2)}
    return minhash_signature(grams)


def _cmvn_store(raw_embs: list[np.ndarray], rows: list[Record]) -> None:
    """C1: 次元ごとに std でスケーリング (CMVN の分散側) してから centroid cosine を取る.

    c0 (エネルギー) が分散の大きさで cosine を支配し「音量類似度」になるのを防ぐ。平均引き算は
    小コーパスで pairwise cosine に -1/(N-1) の偏りを作るため行わない (スケーリングのみ)。
    正規化済み emb を保存するので rescore/dedup は再計算なしに同じ指紋で比較できる (D8)。
    """
    if not raw_embs:
        return
    dim = min(e.shape[0] for e in raw_embs)
    M = np.stack([e[:dim] for e in raw_embs]).astype(np.float64)
    Z = M / (M.std(axis=0) + 1e-9)  # 次元正規化: c0 (エネルギー) が cosine を支配しない (C1)
    centroid = np.median(Z, axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    for z, r in zip(Z, rows, strict=True):  # raw_embs と emb_rows は必ず対で append される
        r.scores["speaker_sim"] = round(float(z @ centroid / (np.linalg.norm(z) + 1e-9)), 3)
        r.scores["emb"] = [round(float(v), 4) for v in z]
