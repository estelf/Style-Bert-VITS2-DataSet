# パイプライン各ステージの処理まとめ

音声・動画の投入から `esd.list` + `raw.zip` ができるまで、**どのステージで何が起きるか**をまとめた
実装リファレンス。設計思想は [DESIGN.md](DESIGN.md)、GPU 手順は [GPU.md](GPU.md)、
「なぜその仕様・その閾値か」の実測と経緯は [DECISIONS.md](DECISIONS.md)、
ユースケース別の操作は [../チートシート.md](../チートシート.md) を参照。

## 全体フロー

```
voice/<話者>/*.mp3|mp4 ...
   │
   ├─ Stage 0 投入・前処理      : バイトハッシュで同一ファイル再投入を除外 (Source 列を作る)
   ├─ Stage 1 音声セグメント化  : Silero-VAD で文単位スライス + 境界補正 → .cache/<utt>.flac
   ├─ Stage 2 発話スコアリング  : 音響スコア (SNR/無音率/F0/ジッタ/埋め込み/内容指紋…) を記録
   │                              [+オプトイン: pyannote 2人同時発話検出 [overlap]]
   ├─ Stage 3 文字起こしと言語系: ASR + G2P 整合 + テキストフィルタ用スコアを記録
   │                              [+オプトイン: 2系統ASRクロス検証 [asr2] / CTCアライメント [alignment] / MLM擬似PPL [lm]]
   ├─ Stage 4 音声品質スコア    : SQUIM 客观品質 (PESQ/STOI/SI-SDR) を記録
   │                              [+オプトイン: SQUIM_SUBJECTIVE mos [stage4] / AudioSet タグ [tags]]
   ├─ (任意) 除去修復           : [repair] 指定ゲート単独落ちの発話だけノイズ除去してスコアを引き直す
   ├─ Stage 5 重複排除          : 完全/ファジィ/音素/音声指紋で代表 1 本にまとめる
   ├─ Stage 5b 多様性選別       : 重複ではないが「似た内容」の発話も間引く
   └─ Stage 6 判定・出力        : ここで初めて閾値で合否 → esd.list / raw.zip / report.html
```

### 押さえておく 3 つの原則

1. **Stage 2〜4 は「記録」だけして「判定」しない。** 各発話の `Record.scores` にスコアを積み上げるだけで、
   採用/除外は Stage 6 の閾値で決まる。だから `rescore` は音声処理・ASR を再実行せずに
   `filters.toml` の閾値だけ変えて納品物を再生成できる。
2. **閾値にコード側デフォルトは無い。** `filters.toml [thresholds]` からキーを消せば（コメント化すれば）
   そのチェックは走らない。逆にキーがあるのにスコアが `None`（取得不能）な発話は「検証不能＝不合格」。
3. **1 レコード＝1 発話。** 全 Stage は `manifest.jsonl` の 1 行（`Record`）をリレー方式でリッチ化していく。
   除外理由も最後まで全部残す（`dropped_by`）。

## データの単位: `Record`（`manifest.jsonl` 1 行）

| フィールド | 意味 |
|---|---|
| `utt_id` | `<ソースハッシュ>-<連番>`。`.cache/<utt_id>.flac` と 1:1 |
| `src` / `start` / `end` | 抽出元ファイル名と、その中の開始／終了秒 |
| `text` / `lang` | ASR 確定セリフ（正規化後）／既定 `JP` |
| `scores` | スコア袋。各 Stage が追記（下記ステージ別の一覧） |
| `dup_group_id` / `dup_reason` | 重複グループ ID と、代表でなくされた理由（`exact`/`fuzzy_text`/`fuzzy_audio`） |
| `diversity_reason` | 多様性選別が落とした理由（`similar:<代表utt>` / `cap:prosody:<バケット>` / `cap:source:<src>` / `target_size`） |
| `status` | `pending` → `adopted` / `excluded` / `pending_review`（Stage 6 で確定） |
| `dropped_by` | 全失格理由の配列（閾値キー・`pipeline_error`・`empty_text`・`dedup:*`・`diversity:*` …） |
| `review_reason` | 自動採用を避けた見送り行き（ユーザー辞書候補・修復音声・`review_*` 閾値） |

> `dup_reason` の列挙値は 3 つだけ。音素三量 Jaccard によるファジィも `fuzzy_audio` ラベルになる
> （`fuzzy_audio` = 「音素レベルで近い」＋「音声指紋で近い」の両方）。

---

## Stage 0: 投入・前処理（`stage0_ingest.py`）

- **目的**: 入力ファイルを「ソース」に正規化し、同一ファイルの再投入を弾く。
- **入力**: CLI が話者名／パスから展開したメディアファイル列。
- **処理**: 各ファイルをチャンク読みで `sha256[:16]` ハッシュ化。ハッシュ一致（＝バイト単位で同一）は
  即除外して `skipped` に記録。**ここではデコードしない**（後段がソース単位でデコードし、
  全波形を同時保持しない）。
- **出力**: `Source(src_hash, path)` の列と、除外した `(path, 理由)` 列。`src_hash` は `utt_id` の接頭辞になる。
- **補助関数**（後段から呼ばれる）: `decode` / `decode_range`（範囲デコード）/ `probe_duration`（ffprobe で尺）/
  `resample_to` / `load_cached`（`.cache` から読む）。

## Stage 1: 音声セグメント化（`stage1_segment.py`）

- **目的**: 「1 行＝1 文」契約どおり、発話単位にスライスして中間キャッシュへ書く。
- **入力**: `Source` 列。
- **処理**:
  1. ソースを 1 本ずつ 44.1kHz モノラルにデコード（**デコードは 1 回だけ**、16k はそこからリサンプル）。
  2. **長尺ソースはチャンク分割デコード**（`decode_chunk_min`、既定 10 分）。チャンク境界で VAD の
     判定が全体デコードとズレないよう前後に広い文脈を取り、発話は「開始時刻」でチャンクに一意に
     割り当てる（取りこぼし・重複なし）。
  3. **Silero-VAD** で音声区間を検出（`vad_threshold` / `min_silence_duration_ms` /
     `min_speech_duration_ms` / `max_speech_duration_s`）。長い塊は破棄せず VAD 側で分割。
  4. **エネルギーベース境界補正**: 騒音床（セグメント周辺のエネルギー分布から推定）を使い、
     始端＝最初の話声フレーム／終端＝最後の話声フレームへ寄せる（語頭・語尾は削らない）。
     補正後の境界が極端に縮んだ場合は元境界へ退避する。
  5. 補正後の境界へ `output_pad_ms`（既定 20ms）の**無音マージンを前後に足す**。
     `speech_pad_ms`（VAD 側の切り出し拡張）は補正窓の内側でトリムされて出力に残らないので、
     納品音声のマージンはこのノブで決める（根拠: DECISIONS 2.6）。
  6. 補正後の境界で `.cache/<utt_id>.flac`（44.1k）に書き出す。ASR・音響スコアも同じ音声から作る。
- **スキップ記録**: 物理下限 `cache_min_sec`（既定 0.2 秒）未満はキャッシュを作らず
  `scores.stage1_skipped="too_short"` だけ残す。ソース処理の例外は `source_error` として残す
  （**どちらも Stage 6 で常に不合格**、音声キャッシュが無い）。
- **出力**: `Record` 列（`utt_id`/`src`/`start`/`end`/`scores.seg_dur` を設定）＋ `.cache/*.flac`。
- **設定**: `[stage1]`（`vad_threshold`, `min_silence_duration_ms`, `min_speech_duration_ms`,
  `max_speech_duration_s`, `speech_pad_ms`, `refine_window_ms`, `output_pad_ms`, `cache_min_sec`,
  `decode_chunk_min`）。

## Stage 2: 発話スコアリング（`stage2_score.py`）

- **目的**: 発話ごとの**音響スコア**を `scores` に積み上げる（判定しない）。
- **入力**: `Record` 列 ＋ `.cache`。
- **処理と生成スコア**:

| スコア | 何を見るか | 検出対象 |
|---|---|---|
| `clip_ratio` / `dc_offset_abs` | クリッピング率／直流オフセット | 学習後に修復不能な破綻 |
| `digital_silence_ratio` | 完全ゼロサンプル率 | 無音スライス混入 |
| `rolloff_hz` / `hf_energy_ratio` | スペクトルロールオフ／7kHz 以上エネルギー比 | 16k/24k アップコンの見せかけ高品質 |
| `snr` / `voiced_ratio` | フレーム RMS を絶対騒音床（`min(p20×3, -55dBFS)`）で分割 | 無音率・有声率（ノイズ/咳のみ） |
| `lufs` | pyloudnorm 統合ラウドネス（メータのブロック長未満は `None`） | 極端な大小 |
| `jitter_local` / `shimmer_local` / `f0_median_hz` | 有声区間内の Praat 風 周期/振幅摂動・F0 中央値 | 嗄れ・叫び・異常ピッチ |
| `reverb_score` | 末尾減衰エネルギー比（残響プロキシ） | ホールかかり |
| `overlap_flag` / `overlap_sec` | `[overlap]` 無効の間は常に `None`。有効時は pyannote segmentation-3.0 が検出した 2 人同時発話の有無・秒数 | 2 人同時発話 |
| `emb` | MFCC 統計を次元スケーリング（CMVN の分散側）した話者ベクトル | `speaker_sim` の材料（Stage 6 で使用） |
| `fp_sig` | MFCC フレーム 2 値化→3 フレーム gram→MinHash 署名 | **内容**指紋（重複排除・多様性で使用） |

- **注意**: `speaker_sim` はここでは作らない（Stage 6 直前に `emb` から頑健 centroid で計算）。
  1 本の異常でランを止めない（例外は `scores.errors` に集約、非有限サンプルも `errors`）。
  指紋計算の失敗は `fp_error` に寄せる（dedup が弱まるだけでレコードは道連れにしない）。
- **設定**: `[overlap]`（オプトインの pyannote 2 人同時発話検出。`enabled`/`model`/`token`/`min_overlap_sec`）。
  `[runtime] device` だけ間接的に使用。

## Stage 3: 文字起こしと言語系チェック（`stage3_asr.py`）

- **目的**: セリフを確定し、G2P 整合・テキスト品質のスコアを記録する。
- **入力**: `Record` 列 ＋ `.cache`（16k で読む）。
- **ASR バックエンド**（`[asr] backend`）:
  - `qwen3-asr`（既定, `Qwen/Qwen3-ASR-1.7B-hf` を transformers native で実行、トークン平均確率を信頼度に）
  - `faster-whisper`（CPU 軽量代替、`model` は `"small"`/`"tiny"` 等。`no_speech_prob`/`compression_ratio`/単語確率も取る）
- **処理と生成スコア**:
  1. テキスト正規化（NFKC、区切り `|` を `・` に保護、**記号は `[text] symbol_readings`（既定 true）で
     日本語読みへ展開**（`%`→「パーセント」等、itn.py の対応表。展開後も残る記号は `residual_symbols` に記録）、
     装飾記号除去、`♪` 等は除去前に検出して `music_flag`）。
  2. **G2P 整合①**: `pyopenjtalk-plus`（学習側と同一エンジン）が通るか → `g2p_ok` / `unknown_tokens` /
     `mora_count`。unknown は**学習側の読みチェック（YomiError）で実際に落ちる語**（pron=「、」の読めない語。
     希少漢字・古いカタカナ等）の数で、語自体は `yomi_bad_words` に残す。`yomi.py` が学習側の正規化＋
     失敗条件を鏡写し（学習側は絵文字等を解析前に消すため、同じ正規化を通してから打診し誤爆を防ぐ）。
  3. **G2P 整合②**: 話速 `mora_rate = mora_count / 実音声長`。話者中央値からの乖離 `mora_rate_dev` は
     `flag_mora_rate_outliers` が Stage 6 直前に算出（中央値はクリーンな発話のみで計算）。
  4. **G2P 整合③**: アライメント信頼度 `align_conf`（whisper は単語確率、qwen はトークン確率を流用）。
  5. テキストフィルタ用: `char_count` / `kana_ratio`（分母は**かな＋英数字**のみ。漢字・句読点で薄めない）/
     `alpha_only` / `filler_ratio`（フィラーの**出現回数**ベース）/ `repeat_block`（同一文言 3 連呼以上のブロック）/
     `music_flag`。
  6. **文末の開き**: `text_tail_open` を 0/1/2 で記録（G2P と同じ `run_frontend` 結果を流用する）。
     **2**=文末記号が全く無い、または動詞・形容詞・助動詞の読みが促音で止まる（「お客様の体温と混じっ」型）。
     **1**=条件節「〜したら。」・連用形中止「〜だし。」・連体止め「〜ような。」。**0**=閉じた文。
     助詞止め（「今から。」）は検出しない。理由文字列は `text_tail_open_reason`（深刻度の定義と実測: DECISIONS 3.9）。
  7. **トークン確率分布**（両バックエンド）: 平均だけでなく `token_prob_min`（最悪トークン）/
     `token_prob_p10`（下位 1 分位・可視化専用）/ `token_nll`（平均負対数尤度）を記録。
  8. **[asr2] 有効時**: 第 2 ASR（別系統推奨）を同じ発話に走らせ、2 系統間 CER を `x_cer` に記録する。
     比較は表層文字でなく**読みレベル**（`reading_key`）。表層値は参考値として `x_cer_surface`、
     正規化の版数は `x_cer_algo`。メインと同じ backend+model だと起動時に警告。
     明らかなゴミ発話（0.3 秒未満・無音率 0.9 超・クリップ 5% 超・SNR -5dB 未満）は推論をスキップして
     `asr2_skipped="obvious_garbage"`（カットオフは固定値）。失敗は `asr2_errors`。
  9. **[alignment] 有効時**: CTC 強制アライメント（`align.py`）で `ctc_align_score`（経路平均後部確率）と
     `unaligned_edge_ratio`（先頭/末尾の「テキストに無い発話」比）を記録。失敗は `align_errors`。
  10. **[lm] 有効時**: MLM 擬似 PPL（`lm.py`）で `lm_ppl` を記録。失敗は `lm_errors`。
- **生成スコア**: `asr_conf`, `align_conf`, `token_prob_min`, `token_prob_p10`, `token_nll`, `g2p_ok`,
  `unknown_tokens`, `yomi_bad_words`, `mora_count`, `mora_rate`, `mora_rate_median`, `mora_rate_dev`,
  `char_count`, `kana_ratio`, `alpha_only`, `filler_ratio`, `repeat_block`, `music_flag`,
  `residual_symbols`, `text_tail_open`, `text_tail_open_reason`
  （whisper のみ `no_speech_prob`, `compression_ratio`。
  オプトイン: `x_cer`, `x_cer_surface`, `x_cer_algo`, `asr2_text`, `asr2_conf`, `asr2_skipped`,
  `ctc_align_score`, `unaligned_edge_ratio`, `lm_ppl`）。
- **設定**: `[asr]`（`backend`, `model`, `language`, `beam_size`, `batch_size`, `compute_type`, `device`）、
  `[asr2]`（第 2 系統。`enabled` でオプトイン）、`[alignment]`（`enabled`, `model`）、
  `[lm]`（`enabled`, `model`）、`[text]`（`symbol_readings`）。

## Stage 4: 音声品質スコア（`stage4_quality.py`）

- **目的**: 知覚／客观品質を記録する。
- **入力**: `Record` 列 ＋ `.cache`（16k、SQUIM の 25ms フレームに切り揃える）。
- **処理**: torchaudio `SQUIM_OBJECTIVE` → `stoi` / `pesq` / `si_sdr`、および集約先 `quality_score`
  （現状 `= pesq`、UTMOS2 差し替え時に置換）。
  - **[stage4] subjective 有効時**（オプトイン）: `SQUIM_SUBJECTIVE` の総合 MOS を `mos` に記録。
    コーパス内で SNR 最良の発話をアンカーにするため、`mos` は**その話者自身の最クリーン発話に対する
    相対品質**（絶対 MOS ではない）。DNSMOS/NISQA を差し替えても同じ `min_mos` ゲート枠が使える。
  - **[tags] 有効時**（オプトイン）: AudioSet タグ（AST）で音楽・笑い・咳・拍手などを**音声から直接**検出し、
    `audio_tags`（上位タグと確率）と `tag_hit`（`reject` リストのタグが `min_prob` 以上でヒット）を記録。
    10 秒窓ごとに推論し、窓間は max-pool（一部分だけ音楽でも検出したい）。
- **生成スコア**: `stoi`, `pesq`, `si_sdr`, `quality_score`（オプトイン: `mos`, `audio_tags`, `tag_hit`）。
  失敗は `mos_errors` / `tag_errors` に寄せる（`errors` に入れてレコードを道連れにしない）。
- **設定**: `[runtime] device`（GPU があれば CUDA で実行）、`[stage4]`（`subjective`）、
  `[tags]`（`enabled`, `model`, `min_prob`, `reject`）。

## (任意) 除去修復（`repair.py`・`[repair]` 有効時のみ、Stage 4 と Stage 6 の間）

- **目的**: ノイズ系ゲート（既定 `targets`: `min_snr_db` / `min_quality_pesq` / `min_stoi` / `min_si_sdr`）
  **単独**で落ちた発話だけ DeepFilterNet で除去し、音響/品質スコアを引き直す。
  他の理由（無音スライス・G2P 不通等）と併せ落ちの発話は対象にしない。
- **約束**: 再検証を通過しても**自動採用せず `pending_review` 行き**。元音声は `.cache_orig/` に保持。
  `emb` はコーパス CMVN スケール済みのため部分集合再スコアで壊れる → バックアップして復元
  （話者本人性は除去で大きく変わらない前提）。
- **導入**: deepfilternet は numpy<2 pins のため lock 外。
  `uv pip install --no-deps deepfilternet deepfilterlib loguru appdirs requests sympy`。

## Stage 5: 重複排除（`stage5_dedup.py`）

- **目的**: ほぼ同一の発話を代表 1 本にまとめる（安い順に 4 段）。
- **入力**: `Record` 列（`text` と `fp_sig` を使う）。
- **処理**: ①テキスト完全一致 → ②テキストファジィ（shingle Jaccard ≥ `text_jaccard`）→
  ③音素三量文字列 Jaccard ≥ `phoneme_jaccard` → ④音声内容指紋 `fp_sig` 一致率 ≥ `audio_jaccard`
  （ズレ切り出し重複）。候補は MinHash/LSH・band 衝突で絞り、Union-Find でグループ化。
  代表は **閾値通過者 → quality → asr_conf → snr** の決定式で選び、他は `dup_reason` を付ける。
- **出力**: `dup_group_id` / `dup_reason`（除外確定は Stage 6）。
- **設定**: `[dedup]`（`text_jaccard`, `phoneme_jaccard`, `audio_jaccard`）。

## Stage 5b: 多様性選別（`stage5_dedup.diversity_select`）

- **目的**: 重複ではないが「似た内容」の発話も間引いて、データセットの多様性を上げる。
- **入力**: 重複排除後に残った候補（閾値通過かつ代表）。
- **処理**: 候補を**品質順**に貪欲選択し、**既に採用済みのどれかと類似しすぎる発話を除外**
  （採用集合の総対ペア類似度を `sim_threshold` 未満に抑える）。類似は**内容の近さ**で測る
  （`metric`: `phoneme` 既定 / `text` / `audio`。話者/音響の `emb` は不採用）。
  `protect_coverage` が真の間、ある音素の唯一の供給源は `similar` / `cap` / `target_size` のいずれでも落とさない。
  追加ノブ（既定 0 = 無効 = 従来の動作）: `gain_lambda`（選択スコアに「新規音素 three-gram 利得」を加算して
  多様性を**最大化**に切り替える）、`max_bucket_share`（F0×話速の三分位バケット別シェア上限。計測不能発話は免除）、
  `max_source_share`（ソース 1 本の採用シェア上限）、`target_count`（目標本数で打ち切り）。
- **出力**: `diversity_reason`（`similar:<代表utt>` / `cap:prosody:<バケット>` / `cap:source:<src>` / `target_size`）。
  選別前後の多様性メトリクス（平均ペア類似度・音素三量文字カバレッジ・ヒストグラム）は `report.html` に出る。
- **設定**: `[diversity]`（`enabled`, `metric`, `sim_threshold`, `protect_coverage`, `gain_lambda`,
  `max_bucket_share`, `max_source_share`, `target_count`）。

## Stage 6: 判定・出力（`stage6_output.py`）

- **目的**: ここで初めて閾値で合否を決め、納品物を書き出す。`run` と `rescore` で共通。
- **処理（この順）**:
  0. **`apply_text_overrides`**: `Data/<話者>/text_overrides.jsonl` があれば読み、テキスト上書きは
     **派生スコア（モーラ数・話速・kana 比率等）を引き直したうえで** dedup・話速中央値より前に反映する。
     `status` 上書きは finalize で自動判定より優先（後述）。
  1. **`recompute_speaker_sim`**: `emb` から主話者 centroid を頑健サブセット（2 パス中央値）で再計算 →
     `speaker_sim`。不良発話で centroid が汚染されない。emb が 3 件未満なら `None`（検証不能）。
  2. **`flag_mora_rate_outliers`**: クリーンな発話だけで話速中央値を再計算 → `mora_rate_dev`。
  3. **`evaluate_checks`**: 各発話を閾値で判定し `failed_checks`（閾値キー単位）を作る。
     - `pipeline_error`（`scores.errors`）、`stage1_skipped`（`too_short`/`source_error`）は常に不合格。
     - `CHECKS` の各ゲート（下表）。`min/max` が同じスコアを見ても閾値キー単位で記録するので
       感度分析が混線しない。
     - 話速レンジ（`min_mora_rate`/`max_mora_rate`。モーラ数が取れない発話はスキップ）。
     - **除外ゲート**: `max_lm_ppl`（既定有効 200）。**見送り行きゲート**: `review_lm_ppl` / `review_x_cer`
       （超過を除外ではなく `pending_review` に回す。既定はコメント-out）。
     - **空テキスト不変条件**: 空セリフ／モーラ 0 は閾値と独立に常に不合格（`empty_text`）。
     - `on_g2p_fail` / `on_unknown_tokens` の扱い（`exclude` / `dict_data`＝見送り行き）。値の誤字は起動時に `ValueError`。
  4. **`dedup` → `diversity_select`**（Stage 5/5b をここで実行）。
  5. **`finalize_statuses`**: `failed_checks` + `dedup:*` + `diversity:*` で `status`/`dropped_by` を確定。
     `review_reason` があれば `pending_review`（`[repair]` で修復された発話は再検証を通過していても
     `review_reason="repaired"` として `pending_review` に回す）。最後に `text_overrides.jsonl` の
     `status_override`（`scores.status_override`）を**自動判定より優先**で最終反映する。
  6. **`write_outputs`**: 契約ガード（**44.1kHz・モノラル固定**、`[output]` が逸脱なら書出し前に `ValueError`）
     → LUFS 正規化＋ピークリミッタ（-1dBFS 超過でスケールし `limited` / `lufs_out` を記録）
     → `raw/*.flac` + `esd.list` + `raw.zip` → 納品前に全ファイルへ契約検証を自走で通す。
     納品用 `raw/` は毎回作り直す（前回の除外済みファイルが混ざらないため）。
  7. **`write_manifest`**（`scores` 含む全レコード）→ **`write_report`**（`report.html`。HTML エスケープ済み）
     → **`write_run_meta`**（`run.json`）。人手レビュー用のページは作らない。
- **`report.html` の中身**（ダーク基調・外部 CDN 依存なしの単一 HTML。表はヘッダでソート、検索ボックスで絞り込み）:
  - **サマリー**: 採用/除外/見送りの本数・採用率・採用音声の合計尺・重複グループ数・多様性間引き数・
    Stage 1 スキップ数・ピークリミッタ作動数と、status の積み上げバー。
  - **見方・凡例**: status の意味、失格理由キーの読み方（`min_*`/`max_*`/`reject_*`、「検証不能＝不合格」）、
    感度表・多様性メトリクス・x_cer と文末の開きの読み方、rescore / `text_overrides.jsonl` での動かし方、
    分類カラー chip の凡例。
  - **除外内訳**: 全失格理由を、分類 chip・日本語の見方・件数バーつきで（個体 ID を含むキーは丸めて集計）。
  - **ゲート別の該当本数と閾値感度**: 有効なゲートを 1 枚に並べ、現在値・該当本数・緩和/厳格化で
    ひっくり返る本数（単独失格のみ数える近似）を併記。
  - **重複排除と多様性選別**: 選別前後の多様性メトリクス（平均ペアwise類似度・音素三量被覆率）と
    採用集合の話速/F0/発話長ヒストグラム。
  - **文末の開き**の分布と severity 2・1 の一覧。
  - **2 系統 ASR の不一致**の分布と、メイン ASR⇄第 2 ASR の出力対比一覧（食い違ったものだけ。
    帯ごとの第 2 系統確信度中央値つき）。
  - **カバレッジ**: 出現音素数・希少音素・F0 レンジ・モーラ数/話速/発話長ヒストグラム。
  - **見送り（`pending_review`）**: 自動採用しなかった一覧（第 2 系統の出力・`asr_conf`/`x_cer`/`lm_ppl`/読めない語つき）。
  - **投入ソース別件数 / Stage 1 スキップ / ピークリミッタ**、**読めない語**。

---

## スコア → ゲート対応表

「どの Stage が作ったスコアを、どの閾値キーが弾くか」の一覧。**既定**は同梱 `filters.toml` の状態
（●=既定有効、○=コメントで用意＝既定無効、―=キー自体が無い）。スコアが `None` でゲートが有効な発話は
「検証不能＝不合格」。

| 生成 Stage | スコア | ゲートキー（方式） | 既定 | 検出対象 |
|---|---|---|---|---|
| 1 | `seg_dur` | `min_seg_sec`(min) / `max_seg_sec`(max_excl) | ● 0.5 / 12.0 | 発話長 |
| 1 | `stage1_skipped` | （閾値なし・常に不合格） | ● | 短すぎ/ソース失敗 |
| 2 | `clip_ratio` | `max_clip_ratio`(max) | ● 0.002 | クリッピング |
| 2 | `dc_offset_abs` | `max_dc_offset`(max) | ● 0.02 | DC オフセット |
| 2 | `digital_silence_ratio` | `max_digital_silence`(max) | ● 0.5 | 無音スライス |
| 2 | `hf_energy_ratio` | `min_hf_energy_ratio`(min) | ● 0.0005 | アップコン見せかけ |
| 2 | `rolloff_hz` | `min_rolloff_hz`(min) | ○ | 帯域不足 |
| 2 | `snr` | `min_snr_db`(min) | ● 5.0 | SNR 低 |
| 2 | `voiced_ratio` | `min_voiced_ratio`(min) | ● 0.1 | 有声率极低 |
| 2 | `lufs` | `min_input_lufs`(min) | ○ | 極端に小さい入力 |
| 2 | `jitter_local` | `max_jitter_local`(max) | ● 0.02 | 周期摂動（嗄れ） |
| 2 | `shimmer_local` | `max_shimmer_local`(max) | ○ | 振幅摂動 |
| 2 | `f0_median_hz` | `min_f0_hz`(min) / `max_f0_hz`(max) | ○ | 異常ピッチ |
| 2 | `reverb_score` | `max_reverb_score`(max) | ● 0.5 | 残響 |
| 2 | `overlap_flag` | `reject_overlap`(flag) | ○（`[overlap]` 有効時のみ計測。無効は None で素通り） | 2 人同時発話 |
| 3 | `asr_conf` | `min_asr_conf`(min) | ● 0.30 | ASR 信頼度低 |
| 3 | `align_conf` | `min_align_conf`(min) | ● 0.30 | 音声とテキスト不一致 |
| 3 | `mora_count` | （`empty_text` 不変条件） | ● | モーラ 0 |
| 3 | `mora_rate` | `min_mora_rate`/`max_mora_rate`(range) | ● 2.0 / 14.0 | 話速の絶対値 |
| 3 | `mora_rate_dev` | `max_mora_rate_dev`(max) | ● 1.2 | 話速の話者中央値からの乖離 |
| 3 | `g2p_ok` | `on_g2p_fail`(action) | ● exclude | G2P 不通（学習側読みチェック YomiError 事前打診を含む） |
| 3 | `unknown_tokens` | `on_unknown_tokens`(action) | ● exclude | 学習側で読めない語（pron=「、」。語は `yomi_bad_words`） |
| 3 | `char_count` | `min_chars`(min) | ● 4 | 文字数 |
| 3 | `kana_ratio` | `min_kana_ratio`(min) | ● 0.5 | 日本語らしさ |
| 3 | `alpha_only` | `reject_alpha_only`(flag) | ● | アルファベットのみ |
| 3 | `filler_ratio` | `max_filler_ratio`(max) | ● 0.6 | フィラー率高 |
| 3 | `repeat_block` | `reject_repetition`(flag) | ● | 同一文言の連呼 |
| 3 | `music_flag` | `reject_music`(flag) | ● | 音楽記号 |
| 3 | `no_speech_prob` | `max_no_speech_prob`(max) | ○（whisper 専用） | ハルシネーション |
| 3 | `compression_ratio` | `max_compression_ratio`(max) | ○（whisper 専用） | 繰り返しガチャ |
| 3 | `token_prob_min` | `min_token_prob`(min) | ● 0.25 | 平均が高くても 1 語だけ怪しい |
| 3 | `token_prob_p10` | （キーなし・可視化専用） | ― | 低確率トークンの分布 |
| 3 | `token_nll` | `max_token_nll`(max) | ● 0.3 | ASR 信頼度（分布全体） |
| 3 | `x_cer` | `max_x_cer`(max) ／ `review_x_cer`(見送り行き) | ● 0.45（`[asr2]` 既定有効。**読みレベル**で比較） | 2 系統 ASR の不一致 |
| 3 | `x_cer_surface` | （キーなし・参考値） | ― | 表層文字での不一致（ゲートに使わない） |
| 3 | `ctc_align_score` | `min_ctc_align`(min) | ○（`[alignment]` 既定有効で記録のみ。**ゲート化は実測で取りやめ**） | 音声↔テキストの実一致 |
| 3 | `unaligned_edge_ratio` | `max_unaligned_edge`(max) | ○（同上。境界ズレは x_cer で取る） | 先頭/末尾の発話混入・境界ズレ |
| 3 | `lm_ppl` | `max_lm_ppl`(max) ＋ `review_lm_ppl`(見送り行き) | ● 200（`[lm]` 既定有効。完全自動） | 文脈的に不自然な文字起こし |
| 3 | `residual_symbols` | `max_residual_symbols`(max) | ○ | 読み上げ展開後も残る記号 |
| 3 | `text_tail_open` | `max_text_tail_open`(max) | ● 1 | 文末の開き（2=途中で切れ、1=言い淀み止め、0=閉じた文） |
| 4 | `pesq`/`quality_score` | `min_quality_pesq`(min) | ● 1.5 | 知覚品質 |
| 4 | `stoi` | `min_stoi`(min) | ○ | 明瞭度 |
| 4 | `si_sdr` | `min_si_sdr`(min) | ○ | 分離度 |
| 4 | `mos` | `min_mos`(min) | ○（`[stage4] subjective` が必要） | 総合 MOS（最クリーン発話に対する相対値） |
| 4 | `tag_hit` | `reject_audio_tags`(flag) | ● true（`[tags]` 既定有効） | 音楽/笑い/咳を音声から直接検出 |
| 6 | `speaker_sim` | `min_speaker_sim`(min) | ● 0.5 | 別人混入 |

ゲートキー以外に、**閾値と独立の不変条件**として `pipeline_error` / `too_short` / `source_error` /
`empty_text` があり、常に不合格。`dedup:*` / `diversity:*` は Stage 5/5b、`manual` は
`text_overrides.jsonl` の `status="excluded"` 由来。

> `no_speech_prob` / `compression_ratio` は faster-whisper 専用。qwen3-asr の間に有効化すると
> スコア欠損で全件落ちるため既定コメント。
>
> ○ のうち `[alignment]` / `[lm]` / `[stage4]` / `[tags]` / `[overlap]` 系は**プロバイダを有効にした `run` の後に限り**
> ゲートをコメント解除できる（`rescore` だけではスコアが manifest に無く「検証不能＝不合格」で全件落ちる。
> `run` 起動時は「ゲート有効なのにプロバイダ無効」を警告する）。
> **`[asr2]` だけが例外**で、`rescore` は `.cache/` から第 2 ASR を走らせて `x_cer` を追加計算する
> （`refresh_asr2_scores`。同じ `x_cer_algo` 版数を持つレコードは触らないので閾値の締め直しは数秒で終わる。
> `reading_key` を変えたら `X_CER_ALGO` を上げるので、古い正規化で計算された値は自動的に引き直される。
> asr2 の**モデル**を変えた場合は検出できないので `run` を使う）。
> 同梱 `filters.toml` では `[asr2]` / `[tags]` / `[lm]` / `[alignment]` が既定有効なので、
> `max_x_cer`・`reject_audio_tags`・`max_lm_ppl` はそのまま動作する
> （旧 manifest への `rescore` では `tag_hit` 欠損にだけ注意）。
> `review_lm_ppl` と `review_x_cer` は CHECKS ではなく `evaluate_checks` 内の見送り閾値で、
> 超過を `review_reason` に入れて `pending_review` に送る。

## 重量検証（オプトイン）の一覧

追加のモデル実行を要する検証軸は、有効化したときだけ `scores` にスコアを**記録**し、
対応ゲート（`[thresholds]`）が合否を決める。採用・棄却の判断経緯は [DECISIONS.md 4 章](DECISIONS.md#4-重量検証オプトインの採用と棄却)。

| 節 | 有効化で増える検証 | 記録スコア | ゲートキー | 既定 | 初回 DL |
|---|---|---|---|---|---|
| `[asr2]` | 第 2 ASR とのクロス検証（別系統を推奨。明らかなゴミ発話は自動スキップ。`x_cer` は**読みレベル**、表層値は `x_cer_surface`） | `x_cer`, `x_cer_surface`, `x_cer_algo`, `asr2_text`, `asr2_conf`, `asr2_skipped` | `max_x_cer` 0.45 / `review_x_cer`（コメント-out） | **オン**（faster-whisper small。run 済みでも `rescore` で x_cer だけ追加計算可） | 選択モデル次第（small で約 0.5GB） |
| `[tags]` | AudioSet タグ（音楽・笑い・咳・拍手を**音声から直接**検出。AST） | `audio_tags`, `tag_hit` | `reject_audio_tags` | **オン（ゲートも有効）** | 約 1.3GB |
| `[lm]` | MLM 擬似パープレキシティ（発音が変わる誤変換の検出。DeBERTa-v2 char-wwm） | `lm_ppl` | `max_lm_ppl`（除外・既定有効 200）/ `review_lm_ppl`（コメント-out） | **オン（除外ゲート有効）** | 約 1GB |
| `[alignment]` | CTC 強制アライメント（text↔audio の実一致＋先頭/末尾の発話混入検出） | `ctc_align_score`, `unaligned_edge_ratio` | `min_ctc_align` / `max_unaligned_edge`（コメント-out） | **オン（記録のみ。ゲート化は見送り）** | 約 1.2GB |
| `[stage4] subjective` | SQUIM_SUBJECTIVE 総合 MOS（SNR 最良発話をアンカーにした**相対値**） | `mos` | `min_mos` | オフ | 約 360MB |
| `[overlap]` | pyannote segmentation-3.0 で 2 人同時発話（`uv sync --extra extras` + HF_TOKEN の規約同意が必要） | `overlap_flag`, `overlap_sec` | `reject_overlap` | オフ | 約 25MB |
| `[repair]` | DeepFilterNet 除去修復（指定ゲート**単独**で落ちた発話だけ。通過しても自動採用せず `pending_review` 行き。元音声は `.cache_orig/`） | `repaired` | —（自動採用しない＝見送り） | オフ | 約 100MB |

## `rescore` の流れ（音声処理・ASR を再実行しない再生成）

```
make_dataset rescore Data/<話者> --config filters.toml
  → manifest.jsonl を読む（scores / emb / fp_sig / text が全部入っている）
  → text_overrides.jsonl があれば先頭で再適用（人手修正を判定に反映）
  → refresh_yomi_scores / refresh_text_tail_scores（テキストから引き直す。音声・モデル実行不要）
  → [asr2] が有効なら refresh_asr2_scores（x_cer_algo が違うレコードだけ .cache から第 2 ASR）
  → .cache/<utt>.flac から音声だけ再取得（LUFS 正規化のため）
  → Stage 6 相当だけ回す: recompute_speaker_sim → flag_mora_rate_outliers
      → evaluate_checks → dedup → diversity_select → finalize → 書き出し
```

`speaker_sim` の centroid・話速中央値・重複排除・多様性選別はすべて manifest の
`emb`/`fp_sig`/`text` から**再計算**されるので、`filters.toml` の閾値も `[dedup]`/`[diversity]` も
`rescore` だけでその場に変更を反映できる。読みチェック（`unknown_tokens`/`yomi_bad_words`）と
文末の開き（`text_tail_open`）も `text` から引き直す（旧 manifest の死んだチェックも自動更新され、
ゲートを有効にして rescore したときに「検証不能」で全件落ちる罠を回避する）。
`[asr2]` は例外的に `rescore` でも走る（run と違いメイン ASR・AST・CTC・LM はやり直さない）。
ただし `[alignment]`/`[lm]`/`[stage4]`/`[tags]`/`[overlap]`/`[repair]` は
**音声・モデル実行を伴うので `run` 側でしか走らない**（`rescore` は manifest にあるスコアを流用するだけ）。

## 生成物（`Data/<モデル名>/`）

| ファイル | 内容 |
|---|---|
| `esd.list` | 1 行 1 文 `<hash>.flac|<話者>|JP|<セリフ>`（raw/ 直下のフラット配置） |
| `raw.zip` | 44.1kHz・モノラル・LUFS 正規化済み flac（納品前に同じ契約検証を自走で通す） |
| `manifest.jsonl` | 全スコア・全失格理由・dup/多様性・status（1 行=1 発話、非有限値は null） |
| `report.html` | 自動判定の内訳＋見方・凡例（上記）。ダーク基調の単一 HTML、外部 CDN 依存なし、表はソート・検索可 |
| `text_overrides.jsonl` | 判定の確定・本文修正の返し先（入力扱い。`utt_id` キーのテキスト/ステータス上書き。あれば常に再適用） |
| `run.json` | 再現性メタデータ（使用した閾値・ASR モデル・ライブラリ版・commit） |
| `.cache/` | 中間音声キャッシュ（セグメント別 44.1k flac。`rescore` の入力。納品物ではない） |
| `.cache_orig/` | `[repair]` で修復した発話の元音声（既定では作られない） |

## 設定セクション（`filters.toml`）

| セクション | 何を制御するか |
|---|---|
| `[paths]` | voice ルート |
| `[output]` | 出力契約（`sample_rate`/`channels` は 44100/1 固定、`target_lufs`） |
| `[runtime]` | 推論 device（`auto`/`cpu`/`cuda`） |
| `[stage1]` | VAD・境界補正・出力マージン・キャッシュ下限・チャンク分割 |
| `[asr]` | ASR バックエンド・モデル・言語 |
| `[asr2]` | 第 2 ASR クロス検証（既定有効。`x_cer` を**読みレベル**で記録） |
| `[stage4]` | SQUIM_SUBJECTIVE の `mos`（オプトイン） |
| `[tags]` | AudioSet タグ付け（**既定有効**。AST。`reject` タグリストと `min_prob`） |
| `[overlap]` | pyannote 2 人同時発話検出（オプトイン。任意依存 + HF_TOKEN） |
| `[alignment]` | CTC 強制アライメント（**既定有効**。スコア記録のみ、ゲートはコメント-out） |
| `[lm]` | MLM 擬似 PPL（**既定有効**。除外ゲート `max_lm_ppl` = 200 を有効） |
| `[text]` | 記号→読み上げ展開（`symbol_readings`、既定 true） |
| `[repair]` | DeepFilterNet 除去修復（オプトイン。対象ゲートリスト。lock 外依存） |
| `[thresholds]` | 全ゲートの閾値（キーを消せば無効） |
| `[dedup]` | 重複排除の類似しきい値 |
| `[diversity]` | 多様性選別の有効化・物差し・しきい値・カバレッジ保護・シェア上限/多様性利得/目標本数 |
