# 判断の経緯・実測根拠（DECISIONS）

**この文書の位置づけ**

- 使い方（コマンド・ユースケース）: [../チートシート.md](../チートシート.md) → [README.md](../README.md)
- 各ステージが今何をするか（実装リファレンス）: [PIPELINE.md](PIPELINE.md)
- 設計思想と実装状況の一覧: [DESIGN.md](DESIGN.md)
- **この文書**: なぜその仕様・その閾値にしたか、何を測って何を棄却したか のアーカイブ

コード中のコメントには「今そう動く理由」の最小限だけを残し、実測値・比較実験・棄却した案はここに集約した。
設定や実装を動かすときは、対応する節の「実測」を読んでからにする（同じ失敗を繰り返さないため）。

コードコメントに残っている旧レビュー番号（`A-1` / `B2` / `C13` / `F9` など、過去ラウンドの指摘 ID）の対応表は
[付録: 旧レビュー番号対応表](#付録-旧レビュー番号対応表) に置く。

---

## 1. 運用方針

### 1.1 人手レビューページを作らず、`text_overrides.jsonl` 1本に寄せる（C6）

- **結論**: 採用/除外は機械の自動判定だけで決め、人が関与する入口は
  ① `filters.toml [thresholds]` を動かす、② `Data/<話者>/text_overrides.jsonl` に `utt_id` キーで
  テキスト／ステータスを書く、の 2 つだけにした。Gradio 等のレビュー UI は作らない。
- **経緯**: 音声プレイヤー付きレビューページは「その場で 1 回しか使えない」上、判定が画面の外に
  残らず再現性が失れる。同じ情報を `manifest.jsonl`（全スコア）＋ `report.html`（判断材料）＋
  `text_overrides.jsonl`（確定事項）に落としておけば、レビューは rescore のループになる。
- **適用順序の約束**: テキスト上書きは**派生スコア（モーラ数・話速・kana 比率・repeat_block 等）を
  引き直したうえで**、dedup・話速中央値・多様性選別より**前**に適用する。順序を守らないと
  「修正後テキストでの重複判定」と「旧テキストの話速中央値」が混ざって壊れる。
  `status` 上書きは `finalize_statuses` の最終段で自動判定より常に優先する。
- **音声の聞き方**: レポートには埋め込まず `.cache/<utt_id>.flac`（納品済みなら `raw/`）を直接開く。
- **コード**: `overrides.py` / `stage6_output.finalize_statuses`

### 1.2 閾値にコード側デフォルトを置かない（E2）

- **結論**: `[thresholds]` だけ `DEFAULTS` から除外し、`filters.toml` を唯一の閾値ソースにしている。
  キーを消す（コメント化する）＝そのチェックは走らない。
- **理由**: TOML には `None` を書けないので、「キーを消す＝無効」が唯一の無効化手段になる。
  コード側に同じキーの既定値を残すと、設定ファイルを削ったときに黙ってゲートが効き続ける。
- **挙動パラメータ**（`[stage1]` / `[asr]` / `[dedup]` / `[diversity]` など）は `DEFAULTS` に置いてよい。
  閾値ではなく動作パラメータなので、既定値があっても設定の意図は崩れない。
- **コード**: `config.py` の `DEFAULTS` / `Config.thresholds`

### 1.3 「検証不能＝不合格」の方針（A-1 / A-2）

- **結論**: 閾値が有効なのにスコアが `None`（プロバイダ未実行・計算失敗・非有限値）の発話は不合格にする。
- **理由**: `nan < 閾値 == False` で「異常値ほど通る」のを防ぐ。`manifest.jsonl` には非有限値を
  記録時点で `null` 化して書き、JSON として読める形に保つ（`allow_nan=False`）。
- **副作用と対処**: 新しいゲートをコメント解除して `rescore` すると、そのスコアを持たない
  旧 manifest は**全件落ちる**。`run` 起動時に「ゲート有効なのにプロバイダ無効」を警告し、
  rescore 側も `x_cer` 欠損を明示する（黙って全件落とさない）。
- **コード**: `manifest._finite_or_none` / `stage6_output._check_failed` / `cli._warn_config`

### 1.4 `run` と `rescore` の責務分担（E1）

- **結論**: `rescore` は `manifest.jsonl` ＋ `.cache/` だけから納品物を再生成する（閾値再判定専用）。
  音声処理・ASR・AST・CTC・LM は再実行しない。
- **例外**:
  - `unknown_tokens` / `yomi_bad_words` と `text_tail_open` は **テキストから引き直す**
    （pyopenjtalk だけ、音声不要・安価）。旧 manifest の死んだチェックも自動で新仕様に更新される。
  - `[asr2]` だけ例外的に **rescore でも音声を引く**（`refresh_asr2_scores`）。第 2 ASR の後付けは
    `run` だとメイン ASR・AST・CTC・LM まで全部やり直しになるため。同じ `x_cer_algo` 版数を持つ
    レコードは触らないので、閾値の締め直しだけの rescore は第 2 ASR を走らせない（実測 1 話者 12 秒）。
- **rescore で最新化される派生値**: `speaker_sim`（主話者 centroid）、話速中央値、dedup、多様性選別。
  いずれも manifest の `emb` / `fp_sig` / `text` から引き直すので、`[dedup]` / `[diversity]` の変更も
  rescore だけで反映できる。
- **コード**: `cli.rescore` / `stage3_asr.refresh_*` / `stage2_score.recompute_speaker_sim`

---

## 2. 投入とセグメント化

### 2.1 話者名とパス指定を「見た目」で分岐する（A5）

- **結論**: 区切り文字を含む / `.` `~` で始まる / メディア拡張子を持つトークンだけパスとして扱い、
  裸のトークンは常に話者名（`voice/<トークン>/`）として解決する。カレント同名へのフォールバックは作らない。
- **理由**: `run test` がカレントの `test/` を黙って読みにいく「静かな衝突」を無くすため。
  誤った入力は終了コード 2 で、`./test`（パス）と `voice/test`（話者名）のどちらを意図したかのヒントを出す。
- **1 実行 = 1 話者契約**: 複数ディレクトリを渡すと話者名が定まらないため拒否する
  （`run voice` のように voice ルート自体を渡すのも同じガードで止まる）。
- **コード**: `cli._is_path_form` / `cli.resolve_inputs`

### 2.2 キャッシュ下限と発話長ゲートを分離する（F8）

- **結論**: 中間キャッシュする物理下限 `[stage1] cache_min_sec`（既定 0.2 秒）と、
  採用/除外を決める閾値 `[thresholds] min_seg_sec`（既定 0.5 秒）は別ノブにする。
- **理由**: 下限を閾値に一致させると、`rescore` で `min_seg_sec` を 0.5 未満に緩めたときに
  救えるはずの発話が既に捨てられている。下限未満の VAD 出力は音声を作らないが
  `scores.stage1_skipped="too_short"` として manifest には残す（「どこへ消えたか」が見える／常に不合格）。
- **ガード**: `min_seg_sec < cache_min_sec` なら起動時に警告（そのゲートの下側は構造的に効かない）。
- **コード**: `stage1_segment.segment_sources` / `cli._warn_config`

### 2.3 `min_silence_duration_ms` は 200ms のまま上げるな（実測）

- **実測**（yui / hikaru / yuka / test_h / korisu / nashu の納品 3,539 件を Stage 1 再現で追跡）:
  文中で切れている発話は 1.7%（61 件）、その分割地点の実無音は p50 = 0.76 秒。
- **比較**: 350ms に上げると「切れ」は 26% しか救えず、代わりに正常な文境界を 328 箇所融合した
  （見合い 20:1 で悪化）。**200ms が最適**。
- **教訓**: 「切れを直したい」は無音判定を緩めるのではなく、`max_text_tail_open`（文末の開き）で
  検出して落とす側に寄せる。境界の融合は取り返しがつかない。
- **コード**: `stage1_segment.segment_sources`（silero-vad 呼び出し）

### 2.4 `max_speech_duration_s` と `max_seg_sec` を揃える（D3）

- **結論**: VAD 側の上限分割 12 秒（`[stage1] max_speech_duration_s`）と発話長ゲート 12 秒未満
  （`[thresholds] max_seg_sec`）を同じ値にしている。
- **理由**: 長い塊は「破棄」ではなく VAD 側で分割する方が回収率が高い。ゲートで捨てる前に
  分割して候補にする。ゲートは `max_excl`（12.0 ちょうどは除外）。
- **コード**: `stage1_segment` / `stage6_output.CHECKS`

### 2.5 境界補正は強制アライメントではなく騒音床ベースのエネルギー補正（C8 / C9 / C11）

- **結論**: DESIGN が候補に挙げていた MFA / whisperX による語境界の精緻化は使わず、
  「騒音床（セグメント周辺のエネルギー分布から推定）を超える最初／最後の話声フレームへ寄せる」
  補正を実装している。窓は既定 70ms（`refine_window_ms`）。
- **理由**: MFA / whisperX は導入コスト（辞書・モデル・実行時間）が大きく、境界 70ms の勝負に
  そこまで掛ける価値がない。始端は「話声始端の少し前」、終端は「話声終端の少し後ろ」に置くことで
  語頭・語尾を削らない挙動にしている（旧実装は語尾を自分で切りにいっていた）。
- **閾値の置き場所**: 騒音床はセグメント**周辺**から推定する（ソース全体平均に依存しない）。
  入力全体の無音率に左右されなくなる。
- **計測基盤**: 境界ズレの実測には CTC 強制アライメント（`[alignment]`・オプトイン）を使い、
  実運用の検出は `x_cer` に期待する（→ [3.7](#37-2-系統-asr-クロス検証と-x_cer-読みレベル比較-c1)）。
- **コード**: `stage1_segment._refine_boundary`

### 2.6 `output_pad_ms`（納品用の無音マージン）を新設した（C12）

- **実測**: 境界補正後に語頭無音 10ms 未満が 6 話者 3,539 件で **48%** に張り付き、
  メロスぺクトル窓際の話頭・語尾が欠けていた。20ms 付与で **17%** まで低下。
- **結論**: 補正の**後**に効く専用ノブ `[stage1] output_pad_ms`（既定 20ms）を新設した。
- **`speech_pad_ms` との役割分担**: `speech_pad_ms` は VAD 側の切り出しを拡げるだけで、
  補正窓（70ms）の内側でトリムされ出力に残らない。既定 0。マージンが欲しいときは `output_pad_ms`。
- **安全弁**: 分割地点の実無音は `min_silence_duration_ms`（200ms）以上あるので、
  補正窓 70ms×2 とマージン 20ms×2 を引いても隣接発話は巻込まれない。
  `seg_dur`・ASR 入力も同じ境界に揃う（C10）。
- **コード**: `stage1_segment.segment_sources`

### 2.7 長尺ソースはチャンク分割デコード（F9）

- **結論**: `[stage1] decode_chunk_min`（既定 10 分、0 で無効＝一括デコード）で範囲デコードする。
  1 本の長尺動画をフル尺でデコードして常時保持し続ける OOM を防ぐ。
- **境界の正しさ**: デコード窓は所有領域を前後 `ctx`（＝最長発話＋無音/音声判定＋補正窓＋余裕）だけ
  広げ、発話は**開始時刻**でチャンクに一意に割り当てる。チャンク境界で VAD のマージ/分割が
  全体デコードとズレても、発話の取りこぼし・重複は起きない。
- **実装上の注意**: `-ss` を `-i` より前に置く高速シークはキーフレーム基準。音声では誤差が小さく、
  前後に重なり込み窓があるので境界の発話はどちらかのチャンクで必ず取れる。
- **コード**: `stage1_segment._chunk_windows` / `stage0_ingest.decode_range`

---

## 3. スコアリングの判断

### 3.1 話者埋め込みは MFCC 統計＋次元スケーリングのプレースホルダ（C1）

- **現状**: WavLM 系 / ERes2Net / TitaNet は入れておらず、MFCC(40) の mean+std を
  コーパス全体で次元ごと std スケーリング（CMVN の分散側）した 80 次元ベクトルを `emb` に保存している。
- **c0（エネルギー）対策**: スケーリングしないと分散の大きい c0 が cosine を支配して
  `speaker_sim` が「音量類似度」になる。
- **平均引き算をしない理由**: 小コーパスでは pairwise cosine に `-1/(N-1)` の偏りが出るため、
  分散スケーリングだけ行っている。
- **`speaker_sim` を Stage 6 直前に置く理由**: 全件中央値から centroid を作ると、無音・音楽・別人の
  多いコーパスで centroid が汚染されて `min_speaker_sim` が機能不全になる。
  (1) 全 emb 中央値で暫定 centroid → (2) 暫定 cosine が中央値以上の頑健群で引き直す 2 パスにした。
  `emb` から計算するので `run`/`rescore` どちらでも最新の採用候補に対して最新化される。
- **母集団が小さいとき**: 3 件未満なら centroid は無意味なので `speaker_sim=None`（検証不能扱い）。
- **差し替え先**: WavLM 系 speaker embedding など。`emb` を差し替えても `speaker_sim` /
  `min_speaker_sim` の枠はそのままでいい（`emb` は manifest に保存される指紋）。
- **コード**: `stage2_score._cmvn_store` / `stage2_score.recompute_speaker_sim`

### 3.2 voiced/unvoiced は絶対騒音床で分ける（C2）

- **結論**: フレーム RMS を `floor = min(セグメント内 p20 × 3, -55 dBFS)` で分割し、
  `snr` と `voiced_ratio` を作る。中央値分割は `voiced_ratio` が常に ≒0.5 で無情報になるため棄却。
- **`min` を取る理由**: 無音を含む収録では `p20×3` が実騒音床になるが、無音ゼロのホットな収録では
  `p20×3` が「静かな発話」になって SNR がダイナミックレンジ化してしまう。絶対床で頭打ちにすると
  収録間のスケールが揃う（`voiced_ratio` は床超過率のまま）。
- **ほぼ無音のセグメント**: 床超フレームが 1 も無い場合は絶対床（-55dBFS）比で算出、
  床下しか無い（＝話声が無い）セグメントは `snr=None`＝検証不能としてゲート側で不合格。
- **コード**: `stage2_score.score_segments`

### 3.3 jitter / shimmer は有声区間内だけ・Praat の local 定義に倣う（C3）

- **問題**: 無声フレームを落とした列に `np.diff` を当てると区間の境界を跨いだ差分が混ざり、
  shimmer は jitter のコピーになっていた。
- **対策**: 連続した pitch 区間（3 フレーム以上）ごとに周期摂動を計算し、振幅摂動は
  ピッチグリッド上の RMS から別に計算する（境界跨ぎ差分を混ぜない）。
- **閾値の再校正**: 定義を直したので `max_jitter_local` は 0.5 → **0.02** に下げた。
  `max_shimmer_local` は実測分布を確認してから有効化する（既定はコメント）。
- **コード**: `stage2_score._voiced_runs` / `score_segments`

### 3.4 LUFS はメータのブロック長基準（C4 / A1）

- **結論**: 「1 秒以上あるとき」ではなく `pyloudnorm` の `meter.block_size`（0.4 秒）以上で計算する。
  計算不能・非有限は `None` を記録する（`-120` や `-inf` を撒かない）。
- **LUFS 正規化側でも同じ**: 納品書き出しで `x * -inf` による nan 撒き散らしを防ぐ
  （→ [5.2](#52-lufs-正規化とピークリミッタg3)）。
- **コード**: `stage2_score.score_segments` / `stage6_output.write_outputs`

### 3.5 残響スコアは末尾減衰エネルギー比のプロキシ（F4）

- **結論**: DESIGN 候補の DRR・自己相関ではなく、100ms 移動平均の末尾値／ピーク値を使う。
  実装は `scipy.ndimage.uniform_filter1d`（O(N)）。`sliding_window_view(4410).mean` の
  O(N×W) を避けるため。
- **コード**: `stage2_score._tail_ratio`

### 3.6 内容指紋 `fp_sig`（D8）

- **結論**: 音声内容の重複検出には、話者埋め込みではなく MFCC フレームの 2 値化コード →
  3 フレーム gram → MinHash 署名（`FP_NUM_PERM=32`, `FP_BANDS=4`）を使う。
  DESIGN.md の「話者埋め込みは話者ベクトルなので内容照合に使わない」の注意に従った形。
  Chromaprint 系の代替。
- **実測の目分量**: ズレ切り出し重複で 0.6 前後、別発話で 0.2 未満（`[dedup] audio_jaccard = 0.5`）。
- **共有**: MFCC フレーム列は 1 回だけ計算し、`emb` と `fp_sig` で共有する（二重計算回避）。
- **版数の共有**: 生成側（stage2）も照合側（stage5）も同じ定数を import する。
  一方だけ変えると band 切り出しが静かに壊れる（D-6）。
- **コード**: `stage2_score._content_fingerprint_from` / `stage5_dedup.FP_NUM_PERM`

### 3.7 2 系統 ASR クロス検証と `x_cer`（読みレベル比較）（C1 / C14）

- **結論**: `[asr2]` で別系統（既定: メイン Qwen3-ASR ⇄ 第 2 faster-whisper small）を走らせ、
  2 系統間 CER を **読みレベル** で `x_cer`、表層値を `x_cer_surface`（参考値）に記録する。
- **読みレベルにした実測根拠**（6 話者 10,369 候補のうち asr2 計測済み 4,050 件）:
  表層文字のままでは中央値 0.11・完全一致 1%。読み化すると中央値 0.00・完全一致 7 割超
  （集計対象により 73〜79%）。差分の大半は whisper が打たない句読点と同音異字
  （例: モフモフのバンテン ⇄ もふもふの番点）で、教師としては実質一致している。
  **表層でゲートを締めると正常発話を大量に落とす**。
- **正規化の細部**: pyopenjtalk の g2p でカタカナ化し、拗音は大字へ寄せ、**文末の `ー` だけ**落とす
  （どうぞ ⇄ どうぞー 対策）。語中の長音は残すのでおばあさん/おばさんは潰れない。
  頭字語中の句点（S.F. / U.S.A.）は先に消す — pyopenjtalk は語中の句点を境にそこだけ読まず、
  「S.F.とか。」から「トカ」だけを返して一致しなくなる（実測で誤検出 1 件）。
  g2p が読めない入力は句読点だけ落とした表層形に退避する（比較不能で全滅させるより弱い指標でも残す）。
- **`max_x_cer = 0.45` に決めた経緯**（採用 3,525 件での超過率）:
  0.3 → 2.9%（103 件）/ 0.4 → 1.5%（53 件）/ 0.45 → 1.25%（44 件）/ 0.5 → 0.8%。
  旧ゲート脱落群では 0.3 で 42.9% = **15 倍の濃度差**なのでゲート自体は明確に効く。
  ただし食い違い 103 件を CTC 強制アライメントで裁定すると 主 ASR 支持 38% / whisper 支持 29% /
  判定不能 33%、しかも CTC は語彙頻度で選ぶので「こめかみ」より「米紙」を支持した（自動裁定に使えない）。
  whisper small はこの素材（ささやき・オノマトペ・口語短文）で負けることが多く、0.3 で締めると
  正しそうな発話まで捨てて register が偏る。そこで「片方が明らかに壊れている」0.45 だけで除外し、
  境界帯は人が選べるようにした。適用結果: 3,525 → 3,486 件（-1.1%）。
  単独脱落 36 件のうち第 2 系統の確信度が 0.5 以上だったのは 8 件だけ（中央値 0.41、
  2 系統が完全一致する行は 0.69）＝脱落の半分は主 ASR の方が正しそう。
  0.6 に緩めると +22 件（3,508）で、`x_cer > 0.6` の明らかな壊れ 14 件だけ落ちる。
- **`review_x_cer`（見送り受け皿）は既定コメント-out**: 0.25 で採用の約 4% が見送り行き。
  完全自動運用を優先するため。境界帯を救いたいときだけコメント解除する。
- **スキップは固定値**: 「明らかなゴミ」（0.3 秒未満・無音率 0.9 超・クリップ 5% 超・SNR -5dB 未満）だけ
  第 2 ASR をスキップする（`asr2_skipped="obvious_garbage"`）。ユーザー閾値から意図的に切り離したのは、
  `rescore` で `min_snr_db` 等を緩めたときにスキップ済みレコードだけ `x_cer` 欠損で全滅する罠を避けるため。
- **実行コスト**: 0.1〜0.3 秒/本（rescore の korisu 826 本 = 1 分 33 秒、6 話者 10,369 本 = 15〜30 分）。
  GPU を他プロセス（ローカル LLM 等）と共有すると数倍遅くなる。初見の見積 0.47 秒/本はモデル起動込み。
- **`X_CER_ALGO` 版数**: `reading_key` の正規化を変えたら**上げる**。rescore は
  「この版数を持たないレコードだけ」引き直す。上げ忘れると古い正規化の値が manifest に残り、
  修正がデータセットに反映されない。`[asr2]` の**モデル**を変えた場合は検出できないので `run` を使う。
- **コード**: `stage3_asr.reading_key` / `_asr2_eligible` / `refresh_asr2_scores`

### 3.8 トークン確率は分布で記録する（C3）

- **結論**: 平均（= `asr_conf`）に加えて `token_prob_min` / `token_prob_p10` / `token_nll` を記録する。
  平均が高いのに 1 語だけ極端に低い＝そこだけ聞き間違い、という典型を拾うため。
- **実測スケール**（6 話者 10,369 候補 / 採用 3,561 件）:
  `token_prob_min` は採用内 p1=0.26 / p5=0.43 / p25=0.68 → 既定 0.25 は採用 0.8%・全候補 1.7% を切る末尾カット。
  `token_nll` は採用内 p25=0.003 / 中央値 0.011〜0.025 / p95=0.11〜0.18 → 既定 0.3 で採用 0.5%・全候補 5.0%。
- **教訓**: 旧コメント値 `min_token_prob = 0.30` は採用 1.7% を落としており、
  `max_token_nll = 3.0` は実測スケールと 10 桁違いの完全な no-op だった。
  **閾値は必ず実測分布から決める**（コメント値を信じない）。
- **`token_prob_p10` はゲート化しない**: 可視化専用（min と nll で十分）。
- **コード**: `stage3_asr._token_stats`

### 3.9 文末の開き `text_tail_open`（C13）

- **深刻度の定義**（G2P と同じ pyopenjtalk の品詞・活用形から決める。学習側と同一エンジン）:
  - **2** = 文末記号が全く無い、または動詞・形容詞・助動詞の読みが促音で止まる
    （「お客様の体温と混じっ」型＝語の途中で発話が終わっている）。納品対象にしない。
  - **1** = 条件節「〜したら。」・連用形中止「〜だし。」・連体止め「〜ような。」。
    続きがありそうだが演技上の言い淀みとして成立する。
  - **0** = 閉じた文。
- **既定 `max_text_tail_open = 1`（2 だけ除外）の根拠**: 実コーパス 3,539 件の分布は
  0 = 3,464 / 1 = 74（2.1%）/ 2 = 1。1 まで落とすかは分布を見て判断する（report.html に一覧、rescore で再調整可）。
- **検出しないもの**: 助詞止め（「今から。」）は口語の普通形なので対象外 — 実コーパスの **34%** が
  該当してしまい、欠陥ではなく文体を落とすことになる。連用タ接続（「待って。」）と
  連用テ接続（「早く。」）も口語では完文になるので対象外。
- **`…` の扱い**: 判定は ASR 生テキスト（raw）で行う。`normalize_text` は装飾記号として `…` を落とすため、
  正規化後テキストだけを見ると「ごめんね…」が終端なし＝切れ目と誤爆する。
  rescore では raw が manifest に無いので正規化後テキストで判定するが、
  実コーパス 3,539 件で正規化後に文末記号が消えた発話は 0 件（run 時の値と食い違わない）。
- **コード**: `stage3_asr.tail_openness` / `is_sentence_terminated`

### 3.10 学習側の読みチェック（YomiError）を鏡写しする（yomi.py）

- **背景**: 旧実装の `unknown_tokens` は pron に出ない `*` / `?` を見ていて**実質死んでいた**。
  学習側 `preprocess.text`（既定 `yomi_error=raise`）が実際に弾くのは
  `pron == "、"` の語（読めない漢字・古いカタカナ）と、カタカナ化に失敗する語。
- **結論**: 学習側 `nlp/japanese/normalizer.py` と `g2p.py::text_to_sep_kata` の失敗条件を
  同一エンジン（pyopenjtalk-plus）で鏡写しし、「学習側で実際に落ちる語」を数える。
  語自体は `yomi_bad_words` に残し、report.html の「読めない語」に出す（ユーザー辞書登録の判断材料）。
- **誤爆防止**: 学習側は正規化（句読点置換＋allow-list 除去）してから `run_frontend` に掛ける。
  絵文字・キリル文字等は allow-list 外で解析前に消える＝学習側では無害なので、
  生テキストにそのまま掛けるのは誤爆。必ず `learner_normalize()` 経由で打診する。
- **意図的な省略**: 数字→漢字変換（num2words）は YomiError の可否に変えないので写さない。
  `word2ph` 合計や `align_tones` のアサーション系も稀なエッジケースとして鏡写ししない。
- **`"adopt"` オプションを削除**: unknown を採用すると学習側 preprocess で必ず落ちる行を納品する。
  そんな読み方からしか得られない特徴は少ないので既定は不採用。`on_unknown_tokens` は
  `exclude` / `dict_data`（見送り行き）のみ、値の誤字は起動時に `ValueError`（D-4）。
- **同期元**: Style-Bert-VITS2 の `nlp/symbols.py::PUNCTUATIONS` /
  `nlp/japanese/normalizer.py` / `nlp/japanese/g2p.py`。学習側を変えたら追従させる。
- **コード**: `yomi.py` / `stage3_asr.g2p_check` / `refresh_yomi_scores`

### 3.11 テキスト系スコアの定義修正（C7）

- **`kana_ratio`**: 分母を「かな＋英数字」のみにした。旧実装の `÷ len(text)` は句読点や漢字に薄められ、
  「本日、会議は中止となりました」のような漢字多めの正常発話を `min_kana_ratio` で誤爆していた。
  英数もかなもない発話（純漢字・純記号）は「非日本語の証拠」が無いので 1.0 とする
  （純記号ゴミは `residual_symbols` と `min_chars` が別途捕まえる）。
- **`filler_ratio`**: 出現回数ベースに修正。旧実装は各語 1 回出現かどうかだけで、
  「えっと」×3 でも 3 文字分しか数えていなかった。`max_filler_ratio` は 0.8 → **0.6**（0.8 は甘すぎた）。
- **`text_features()` に集約**: メイン ASR / asr2 / 人手オーバーライドで同じ定義を使う
  （唯一の計算源。定義がズレるとオーバーライド時だけ判定が変わる）。
- **コード**: `stage3_asr.text_features`

### 3.12 記号は読み上げ展開してから G2P にかける（C5）

- **結論**: `[text] symbol_readings`（既定 true）で `%` → パーセント、`℃` → 度 等を展開してから G2P。
  展開後も残る記号は `residual_symbols` に記録（`max_residual_symbols` ゲートで弾ける）。
- **写さない記号**: `+` / `-` のように読みが文脈依存（たす/プラス/ふたつ…）なものは、
  誤読リスクの方が大きくて意図的に対応表に入れていない。
- **英語→カタカナ変換は未実装**: 発音辞書（judict 等）が必要。学習側 dict_data 連携とセットで今後の課題。
- **コード**: `itn.py` / `stage3_asr.normalize_text`

### 3.13 ASR バックエンドの既定（Stage 3）

- **既定**: `backend = "qwen3-asr"`（`Qwen/Qwen3-ASR-1.7B-hf` を transformers native で実行、
  greedy generate のトークン平均確率を信頼度に）。初回に約 4GB をダウンロード。
- **CPU 軽量代替**: `backend = "faster-whisper"` + `model = "small"` / `"tiny"`。
  whisper 固有の `no_speech_prob` / `compression_ratio` も取れる（qwen でこれらのゲートを有効にすると
  スコア欠損で全件落ちるため、既定はコメント-out）。
- **`align_conf`**: whisper は単語確率、qwen はトークン確率を流用（実質 `asr_conf` と同じものを見ていた。
  実アライメントが欲しいときは `[alignment]` → [4.1](#41-ctc-強制アライメントのゲート化は見送りc2)）。
- **コード**: `stage3_asr._asr_qwen` / `_asr_whisper`

---

## 4. 重量検証（オプトイン）の採用と棄却

同梱 `filters.toml` は `[asr2]` / `[tags]` / `[lm]` / `[alignment]` を既定有効にしている。
ゲートは `max_x_cer`・`reject_audio_tags`・`max_lm_ppl`（いずれも除外）だけ既定有効で、
**人手レビュー行きを作らない完全自動運用**。他ゲートは初回 run の実測分布を見てから締める。

### 4.1 CTC 強制アライメントのゲート化は見送り（C2）

- **記録はする**（`[alignment]` 既定有効）: `ctc_align_score`（経路平均後部確率）、
  `unaligned_edge_ratio`（先頭/末尾の「テキストに無い発話」比）。
- **モデル選定**: `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`（Common Voice 日本語の orthographic
  文字語彙）。ASR 出力（漢字混じり）をそのままターゲットにでき、実測分離は 一致 0.76 / 別テキスト 0.00。
  音素ハブート系は 0.53 / 0.23 と分離が弱く採用外。
- **ゲート化を取りやめた実測**（6 話者 10,369 候補 ＋ 別系統 ASR との突き合わせ）:
  1. 発話長に強く従属する（スコア自体が 1/T 依存）。`min_ctc_align = 0.15` で旧採用からの脱落が
     <1s 13.6% / 1-2s 4.3% / 2-3s 0.8% / 3-5s 0.5%。`max_unaligned_edge = 0.5` は <1s で 8.8%、
     2 秒以上で 0%。短文は枠数が小さく 1 枠の誤差が効く。
  2. 落ちた側の品質が採用群と変わらない。ctc 単独脱落の whisper 一致率 67% / edge 単独 80% /
     対照（採用のまま）78%。`ctc=0.0` の「ゆっくり息を吸って。」「グリグリ。」「くるくる。」、
     `edge=0.737` の「どうぞ。」(1.16s) を whisper も本文と同一に聴いた＝CTC 側のアライメント失敗。
  3. 旧コメント値はさらに悪く、`min_ctc_align = 0.5` は採用の 18.3%、
     `max_unaligned_edge = 0.25` は 21.1% を落とす。
- **結論**: 記録のみ（ゲートはコメント-out）。2 文混入・境界ズレの実質的な検出は `x_cer` に期待する。
- **注意**: 発音が同じで漢字/仮名表記が違う場合（ASR と CTC モデルの漢字変換のズレ）はスコアが下がる。
- **コード**: `align.py`

### 4.2 MLM 擬似 PPL は除外ゲート 200 で採用（C4）

- **前提**: 発音が変わる誤変換（「全然大丈夫」→「全然大平」等）は文としても不自然になる。
  同音異字（発音が同じ誤変換）は無害なので上げないこと。
- **実測分布**（1,688 件）: adopted p95 ≈ 59。100〜200 帯は短文口語断片（「どうしたの。」「じゃあ。」）が
  中心で採用可。>200 に同音異字疑似（氷柱/兵衆/悲願花）が寄る。もっと詰めるなら 100。
- **モデル**: `ku-nlp/deberta-v2-large-japanese-char-wwm`（SentencePiece 文字トークナイザ）。
  bert-japanese 系は char 版でも MeCab（fugashi+unidic）を要求するため、追加パッケージなしで動くこちらを既定に。
- **実装上の注意**: 「1 位置ずつ mask して真トークンの対数確率を拾う」方式を**バッチ 1 回**に畳む。
  全位置同時 mask は文脈そのものを殺して予測がほぼ一様になるため不可。
- **二段構え**: `max_lm_ppl`（除外・既定有効 200）＋ `review_lm_ppl`（見送り行き・既定コメント-out）。
- **コード**: `lm.py`

### 4.3 AudioSet タグ（AST）を既定有効にした（B2）

- **理由**: 既存の `music_flag` は ASR が `♪` を吐くかに依存していて検出漏れが大きく、
  笑い・咳・拍手は全く見ていなかった。AST は transformers 既存依存で動くため新規パッケージ不要
  （初回に約 1.3GB）。
- **実装**: 10 秒窓ごとに sigmoid を取り、**窓間 max-pool**（一部分だけ音楽でも検出したい）。
  `min_prob`（既定 0.25）以上でヒット、`reject` の AudioSet クラス名で増減する。
- **ゲート**: `reject_audio_tags`（既定有効）。`tag_hit` を持たない旧 manifest への rescore は
  全件「検証不能＝不合格」になる点に注意。
- **コード**: `tags.py`

### 4.4 SQUIM_SUBJECTIVE の `mos` は「相対 MOS」（B1）

- **結論**: `[stage4] subjective`（既定オフ、初回約 360MB）で `mos` を記録する。
  モデルは「クリーンな参照音声」アンカーを要求するので、コーパス内で SNR 最良の発話をアンカーに使う
  → `mos` は**その話者自身の最クリーン発話に対する相対品質**（絶対 MOS ではない）。
- **枠の再利用**: DNSMOS / NISQA を差し替えても同じ `min_mos` ゲート枠が使える。
  `min_mos` を締める前に実測分布を見ること。
- **集約先 `quality_score`**: 現状 `= pesq`。UTMOS2 を差し替えたときに置かれる前提の別キー。
- **コード**: `stage4_quality._subjective_mos`

### 4.5 pyannote 2 人同時発話（B3）

- **オプトイン**: `[overlap] enabled` で segmentation-3.0 の powerset 出力から重複秒を測り、
  `overlap_flag` / `overlap_sec` を上書きする（既定は `overlap_flag=None` で素通り）。
- **任意依存**: `uv sync --extra extras` ＋ HF_TOKEN（モデルの利用規約同意付き）。
- **fail-fast**: 未導入のまま有効化したら黙って `None` にせず、導入手順を告げるエラーを出す。
- **コード**: `overlap.py`

### 4.6 除去修復 `[repair]` は自動採用しない（B8）

- **対象**: `targets` に挙げたノイズ系ゲート（既定 `min_snr_db` / `min_quality_pesq` / `min_stoi` /
  `min_si_sdr`）**単独**で落ちた発話だけ。他の理由と併せ落ちの発話は対象にしない。
- **約束**: 再検証を通過しても**自動採用せず `pending_review` 行き**。除去アーティファクトは
  学習後に効いてくるので、機械の「直った」の自己申告を鵜呑みにしない。
- **元音声**: `.cache_orig/` に保持（`.cache` が修復音声で上書きされても rescore / 手動復元できるように）。
- **`emb` の扱い**: `emb` はコーパス全体の CMVN スケーリング済み値。`score_segments` を部分集合で
  再実行すると単一レコードの std で割って壊れるため、バックアップして復元する
  （話者本人性は除去で大きく変わらない前提。`fp_sig` 等の内容系は再計算が正しいのでそのまま採用）。
- **依存**: deepfilternet は numpy<2 pins のため lock 外。
  `uv pip install --no-deps deepfilternet deepfilterlib loguru appdirs requests sympy`
  （`--no-deps` なしだとプロジェクトの numpy 2.x を downgrade して壊す）。
- **コード**: `repair.py`

### 4.7 追加検証の失敗は `errors` に入れない

- **結論**: 追加モデル（AST / CTC / LM / SQUIM_SUBJECTIVE / pyannote）の失敗は
  `mos_errors` / `tag_errors` / `align_errors` / `lm_errors` / `overlap_errors` に寄せる。
- **理由**: `errors` は「常に不合格」のキーなので、ゲートが無効な間に追加検証が落ちただけで
  レコードを道連れにしないため。ゲートが有効ならスコア欠損＝検証不能＝不合格になるので、
  結果として必要なときだけ落ちる。
- **コード**: `align.py` / `lm.py` / `tags.py` / `overlap.py` / `stage4_quality.py`

---

## 5. 出力と納品契約

### 5.1 契約は 44.1kHz・モノラル固定、書き出し前に止める（A-4 / A7）

- **結論**: `[output] sample_rate` / `channels` が契約（44100 / 1）から逸脱していたら、
  全作業の後ではなく**書き出し前に** `ValueError` で止まる。
- **自走検証**: `raw.zip` を納める前に、学習側 preprocess と同じ検証（sr・チャンネル数）を
  全ファイルへ通す。
- **コード**: `stage6_output.write_outputs` / `_contract_check`

### 5.2 LUFS 正規化とピークリミッタ（G3）

- **手順**: `.cache` から 44.1k で読む → LUFS 正規化（目標 `target_lufs` = -23.0）→
  ピークが -1dBFS を超えたらスケール → flac 書き出し。
- **`limited` / `lufs_out`**: リミッタが作動して LUFS ターゲットが崩れたことを黙らせないため、
  作動フラグと実測 LUFS を manifest に残す（report.html に作動本数）。
- **正規化不可のクリップ**: `ValueError` は握りつぶしてそのまま納品し、直後のピークリミッタで守る。
- **コード**: `stage6_output.write_outputs`

### 5.3 `report.html` の設計方針

- **単一 HTML・外部 CDN 依存なし・ダーク基調**。ブラウザで開くだけ。印刷時は明色に反転。
- **音声は埋め込まない**（単一 HTML で完結させるため）。聴くなら `.cache/<utt_id>.flac` を直接開く。
- **「数字を並べても判断材料にならない」**ので、失格理由キーの凡例・見方・動かし方
  （rescore / `text_overrides.jsonl`）を同じページに置く。分類カラー chip の凡例もここ。
- **閾値感度表（G1）**: 「そのゲート 1 つだけ緩めた/締めたときにひっくり返る本数」は、
  他の理由でも落ちている発話を数えない近似なので、実際の rescore での増減はこれより少ない。
  緩和 = `min_*` は 20% 下げる / `max_*` は 25% 上げる、厳格化 = `min_*` は 20% 上げる / `max_*` は 20% 下げる。
  flag ゲートは緩める/締めるが定義できないので感度列は空。
- **多様性メトリクス（A6）**: 平均ペアwise類似度は総対ペアのサンプル近似（固定シードで再現可能、
  大コーパスでも有界）。
- **x_cer 一覧は食い違ったものだけ**（`x_cer = 0` の完全一致を 60 行並べてもノイズなので）。
  帯ごとの「第 2 確信度」中央値を添える — 食い違い帯で確信度が低ければ除外の証拠は弱い。
- **コード**: `stage6_output.write_report`

### 5.4 納品用 `raw/` は毎回作り直す（A3）

- 前回の除外済みファイルが混ざらないよう、`write_outputs` の冒頭で削除して作り直す。
  中間音声は `.cache/`（rescore 入力・納品物ではない）で分離している。

### 5.5 空セリフ・モーラ 0 は閾値と独立の不変条件（A-2）

- `min_chars` を消していても常に不合格（`empty_text`）。Stage 1 の `too_short` / `source_error`
  （音声キャッシュが存在しない）も閾値と無関係に常に不合格。

---

## 6. 重複排除と多様性選別

### 6.1 重複排除は 4 段・LSH で候補を出す（F3 / B4 / B5）

- **順序（安い順）**: ①テキスト完全一致（ハッシュで O(N)）→ ②テキストファジィ（shingle Jaccard ≥
  `text_jaccard`）→ ③音素三量文字列 Jaccard ≥ `phoneme_jaccard` → ④音声内容指紋 `fp_sig` 一致率 ≥
  `audio_jaccard`（ズレ切り出し重複）。
- **O(N²) を取らない**: MinHash band 衝突（テキスト・音素）と band LSH（音声指紋）で候補ペアを出してから
  厳密判定する。グループ化は Union-Find。
- **理由の引き継ぎ**: マージ時に優先度（`exact` < `fuzzy_text` < `fuzzy_audio`）の高い理由をグループへ
  引き継ぐ（B4）。
- **代表の選び方**: 「閾値チェックを通過したもの」を最優先、次に `quality → asr_conf → snr`
  （B5）。代表候補が後で閾値落ちしてグループごと消えるのを防ぐ（`evaluate_checks` が先に走る前提）。
- **`dup_reason` の実運用上の注意**: 音素三量 Jaccard によるファジィも `fuzzy_audio` ラベルになる
  （列挙値は `exact` / `fuzzy_text` / `fuzzy_audio` の 3 つ）。`fuzzy_audio` は
  「音素レベルで近い」＋「音声指紋で近い」の両方を指す。判定キーを分けたい場合は列挙値の追加が必要。
- **rescore での再実行**: `emb` / `fp_sig` / `text` から引き直すので、`[dedup]` の閾値変更は rescore だけで反映される。

### 6.2 多様性選別（Stage 5b）の物差しは「内容」（A1 / A2 / A4 / A5）

- **動機**: 重複排除はほぼ同一の発話を代表 1 本にまとめるだけなので、
  「重複ではないものの似た」発話は残る。それを間引いてデータセットの多様性を上げる。
- **アルゴリズム**: 重複排除後に残った候補を**品質順**に貪欲選択し、採用済みのどれかと
  `sim_threshold` 以上で類似していれば除外する（採用集合の総対ペア類似度を閾値未満に抑える独立集合）。
- **`metric` は内容ベースのみ**: `phoneme`（既定・音素三量文字列 Jaccard）/ `text`（shingle）/
  `audio`（内容指紋一致率）。**話者/音響ベクトル `emb` は不採用** — 単一話者コーパスでは全発話が
  emb 上で近接し、別内容まで間引いてしまう。
- **`sim_threshold = 0.6` の実測**: 別々の発話は音素三量 Jaccard ≤ 0.05、似た内容は ~0.67、
  完全一致 ~1.0。dedup（0.90）より**緩く**置き、「似た内容」だけを間引いて別発話には触れない。
  `sim_threshold >= dedup 閾値` なら警告（多様性選別は何も間引けない）。
- **`protect_coverage`**: ある音素の**唯一の供給源**は `similar` / `cap` / `target_size` のいずれでも落とさない
  （多様性のために音素カバレッジを犠牲にしない）。
- **追加ノブ（既定 0 = 無効 = 従来の「品質順+類似間引き」と同動作）**:
  - `gain_lambda`: 選択スコアに「新規音素 three-gram 利得」を加算し、間引きから**多様性最大化**へ切り替える。
    唯一の供給源保護を連続値に一般化した形。`target_count` と併用すれば K 本の多様性最大化選択。
  - `max_bucket_share`: F0 三分位 × 話速三分位のバケット別シェア上限（読み上げ調などへの偏りを整流化。
    韻律は `emb` と違い「話者」ではなく「話し方」の軸）。計測不能発話は免除。
  - `max_source_share`: 投入ソース 1 本の最大採用シェア（長尺動画 1 本にスタイルを引きずられるのを防ぐ）。
  - `target_count`: 目標本数に達したら選択を打ち切る（超過分は `target_size` で除外）。
- **除外理由**: `diversity:similar:<代表utt>` / `diversity:cap:prosody:<バケット>` /
  `diversity:cap:source:<src>` / `diversity:target_size`。report.html に除外本数と選別前後のメトリクスを出す。

---

## 7. 環境・依存・品質ゲート

### 7.1 pytorch index は cu128 既定（F7）

- **現状**: `pyproject.toml` の `[[tool.uv.index]] name = "pytorch-gpu"` が
  `https://download.pytorch.org/whl/cu128` を指しており、`uv sync` すると GPU wheel が入る
  （検証環境: torch 2.11.0+cu128）。
- **CPU 運用に戻す**: 同じ index の url を `https://download.pytorch.org/whl/cpu` に差し替えて
  `uv lock && uv sync`（手順は [GPU.md](GPU.md)）。
- **torch を optional-dependencies に分離しない**: 既定の `uv sync` で torch が消えて壊れるため、
  index 差し替え手順を明示する形にしている。
- **深追い**: `[runtime] device = "auto"` のままなら、GPU を挿すだけで Qwen3-ASR・SQUIM・
  faster-whisper が自動で CUDA に載る。

### 7.2 任意依存の入れ方

| 機能 | 入れ方 | 備考 |
|---|---|---|
| pyannote（`[overlap]`） | `uv sync --extra extras` | HF_TOKEN ＋ pyannote モデルの利用規約同意が必要 |
| deepfilternet（`[repair]`） | `uv pip install --no-deps deepfilternet deepfilterlib loguru appdirs requests sympy` | numpy<2 pins のため lock 外。`--no-deps` 必須 |
| AST / CTC / LM / SQUIM | 追加導入なし（transformers / torchaudio 既存依存） | 初回ダウンロード量だけ注意 |

### 7.3 品質ゲートの取り決め

| コマンド | 役割 | 取り決め |
|---|---|---|
| `uv run pytest test/` | スモーク＋入力解決の契約テスト | ゴールデン方式。tiny ASR で数秒音声から 1 周＋rescore 往復 |
| `uv run ruff check src test` | リント専用 | 書式は black 担当（`ruff format` は使わない）。日本語コメントで誤爆する `RUF001/002/003` と `E501` は無効。`BLE001`（広い except）は有効のまま意図した箇所だけ `# noqa` |
| `uv run black src test` | 整形 | `line-length = 100`（日本語コメント・HTML 文字列で行が伸びるため。既定 88 では超える） |
| `uv run pyright` | 型検査（`src` と `test`） | `uvx pyright` はプロジェクトの `.venv` を使わず torch / numpy の import 解決に失敗して誤検知するため使わない |

---

## 8. 未実装・今後の課題

| 項目 | 現状 | 補足 |
|---|---|---|
| 話者埋め込み | MFCC 統計プレースホルダ | WavLM 系 / ERes2Net / TitaNet への差し替え余地（`emb` を差し替えても `speaker_sim` 枠はそのまま） |
| 強制アライメント境界補正 | 騒音床ベースのエネルギー補正 | MFA / whisperX は未採用。CTC は計測基盤として実装済み |
| 音声品質スコア | SQUIM_OBJECTIVE（`quality_score = pesq`） | UTMOS2 差し替え時に `quality_score` を置換する前提 |
| 英語→カタカナ表記 | 未実装 | 発音辞書（judict 等）が必要。学習側 dict_data 連携とセット |
| 感情クラスタ（emotion2vec） | 未実装 | DESIGN の候補どおり未着手 |
| TTS 往復検証 | 未実装 | 学習済み Style-Bert-VITS2 との統合が必要 |
| `dict_data` 実連携 | 見送り（`pending_review`）まで | ユーザー辞書候補の振り分け・一覧化まで実装。辞書ファイルへの書き出しは未実装 |

---

## 付録: 旧レビュー番号対応表

コード・設定コメントに残る `A`/`B`/`C`/`D`/`E`/`F`/`G` 系と `P0-P2` の指摘は、過去ラウンドのレビュー ID。
判断の中身はこの文書の各節にある。

| ID | 論点 | 節 |
|---|---|---|
| A-1 / A1 | 感度分析を閾値キー単位で数える／非有限値を検証不能にする | [1.3](#13-検証不能不合格-の方針a-1--a-2) |
| A-2 / A2 | 空テキスト不変条件・非有限値の null 化 | [1.3](#13-検証不能不合格-の方針a-1--a-2) / [5.5](#55-空セリフ・モーラ-0-は閾値と独立の不変条件a-2) |
| A-3 | 計算済みなのに未配線だった指標（F0 等）をゲート化 | [PIPELINE のゲート対応表](PIPELINE.md#スコア--ゲート対応表) |
| A-4 | 出力契約を 44.1k/mono 固定で書き出し前に止める | [5.1](#51-契約は-441khz・モノラル固定書き出し前に止めるa-4--a7) |
| A3 | 納品用 `raw/` を毎回作り直し、中間キャッシュと分離 | [5.4](#54-納品用-raw-は毎回作り直すa3) |
| A5 | 素文件名でも話者名が空にならない／入力解決の見た目分岐 | [2.1](#21-話者名とパス指定を見た目で分岐するa5) |
| A6 | 多様性選別前後のメトリクスをレポートに出す | [5.3](#53-reporthtml-の設計方針) / [6.2](#62-多様性選別stage-5bの物差しは内容a1--a2--a4--a5) |
| A1/A2/A4/A5（diversity） | `gain_lambda` / `max_bucket_share` / `target_count` / `max_source_share` | [6.2](#62-多様性選別stage-5bの物差しは内容a1--a2--a4--a5) |
| A7 | 納品前に契約検証を全ファイルへ自走で通す | [5.1](#51-契約は-441khz・モノラル固定書き出し前に止めるa-4--a7) |
| B | 計算済みスコアのゲート化（digital_silence / voiced_ratio / stoi / si_sdr / lufs / overlap） | [PIPELINE のゲート対応表](PIPELINE.md#スコア--ゲート対応表) |
| B1 | SQUIM_SUBJECTIVE を相対 MOS として記録 | [4.4](#44-squim_subjective-の-mos-は相対-mosb1) |
| B2 | AudioSet タグ（AST） | [4.3](#43-audioset-タグastを既定有効にしたb2) |
| B3 | pyannote 2 人同時発話 | [4.5](#45-pyannote-2-人同時発話b3) |
| B4 / B5 | union-find の理由引き継ぎ／代表は閾値通過者優先 | [6.1](#61-重複排除は-4-段・lsh-で候補を出すf3--b4--b5) |
| B8 | 除去修復（自動採用しない） | [4.6](#46-除去修復-repair-は自動採用しないb8) |
| C1 | 2 系統 ASR クロス検証（読みレベル `x_cer`） | [3.7](#37-2-系統-asr-クロス検証と-x_cer-読みレベル比較-c1--c14) |
| C2 | CTC 強制アライメント | [4.1](#41-ctc-強制アライメントのゲート化は見送りc2) |
| C3 | jitter/shimmer の定義／トークン確率の分布 | [3.3](#33-jitter--shimmer-は有声区間内だけpraat-の-local-定義に倣うc3) / [3.8](#38-トークン確率は分布で記録するc3) |
| C4 | MLM 擬似 PPL／LUFS のブロック長基準 | [4.2](#42-mlm-擬似-ppl-は除外ゲート-200-で採用c4) / [3.4](#34-lufs-はメータのブロック長基準c4--a1) |
| C5 | 記号の読み上げ展開・残留記号 | [3.12](#312-記号は読み上げ展開してから-g2p-にかけるc5) |
| C6 | `text_overrides.jsonl` への集約 | [1.1](#11-人手レビューページを作らず-text_overridesjsonl-1本に寄せるc6) |
| C7 | `kana_ratio` / `filler_ratio` の定義修正 | [3.11](#311-テキスト系スコアの定義修正c7) |
| C8 / C9 / C11 | 境界補正（side 考慮・騒音床の局所推定・`speech_pad_ms` は出力に残らない） | [2.5](#25-境界補正は強制アライメントではなく騒音床ベースのエネルギー補正c8--c9--c11) |
| C10 | 44.1k 出力と 16k ASR 入力を同一境界から作る | [2.6](#26-output_pad_ms納品用の無音マージンを新設したc12) |
| C12 | `output_pad_ms`（納品用マージン） | [2.6](#26-output_pad_ms納品用の無音マージンを新設したc12) |
| C13 | 文末の開き `text_tail_open` | [3.9](#39-文末の開き-text_tail_openc13) |
| C14 | 2 系統不一致の見送り受け皿 `review_x_cer` | [3.7](#37-2-系統-asr-クロス検証と-x_cer-読みレベル比較-c1--c14) |
| D1 | ハルシネーション検出（`no_speech_prob` / `compression_ratio` / 繰り返しブロック） | [3.13](#313-asr-バックエンドの既定stage-3) / [PIPELINE のゲート対応表](PIPELINE.md#スコア--ゲート対応表) |
| D2 | クリッピング・DC オフセット・デジタル無音 | [PIPELINE のゲート対応表](PIPELINE.md#スコア--ゲート対応表) |
| D3 | 長い塊は VAD 側で分割する | [2.4](#24-max_speech_duration_s-と-max_seg_sec-を揃えるd3) |
| D4 | silero-vad の主要パラメータを露出 | [2.3](#23-min_silence_duration_ms-は-200ms-のまま上げるな実測) |
| D5 | 音楽記号を `music_flag` として残す | [PIPELINE の Stage 3](PIPELINE.md#stage-3-文字起こしと言語系チェックstage3_asrpy) |
| D7 | `overlap_flag` は既定 `None` 記録 | [4.5](#45-pyannote-2-人同時発話b3) |
| D8 | 音声内容指紋（MinHash）によるズレ切り出し重複 | [3.6](#36-内容指紋-fp_sigd8) |
| D-4 | `on_g2p_fail` / `on_unknown_tokens` の値を起動時に検証 | [3.10](#310-学習側の読みチェックyomierrorを鏡写しするyomipy) |
| D-6 | MinHash 定数を生成側・照合側で共有 | [3.6](#36-内容指紋-fp_sigd8) |
| E1 | `rescore`（閾値変更だけの再判定） | [1.4](#14-run-と-rescore-の責務分担e1) |
| E2 | 閾値にコード側デフォルトを置かない | [1.2](#12-閾値にコード側デフォルトを置かないe2) |
| E3 | g2p / unknown の扱いを filters.toml で決める | [3.10](#310-学習側の読みチェックyomierrorを鏡写しするyomipy) |
| E4 | 全失格理由を `dropped_by` に残す | [1.3](#13-検証不能不合格-の方針a-1--a-2) / [PIPELINE の Stage 6](PIPELINE.md#stage-6-判定・出力stage6_outputpy) |
| E5 | `run.json` に再現性メタデータ | [PIPELINE の生成物](PIPELINE.md#生成物datamodel_name) |
| E6 | `dict_data`（ユーザー辞書候補）の受け皿 | [8](#8-未実装・今後の課題) |
| F1 / F2 | ハッシュのチャンク読み／フル尺波形を常時保持しない | [2.7](#27-長尺ソースはチャンク分割デコードf9) |
| F3 | 重複排除の LSH | [6.1](#61-重複排除は-4-段・lsh-で候補を出すf3--b4--b5) |
| F4 | 残響プロキシの O(N) 実装 | [3.5](#35-残響スコアは末尾減衰エネルギー比のプロキシf4) |
| F5 | faster-whisper の `batch_size` 対応 | [3.13](#313-asr-バックエンドの既定stage-3) |
| F6 | デコードは 1 回だけ（16k は 44.1k からリサンプル） | [2.5](#25-境界補正は強制アライメントではなく騒音床ベースのエネルギー補正c8--c9--c11) |
| F7 | pytorch index の扱い | [7.1](#71-pytorch-index-は-cu128-既定f7) |
| F8 | キャッシュ下限とゲート閾値の分離 | [2.2](#22-キャッシュ下限と発話長ゲートを分離するf8) |
| F9 | 長尺ソースのチャンク分割デコード | [2.7](#27-長尺ソースはチャンク分割デコードf9) |
| G1 | 閾値感度分析表 | [5.3](#53-reporthtml-の設計方針) |
| G2 | 音素被覆率・モーラ分布・F0 レンジ | [5.3](#53-reporthtml-の設計方針) |
| G3 | ピークリミッタ作動を黙らせない | [5.2](#52-lufs-正規化とピークリミッタg3) |
