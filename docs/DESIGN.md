# Style-Bert-VITS2-DataSet 設計ドキュメント

本リポジトリは [Style-Bert-VITS2-Train](https://github.com/estelf/Style-Bert-VITS2-Train) 用のデータセットを、**音声・動画ファイルの投入だけで全自動生成する**ことを目的とするツール群である。

- 学習側の契約: `Data/<モデル名>/` に `esd.list`（1行1文 `<音声ファイル名>|<話者名>|JP|<セリフ>`）+ `raw.zip`（44.1kHz・モノラル、スタイル別フォルダ可）を置く
- 投入側の契約: 1話者分の音声・動画を `voice/<話者名>/` 直下にフラット配置する（スタイルフォルダ・サブフォルダは作らない）
- 本リポジトリの責任範囲: 生の投入 → スライス・ASR・吟味・正規化 → 完成済みデータセット出力
- 設計原則: **全工程が「スコアを manifest に積み上げ、最後に閾値で採用/除外を決める」形**で統一される。フィルタロジックと閾値設定を分離し、人間介入なしで「どこで何がなぜ落ちたか」を説明可能にする

## 全体パイプライン

```
input → Stage 0 抽出 → Stage 1 セグメント化 → Stage 2 発話スコアリング
      → Stage 3 ASR・G2P整合 → Stage 4 音声品質スコア → Stage 5 重複排除 → Stage 5b 多様性選別
      → Stage 6  出力(esd.list + raw.zip + レポート)
```

CLI は1本を想定:

```bash
uv run make_dataset run <話者名> --config filters.toml            # voice/<話者名>/ を展開して読む (既定の投入形態)
uv run make_dataset run <input...> -m <モデル名> --config filters.toml
uv run make_dataset rescore Data/<モデル名>/ --config filters.toml   # 閾値変更だけの再判定 (manifest + 中間キャッシュから)
```

### 投入の解決（`run` の引数）

音声・動画は `voice/<話者名>/` にフラット配置する。voice ルートは `filters.toml` の `[paths] voice`（既定 `"voice"`、カレント基準）で変更できる。**話者名とパス指定は「見た目」で分岐**する。`voice/<トークン>` がカレントの同名ファイル・ディレクトリと静かに衝突するのを防ぐため、話者名からカレントへのフォールバックは作らない。話者名と `Data/<モデル名>/` は解決できたディレクトリ名から自動採用する（`--speaker` / `-m` で上書き可）。

| 引数の見た目 | 扱い | 解決される内容 |
|---|---|---|
| 区切り文字を含む / `.` `~` で始まる / メディア拡張子を持つ | **パス指定** | カレント基準（または絶対パス）のファイル 1 本、またはディレクトリ中のメディアファイルを名前順に展開（サブフォルダは辿らない） |
| それ以外の裸のトークン | **話者名** | `<voice>/<話者名>/` のみ。カレントに同名があっても解決に割り込まない |
| 解決先が存在しない | **エラー（終了コード 2）** | 処理は走らせない。`./<トークン>`（パス）と `voice/<トークン>`（話者名）のどちらを意図したかのヒントを出す |

- `run speeker1`（話者名）と `run voice/speeker1`（パス）は同じ結果になるが、`run speeker1` は**常に voice 配下**を読む
- **1実行 = 1話者分（同じディレクトリのファイル）が契約**。複数ディレクトリを渡すと話者名が定まらないため拒否する（`run voice` のように voice ルート自体を渡すことも同じガードで止まる）
- 対応拡張子: mp3 wav flac m4a ogg opus aac wma mp4 m4v mov mkv webm avi mpg mpeg ts（Stage 1 が ffmpeg で拡張子非依存にデコードするので、動画もそのまま投入できる）

---

## Stage 0: 投入・前処理

- 引数の解決（話者名 → `voice/<話者名>/`、1話者契約のガード）は CLI の入力展開までで行い、Stage 0 は確定したファイル列を受ける
- ffmpeg で音声抽出し 44.1kHz float wav 化、ファイルにハッシュIDを付与
- バイトハッシュ一致（同一ファイル再投入）はここで即除外

## Stage 1: 音声セグメント化

- **Silero-VAD** で音声区間を検出し文単位にスライス（契約は1行=1文）
- **強制アライメント境界補正**: VAD 境界は曖昧なので、MFA / whisperX 系アライメント＋モーラ数逆算で単語境界±約70ms に精緻化。語尾切れ・先端ノイズ混入を防ぐ
- 無音パディングのトリム、極端な長短セグメントはスコアとして記録
- **長尺ソースのチャンクデコード**: 1 本の長い動画などでラン全体が OOM しないよう、`decode_chunk_min`（既定 10 分）ごとに分割デコードする。チャンク境界で VAD のマージ/分割が全体デコードとズレないよう前後に「最長発話＋無音/音声判定＋補正窓」ぶんの文脈を持たせ、発話の取りこぼし・重複が起きないようにする（所有領域は開始時刻で一意に割り当て）
- **キャッシュ下限とゲート閾値の分離**: 中間キャッシュする物理下限 `cache_min_sec`（既定 0.2 秒）未満の VAD 出力は音声を作らず manifest に `too_short` として残すだけ。採用/除外は閾値 `min_seg_sec` が決める（rescore で `min_seg_sec` を緩めたときに救えるよう、両者は分離）

## Stage 2: 発話ごとのスコアリング（1セグメント＝1レコード）

各スコアは `manifest.jsonl` に記録され、除外可否は Stage 6 出力時の閾値設定で決まる。

| スコア | 手段（候補） | 検出対象 |
|---|---|---|
| `speaker_sim` | MFCC 統計埋め込み（次元スケーリング）を manifest の `emb` に保存し、**主話者 centroid とのコサインを Stage 6 直前に頑健サブセットから 2 パスで再計算** | 別人混入、加工ノイズで別人化した音声（全件中央値だと不良発話で centroid が汚染されるため、暫定 centroid から遠い外れ値を落として引き直す） |
| `overlap_flag` | pyannote overlap 等 | 2人同時発話（被り） |
| `reverb_score` | DRD（直接音/残響音比）・自己相関 | ホールかかり音声、オケ帯域漏れ |
| `rolloff_hz` | スペクトルロールオフ推定 | 16k/24k をアップコンしただけの見せかけ高品質 |
| `jitter / shimmer / F0分布` | praat-parselmouth | 叫び・嗄れ・異常ピッチ |
| `snr`, `lufs` | torchaudio SQUIM / pyloudnorm | 無音率・ダイナミクス異常 |

- BGM/ボーカル分離は**行わない**（BGMなしのクリーンな投入を前提とし、汚い入力は検出して弾く方針）
- ASR は既定で Qwen3-ASR に加え whisper（asr2）も走らせ、両者の読み差分（x_cer）でガチャ文字起こしを検出する。
  当初は「単一モデル＋G2P 整合チェック（モーラ数×話速ズレ・アライメント尤度）」だけで検知する設計だったが、
  検知率が低く、二重掛けのランニングコストが下回ったため採用に転じた（経緯は [DECISIONS.md §3](DECISIONS.md)）

## Stage 3: 文字起こしと言語系チェック

ASR は **Qwen/Qwen3-ASR-1.7B** を第一候補（日本語特化・置信度が出せる）。

**G2P 整合チェック（3段構え）** — 学習側と同一エンジン（pyopenjtalk-plus）で検証するのがミソ。通れば学習側で raise しないことが保証される。

1. **G2P が通るか**: 学習側の読みチェック（YomiError）の事前打診。学習側 `text_to_sep_kata` の失敗条件（pron=「、」の読めない語・カタカナでない読み）を同一エンジン＋同一正規化で鏡写しし、学習側で実際に落ちる語（希少漢字・古いカタカナ・正規化を生き残る記号）を unknown トークンとして数える（語自体は `yomi_bad_words` に残す）。「ユーザー辞書候補」または「除外」へ自動振り分け（`dict_data/` 連携）
2. **モーラ数 × 話速 vs 実音声長**: テキストからモーラ数を確定し、音声長÷モーラ数が話者中央値から乖離ならフラグ。2文混入の自動分割トリガにも、ASR 欠落・ガチャ検出にも効く
3. **アライメント尤度**: ASR テキストと音声の不一致そのものを alignment score / whisperX 確信度で検出

整形系:
- ITN 正規化（数字・記号の読み、学習側 dict_data とセットで管理）
- 句読点・全角/半角正規化、英語→カタカナ表記ルール
- テキストフィルタ: 文字種が少ない / 文字数が少ない / カタカナのみ / アルファベットのみ / フィラー率高等を棄却

## Stage 4: 音声品質スコア

- **UTMOS2**（知覚品質）＋ SQUIM 系 → `quality_score` に集約
- ここまで全スコアは「除外」ではなく「記録」止まり。閾値は `filters.toml` で一元管理し、再生成なしで挙動変更可能にする

## Stage 5: 重複排除（この順番で安い順に）

1. テキスト完全一致（正規化後）
2. テキストファジィ（shingle Jaccard / 編集距離。ASR ガチャによる微差も吸収）
3. 音素列 MinHash/LSH（O(N²) を避ける）
4. ズレ切り出し音声フィンガープリント（同一動画クリップの位置ズレ重複。speaker embedding は話者ベクトルなので内容照合には使わない）

判定後は**グループ代表1本のみ自動採用**（基準: quality → ASR confidence → SNR の決定式）。
manifest に `dup_group_id` / `dup_reason=exact|fuzzy_audio|fuzzy_text` / `dropped_by` を残す。

## Stage 5b: 多様性選別（重複ではなく「似た」データも間引く）

重複排除はほぼ同一の発話を代表 1 本にまとめるだけなので、その後の多様性選別（`[diversity]`）がデータセット全体の多様性を上げる。

- 重複排除後に残った候補（閾値通過かつ代表）を**品質順**に貪欲選択し、**既に採用済みのどれかと類似しすぎる発話を除外**する。採用集合の総対ペア類似度を `sim_threshold` 未満に抑える独立集合を greedily 作る（＝どの 2 本も似ていない集合）。
- 類似の物差しは**内容の近さ**で測る `metric`（`phoneme` 既定＝音素三量文字列 Jaccard / `text`＝shingle / `audio`＝内容指紋一致率）。話者/音響ベクトル `emb` は単一話者コーパスで別内容まで間引くため不採用。`sim_threshold` は dedup より**緩く**（既定 0.6、dedup 閾値未満）して「似た」レベルまで間引く。集合系・指紋は LSH/band 衝突で候補を出して厳密判定（大コーパスでも O(N²)/メモリ有界）。実測: 別発話は音素三量 Jaccard ≤0.05、似た内容 ~0.67、完全一致 ~1.0。
- `protect_coverage`：ある音素の**唯一の供給源**は多様性名目では落とさない（多様性のために音素カバレッジを犠牲にしない）。
- 除外理由は `diversity_reason`（`similar:<代表utt>`）→ `dropped_by` に `diversity:...`。`emb`/`fp_sig`/テキストから再計算するので **rescore で `metric`/閾値を再調整できる**（ASR 再実行不要）。


## Stage 6: 出力

```
Data/<モデル名>/
  ├─ esd.list        # <hash>.flac|<話者>|JP|<セリフ>（raw/直下のフラット配置）
  ├─ raw.zip         # 44.1k/mono・LUFS正規化済み flac
  ├─ manifest.jsonl  # 全スコア・除外理由(全件)・dupグループ・status
  ├─ report.html     # 除外内訳、閾値感度分析、多様性選別の除外数、Stage1スキップ、音素/F0カバレッジ、モーラ分布（HTMLエスケープ済み）
  ├─ run.json        # 再現性メタデータ（閾値・ASRモデル・ライブラリ版・commit）
  └─ .cache/         # 中間音声キャッシュ（rescore 用。納品 raw/ とは分離）
```

- 出力前に学習側 `preprocess/check_dataset.py` と同じ検証（44.1kHz/モノラル等）を自走して通してから納品。契約は **44.1kHz・モノラル固定**で、`[output]` がこれ以外なら書き出し前に明確なエラーで止まる（全作業後に遅れて落ちない）
- **閾値と独立の不変条件**: 空セリフ / モーラ0 は学習不能なので、対応キーを消していても常に不合格（`empty_text`）。Stage 1 の `too_short`/`source_error`（音声キャッシュ無し）も閾値と無関係に常に不合格
- **ゲート配線**: 計算済みなのに採用/除外に使っていなかった指標を `[thresholds]` に接続（既定有効: `max_digital_silence` / `min_voiced_ratio`。コメントで用意: `min_f0_hz`/`max_f0_hz`/`min_stoi`/`min_si_sdr`/`min_input_lufs`/`reject_overlap`）。感度分析は閾値キー単位で数える（min/max が同じスコアを見ても混線しない）
- `on_g2p_fail` / `on_unknown_tokens` の値は起動時に検証（誤字は即エラー）。unknown は学習側で読めない語の数で、それを含むデータは**既定で不採用**（そんな読み方からしか得られない特徴は少なく、納品しても学習側 preprocess で落ちるだけなので旧 `"adopt"` オプションは削除）。読めない語は `yomi_bad_words` に残り report.html に一覧化される（ユーザー辞書登録の決定材料）
- カバレッジレポート: 音素被覆率・モーラ数分布・F0範囲などから「何を追加投入すべきか」を自動コメント

## データ契約（レコード単位）

`manifest.jsonl` 1行 = 1発話。例:

```json
{"utt_id": "23d9900d1e68ea43-000", "src": "input.mp4", "start": 12.3, "end": 15.1,
 "text": "…", "lang": "JP",
 "scores": {"speaker_sim": 0.87, "snr": 22.1, "lufs": -23.0, "pesq": 3.9,
            "asr_conf": 0.94, "mora_rate": 5.2, "align_conf": 0.91},
 "dup_group_id": null, "dup_reason": null, "diversity_reason": null,
 "status": "adopted", "dropped_by": null}
```

- `diversity_reason` は多様性選別（Stage 5b）が「既採用と類似すぎて」落とした理由（`similar:<代表utt>`）。除外時は `dropped_by` に `diversity:...` として現れる

- 人間レビューの受け皿は `status=pending_review` として実装済み（`[review]` で有効化）。
  将来のレビュー GUI（Gradio で再生→校正→OK/NG）も同じ構造に乗せられる

## 技術スタック（設計候補と実装状況）

| 用途 | 設計候補 | 現状 |
|---|---|---|
| VAD / スライス | Silero-VAD | 実装。エネルギーベース境界補正つき（強制アライメント代替） |
| 強制アライメント | MFA（ja）、whisperX | 不採用。境界補正はエネルギーベースで代替。text↔audio 検証として CTC 強制アライメント（`[alignment]`）を実装 |
| ASR | Qwen3-ASR-1.7B（transformers、bf16）＋ whisper（asr2、既定 small。読み差分でガチャ検知） | 実装（両系統が既定有効） |
| 話者埋め込み | WavLM系 speaker embedding、ERes2Net/TitaNet | **プレースホルダ**（MFCC 統計）。speaker_sim ゲートは動くが、実エンコーダへの差し替えは未着手 |
| G2P | pyopenjtalk-plus（学習側と同一・必須） | 実装 |
| 音声解析 | praat-parselmouth、torchaudio SQUIM、pyloudnorm | 実装（jitter/shimmer/F0、SNR/STOI、LUFS） |
| UTMOS | UTMOS2（ノイジー品質評価） | 未実装。非侵入 MOS（SQUIM_SUBJECTIVE、`[stage4]`）が近似の役割を担う |
| 感情クラスタ | emotion2vec（オプション） | 未実装 |
| CLI / 設定 | typer + filters.toml（uv 構成は学習側と統一） | 実装 |

## テスト方針

- 学習側に習うゴールデン方式: 数秒分の音声でパイプライン1周のスモークテスト、生成された esd.list/raw.zip が契約検証を通ることを確認
- 各フィルタの閾値変更が manifest 再生成だけで可能であることを契約チェック
- 入力解決の契約テスト: 話者名とパス指定の判別（カレント同名と衝突しない）、1話者ガード

### 品質ゲート（`uv run` 経由で常に緑を維持）

| コマンド | 役割 | 取り決め |
|---|---|---|
| `uv run pytest test/` | スモーク + 入力解決の契約テスト | ゴールデン方式。tiny ASR で数秒音声から1周 + rescore 往復 |
| `uv run ruff check src test` | リント | 書式は black 担当（`ruff format` は使わない）。既定の広域ルールは select で選び直し、日本語コメントで誤検知する `RUF001/002/003` と `E501` は無効。`BLE001`（広い except）は有効のまま、意図した箇所だけ `# noqa` |
| `uv run black src test` | 整形 | `line-length = 100`（日本語コメント・HTML 文字列で行が伸びるため。既定 88 では既存の 83 行が超過した） |
| `uv run pyright` | 型検査 | `src` と `test`。`uvx pyright` はプロジェクトの `.venv` を使わず import 解決に失敗して誤検知するため使わない |
