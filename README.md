# Style-Bert-VITS2-DataSet

[Style-Bert-VITS2-Train](https://github.com/estelf/Style-Bert-VITS2-Train) 用のデータセットを、
音声・動画ファイルの投入だけで全自動生成するツール群。
`voice/<話者名>/` にファイルを置いて 1 コマンド実行すると、スライス・文字起こし・吟味・正規化を
通して `esd.list` + `raw.zip` + 判定レポートが出てくる。

**ドキュメントの読み分け**

| 知りたいこと | 場所 |
|---|---|
| まず動かす・ユースケース別の操作 | [チートシート.md](チートシート.md) |
| 各ステージが今何をするか（スコア・ゲート対応表） | [docs/PIPELINE.md](docs/PIPELINE.md) |
| 設計思想と実装状況（DESIGN 候補 vs 実装） | [docs/DESIGN.md](docs/DESIGN.md) |
| なぜその仕様・その閾値にしたか（実測・棄却した案） | [docs/DECISIONS.md](docs/DECISIONS.md) |
| GPU (CUDA) で動かす | [docs/GPU.md](docs/GPU.md) |

## 最短手順

```bash
uv sync                                   # 依存導入（既定は GPU wheel。CPU 運用は docs/GPU.md）
# 音声・動画は voice/<話者名>/ 直下にフラット配置（スタイルフォルダ・サブフォルダは作らない）
uv run make_dataset run speeker1 --config filters.toml     # -> Data/speeker1/
```

`Data/speeker1/report.html` をブラウザで開くと、採用/除外の内訳と「どの閾値をどう動かすか」が分かる
（ダーク基調の単一 HTML・外部 CDN 依存なし）。納品物は `esd.list` + `raw.zip`。

```bash
# 閾値を変えて再判定だけする（音声処理・ASR を再実行せず納品物を再生成）
uv run make_dataset rescore Data/speeker1 --config filters.toml

# レポートの見た目だけ先に確認する（合成レコード。音声・モデル不要）
uv run python scripts/preview_report.py    # -> Data/_report_preview/report.html
```

## 入力のルール

- 引数は**区切り文字を含むか否か**で扱いが決まる。`speeker1` のように区切りを含まないトークンは常に
  話者名として `voice/speeker1/` を読み、カレントに同名のファイル・ディレクトリがあってもそちらは読まない。
  voice 配下以外を読ませたいときは `./speeker1`・`../raw/speeker1`・`/data/voice/speeker1` のように
  パスで書く（メディア拡張子を持つ `clip.mp3` はファイル指定）。
- **1 実行 = 1 話者分（同じディレクトリのファイル）**。複数ディレクトリにまたがる入力は話者名が
  定まらないため拒否する。
- 話者名と `Data/<話者名>/` のディレクトリ名は、既定で入力ディレクトリ名から自動採用される
  （`--speaker` / `-m` で上書き可）。voice ルートは `[paths] voice`（既定 `"voice"`）で変更可。
- 受け付ける拡張子: mp3 wav flac m4a ogg opus aac wma mp4 m4v mov mkv webm avi mpg mpeg ts
  （ffmpeg でデコードできるもの。動画もそのまま投入できる）。

## 生成物（`Data/<モデル名>/`）

| ファイル | 内容 |
|---|---|
| `esd.list` | 1 行 1 文 `<hash>.flac|<話者>|JP|<セリフ>`（raw/ 直下のフラット配置） |
| `raw.zip` | 44.1kHz・モノラル・LUFS 正規化済み flac（納品前に同じ契約検証を全ファイルへ自走で通す） |
| `manifest.jsonl` | 全スコア・全失格理由（`dropped_by`）・dup/多様性・status（1 行=1 発話、非有限値は null） |
| `report.html` | 自動判定の内訳＋見方・凡例（ゲート別感度／多様性メトリクス／文末の開き／2 系統 ASR／カバレッジ／見送り／読めない語）。表はソート・検索可 |
| `text_overrides.jsonl` | 判定の確定・本文修正の返し先（`utt_id` キー。あれば `run`/`rescore` の先頭で再適用され、自動判定より常に優先） |
| `run.json` | 再現性メタデータ（使用した閾値・ASR モデル・ライブラリ版・commit） |
| `.cache/` | 中間音声キャッシュ（セグメント別 44.1k flac。`rescore` の入力。納品物ではない） |
| `.cache_orig/` | 修復（`[repair]`）で差し替えた発話の元音声（既定では作られない） |

## 押さえておく 4 つの原則

1. **Stage 2〜4 は「記録」だけして「判定」しない。** 各発話の `scores` にスコアを積み上げ、
   採用/除外は Stage 6 の閾値で決まる。だから `rescore` は音声処理・ASR を再実行せずに
   閾値だけ変えて納品物を再生成できる（例外: `unknown_tokens` / `text_tail_open` はテキストから、
   `[asr2]` はキャッシュ音声から `x_cer` だけ引き直す）。
2. **閾値にコード側デフォルトは無い。** `filters.toml [thresholds]` からキーを消せば（コメント化すれば）
   そのチェックは走らない。挙動パラメータのデフォルトは `src/make_dataset/config.py` の `DEFAULTS`。
3. **検証不能＝不合格。** 閾値が有効なのにスコアが取得できない発話は不合格になる。
   新しいゲートをコメント解除するのは、対応するプロバイダを有効にした `run` の**後**にすること
   （旧 manifest への `rescore` で全件落ちる。`run` 起動時に警告を出す）。
4. **人手レビューページは作らない。** 人が動かすのは閾値と `text_overrides.jsonl` だけ。
   機械が自動採用を避ける `pending_review`（修復音声・ユーザー辞書候補・`review_*` 閾値の受け皿）は
   納品に含まれず、`report.html` の「見送り」に出る。

## 設定の要点（`filters.toml`）

- **ASR**: 既定は **Qwen3-ASR-1.7B**（`[asr] backend = "qwen3-asr"`、初回に約 4GB をダウンロード）。
  CPU だけで軽く回したいときは `backend = "faster-whisper"` + `model = "small"`（または `"tiny"`）。
- **オプトインの重量検証**: 同梱設定は `[asr2]`（第 2 ASR クロス検証）/ `[tags]`（AudioSet タグ）/
  `[lm]`（MLM 擬似 PPL）/ `[alignment]`（CTC アライメント）が既定有効（初回ダウンロード合計 約 3.5GB）。
  ゲートは `max_x_cer`・`reject_audio_tags`・`max_lm_ppl`（いずれも除外）のみ既定有効で、
  人手レビュー行きを作らない完全自動運用。`[stage4] subjective` / `[overlap]` / `[repair]` は既定オフ。
  各軸のねらいと採用/棄却の判断は [docs/PIPELINE.md](docs/PIPELINE.md#重量検証オプトインの一覧) と
  [docs/DECISIONS.md](docs/DECISIONS.md#4-重量検証オプトインの採用と棄却)。
- **発話長**: ゲートは `[thresholds] min_seg_sec`（0.5 秒以上）・`max_seg_sec`（12 秒未満）。
  中間キャッシュの物理下限は `[stage1] cache_min_sec`（0.2 秒）で分離している
  （`min_seg_sec` を緩めたときに救えるようにするため）。
- **納品音声の語頭・語尾**: `[stage1] output_pad_ms`（既定 20ms）の無音マージンを付ける。
  `speech_pad_ms` は VAD 側の切り出しを拡げるだけで境界補正窓の内側でトリムされ、出力には残らない。
- **多様性選別（Stage 5b）**: 重複排除の後に「重複ではないものの似た」発話も間引く（`[diversity]`、既定有効）。
  音素カバレッジ保護・韻律/ソース別シェア上限・多様性利得・目標本数が選べる。
- **出力契約**: 44.1kHz・モノラル固定。`[output]` をこれ以外にすると書き出し前に明確なエラーで止まる。

## 判定の確定（`text_overrides.jsonl`）

`Data/<話者>/text_overrides.jsonl` に 1 行 1 レコードで置くと、`run`/`rescore` の先頭で再適用される。

```json
{"utt_id": "ab12cd34ef567890-003", "text": "修正後のセリフ"}
{"utt_id": "ab12cd34ef567890-007", "status": "adopted"}
```

テキスト上書きは派生スコア（モーラ数・話速・kana 比率等）を引き直し、
**dedup・話速中央値・多様性選別より前**に効く。`status` 上書き（`adopted`/`excluded`/`pending_review`）は
自動判定より常に優先される。書き方は `report.html` の「見方・凡例 6」に出る。

## GPU で使う

`[runtime] device = "auto"` のままなら、GPU を挿すだけで Qwen3-ASR・SQUIM・faster-whisper が
自動で CUDA に載る。既定の uv index は cu128（GPU wheel）。CPU 運用に戻す手順も含めて
**[docs/GPU.md](docs/GPU.md)** を参照。

## テスト・開発ツール

```bash
uv sync                    # 依存導入（dev グループ: black / pyright / ruff / pytest を含む）
uv run pytest test/        # ゴールデン方式スモークテスト（tiny ASR で数秒音声から1周 + rescore 往復）
uv run ruff check src test # リント（書式は black 側が担当。ruff format は使わない）
uv run black src test      # 整形（line-length = 100）
uv run pyright             # 型検査（src と test。設定は pyproject の [tool.pyright]）
```

型検査は `uv run pyright` で実行する（`uvx pyright` だとプロジェクトの `.venv` を使わず、
torch / numpy などの import 解決に失敗して誤検知が出る）。
