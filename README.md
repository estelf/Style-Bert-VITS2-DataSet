# Style-Bert-VITS2-DataSet

Style-Bert-VITS2-Train 用データセットを、音声・動画ファイルの投入だけで全自動生成するツール群（実装ドラフト）。
設計は [docs/DESIGN.md](docs/DESIGN.md) を参照。

## 使い方

```bash
# 音声・動画は voice/<話者名>/ に置く (voice ルートは filters.toml の [paths] voice で変更可)
#   voice/speeker1/*.mp3, voice/speeker2/*.mp4 ...
uv run make_dataset run speeker1 --config filters.toml

# ディレクトリ指定・ファイル glob でも同じ (1 実行 = 1 話者分 = 同じディレクトリのファイル)
uv run make_dataset run voice/speeker1 --config filters.toml
uv run make_dataset run voice/speeker1/*.mp3 --config filters.toml

# 閾値を変えて再判定だけしたいとき (音声処理・ASR を再実行せず納品物を再生成)
uv run make_dataset rescore Data/speeker1 --config filters.toml
```

引数は**区切り文字を含むか否か**で扱いが決まる。`speeker1` のように区切りを含まないトークンは常に話者名として `voice/speeker1/` を読み、カレントに同名のファイル・ディレクトリがあってもそちらは読まない。voice 配下以外をパスで読ませたいときは `./speeker1`・`../raw/speeker1`・`/data/voice/speeker1` のように書く（メディア拡張子を持つ `clip.mp3` はファイル指定）。

入力は全部で1話者分のデータセット。話者名と `Data/<話者名>/` のディレクトリ名はデフォルトで入力ディレクトリ名から自動採用される（`--speaker` / `-m` で上書き可）。音声ファイルはスタイルフォルダを作らず `raw/` 直下にフラット配置する（入力も `voice/<話者名>/` 直下にフラット配置、サブフォルダは展開しない）。
受け付ける拡張子: mp3 wav flac m4a ogg opus aac wma mp4 m4v mov mkv webm avi mpg mpeg ts（ffmpeg でデコードできるもの）。
ASR は既定で **Qwen3-ASR-1.7B**（`[asr] backend = "qwen3-asr"`、初回に約 4GB をダウンロード）を使う。CPU だけで軽く回したいときは `backend = "faster-whisper"` + `model = "small"`（または `"tiny"`）に差し替える。
閾値・挙動設定は `filters.toml` に一元化されている。挙動パラメータのデフォルトは `src/make_dataset/config.py` の `DEFAULTS` にあるが、
**閾値 (`[thresholds]`) のコード側デフォルトは無い — キーを消せば該当チェックは無効化される**（キーが残っていてスコアが取得できない発話は「検証不能=不合格」）。
発話長ゲート（既定: **0.5 秒以上・12 秒未満**）も `[thresholds] min_seg_sec / max_seg_sec` で管理しており、キーを消せば無効化される。
`rescore` は中間キャッシュ `.cache/` と `manifest.jsonl` から納品物を再生成するため、`run` したディレクトリに `.cache/` があることが必要。

## 生成物 (`Data/<モデル名>/`)

- `esd.list` — 1行1文 `<hash>.flac|<話者>|JP|<セリフ>`（raw/直下のフラット配置）
- `raw.zip` — 44.1kHz・モノラル・LUFS正規化済み flac（納品前に同じ契約検証を全ファイルへ自走で通す）
- `manifest.jsonl` — 全スコア・全失格理由 (dropped_by)・dupグループ・status（1行=1発話。非有限値は null 化され厳密 JSON）
- `report.html` — 除外内訳・閾値感度分析・音素被覆率・モーラ分布・F0レンジ・人手レビュー待ち一覧
- `run.json` — 再現性メタデータ（使用した閾値・ASR モデル・ライブラリ版・commit hash）
- `.cache/` — 中間音声キャッシュ（セグメント別 44.1k flac。rescore の入力。納品物ではない）

## GPU で使う

手順は **[docs/GPU.md](docs/GPU.md)**（pytorch index の cu128 への差し替え → `uv lock && uv sync` → `[runtime] device` 確認）。
既定の `[runtime] device = "auto"` のままなら、GPU を挿すだけで Qwen3-ASR・SQUIM・faster-whisper が自動で CUDA に載る。

## ドラフト段階の既知の差し替えポイント（DESIGN.md との差分）

| Stage | DESIGN の候補 | ドラフト実装 |
|---|---|---|
| 1 | MFA/whisperX アライメント境界補正 | 騒音床ベースの side 考慮エネルギー境界補正（語尾は切らない） |
| 2 | WavLM系 speaker embedding / ERes2Net | MFCC 統計埋め込み + 次元スケーリング (CMVN) cosine（プレースホルダ） |
| 2 | pyannote overlap | 未実装 (`overlap_flag=None`。diarization モデルが必要) |
| 2 | DRR・自己相関リバーブ検出 | 末尾減衰エネルギー比プロキシ |
| 3 | Qwen3-ASR-1.7B | **既定バックエンド**: `backend="qwen3-asr"` で transformers native 実行（`Qwen/Qwen3-ASR-1.7B-hf`）。CPU で軽く回すときは `backend="faster-whisper"` に切替。信頼度はトークン平均確率 |
| 3 | ITN 正規化・英語→カタカナ | 未実装（NFKC + 記号除去 + 区切り文字保護のみ。学習側 dict_data 連携とセットで今後の課題） |
| 4 | UTMOS2 + SQUIM | torchaudio SQUIM_OBJECTIVE のみ (`quality_score=pesq`) |
| 5 | ④音声フィンガープリント重複排除 | MFCC フレーム列の MinHash 内容指紋 (`fp_sig`) で実装（Chromaprint 系の代替） |
| 6 | 感情クラスタ (emotion2vec) | 未実装 |

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

