# Style-Bert-VITS2-DataSet

Style-Bert-VITS2-Train 用データセットを、音声・動画ファイルの投入だけで全自動生成するツール群（実装ドラフト）。
設計は [docs/DESIGN.md](docs/DESIGN.md) を参照。

## 使い方

```bash
uv run make_dataset <input...> [--speaker 話者名] [--model-name モデル名] --config filters.toml
# 例: テストデータで1周 (話者名・データセットディレクトリ名は入力ディレクトリ名 haruka_ran_nobgm から自動採用)
uv run make_dataset test/asset/haruka_ran_nobgm/*.mp3 --config filters.toml
```

入力は全部で1話者分のデータセット。話者名と `Data/<名前>/` のディレクトリ名はデフォルトで入力ディレクトリ名から自動採用される（`--speaker` / `-m` で上書き可）。音声ファイルはスタイルフォルダを作らず `raw/` 直下にフラット配置する。
閾値・挙動設定は `filters.toml` に一元化されており、キーを変えるだけで再生成なしに挙動を変えられる
（デフォルト値は `src/make_dataset/config.py` の `DEFAULTS`）。

## 生成物 (`Data/<モデル名>/`)

- `esd.list` — 1行1文 `<hash>.flac|<話者>|JP|<セリフ>`（raw/直下のフラット配置）
- `raw.zip` — 44.1kHz・モノラル・LUFS正規化済み flac（納品前に同じ契約検証を自走で通す）
- `manifest.jsonl` — 全スコア・除外理由・dupグループ・status（1行=1発話）
- `report.html` — 除外内訳・ソース別件数・F0レンジなどの可視化

## ドラフト段階の既知の差し替えポイント（DESIGN.md との差分）

| Stage | DESIGN の候補 | ドラフト実装 |
|---|---|---|
| 1 | MFA/whisperX アライメント境界補正 | エネルギー谷による ±70ms 補正のみ |
| 2 | WavLM系 speaker embedding / ERes2Net | MFCC 統計埋め込みの cosine（プレースホルダ） |
| 2 | pyannote overlap | 未実装 (`overlap_flag=None`) |
| 2 | DRR・自己相関リバーブ検出 | 末尾減衰エネルギー比プロキシ |
| 3 | Qwen3-ASR-1.7B | バックエンド切替済み: `backend="qwen3-asr"` で transformers native 実行（既定は CPU で速い faster-whisper small）。信頼度はトークン平均確率 |
| 4 | UTMOS2 + SQUIM | torchaudio SQUIM_OBJECTIVE のみ (`quality_score=pesq`) |
| 5 | ④音声フィンガープリント重複排除 | 未実装（テキスト/音素ベースのみ） |
| 6 | 音素被覆率・モーラ分布ヒストグラム | レポートは除外内訳・ソース別件数・F0レンジまで |

## テスト

```bash
uv run pytest test/        # ゴールデン方式スモークテスト（tiny ASR で数秒音声から1周）
```
