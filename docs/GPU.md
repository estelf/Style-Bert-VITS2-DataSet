# GPU (CUDA) で実行する手順

既定は CPU wheel 固定（`pyproject.toml` の `pytorch-cpu` index）。GPU マシンでは以下の手順で切り替える。
切り替えると **Qwen3-ASR（既定 ASR）・SQUIM 品質スコア・faster-whisper（選択時）** が GPU で走る。

## 1. 前提チェック

```bash
nvidia-smi   # NVIDIA ドライバが入っていて GPU が見えること
```

`nvidia-smi` の `CUDA Version` が 12.8 以上なら cu128 を使う（下表を参照）。

| nvidia-smi の CUDA Version | 使う index url |
|---|---|
| 12.8 以上 | `https://download.pytorch.org/whl/cu128` |
| 12.6 〜 12.7 | `https://download.pytorch.org/whl/cu126` |
| それ以下 | ドライバ更新を推奨。古いままなら対応する cuXXX を選ぶ |

## 2. pyproject.toml の index を差し替えて再インストール

`pyproject.toml` の末尾にある pytorch-cpu index の url を書き換える:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu128"   # ← /cpu を /cu128 などへ
explicit = true
```

そのうえで再解決・再インストール:

```bash
uv lock && uv sync
```

## 3. 動作確認

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True <GPU名> と出れば OK
```

## 4. filters.toml の設定

```toml
[runtime]
device = "auto"   # 既定値のまま for OK。CUDA があれば自動で cuda。cpu 固定したいなら "cpu"
```

- **Qwen3-ASR（既定）**: `[runtime] device` に従い、bf16 で GPU に載る。追加設定不要。
- **faster-whisper を選んだ場合**: `[asr] device = "auto"` のままで GPU 検出される。
  `compute_type = "int8"` は CPU 量子化なので、cuda のときは実装側で `float16` へ自動切替される
  （明示したいなら `compute_type = "float16"`）。
- **SQUIM（Stage 4 品質スコア）**: `[runtime] device` に従う。追加設定不要。

CPU 実行に戻したいときは `[runtime] device = "cpu"` にする（wheel は GPU 版のままでも動く）。

## VRAM の目安とトラブルシューティング

| モデル | 精度 | 目安 VRAM |
|---|---|---|
| Qwen/Qwen3-ASR-1.7B-hf | bf16 | 約 4〜5 GB |
| faster-whisper small | float16 | 約 2 GB |
| SQUIM_OBJECTIVE | fp32 | ほぼ無視できる |

- **OOM になる**: `[asr] model` を `"Qwen/Qwen3-ASR-0.6B"`（transformers 版が公開されている場合）や
  faster-whisper `"tiny"` に落とす。他プロセス（ブラウザ等）の VRAM 占有も確認する。
- **`torch.cuda.is_available()` が False**: 手順 2 の `uv sync` がキャッシュを再利用して CPU wheel のまま
  になっていないか確認。`uv cache clean torch torchaudio && uv sync` で再取得する。
- **faster-whisper だけ cuDNN エラー**: ctranslate2 がシステムの cuDNN を要求するケースがある。
  `uv add nvidia-cudnn-cu12` で解決することが多い（Qwen3-ASR / SQUIM には不要）。
- **モデルのダウンロード先を変えたい**: `HF_HOME=/path/to/cache` を設定する（既定は `~/.cache/huggingface`）。
  Qwen3-ASR-1.7B は初回に約 4GB をダウンロードする。

## 補足: rescore は GPU 不要

`make_dataset rescore` は閾値再判定と納品物再生成だけで、音声処理・ASR を再実行しない。
dedup（MinHash）とレポート生成は CPU で完結するため、GPU 環境の切り替えなしで使える。
