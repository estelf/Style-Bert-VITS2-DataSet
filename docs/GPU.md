# GPU（CUDA）で実行する

## 既定: uv sync するだけで GPU が使える

`uv.lock` は torch を **CUDA 12.8 版（cu128）** で解決している。
NVIDIA ドライバが入った Linux マシンでは `uv sync` するだけで GPU が使える。

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.11.0+cu128 True
```

`torch.cuda.is_available()` が `False` の場合、`uv run make_dataset run <話者>` は自動的に CPU で走る
（正しさは同じ、速度だけ遅い）。

## CPU 運用に戻す（任意）

GPU ノードと CPU ノードで lock を共有できない場合、`pyproject.toml` の torch インデックスを
CPU 差し替え、その lock を `uv.lock` としてコミットする。

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

```bash
uv lock && uv sync
```

この差し替えは lock の解決先が変わるだけで、コードの変更は不要。
`uv.lock` を差し替えた場合は README の「インストール」にその旨を明記する。

## 速度の目安

| 環境 | 10分音声の目安 |
|---|---|
| CPU | 約 10 分 |
| RTX 4090 | 約 1〜2 分 |

`[asr2]`（whisper 二重掛け）と `[alignment]`（forced alignment）を有効にすると ASR 側の時間が増える。
速度より正確性を落としたい場合は `filters.toml` の `[asr2] enabled = false`、
`[alignment] enabled = false` で個別に無効化できる。

## 制約

- bf16 前提。非対応 GPU では `filters.toml` の `[asr] torch_dtype = "float32"` にする
- 初回実行時にモデルをダウンロードする。メイン ASR（Qwen3-ASR）で約 4GB、既定有効の追加機能
  （tags・LM・alignment）で合計約 3.5GB、asr2（whisper small）で約 0.5GB をキャッシュする
- CPU 実行時に限らず、モデルのダウンロードはネットワーク越しに一度だけ発生する
