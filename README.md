# I-Parakeet 再現実装

論文 **"I-Parakeet: Integer-Only Conformer ASR on Mobile NPU"**（Taichi Nishimura, arXiv [2609.30846](https://arxiv.org/abs/2609.30846)）の再現を目指すリポジトリ。

NVIDIA Parakeet-CTC-0.6B を浮動小数点演算なし・CPU フォールバックなしでスマートフォン NPU（Snapdragon 7s Gen 3, QNN）上で動かす研究。
論文値: LibriSpeech test-other WER 4.97%（実機）/ 5.32%（整数シミュレータ）、RTF 0.048、ピークメモリ 612 MiB。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/01_paper_summary.md](docs/01_paper_summary.md) | 論文の内容整理（手法・実験・数値）と、本文に書かれていない点の一覧 |
| [docs/02_technical_survey.md](docs/02_technical_survey.md) | 再現のための技術調査（NeMo の実装、整数カーネル、QNN ツールチェーン、評価方法） |
| [docs/03_reproduction_plan.md](docs/03_reproduction_plan.md) | マイルストーン（M0〜M8）ごとの目的・ゴール、リスク |
| [docs/04_results.md](docs/04_results.md) | 実行結果、論文に書かれていない事項の決定、残りの実行手順 |

## 現状

マイルストーンごとの目的・ゴールは [docs/03_reproduction_plan.md](docs/03_reproduction_plan.md)、達成状況は [docs/04_results.md](docs/04_results.md) を参照。

M0〜M7 のコードはすべて実装済みで、テストは 164 件すべて通過している（TDD、uv で環境管理）。
一方、実データ・実機が必要な数値（WER、RTF、Fig. 2 の実測値）は未測定である。調査・実装を行ったクラウド環境では、モデル・データのダウンロードがブロックされており、GPU・QAIRT SDK・実機もなかったため。

| # | マイルストーン | 状態 |
|---|---|---|
| M0 | Swish 近似の再現 | ✅ 論文の係数と Table 3 の最大誤差を再現 |
| M1 | 評価基盤と FP32 ベースライン | 🟡 実装・テスト済み、実データでは未測定 |
| M2 | スタンドアロン FP32 参照実装 | ✅ 実際の NeMo 3.0.0 モジュールと数値一致（小型構成） |
| M3 | レンジ解析（Fig. 2） | 🟡 実装・テスト済み、実チェックポイントでは未実行 |
| M4 | 整数シミュレータ | ✅ 全カーネル・float 非混入・INT32 検査・全レシピ |
| M5 | シミュレーション実験（Table 2, 3） | 🟡 パイプラインと順位判定は完成、実データでは未実行 |
| M6 | NPU グラフと integer-only 検証 | 🟡 ONNX・エンコーディング・QDQ の静的検証は通過、QAIRT と実機は未実施 |
| M7 | 実機評価（Table 1） | 🟡 ツールは完成（onnxruntime を実機の代わりにして検証）、実機は未実施 |
| M8 | 結果まとめ | ✅ docs/04_results.md |

## セットアップとテスト

```bash
uv sync --extra deploy
uv run pytest            # 164 tests
uv run python scripts/fit_swish_approx.py          # M0
uv run python -m scripts.kernel_report --out results/kernels
```

モデル・データがある環境での実行手順は [docs/04_results.md §4](docs/04_results.md) を参照。

## 構成

| パス | 内容 |
|---|---|
| `iparakeet/eval/` | Whisper 正規化・WER・マニフェスト・RTF・推論ランナー（M1） |
| `iparakeet/model/` | NeMo 互換の前処理と FastConformer-CTC、`.nemo` ローダ、入力長バケット（M2） |
| `iparakeet/analysis/` | レンジ解析・キャリブレーション統計・Fig. 2・カーネル精度（M3） |
| `iparakeet/quant/`, `iparakeet/intops/` | 固定小数点・float 混入検出・厳密な整数 GEMM、各整数カーネル（M4） |
| `iparakeet/sim/` | integer-only Parakeet、レシピ、評価、論文表との比較（M4, M5） |
| `iparakeet/deploy/` | ONNX グラフ・エンコーディング・QDQ・QAIRT テンプレート・実機評価（M6, M7） |
| `scripts/` | 各マイルストーンの CLI |
| `tests/` | テスト（NeMo との数値一致用フィクスチャを含む） |
