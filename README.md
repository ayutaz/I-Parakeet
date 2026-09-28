# I-Parakeet 再現実装

論文 **"I-Parakeet: Integer-Only Conformer ASR on Mobile NPU"**（Taichi Nishimura, arXiv [2609.30846](https://arxiv.org/abs/2609.30846)）の再現を目指すリポジトリ。

NVIDIA Parakeet-CTC-0.6B を浮動小数点演算なし・CPU フォールバックなしでスマートフォン NPU（Snapdragon 7s Gen 3, QNN）上で動かす研究。
論文値: LibriSpeech test-other WER 4.97%（実機）/ 5.32%（整数シミュレータ）、RTF 0.048、ピークメモリ 612 MiB。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/01_paper_summary.md](docs/01_paper_summary.md) | 論文の内容整理（手法・実験・数値）と、本文に書かれていない点の一覧 |
| [docs/02_technical_survey.md](docs/02_technical_survey.md) | 再現のための技術調査（NeMo の実装、整数カーネル、QNN ツールチェーン、評価方法） |
| [docs/03_reproduction_plan.md](docs/03_reproduction_plan.md) | フェーズ別の再現計画、成功基準、リスク |

## 現状

- [x] 論文調査・技術調査・計画
- [x] Swish の minimax 近似（§3.2, Table 3 の最大誤差）: `python scripts/fit_swish_approx.py` で論文の係数 a* = −0.1240, c* = 2.4632 と 5 条件すべての最大誤差を再現
- [ ] Phase 0: FP32 ベースライン（1.87 / 3.76）
- [ ] Phase 1〜6: `docs/03_reproduction_plan.md` を参照

## 実行例

```bash
pip install numpy scipy
python scripts/fit_swish_approx.py
```
