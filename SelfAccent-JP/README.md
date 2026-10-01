# SelfAccent-JP

**録音なしで、raw text の日本語 TTS に「読み＋ピッチアクセント」タグを教える**

**"Self-Distilled Pronunciation and Accent Control for Neural TTS"**（arXiv [2609.17234](https://arxiv.org/abs/2609.17234)）の非公式実装（日本語向け）。

```
魑魅魍魎が跋扈する。                                     ← 骨格が誤読する
<PHON_START>チ'ミ/モーリョー<PHON_END>が<PHON_START>バ'ッコ<PHON_END>する。   ← 読みとアクセントを指定
```

CosyVoice2 や Sarashina2.2-TTS のように raw text を直接読む TTS は、G2P を持たないので、難読語の読みとアクセントを直せない。
本手法では、凍結した骨格自身が正しく読める**一般語**の出力を教師にし、同じ文の語をタグに置き換えた入力へ**自己蒸留**する。
学習するのは LoRA とタグ記号の埋め込み行だけで、**新しい録音は不要**。

> 注意: 論文の本文（PDF）は調査環境から読めなかった。アブストラクトと検索要約、および前身の UtterTune（MIT）をもとにした実装である。論文に書かれていない点と、本実装での決定は [docs/03_architecture.md §5](docs/03_architecture.md) にまとめた。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/01_survey.md](docs/01_survey.md) | 直近 90 日（2026-07〜09）の TTS / VC / 音声生成論文の調査と、この論文を選んだ理由 |
| [docs/02_oss_comparison.md](docs/02_oss_comparison.md) | 既存 OSS（UtterTune、VOICEVOX、Style-BERT-VITS2、CosyVoice2 など）との違い |
| [docs/03_architecture.md](docs/03_architecture.md) | 構成、データの流れ、実在の骨格（CosyVoice2 など）への移植方法、論文に書かれていない点の決定 |
| [docs/04_benchmark.md](docs/04_benchmark.md) | ベンチマーク方法（論文規模の聴取評価と、CPU で回る自動評価） |
| docs/05_results.md（作成予定） | 小規模データでの動作確認の結果 |

## 仕組み

```
一般語を含む文 ──(採掘)──▶ 教師: 「橋を見ました。」
                            生徒: 「<PHON_START>ハシ'<PHON_END>を見ました。」
凍結した骨格(教師の文) ──▶ 目標（mel / 音声トークン）
骨格＋[タグ埋め込み＋LoRA](生徒の文) ──骨格自身の損失──▶ 目標に合わせる（adapter だけ学習）
推論: 任意の語（未知語も含む）をタグで書く
```

タグの記法は UtterTune と互換:

- `'` はアクセント核の直後に置く。`'` がなければ平板型
- `/` はアクセント句の区切り

## セットアップ

```bash
cd SelfAccent-JP
uv sync
uv run pytest
```

依存: PyTorch、pyopenjtalk-plus（辞書・HTS 音声を同梱）、pyworld、numpy / scipy / soundfile。GPU は不要。

## 使い方

### 1. 小規模データでの全工程（CPU で約 1.5 時間）

```bash
bash scripts/run_toy_experiment.sh
```

内訳:

```bash
# 1) トイコーパス: 一般語 141 語 × 定型文。OpenJTalk の HTS 音声で読み上げる（約 40 分）
uv run python -m scripts.make_toy_corpus --out data/toy
# 2) raw text 骨格（TinyMatcha、5M パラメータ）の学習。実在の骨格を使うなら不要
uv run python -m scripts.train_backbone --data data/toy --out exp/backbone --steps 8000 --threads 3
# 3) 自己蒸留ペアの採掘（テキストだけ）
uv run python -m scripts.mine_pairs --sentences data/toy/mine.jsonl \
    --count-corpus data/toy/train.jsonl --out exp/pairs.jsonl --min-count 3 --identity-ratio 0.1
# 4) 自己蒸留（教師の生成 → タグ埋め込み＋LoRA の学習）
uv run python -m scripts.train_selfdistill --backbone exp/backbone/model.pt \
    --pairs exp/pairs.jsonl --out exp/adapter --steps 1500
# 5) ベンチマーク（判定器の検証 → raw / kana / tag / 全アクセント型の掃引）
uv run python -m scripts.benchmark validate-judge --items data/toy/eval_difficult.jsonl --out results/judge_validation.json
uv run python -m scripts.benchmark run --backbone exp/backbone/model.pt --adapter exp/adapter/adapter.pt \
    --items data/toy/eval_difficult.jsonl --out results/benchmark_difficult.json
```

### 2. 推論

```bash
# タグを直接書く
uv run python -m scripts.infer --backbone exp/backbone/model.pt --adapter exp/adapter/adapter.pt \
    --text "昨日、<PHON_START>ソ'ゴ<PHON_END>について話しました。" --out out.wav
# 平文中の語にタグを付ける（G2P の出力などをそのまま渡せる）
uv run python -m scripts.infer ... --text "齟齬について調べました。" --tag "齟齬=ソ'ゴ" --out out.wav
# アクセントを変えてみる（平板型）
uv run python -m scripts.infer ... --text "齟齬について調べました。" --tag "齟齬=ソゴ" --out out_heiban.wav
```

波形は Griffin-Lim で作る。mel は HiFi-GAN 互換（22.05 kHz、hop 256、80 bins）なので、`--mel-out` で書き出せば外部のボコーダにも渡せる。

### 3. 自分の骨格に適用する

`selfaccent.backbones.base.Backbone` を実装すれば、`selfaccent.distill` の処理（教師の生成・adapter の学習・保存）をそのまま使える。
必要なメソッドは 5 つ: `synthesize`、`training_loss`、`add_tag_tokens`、`adapter_root`、`default_lora_targets`。
CosyVoice2 系と CFM 系への対応づけは [docs/03_architecture.md §4](docs/03_architecture.md) を参照。

## 小規模データでの結果

実行中（トイ骨格の学習 → 自己蒸留 → ベンチマーク）。完了後に docs/05_results.md にまとめる。

## 構成

| パス | 内容 |
|---|---|
| `selfaccent/tags.py` | タグ記法（モーラ、アクセント核、句境界、高低パターン） |
| `selfaccent/frontend.py` | pyopenjtalk による解析と HTS オラクル（任意の読み・アクセントで合成） |
| `selfaccent/mining.py` | 自己蒸留ペアの採掘 |
| `selfaccent/lora.py` | LoRA、`ExtendedEmbedding`、adapter の state dict |
| `selfaccent/distill.py` | 教師の生成、生徒の準備と学習、adapter の保存・読み込み |
| `selfaccent/backbones/` | `Backbone` プロトコル、TinyMatcha（raw text の CFM-TTS）、MAS |
| `selfaccent/audio.py`, `metrics.py`, `benchmark.py` | mel・Griffin-Lim・F0、DTW、アクセント判定、ベンチマーク |
| `selfaccent/data/` | トイ実験の語彙・定型文・データ分割 |
| `scripts/` | CLI（コーパス作成、骨格の学習、採掘、自己蒸留、推論、ベンチマーク） |
| `tests/` | pytest（TDD で実装） |

## ライセンスと謝辞

- コード: MIT
- タグの記法と LoRA の既定値は UtterTune（S. Kato, MIT）にならった。
- トイデータは、pyopenjtalk-plus（MIT）が同梱する OpenJTalk 辞書と HTS 音声 "Mei"（© Nagoya Institute of Technology, CC BY 3.0）で生成する。生成した音声はリポジトリに含めない。
- 論文: S. Kato, "Self-Distilled Pronunciation and Accent Control for Neural TTS", arXiv:2609.17234, 2026.
