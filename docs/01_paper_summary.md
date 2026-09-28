# 01. 論文調査: I-Parakeet: Integer-Only Conformer ASR on Mobile NPU

- 対象: arXiv [2609.30846v1](https://arxiv.org/abs/2609.30846)（2026-09-25 投稿, cs.CL）
- 著者: Taichi Nishimura（Sony Interactive Entertainment, Tokyo）
- 分量: 本文 4 ページ + 参考文献（ICASSP 形式）
- コード: 論文中に公開リンクなし

> 本ドキュメントは論文 PDF（v1）を通読して整理したもの。
> 記号: **[論文]** = 本文に明記 / **[推定]** = 本文からの推測（要検証） / **[不明]** = 本文に記載なし。

---

## 1. 一言でいうと

NVIDIA の **Parakeet-CTC-0.6B**（FastConformer-XL + CTC）を、**浮動小数点演算ゼロ・CPU フォールバックゼロ**で
**ミッドレンジのスマートフォン NPU（Snapdragon 7s Gen 3）上で動かした**研究。
LibriSpeech test-other で **WER 4.97%**、**RTF 0.048**（CPU 版 parakeet.cpp の 7.5 倍速）、ピークメモリ **612 MiB**。

## 2. 動機 [論文]

- Parakeet-CTC は 0.6B パラメータで、FP32 重みだけで 2.4 GB。INT8 化すれば 600 MB。
- モバイル SoC の NPU は整数演算向けに作られており、推論を丸ごと NPU に載せれば CPU/GPU を空けられ、電力も下がる。
- しかし既存の量子化 ASR の多くは **integer-only ではない**。正規化・Softmax・非線形活性化で FP16/FP32 に戻している。
- I-BERT 系の研究（I-BERT, Integer-only zero-shot quantization for ASR）は integer-only を示したが、**実機検証は GPU のみ**。
  **integer-only Conformer が実際にモバイル NPU で動くかは未検証**だった。
- 本論文での「integer-only」の定義は **デプロイ時の性質**：「浮動小数点の演算子がない」かつ「CPU フォールバックがない」。

## 3. ベースモデル（Parakeet-CTC-0.6B）の整理 [論文]

```
log-mel M ∈ R^{80×T0}
 └ Pre-Encoder: stride-2 conv ×3 (+ReLU) で 8 倍ダウンサンプル → Linear で d=1024
     H(0) ∈ R^{L×d},  L = ceil(T0/8)
 └ Encoder: Conformer block ×24
     Z1 = X  + 0.5·FFN1(LN(X))                      (1)
     Z2 = Z1 + MHSA(LN(Z1))        ← 相対位置 MHSA    (2)
     Z3 = Z2 + Conv(LN(Z2))                          (3)
     H  = LN(Z3 + 0.5·FFN2(LN(Z3)))                   (4)
     Conv(U) = PW2(Swish(BN(DW(GLU(PW1(U))))))  DW: K=9 (5)
     FFN = Linear → Swish → Linear
 └ CTC head: Linear → greedy decoding
```

## 4. 量子化の基本方式 [論文 §2.2]

| 項目 | 内容 |
|---|---|
| 量子化方式 | **全テンソル一様・対称量子化**。`q = round(clip(x, -α, α) / S)`, `S = α / (2^{b-1} - 1)` (Eq. 6) |
| ビット幅 | 既定 **b = 8**。例外は BatchNorm 出力のみ **b = 16** |
| クリップ範囲 α | キャリブレーションデータ（**LibriSpeech dev-other**）からオフラインで決定。既定は min–max（max\|x\|） |
| 活性化 | **テンソル単位（per-tensor）** に α を 1 つ |
| 重み | **チャネル単位（per-channel）** |
| 線形層 | `q_W · q_x` を **INT32 で累積** → 固定小数点乗数で再量子化: `q_y = round(2^{-n} · m · q_W q_x)`, `m = round(2^n S_W S_x / S_y)`, **n = 16** (Eq. 7)。実行時は整数積・整数乗算・右シフトのみ（除算なし） |
| 畳み込み | 線形層と同じ扱い |
| BatchNorm | 直前の depthwise conv に **畳み込み（fold）** |
| LayerNorm / Softmax | **I-BERT の整数カーネル**を使用 |
| 各部のビット幅（Fig. 1b） | Pre-Encoder: INT8 (p99.9) / FFN: INT8 I-LayerNorm, INT8 Linear, INT8 I-Swish / MHSA: INT8 I-LayerNorm, INT8 I-MHSA / ConvBlock: INT8 & INT16 / CTC head: INT8 |

## 5. 提案手法（3 つの貢献）

### 5.1 整数版 相対位置 Self-Attention（§3.1）[論文]

1 ヘッド分。`Q, K, V ∈ R^{L×dk}`、`P ∈ R^{(2L−1)×dk}` は正弦波相対位置埋め込みを射影したもの、`u, v` は学習済みバイアス。

- content 分岐: `Ac = (Q + 1uᵀ) Kᵀ`
- position 分岐: `Ap = Φ((Q + 1vᵀ) Pᵀ)`、Φ は (2L−1) 個の相対位置を L 個の絶対キー位置へ並べ替える relative shift
- `A = softmax((Ac + Ap)/√dk)`, `O = A V` (Eq. 8, 9)

**問題**: 2 分岐は独立にキャリブレーションされ、`Ac ≈ Sc·qc`（`Sc = SQ·SK`）、`Ap ≈ Sp·qp`（`Sp = SQ·SP`）。
`Sc ≠ Sp` なので `qc + qp` は `Ac + Ap` を表さない。

**解決**: 「スケール変換 + 加算 + 1/√dk」を **1 回の再量子化に融合**する（融合後スコアのグリッドを `Ss` とする）。

```
qs = round( Sc/(Ss√dk) · qc + Sp/(Ss√dk) · Φ(qp) )            (10)
   = round( 2^{-n} · ( mc·qc + mp·Φ(qp) ) )                    (11)
mc = round(2^n · Sc / (Ss√dk)),  mp = round(2^n · Sp / (Ss√dk))
```

効率化の 2 つの観察:
1. `P` は系列長 L にのみ依存 → **オフラインで量子化し INT8 定数 `qP` として保持**。
2. Φ は値の移動とゼロ挿入だけで算術を含まない → `Φ(S·q) = S·Φ(q)`。
   よって **整数テンソル `qp` に直接 Φ を適用でき、静的なインデックスマップとして実装**できる。

Softmax と `A·V` は I-BERT と Eq. (7) に従う。

### 5.2 minimax 最適化 整数 Swish（§3.2）[論文]

- `sw(x) = x·σ(x)`、`σ(x) = (1 + tanh(x/2))/2`。tanh を二次多項式で近似（I-BERT と同じ関数形）:
  ```
  tanh^(u) = sgn(u) · [ a·(min(|u|, c) − c)^2 + 1 ]            (12)
  ```
- I-BERT は (a, c) を **tanh への L2（最小二乗）フィット**で決める。本論文はこれが Parakeet に不適と指摘:
  1. Swish では σ の誤差に x が掛かるため、大きい |x| で誤差が拡大（x=4 では x=0.1 の 40 倍）→ **平均（L2）より最大誤差（L∞）**を抑えるべき。
  2. tanh に最適な係数が Swish 出力に最適とは限らない → **Swish 出力に直接フィット**すべき。
- 定式化:
  ```
  (a*, c*) = argmin_{a,c} max_x | sw(x) − x(1 + tanh^(x/2; a, c))/2 |     (13)
  ```
  数値的に解いて **a* = −0.1240, c* = 2.4632**。

> **本リポジトリで再現済み**: `scripts/fit_swish_approx.py` で Eq. (13) を解くと
> a = −0.1240, c = 2.4632、最大誤差 0.0386 となり論文と一致。
> Table 3 の他の 4 条件（L2/Swish 0.045, L∞/tanh 0.068, L2/tanh 0.073, Hard-Swish 0.142）の最大誤差もすべて一致した。

### 5.3 層ごとの活性化レンジ解析（§3.3）[論文]

既定設定（全テンソル INT8 + min–max）が破綻する箇所は 2 つだけ。それぞれ最小限の変更で対処し、他は既定のまま。

| 破綻箇所 | 解析（Fig. 2） | 対策 |
|---|---|---|
| ConvBlock の **BatchNorm 出力** | BN のチャネル別スケール `γc/√(σc²+ε)` の max/median 比が、いくつかの層で **10^4 超**（最大 10,759、終盤層でも 1,136）。per-tensor INT8 では大多数のチャネルに量子化レベルがほぼ残らない | BN 出力を **b = 16（per-tensor INT16）** |
| **Pre-Encoder** の活性化 | 観測最大値 / 99.9 パーセンタイルの比が pre-encoder で **11–21 倍**、encoder 内では **2–3 倍** | pre-encoder のみ **α = p99.9**、encoder は min–max のまま（**hybrid scaling**） |

## 6. 実験

### 6.1 実機評価（§4.1, Table 1）[論文]

**設定**
- 端末: **Nothing Phone (3a)**, Snapdragon **7s Gen 3 (SM7635)**（ミッドレンジ）
- NPU ランタイム: **QNN SDK（QAIRT 2.47）**
- NPU は静的シェイプ必須 → NPUsper に倣い **3〜35 秒の入力長ごとにグラフをコンパイル**し、各発話を収まる最小のグラフへ振り分け。
  **無音特徴でパディング**（処理フレームが 23% 増）。RTF はこのオーバーヘッド込み。
- WER は **Whisper の英語テキスト正規化器**で計算。
- ベースライン: **parakeet.cpp**（CPU 2 スレッド、FP16 重み / q8_0 重み、計算はともに FP32）、
  および **Qualcomm 標準ツールチェーン**で変換した FP16 版と INT8 PTQ 版。

**結果（LibriSpeech test-other）**

| 手法 | Backend | 重み / 計算 | WER | RTF | Peak mem (MiB) |
|---|---|---|---|---|---|
| parakeet.cpp | CPU | FP16 / FP32 | 3.76 | 0.36 | 1517 |
| parakeet.cpp | CPU | INT8 / FP32 | 3.76 | 0.42 | 1045 |
| 標準 FP16 | NPU | FP16 / FP16 | 100.0 | – | – |
| 標準 INT8 PTQ | NPU | INT8 / INT8 | 100.0 | – | – |
| **I-Parakeet** | NPU | INT8 / INT8 & INT16 | **4.97** | **0.048** | **612** |

**考察（論文の主張）**
- I-Parakeet は全体が NPU 上で動作し、FP16 parakeet.cpp の 7.5 倍速、ピークメモリ 60% 減、WER は +1.21 pt。
- 重みだけ INT8 にしても CPU では速くならない → 高速化は NPU の整数演算によるもの。
- **実機 WER 4.97% がシミュレータ 5.32% より良い理由**: 実機では Swish の sigmoid を**ハードウェア LUT**で評価しており、多項式近似を使っていないため。
- 標準 FP16 が 100% WER: pre-encoder 活性化が **FP16 のレンジを超える**ことと整合。
- 標準 INT8 PTQ が 100% WER: ツール標準の PTQ ではこのモデルに不十分。

### 6.2 シミュレーションでの精度分析（§4.2, Table 2, 3）[論文]

**設定**: デプロイグラフを模した **PyTorch 整数演算実装**（ただし Swish は §3.2 の多項式近似）。LibriSpeech と Common Voice で評価。
比較する integer-only ベースライン（どちらも §3.1 の整数 Attention は共通）:
- **I-BERT recipe**: 全テンソル INT8 min–max + Swish は tanh への L2 フィット
- **Naive INT8**: 全テンソル INT8 min–max + 提案の minimax Swish（= I-Parakeet から §3.3 の 2 対策を抜いたもの）

**Table 2: シミュレーション WER (%)**

| モデル | LS test-clean | LS test-other | Common Voice test |
|---|---|---|---|
| Parakeet-CTC (FP32) | 1.87 | 3.76 | 10.55 |
| I-BERT recipe | 3.41 | 7.41 | 19.05 |
| Naive INT8 | 3.01 | 6.38 | 16.54 |
| **I-Parakeet** | **2.61** | **5.32** | **14.70** |

**Table 3: アブレーション（LibriSpeech test-other WER, 1 要素ずつ変更）**

| 区分 | 設定 | Swish 最大誤差 | WER |
|---|---|---|---|
| Swish 近似 | **L∞ fit to Swish（提案）** | 0.039 | **5.32** |
| | L2 fit to Swish | 0.045 | 5.49 |
| | L∞ fit to tanh | 0.068 | 5.81 |
| | L2 fit to tanh | 0.073 | 5.95 |
| | Hard-Swish | 0.142 | 6.33 |
| BN 出力精度 | **per-tensor INT16（提案）** | – | **5.32** |
| | per-channel INT16（NPU では活性化に使えない） | – | 5.29 |
| | per-tensor INT8 (min–max) | – | 5.91 |
| スケール較正 | **hybrid（pre-enc p99.9 + enc min–max, 提案）** | – | **5.32** |
| | min–max のみ | – | 5.64 |
| | p99.9 のみ | – | 8.03 |

**読み取り**
- Swish の最大誤差の順位と WER の順位が一致。「L∞ vs L2」と「Swish vs tanh にフィット」は独立に効く。
- per-tensor INT16 は per-channel INT16 と 0.03 pt 差 → デプロイ可能な形でほぼ上限に到達。
- p99.9 を全体に適用すると大幅悪化（8.03）→ クリップは裾の重い分布でのみ有効。
- I-Parakeet は FP32 比 +0.74 / +1.56 pt（clean / other）。Naive INT8 から 0.40 / 1.06 pt 回復（= レンジ対策 2 つの寄与）。

## 7. 本文に書かれていない / 曖昧な点（再現時に決める必要がある項目）

| # | 項目 | 状況 | 本リポジトリでの仮決め（`03_reproduction_plan.md` 参照） |
|---|---|---|---|
| 1 | GLU の sigmoid の整数化 | [不明] シミュレータで何を使ったか書かれていない | Swish 用係数は x を掛ける前提で原点付近が不連続（下記メモ）なので流用不可。**INT8 LUT（実機と同等）** を既定、tanh への L∞ フィットを比較 |
| 2 | 残差加算の再量子化 | [不明] | Eq. (11) と同形（2 入力それぞれに固定小数点乗数 → 加算）。0.5 倍は乗数に吸収 |
| 3 | `Q+u` と `Q+v` のスケール | 本文は `Sc = SQ·SK`, `Sp = SQ·SP` と同じ SQ を使用 | u, v を Q の INT32 バイアスに畳み込み、共通スケール SQ で INT8 化（2 系統を出す） |
| 4 | 融合スコア `qs` のビット幅 | [不明]（既定 b=8 の原則からは INT8） | INT8 を既定、INT16 と比較 |
| 5 | I-BERT カーネルの詳細設定（Softmax 出力ビット、LN の出力ビット・内部シフト） | 「I-BERT に従う」のみ | HuggingFace transformers の `ibert/quant_modules.py` 準拠（Softmax 出力 8bit, LN 出力 8bit） |
| 6 | 実機で LayerNorm / Softmax を何で実装したか | [不明]（I-BERT カーネルを QNN の基本演算で組んだのか、QNN ネイティブ量子化 op か） | まず **QNN ネイティブの量子化 LayerNorm / Softmax / Sigmoid** を使い、全 op が HTP・固定小数点で動くことを確認 |
| 7 | 丸め（Eq. 7 の ⌊·⌉） | 記号上は round。実装の丸めモードは不明 | `(m·acc + 2^{n−1}) >> n`（round-half-up）を既定 |
| 8 | キャリブレーション量 | dev-other を使うことのみ明記 | dev-other 全体（2,864 発話）を既定、サブセットでの感度も確認 |
| 9 | パーセンタイルの計算方法 | 「\|x\| の 99.9%」 | 全キャリブレーション値をプールした \|x\| のヒストグラムから p99.9 |
| 10 | バケット（入力長グラフ）の刻み | 3〜35 秒、パディング +23% | test-other の長さ分布から +23% になる刻みを逆算して決定 |
| 11 | 「無音特徴」の中身 | [不明] | 正規化後 log-mel におけるデジタル無音のフレーム（候補: 0 ベクトル / 無音波形の特徴）を比較 |
| 12 | 特徴抽出・CTC デコードの実行場所と RTF 計測範囲 | [不明]（「ネットワーク全体が整数」とのみ） | log-mel と argmax 後処理は CPU。RTF は「特徴抽出込み」「NPU 推論のみ」の両方を記録 |
| 13 | Common Voice のバージョン・言語・サブセット | [不明] | 英語 test。FP32 で 10.55% に近くなる版を探索して特定 |
| 14 | 標準ツールチェーン FP16 / INT8 PTQ の設定 | [不明] | QAIRT 既定設定（`qnn-onnx-converter` 既定 + min–max 較正）で再現を試みる |
| 15 | xscaling の有無 | [不明] | NeMo の XL 推奨設定は `xscaling: false`。実チェックポイントの config で確認 |

> **メモ（Swish 係数の性質）**: a*·c*² = −0.7524 なので `tanh^(0+) = 0.2476`。
> つまり提案係数の tanh 近似は原点で ±0.25 ジャンプする（σ^ は 0.376 ↔ 0.624）。
> Swish では x≈0 で x を掛けるので無害だが、**GLU のように σ を単独で使う箇所に流用すると大きな誤差**になる。
> 「Swish 出力に直接フィットする」ことの帰結であり、GLU の扱いが本文にない点は再現上の注意点。

## 8. 再現目標（まとめ）

| レベル | 指標 | 論文値 |
|---|---|---|
| Swish 近似 | (a*, c*), 各近似の最大誤差 | −0.1240, 2.4632 / 0.039, 0.045, 0.068, 0.073, 0.142 → **再現済み** |
| レンジ解析 | BN スケール max/median（層別）, pre-enc の max/p99.9 | 最大 10,759 / pre-enc 11–21×, enc 2–3× |
| FP32 | LS clean/other, CV | 1.87 / 3.76 / 10.55 |
| シミュレーション | LS clean/other, CV（Table 2）+ Table 3 全行 | I-Parakeet 2.61 / 5.32 / 14.70 |
| 実機 | LS test-other WER, RTF, Peak mem | 4.97 / 0.048 / 612 MiB |
