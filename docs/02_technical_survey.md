# 02. 再現のための技術調査

論文（`01_paper_summary.md`）の各要素を「何を・どう実装すれば再現できるか」の観点で調べた結果。
NeMo の実装は GitHub の `NVIDIA/NeMo` main ブランチのソースを直接読んで確認した。

---

## 1. ベースモデル Parakeet-CTC-0.6B の実装詳細

### 1.1 構成（NeMo ソース + 論文 Fig. 1a）

| ブロック | NeMo 実装 | 形状・設定 | 非線形 / 数値的に敏感な演算 |
|---|---|---|---|
| 前処理 | `AudioToMelSpectrogramPreprocessor` | 16 kHz, 窓 25 ms / シフト 10 ms, Hann, n_fft 512, 80 mel, log, `normalize: per_feature` | log, 発話単位の平均・分散正規化（CPU・FP で実行する想定） |
| Pre-Encoder | `ConvSubsampling(subsampling='dw_striding', factor=8, conv_channels=256)` | Conv2d(1→256, k3, s2) → ReLU → [DW Conv2d(256, k3, s2) → PW Conv2d(256→256) → ReLU] ×2 → flatten(256×10) → Linear(2560→1024) | ReLU のみ。ただし**活性化の裾が重い**（論文 §3.3） |
| 位置埋め込み | `RelPositionalEncoding` | 正弦波、相対位置 (L−1)…−(L−1) の 2L−1 個。xscaling は XL 推奨設定で `false`（実 config で要確認） | なし（定数） |
| Conformer ×24 | `ConformerLayer` | d=1024, heads=8 (dk=128), d_ff=4096, macaron FFN (×0.5) | LayerNorm ×5/層 |
| FFN | `ConformerFeedForward` | Linear(1024→4096) → Swish → Linear(4096→1024) | **Swish** |
| MHSA | `RelPositionMultiHeadAttention` | `linear_q/k/v/out`, `linear_pos`（バイアスなし）, `pos_bias_u/v` (h×dk), `rel_shift` | **Softmax**、**2 分岐スコアの加算** |
| ConvBlock | `ConformerConvolution` | PW1 Conv1d(1024→2048) → GLU → DW Conv1d(1024, K=9) → BatchNorm1d → Swish → PW2 Conv1d(1024→1024) | **GLU の sigmoid**、**BN 出力のレンジ**、Swish |
| CTC head | `ConvASRDecoder` | Conv1d(1024 → vocab+1, k=1) | argmax のみ（greedy） |

パラメータ概算: 1 層あたり FFN 16.8M + MHSA 5.2M + Conv 3.2M ≈ 25.2M、×24 ≈ 605M（+ pre-encoder, head）。

### 1.2 相対位置 MHSA の NeMo 実装（`multi_head_attention.py`）

```python
q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
p = self.linear_pos(pos_emb)                        # (1, 2L-1, h, dk)
matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
matrix_bd = self.rel_shift(matrix_bd)               # pad(1,0) → view → [1:] → view
matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
scores = (matrix_ac + matrix_bd) / self.s_d_k
```

**検証済みの事実（本調査で numpy により確認）**
- NeMo の `rel_shift` + スライスは、静的インデックスマップ
  **`out[i, j] = x[i, L − 1 − i + j]`**（i: クエリ, j: キー, インデックス範囲 0…2L−2）と完全一致（L = 1, 2, 5, 37, 438 で一致を確認）。
  → 論文の「Φ は静的インデックスマップとして実装」は **Gather（定数インデックス）1 個**で書ける。
  pad で入るゼロは最終スライスで捨てられる領域にしか現れない。
- `pos_emb` は最大長で作った正弦波表の**中央スライス**なので、`P_L = linear_pos(pos_emb_L)` は
  最長バケットの `P_Lmax` の連続スライス（行 `Lmax−L … Lmax+L−2`）になる。
  → 複数の入力長グラフ間で **P 定数を 1 つ共有**できる（メモリ節約。論文のピーク 612 MiB と整合する設計）。

### 1.3 シーケンス長と計算量の目安

| 入力長 | log-mel フレーム T0 | エンコーダ長 L ≈ T0/8 | P 定数 (INT8, 24 層) | スコア (INT32, 8 ヘッド/層) |
|---|---|---|---|---|
| 3 s | 300 | 38 | 1.8 MB | 46 KB |
| 35 s | 3,500 | 438 | 21.5 MB | 6.1 MB |

線形層の MAC はフレームあたり約 0.6 G → 35 秒で約 0.27 TMAC（注意機構込み）。

---

## 2. 整数演算の構成要素と実装方針

### 2.1 量子化・再量子化（論文 Eq. 6, 7）

- 量子化: `q = round(clip(x, −α, α)/S)`, `S = α/(2^{b−1}−1)`（対称, ゼロ点なし）。
- 再量子化: `q_y = (m · acc + 2^{n−1}) >> n`, `m = round(2^n · S_W S_x / S_y)`, **n = 16**。
  - 注意: `S_W S_x / S_y` が小さいと m が小さな整数になり、スケールの相対誤差が大きくなる
    （例: 比 1e−4 → m = round(6.55) = 7 で 7% の誤差）。論文は n = 16 と明記しているので既定はそれに従い、
    **テンソルごとに m の相対誤差をログ出力**し、n を大きくした場合との感度も見る。
    実機の QNN は内部で 32 bit 級の乗数を使うと思われ、シミュレータ（5.32%）と実機（4.97%）の差の一因になり得る。
- バイアス: INT32、スケール `S_W·S_x`。
- **BN の畳み込み**: `W' = W·γ/√(σ²+ε)`, `b' = (b−μ)·γ/√(σ²+ε) + β` を DW conv に吸収 → 重みは per-channel INT8、出力テンソルを INT16。

### 2.2 演算ごとの整数化方式

| 演算 | 論文の方式 | シミュレータでの実装 | 実機（QNN HTP）での実装 |
|---|---|---|---|
| Linear / Conv / DW Conv | Eq. 7 | INT8×INT8→INT32 累積 + 固定小数点再量子化 | 量子化 FullyConnected / Conv2d / DepthWiseConv2d |
| BatchNorm | DW conv に fold、出力 INT16 | 同左 | fold 済み DW conv の出力エンコーディングを 16 bit |
| LayerNorm | I-BERT | I-BERT `IntLayerNorm`（整数 sqrt を Newton 法で） | [不明] QNN ネイティブ量子化 LayerNorm を第一候補 |
| Softmax | I-BERT | I-BERT `IntSoftmax`（`exp` を ln2 分解 + 二次多項式、出力 8 bit） | [不明] QNN ネイティブ量子化 Softmax（`beta` で 1/√dk を吸収可能） |
| Swish | minimax 二次多項式（Eq. 12, 13） | 整数多項式評価（§3.2 参照） | **x · Sigmoid(x)**、Sigmoid は HW の LUT（論文 §4.1） |
| GLU | [不明] | INT8 LUT sigmoid（既定）/ tanh への L∞ フィット多項式 | 量子化 Sigmoid（LUT）+ 乗算 |
| 相対位置スコア | Eq. 10, 11（2 乗数融合） | 同左（§3.1） | MatMul ×2 → Gather（Φ）→ 量子化 Add（出力スケール Ss） |
| 残差加算 | [不明] | 2 入力に各々乗数を掛けて加算（0.5 は乗数に吸収） | 量子化 ElementWiseAdd |
| ReLU | 自明 | clamp(q, 0) | ReLU |
| CTC head | INT8 Linear | 同左 + argmax | FullyConnected（argmax は CPU 後処理でも可） |

### 2.3 I-BERT カーネル（参照実装）

HuggingFace transformers の `src/transformers/models/ibert/quant_modules.py`（Apache-2.0）に
`IntSoftmax`, `IntLayerNorm`, `IntGELU`, `QuantAct`, `QuantLinear` がある（取得して内容を確認済み）。

- `IntSoftmax`: `x − max` → `int_exp`（`x = q·(−ln2) + r` に分解、`exp(r) ≈ 0.3585(r+1.353)² + 0.344`、`2^{−q}` はシフト）
  → 和で割る（`2^32 / sum` の整数除算）→ 出力 `output_bit`（既定 8）bit、スケール `1/2^8`。
- `IntLayerNorm`: 平均・分散を整数で計算、`sqrt` は整数 Newton 反復、オーバーフロー回避の動的シフトあり。出力 8 bit。
- 注意: これらは「整数値を保持した float テンソル」で整数演算をエミュレートしている。
  本再現では **int32/int64 テンソルで実装し直し、float が混入していないことをテストで保証**する方針。

### 2.4 オーバーフローに注意すべき箇所

| 箇所 | 最大値の目安 | 対策 |
|---|---|---|
| INT8 GEMM 累積（K=4096） | 4096·127·127 ≈ 6.6e7 < 2^31 | INT32 で OK |
| Swish（INT16 入力: ConvBlock 内） | 多項式項 `(q−c)²` が ~2^30、さらに x を掛けると INT32 を超える | I-BERT 同様、途中で右シフトしてから x を掛ける |
| LayerNorm の分散 | 入力が INT32 残差の場合 Σ(x−μ)² が INT64 級 | I-BERT の動的シフト（`shift`）方式 |
| Softmax | `2^32 / sum` | I-BERT 同様 |

シミュレータは内部 int64 で計算しつつ、**INT32 を超えたら assert** するデバッグモードを持たせる。

---

## 3. 3 つの貢献の実装方針

### 3.1 整数相対位置 MHSA

```
acc_q  = W_q · q_x                                  (INT32)
q_u    = requant(acc_q + bias(b_q + u), S_Q)         (INT8)  ← u は INT32 バイアスに畳み込み
q_v    = requant(acc_q + bias(b_q + v), S_Q)         (INT8)
q_k, q_val = requant(...)                            (INT8)
q_P    = quantize(P_Lmax) の該当スライス            (INT8 定数, 層ごと)
qc     = q_u · q_kᵀ                                  (INT32, (h, L, L))
qp     = q_v · q_Pᵀ                                  (INT32, (h, L, 2L−1))
Φ(qp)  = gather(qp, idx[i, j] = L−1−i+j)             (INT32, (h, L, L)) ← 算術なし
qs     = (mc·qc + mp·Φ(qp) + 2^{n−1}) >> n           (INT8, スケール Ss)   Eq. (11)
attn   = IntSoftmax(qs, Ss)                          (INT8, スケール 1/2^8)
out    = requant(attn · q_val, S_O) → linear_out
```

- 論文では `Sc = SQ·SK`, `Sp = SQ·SP` と **Q 側スケールを共通 SQ** としている → `q+u` と `q+v` を同一スケールで量子化。
- `Ss` は `(Ac + Ap)/√dk` の min–max でキャリブレーション。
- QNN での表現: `MatMul(q_u, kᵀ)` と `MatMul(q_v, Pᵀ)` → `Gather(indices=const)` → `ElementWiseAdd`（出力エンコーディング = Ss·√dk 相当、Softmax の `beta=1/√dk`）。
  QNN の量子化 Add は入力スケールが異なっても内部で再量子化するので、Eq. (11) はグラフ上ではこの Add に相当する。

### 3.2 minimax Swish

- **係数の最適化は再現済み**: `scripts/fit_swish_approx.py`（x ∈ [−30, 30], 刻み 1e−3 のグリッド + Nelder–Mead）。

  | 条件 | a | c | Swish 最大誤差 | 論文 |
  |---|---|---|---|---|
  | L∞ fit to Swish（提案） | −0.1240 | 2.4632 | 0.0386 | 0.039 |
  | L2 fit to Swish | −0.1381 | 2.3728 | 0.0448 | 0.045 |
  | L∞ fit to tanh | −0.2182 | 2.1032 | 0.0675 | 0.068 |
  | L2 fit to tanh | −0.2304 | 2.0506 | 0.0733 | 0.073 |
  | Hard-Swish `x·ReLU6(x+3)/6` | – | – | 0.1423 | 0.142 |

  グリッド幅を ±8〜±60 で変えても結果は不変（誤差は |x| が大きいと指数的に 0 に近づくため）。
- **整数評価**（入力 `q_x`, スケール `S`。I-BERT の `IntGELU` と同型）:
  ```
  u のスケール S_u = S/2（x/2 はスケール側に吸収）
  c_int = floor(c / S_u)
  t     = min(|q_x|, c_int) − c_int
  poly  = t² + floor(1 / (a·S_u²))           # tanh^ ≈ a·S_u²·poly  （a<0 の符号に注意）
  tanh_q = sgn(q_x) · poly                    # スケール a·S_u²
  sig_q  = tanh_q + 1/(a·S_u²) の整数         # (1 + tanh)/2 → スケール a·S_u²/2
  sw_q   = requant(q_x · sig_q)               # INT16 入力時はシフトしてから乗算
  ```
- ConvBlock 内の Swish は **INT16 入力**（BN 出力）になる点に注意。
- 実機では多項式ではなく **HW の LUT sigmoid** を使う（論文 §4.1）。シミュレータに「LUT モード」も持たせると実機との比較がしやすい。

### 3.3 レンジ解析と 2 つの対策

- **Fig. 2(a) はデータ不要**: チェックポイントの BN パラメータから層ごとに `max_c(s_c)/median_c(s_c)`, `s_c = |γ_c|/√(σ_c²+ε)` を計算するだけ。
  論文値: 最大 10,759（序盤の層）, 1,136（終盤の層）, 多くの層は 10^1 程度。
- **Fig. 2(b)**: dev-other で全活性化テンソルの `max|x|` と `p99.9(|x|)` を集計し比を取る（pre-enc 11–21×, encoder 2–3×）。
  - 全値を保持すると重いので、テンソルごとに「1 パス目で max、2 パス目で固定ビンのヒストグラム」の 2 パス方式で p99.9 を求める。
- 対策 1: 融合後 DW conv（=BN 出力）のテンソルを b = 16。
- 対策 2: pre-encoder 内の活性化テンソル（Conv 出力 / ReLU 出力 / Linear 出力）の α を p99.9、encoder 以降は min–max。
- 補足: 論文は標準ツールチェーンの FP16 変換が 100% WER になる理由を「pre-encoder 活性化が FP16 のレンジ（±65504）を超えるため」としている。
  レンジ解析のついでに **FP16 の最大値を超えるテンソルがあるか**も確認する。

---

## 4. 実機デプロイのツールチェーン

### 4.1 論文の構成

| 項目 | 論文 |
|---|---|
| 端末 | Nothing Phone (3a), Snapdragon 7s Gen 3 (SM7635) |
| ランタイム | QNN SDK（QAIRT 2.47）, HTP バックエンド |
| 静的シェイプ | 3〜35 秒の長さ別グラフ、各発話を収まる最小グラフへ（NPUsper 方式）。無音特徴パディングで +23% フレーム |
| CPU ベースライン | parakeet.cpp, 2 スレッド, FP16 / q8_0 GGUF（計算は FP32） |
| 失敗ベースライン | 標準ツールチェーンによる FP16 変換、INT8 PTQ |

### 4.2 候補ツールと役割

| ツール | 用途 | 備考 |
|---|---|---|
| **QAIRT（Qualcomm AI Runtime）SDK 2.47** | ONNX → QNN 変換・量子化（エンコーディング上書き）、HTP コンテキストバイナリ生成、`qnn-net-run` で実機実行・プロファイル | 論文と同じ。Qualcomm アカウントで入手。SDK は再配布不可 |
| **Qualcomm AI Hub**（PyPI `qai-hub` 0.55.0） | クラウド上の実機でコンパイル・プロファイル・推論（w8a16 / int16 対応、QNN コンテキストバイナリ出力） | 端末を持っていなくても開発・計測できる。Nothing Phone (3a) / SM7635 が利用可能かは要確認 |
| AIMET（PyPI `aimet-torch` 2.40.0） | Qualcomm 公式の量子化シミュレーション・エンコーディング出力 | 本再現では独自の整数シミュレータが主。エンコーディング JSON 形式の参考 |
| ONNX Runtime QNN EP（`onnxruntime-qnn` 2.6.0） | QDQ ONNX を HTP で実行 | 代替経路 |
| ExecuTorch QNN backend（`executorch` 1.5.1） | PyTorch から直接 QNN へ | 代替経路 |
| parakeet.cpp（`mudler/parakeet.cpp`, ggml） | CPU ベースライン。parakeet-ctc-0.6b 対応、f16 / q8_0 GGUF あり | Android NDK でビルドして実機計測 |

### 4.3 想定フロー（QAIRT 直接）

1. 整数シミュレータのキャリブレーション結果から、**テンソルごとのエンコーディング**（scale, bitwidth, symmetric）を JSON で出力
   （BN 出力 = 16 bit, pre-enc = p99.9 由来の scale, 重み = per-channel）。
2. バケットごとの固定長 ONNX を出力（Swish は `x·Sigmoid(x)`、Φ は定数 Gather、P は定数）。
3. QAIRT のコンバータ / クオンタイザに**エンコーディングの上書き**を与えて量子化 DLC / モデルを作る。
   フォールバックで FP16 になるテンソルがないこと（全テンソルが固定小数点型）を変換ログ・モデル情報で確認。
4. HTP 向けコンテキストバイナリを生成。**全バケットを 1 コンテキストに入れ、重みを共有**する（HTP の weight sharing 機能を想定。要確認）。
5. 実機で `qnn-net-run`（初期確認）→ 自作ランナー（C++、QNN API）で全発話の推論・時間・ピークメモリを計測。
   HTP バックエンドはグラフ全体を HTP で確定（finalize）するので、成功すれば「CPU フォールバックなし」。

> QAIRT 2.47 のツール名・オプション名（`qairt-converter` / `qairt-quantizer` / `qnn-onnx-converter` 等）は
> バージョンで変わっているため、SDK 同梱ドキュメントで確認してから `deploy/` のスクリプトを書く。

---

## 5. 評価方法

| 項目 | 内容 |
|---|---|
| データ | LibriSpeech test-clean（2,620 発話, 約 5.4 h）, test-other（2,939 発話, 約 5.3 h）, dev-other（キャリブレーション, 2,864 発話）, Common Voice 英語 test（版は [不明]） |
| テキスト正規化 | **Whisper の EnglishTextNormalizer**（`openai-whisper` の `whisper.normalizers` または PyPI `whisper-normalizer`） |
| WER | `jiwer`（正規化後のコーパス WER） |
| デコード | CTC greedy（argmax → 連続重複除去 → blank 除去 → SentencePiece デコード） |
| RTF | 処理時間 / 音声長。論文はパディング分込み。特徴抽出を含むかは [不明] → 両方記録 |
| ピークメモリ | 実機プロセスのピーク RSS（MiB） |

**バケット刻みの逆算**: 論文は「3〜35 秒」「パディングで +23%」としか書いていない。
test-other の発話長分布に対して候補の刻み（1 s, 2 s, 5 s, 等比など）でのパディング率を計算し、23% に近いものを採用する。

---

## 6. 計算資源と、この環境の制約

- 整数シミュレータのコストは FP32 推論とほぼ同じ規模（フレームあたり約 0.6 GMAC）。
  test-other 1 回の評価で約 1.4e14 MAC。
  - GPU: `torch._int_mm`（INT8×INT8→INT32）で高速に実行可能。
  - CPU: float64 でエミュレートすると（|累積| < 2^53 なので厳密）、4 コアで 1 評価あたり数時間。Table 3 の全条件を回すなら GPU を推奨。
  - float32 で厳密にしたい場合は K を 1024 ずつに分割（1024·127·127 < 2^24）して部分和を int64 で合計する方法もある。
- **このクラウド環境（調査時点）**: GPU なし（4 CPU / 15 GB RAM）。`huggingface.co`, `arxiv.org`, `openslr.org` 等はネットワークポリシーでブロック。
  PyPI と GitHub（raw）は利用可。→ モデル・データのダウンロードが必要なフェーズは、許可ドメインを追加するか手元の GPU マシンで実行する必要がある。

---

## 7. 関連研究（論文が引用しているもの中心）

| 文献 | 本再現での役割 |
|---|---|
| I-BERT (Kim et al., ICML 2021) | LayerNorm / Softmax の整数カーネル、Swish 近似の関数形、比較ベースライン「I-BERT recipe」 |
| Integer-only zero-shot quantization for ASR (Kim et al., ICASSP 2022) | 「I-BERT recipe」の ASR 版 |
| Fast Conformer (Rekesh et al., ASRU 2023) | Parakeet のエンコーダ |
| Transformer-XL (Dai et al., ACL 2019) | 相対位置注意と relative shift |
| MobileNetV3 (Howard et al., ICCV 2019) | Hard-Swish（Table 3 の比較対象） |
| NPUsper (Lee et al., arXiv 2607.01108, 2026) | 入力長別グラフへの振り分け方式 |
| parakeet.cpp (mudler) | CPU ベースライン |
| Fixed-point arithmetic (R. Yates) | 固定小数点乗数による再量子化 |
| LLM.int8() / SmoothQuant | 外れ値によるレンジ問題の背景 |
