# 03. 再現実装計画

## 0. ゴールと成功基準

論文の主張を **ハードウェア非依存の部分（シミュレーション）** と **実機部分** に分け、前者を確実に再現してから後者に進む。

| レベル | 再現対象 | 成功基準（許容幅） |
|---|---|---|
| L0 Swish 近似 | §3.2 の (a*, c*) と Table 3 の最大誤差 | 小数第 3 位まで一致 → **達成済み**（`scripts/fit_swish_approx.py`） |
| L1 FP32 | Table 2 の FP32 行 1.87 / 3.76 / 10.55 | ±0.05 pt |
| L2 レンジ解析 | Fig. 2(a)(b) | BN 比の層別プロファイル（最大 ~10^4）と pre-enc 11–21× / enc 2–3× が定性的に一致 |
| L3 シミュレーション | Table 2（I-BERT recipe / Naive INT8 / I-Parakeet）と Table 3 全行 | I-Parakeet は ±0.3 pt。**全アブレーションの順位が一致**すること |
| L4 実機 | Table 1（WER 4.97, RTF 0.048, 612 MiB）+ 標準ツールチェーンの 100% WER | WER ±0.3 pt。RTF とメモリは同一端末なら ±20%、別端末なら参考値 |

「integer-only」の検証基準（論文の定義に従う）:
- シミュレータ: 推論経路の全テンソルが整数型であり、float は**オフラインの定数計算（m, LUT, 定数量子化）にのみ**使われることをテストで保証。
- 実機: グラフ内の全テンソルが固定小数点型（FP16/FP32 テンソルなし）で、HTP 上でグラフ全体が finalize されること。

## 1. 全体方針

```
Phase 0 環境・データ ─▶ Phase 1 FP32 参照実装 ─┬▶ Phase 2 レンジ解析 ─┐
                                                └▶ Phase 3 整数シミュレータ ─┴▶ Phase 4 シミュレーション実験 ─▶ Phase 6 まとめ
                                                                    └▶ Phase 5 実機デプロイ ───────────────┘
```

- **NeMo は「正解の参照」としてのみ使い、量子化対象のモデルは自前のスタンドアロン PyTorch 実装にする。**
  NeMo のモジュールに量子化を差し込むより、全演算が明示的な自前実装の方が「どこで float が入るか」を完全に制御できる。
- シミュレータと実機グラフは **同一のキャリブレーション結果（エンコーディング）** を共有する。
- 論文に書かれていない点（`01_paper_summary.md` §7 の 15 項目）は既定値を置き、**感度を測って記録**する。

## 2. フェーズ詳細

### Phase 0: 環境・データ準備（1〜2 日）

- Python 環境: PyTorch, NeMo（チェックポイント読込と FP32 参照用。PyPI `nemo-toolkit` 3.0.0）, `jiwer`, Whisper 正規化器, `soundfile`, `sentencepiece`
- モデル: `nvidia/parakeet-ctc-0.6b`（.nemo）→ `model_config.yaml`, 重み, トークナイザを展開して config を確認（xscaling, vocab サイズ, 前処理設定）
- データ: LibriSpeech dev-other / test-clean / test-other、Common Voice 英語 test（版は FP32 WER 10.55% に合うものを探す）
- 成果物: `scripts/eval_fp32.py`（NeMo で推論 → Whisper 正規化 → WER）
- 完了条件: **FP32 WER 1.87 / 3.76**（±0.05）

### Phase 1: スタンドアロン FP32 参照実装（3〜4 日）

- `iparakeet/model/`: 前処理（NeMo 互換 log-mel, CPU/FP）, pre-encoder, Conformer ×24, CTC head, greedy デコード
  - BN を DW conv に fold した形でも実装（量子化前提の形）
  - relative shift を **静的 Gather（`idx[i,j] = L−1−i+j`）** で実装し、NeMo の pad/reshape 版と一致をテスト
  - P を最長バケットで 1 回計算し、長さ L のグラフではスライスを使う
- `iparakeet/model/buckets.py`: 入力長バケットへの振り分けと無音特徴パディング
- 完了条件:
  - NeMo とのロジット差 max\|Δ\| < 1e−4、dev-other の書き起こしが完全一致
  - バケット＋無音パディングありの FP32 WER も測る（パディング自体の影響を量子化の影響と切り分けるため）
  - test-other の長さ分布からパディング率 23% になるバケット刻みを決定

### Phase 2: レンジ解析 = Fig. 2 の再現（1〜2 日）

- `scripts/range_analysis.py`
  - (a) BN スケール `|γ|/√(σ²+ε)` の層別 max/median（データ不要）
  - (b) dev-other 上の全活性化テンソルの `max|x|` と `p99.9|x|`（2 パス: max → ヒストグラム）、比をプロット
  - FP16 の最大値（65504）を超えるテンソルの有無（標準 FP16 変換が壊れる理由の確認）
- 完了条件: 論文の Fig. 2 と同じ傾向（BN 比が一部の層で 10^3〜10^4、pre-enc 11–21×、encoder 2–3×）

### Phase 3: 整数シミュレータ（1〜1.5 週）

モジュール構成:

| モジュール | 内容 |
|---|---|
| `quant/fixed_point.py` | Eq. 6 の量子化、Eq. 7 の再量子化（m, n=16, round-half-up シフト）、m の相対誤差ログ |
| `quant/observers.py` | min–max / パーセンタイル（ヒストグラム）オブザーバ、per-channel 重み量子化 |
| `quant/qconfig.py` | テンソル名 → (bit 幅, 較正方式) の割当。レシピ（I-BERT recipe / Naive INT8 / I-Parakeet / 各アブレーション）を YAML で定義 |
| `intops/linear.py` | INT8 GEMM / Conv / DW Conv（GPU は `torch._int_mm`、CPU は float64 厳密エミュレーション） |
| `intops/layernorm.py`, `intops/softmax.py` | I-BERT カーネルを int32/int64 テンソルで再実装 |
| `intops/swish.py` | 二次多項式 Swish（係数は L∞/L2 × Swish/tanh の 4 種）、Hard-Swish、LUT Swish（実機模擬） |
| `intops/sigmoid.py` | GLU 用 sigmoid（LUT / 多項式） |
| `intops/relpos_mhsa.py` | Eq. 10–11 の融合スコア、定数 `q_P`、Gather による Φ |
| `intops/residual.py` | スケールの異なる 2 入力の整数加算（×0.5 を乗数に吸収） |
| `sim/int_parakeet.py` | 上記を組み立てた integer-only Parakeet-CTC |
| `sim/calibrate.py` | dev-other でキャリブレーション → エンコーディング JSON を出力（実機と共有） |

テスト（`tests/`）:
- 各カーネルの float 参照との誤差上限（例: I-Swish の出力誤差 ≤ 0.039 + 量子化誤差）
- relative shift の Gather と NeMo 版の一致
- 「float 非混入」テスト: 推論関数内で float テンソルが生成されたら失敗させる（`torch` の dtype チェック / フック）
- INT32 オーバーフロー検出（デバッグモード）

完了条件: 全テストが通り、FP32 と層ごとの SQNR を比較できる。

### Phase 4: シミュレーション実験 = Table 2, 3 の再現（1 週）

| 実験 | 設定 | 論文値（test-other） |
|---|---|---|
| I-BERT recipe | 全 INT8 min–max + L2/tanh Swish | 7.41 |
| Naive INT8 | 全 INT8 min–max + L∞/Swish | 6.38 |
| **I-Parakeet** | Naive INT8 + BN 出力 INT16 + pre-enc p99.9 | **5.32** |
| Swish 近似 ×5 | L∞/Swish, L2/Swish, L∞/tanh, L2/tanh, Hard-Swish | 5.32 / 5.49 / 5.81 / 5.95 / 6.33 |
| BN 出力 ×3 | per-tensor INT16, per-channel INT16, per-tensor INT8 | 5.32 / 5.29 / 5.91 |
| 較正 ×3 | hybrid, min–max のみ, p99.9 のみ | 5.32 / 5.64 / 8.03 |

- test-clean と Common Voice でも Table 2 の 3 モデルを評価。
- 論文と差が出た場合の切り分け手順:
  1. 1 層ずつ整数版を FP32 版に戻して、誤差の大きい層・演算を特定
  2. 未記載事項（GLU sigmoid、スコアのビット幅、I-BERT の出力ビット、丸め、n、パディング有無）の感度を測る
  3. それでも合わない点は「未記載事項による差」として記録（論文著者への問い合わせ候補）
- 追加実験（任意）: LUT Swish（実機模擬）で WER が 5.32 → 4.97 付近に下がるかを確認（論文 §4.1 の説明の検証）

### Phase 5: 実機デプロイ = Table 1 の再現（2〜3 週）

1. **グラフ出力**: バケットごとの固定長 ONNX（Swish = `x·Sigmoid(x)`、Φ = 定数 Gather、P = 共有定数）＋ Phase 3 のエンコーディング JSON
2. **QAIRT 2.47 で変換・量子化**: エンコーディング上書きを適用。全テンソルが固定小数点型であることを確認
3. **HTP コンテキストバイナリ生成**: 全バケットを 1 コンテキストにまとめ重み共有。グラフが大きすぎる場合はエンコーダを分割（分割しても全て NPU 上）
4. **実機ランナー**: まず `qnn-net-run` で動作確認 → C++ ランナー（QNN API）で test-other 全発話の推論・時間・ピーク RSS を計測
5. **評価**: WER（Whisper 正規化）、RTF（パディング込み。特徴抽出込み / なしの両方）、ピークメモリ
6. **ベースライン**:
   - parakeet.cpp を Android NDK でビルドし、2 スレッド・f16 / q8_0 で RTF・メモリ計測（論文: RTF 0.36 / 0.42, 1517 / 1045 MiB）
   - 標準ツールチェーンによる FP16 変換と INT8 PTQ（既定設定）→ 100% WER になることを確認
- 端末: 論文と同じ Nothing Phone (3a)（SM7635）が理想。無い場合は他の Snapdragon 端末または Qualcomm AI Hub のクラウド端末で代替し、RTF は参考値扱い。

### Phase 6: まとめ（2〜3 日）

- 論文の各表・図との対応表（再現値 / 論文値 / 差 / 原因の推定）を `docs/04_results.md` にまとめる
- 未記載事項ごとに採用した設定と感度を記録

## 3. リポジトリ構成案

```
iparakeet/
  model/     parakeet.py, load_nemo.py, frontend.py, buckets.py
  quant/     fixed_point.py, observers.py, qconfig.py
  intops/    linear.py, layernorm.py, softmax.py, swish.py, sigmoid.py, relpos_mhsa.py, residual.py
  sim/       int_parakeet.py, calibrate.py
  eval/      datasets.py, normalizer.py, wer.py, rtf.py
  deploy/    export_onnx.py, encodings.py, qnn/（変換スクリプト）, android/（C++ ランナー）
configs/     recipes/*.yaml（ibert_recipe, naive_int8, iparakeet, ablations）
scripts/     fit_swish_approx.py（済）, eval_fp32.py, range_analysis.py, eval_sim.py, run_ablation.py
tests/
docs/
```

## 4. 未記載事項の仮決め（Phase 3〜5 の既定値）

| 項目 | 既定値 | 感度を見る代替案 |
|---|---|---|
| GLU の sigmoid | INT8 LUT | tanh への L∞ フィット多項式 |
| 融合スコア qs のビット幅 | INT8 | INT16 |
| I-BERT Softmax / LN の出力 | 8 bit | 16 bit |
| 再量子化の丸め | round-half-up（加算 + 右シフト） | floor |
| 固定小数点乗数の n | 16（論文） | 24, 31 |
| キャリブレーション | dev-other 全体 | 256 / 512 発話のサブセット |
| p99.9 を適用する範囲 | pre-encoder 内の全活性化（Linear 出力を含む） | Linear 出力を除く |
| 無音特徴 | 正規化後 log-mel で 0 ベクトル | 無音波形から計算した特徴 |
| シミュレーション時のパディング | なし（発話長そのまま） | 実機と同じバケット＋パディング |

## 5. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| 未記載事項が多く、シミュレーション WER が一致しない | L3 | 感度分析で幅を示す。順位の一致を主目標にする |
| QAIRT 2.47 / 端末が入手できない | L4 | 近いバージョン・別端末・AI Hub で代替し、差を明記 |
| n=16 の固定小数点乗数で精度不足 | L3 | m の誤差をログ出力、n の感度を報告 |
| INT16 経路・LN・Softmax でのオーバーフロー | L3 | int64 計算 + INT32 範囲チェックのデバッグモード |
| 0.6B × 複数バケットのコンテキストが大きすぎる / コンパイルが重い | L4 | 重み共有、バケット数削減、エンコーダ分割 |
| 計算資源（この環境は GPU なし、HF / OpenSLR / arXiv がブロック） | 全体 | GPU マシンで実行、または環境のネットワーク許可リストに追加 |
| ライセンス | 公開時 | Parakeet は CC-BY-4.0（量子化モデルは帰属表示付きで配布可）。QAIRT SDK と論文 PDF はリポジトリに含めない |

## 6. 着手前に決めたいこと

1. **実機**: Nothing Phone (3a) か他の Snapdragon 端末は用意できるか。Qualcomm AI Hub のアカウントはあるか。
2. **計算資源**: シミュレータ実験（Table 2/3 で 14 条件 × 複数テストセット）を回す GPU マシンはあるか。
3. **スコープ**: シミュレーション（L0〜L3）を先に完了させ、その後実機（L4）に進む順序でよいか。Common Voice も対象にするか。
4. **実行環境**: このクラウド環境で進める場合、`huggingface.co`, `openslr.org`（LibriSpeech）, Common Voice の配布元などをネットワーク許可に追加する必要がある。

## 7. 目安スケジュール（1 人・GPU あり想定）

| 週 | 内容 |
|---|---|
| 1 | Phase 0, 1（FP32 一致まで）, Phase 2 |
| 2〜3 | Phase 3（整数カーネル + テスト + 組み立て） |
| 4 | Phase 4（Table 2, 3） |
| 5〜7 | Phase 5（実機。ツールチェーンの試行錯誤込み） |
| 8 | Phase 6（まとめ） |
