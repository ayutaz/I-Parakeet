# 04. 実行結果（M0〜M8）

計画（`03_reproduction_plan.md`）の M0〜M8 を実行した結果をまとめる。
実装はすべて TDD（テストを先に書き、失敗を確認してから実装）で行い、Python 環境は uv で管理している。

## 0. 先に結論

- **コードは M0〜M7 の全マイルストーン分を実装済みで、テストはすべて通過している**（`uv run pytest`: 175 件 pass）。
- **実データ・実機が必要な数値（FP32 / シミュレーションの WER、Fig. 2 の実測値、実機の WER・RTF・メモリ）は、この環境では測定できていない。**
  - この環境では、実チェックポイント（`huggingface.co`）とデータ（`openslr.org`）がネットワークポリシーでブロックされている（2026-09-28 時点で再確認済み）。
  - GPU がない（CPU 4 コア / 15 GB）。
  - Qualcomm 実機と QAIRT SDK がない。
- そのため、実データの代わりに次の方法で検証した。
  - 実際の NeMo 3.0.0 モジュール（小型構成、ランダム重み）との数値一致
  - 合成データでの end-to-end 実行
  - onnxruntime による実機の代用
- 実データを使える環境で残りを実行する手順は §4 にまとめた。コマンドを順に実行すれば、各マイルストーンのゴールを判定できる。

## 1. マイルストーン別の達成状況

凡例: ✅ 達成 / 🟡 ツールは完成・テスト済みだが、実データ・実機での実行が未実施 / ⛔ この環境では実行不可

### M0: Swish 近似の再現 — ✅ 完了

| ゴール | 結果 |
|---|---|
| [必達] (a*, c*) = (−0.1240, 2.4632) | ✅ 一致（`scripts/fit_swish_approx.py`） |
| [必達] 5 条件の最大誤差が小数第 3 位まで一致 | ✅ 0.039 / 0.045 / 0.068 / 0.073 / 0.142 |

### M1: 評価基盤と FP32 ベースライン — 🟡

| ゴール | 結果 |
|---|---|
| [必達] FP32 WER 1.87 / 3.76 | ⛔ モデル・データを取得できず未測定。`scripts/eval_fp32.py` は実装・テスト済み |
| [必達] 共通評価スクリプト | ✅ Whisper 正規化 + コーパス WER（空参照の除外を含む）、LibriSpeech / Common Voice のマニフェスト、RTF |
| [目標] Common Voice の版の特定 | ⛔ 未実施（データ未取得） |
| [必達] 実チェックポイントの config 確認 | ⛔ 未実施。ただし `.nemo` ローダは NeMo の config / state dict 形式を直接読める |

### M2: スタンドアロン FP32 参照実装 — ✅（実チェックポイントでの確認のみ 🟡）

| ゴール | 結果 |
|---|---|
| [必達] NeMo とのロジット差 < 1e−4 | ✅ **実際の NeMo 3.0.0 モジュール**と一致。小型構成・ランダム重み、`xscaling` の true/false 両方で確認。前処理の特徴量、パディングありのバッチ、切り出した単発推論のいずれも 1e−4 以内（`tests/test_nemo_parity.py`、参照データは `tests/nemo_parity/generate_reference.py` で生成）。実チェックポイントでの dev-other 書き起こし一致は ⛔ 未実施 |
| [必達] BN fold / gather による Φ / P スライス共有でも一致 | ✅ 単体テスト済み（Φ は `out[i,j] = x[i, L−1−i+j]`） |
| [必達] バケット＋無音パディング時の FP32 WER | 🟡 実装済み（`BucketSet`, `pad_features`）。実データでは未測定 |
| [目標] パディング率 23% になるバケット刻み | 🟡 `scripts/choose_buckets.py` は実装済み。実データでは未実行 |

### M3: レンジ解析（Fig. 2） — 🟡

| ゴール | 結果 |
|---|---|
| [必達] Fig. 2(a) の BN スケール比 | 🟡 計算・描画は実装済み（重みだけで計算できる）。実チェックポイントでは未実行 |
| [必達] Fig. 2(b) の max/p99.9 | 🟡 2 パス集計（max → ヒストグラム）は実装済み。numpy のパーセンタイルとビン幅以内で一致することをテスト済み |
| [必達] INT16 / p99.9 を適用するテンソル一覧 | ✅ タップ名の規約で確定（BN 出力 = `L*.conv.dw`、pre-encoder = `pre.*`） |
| [目標] FP16 レンジを超えるテンソルの特定 | 🟡 検出ロジックは実装・テスト済み |

### M4: 整数シミュレータ構築 — ✅

| ゴール | 結果 |
|---|---|
| [必達] 全整数カーネルの単体テスト | ✅ Swish（4 種の係数 / Hard-Swish / LUT）、GLU sigmoid、LayerNorm（整数 Newton sqrt）、Softmax、Linear / Conv / DW Conv、残差加算、融合スコア（Eq. 11）、整数相対位置 MHSA |
| [必達] float 非混入テスト | ✅ `NoFloatMode`（`TorchFunctionMode`）で、推論中の全 torch 演算の出力が整数型であることを検査。全レシピで通過 |
| [必達] INT32 オーバーフロー 0 件 | ✅ 小型モデルで検査済み（`int32_check`）。実モデル・dev-other 全体では未実行 |
| [必達] ほぼ無損失の設定で FP32 との差 ≤ 0.1 pt | 🟡 代替として、小型モデルで SQNR > 25 dB と argmax 一致率 > 90% を確認。実データの WER 差は未測定 |
| [必達] 11 条件を config の切り替えだけで実行 | ✅ `iparakeet/sim/recipe.py`（診断用の追加レシピを含めて 18 種） |
| [目標] GPU で test-other を 1 時間以内 | ⛔ GPU なし（CUDA では `torch._int_mm` の経路を用意済み） |

### M5: シミュレーション実験（Table 2, 3） — 🟡

| ゴール | 結果 |
|---|---|
| [必達] Table 2 の順位一致 | 🟡 自動判定（`check_table2_order`）と表の生成は実装済み。実データでは未実行 |
| [必達] Table 3 の各グループの順位一致 | 🟡 同上。差が 0.1 pt 未満の組は同等として扱う |
| [目標] 2.61 / 5.32 / 14.70（±0.3） | ⛔ 未測定 |
| [必達] 未記載事項の感度表 | 🟡 感度分析用のレシピ（LUT Swish、I-BERT Softmax、INT16 スコア、GLU 多項式、n=24、INT8 ヘッド）を用意済み |
| 合成データでの end-to-end | ✅ `tests/test_smoke_pipeline.py` で、prepare_data → eval_fp32 → range_analysis → run_ablation を実際の CLI で実行 |

### M6: NPU グラフ構築と integer-only 検証 — 🟡（静的な代替検証は ✅）

| ゴール | 結果 |
|---|---|
| [必達] QAIRT で変換からコンテキスト生成まで | ⛔ SDK がない。変換スクリプトのテンプレートを生成（HTP の重み共有、全バケットを 1 コンテキストに） |
| [必達] FP テンソル 0 個・HTP での finalize | ✅（代替検証）エンコーディングのない活性化 0 件、QDQ グラフで量子化されていない演算入力 0 件。実機での finalize は ⛔ |
| [必達] シミュレータと実機の出力一致（WER 差 ≤ 0.2） | 🟡 代替として、onnxruntime 上の QDQ グラフと整数シミュレータ（LUT Swish）の argmax 一致率 ≥ 80% を確認（小型モデル）。FP の ONNX グラフは PyTorch と 1e−4 以内で一致 |
| [目標] 重み共有で約 600 MB＋α | ⛔ 実機が必要 |

### M7: 実機評価（Table 1） — 🟡

| ゴール | 結果 |
|---|---|
| [必達] test-other 全発話の NPU 推論と計測 | ⛔ 実機なし。入力作成・結果集計のツールは実装済みで、onnxruntime を実機の代わりにした end-to-end テストが通過 |
| [目標] WER 4.97 / RTF 0.048 / 612 MiB | ⛔ |
| [必達] parakeet.cpp の CPU ベースライン | ⛔（Android NDK でのビルドと実機が必要） |
| [必達] 標準ツールチェーンの破綻確認 | ⛔ |
| [必達] RTF の計測範囲の明記 | ✅ レイテンシ CSV（発話ごとの NPU 時間）とパディング込みの RTF を分けて集計する形にした |

### M8: 結果まとめ — ✅（本ドキュメント）

## 2. 実装中に分かったこと・決めたこと

### 2.1 論文に書かれていない事項の最終決定

| 項目 | 採用した実装 | 理由・根拠 | 感度分析用のレシピ |
|---|---|---|---|
| GLU の sigmoid | INT8 LUT | 提案の Swish 係数は原点で ±0.25 不連続になるため、GLU には流用できない（M0 で確認） | `iparakeet_glu_poly` |
| 融合スコアのビット幅 | INT8（論文の既定 b=8） | Fig. 1b の「INT8 I-MHSA」に従う | `iparakeet_int16_scores` |
| Softmax の範囲縮約 | **log2 固定小数点**（I-BERT と同じ多項式・シフト構造） | I-BERT の `floor(−ln2/S)` は INT8 スコアでは粗く、確率の誤差が α=32 で 0.045 になる。log2 方式は 0.004（§3） | `iparakeet_ibert_softmax` |
| CTC ヘッドの出力 | 共通の INT16 グリッドに再量子化してから argmax | **テストで見つかったバグ**: 重みのスケールがクラスごとに違うため、INT32 累積値のまま argmax を取ると誤る | `iparakeet_head8` |
| 固定小数点乗数の n | Eq. 7 の線形層・残差加算は **n=16**（論文どおり）。論文に指定のない非線形カーネルは、m が 15 ビットになるよう n を自動選択 | n=16 の乗数の相対誤差は小型モデルで最大 0.50%、中央値 0.10%（n=24 なら 0.0015%） | `iparakeet_n24` |
| LayerNorm | I-BERT と同じ構造（整数平均 → 整数 sqrt → 整数逆数 → チャネルごとの固定小数点出力）。逆数は 2^23 スケール | int64 のオーバーフローを避けつつ、約 14 ビットの精度を確保 | – |
| 位置定数 P の量子化 | 最長バケットの P_max で決めた 1 つの INT8 スケールを全バケットで共有 | P_L は P_max の連続スライスなので、定数を 1 つにできる | – |
| 入力の量子化 | モデルの境界（特徴抽出は CPU・FP、グラフ入力で INT8 化） | 実機グラフの入力量子化と同じ位置 | – |
| 無音特徴 | 正規化後の 0 ベクトル | NeMo の pad_value と同じ | – |
| per-channel INT16（Table 3 の比較用） | チャネルごとの min–max | 論文でも「実機では使えない」とされる比較用の条件 | `bn_per_channel_int16` |

### 2.2 テストで検出・修正した問題

1. **CTC ヘッドの argmax**: クラスごとに違うスケールの INT32 累積値を比較していた。回帰テストを追加し、共通グリッドへの再量子化で修正した（M5 のコミット）。
2. **Softmax の精度**: 粗い INT8 グリッドで I-BERT の i-exp をそのまま使うと、確率の誤差が 0.05 に達することを単体テストで検出した。範囲縮約の方式を切り替え可能にし、既定を log2 にした。
3. **パーセンタイル較正の利点の性質**: 平均二乗誤差では、クリップされた外れ値の影響で p99.9 が不利になる。論文が述べる利点は「大部分の値（|x| ≤ p99.9）の分解能の回復」であり、テストもこの性質を検証する形にした。

## 3. データなしで測れた数値

### 3.1 Swish 近似（M0、論文 Table 3 の Max err.）

| 近似 | 論文 | 再現 |
|---|---|---|
| L∞ fit to Swish（a=−0.1240, c=2.4632） | 0.039 | 0.0386 |
| L2 fit to Swish（a=−0.1381, c=2.3728） | 0.045 | 0.0448 |
| L∞ fit to tanh（a=−0.2182, c=2.1032） | 0.068 | 0.0675 |
| L2 fit to tanh（a=−0.2304, c=2.0506） | 0.073 | 0.0733 |
| Hard-Swish | 0.142 | 0.1423 |

### 3.2 整数カーネル単体の精度（`uv run python -m scripts.kernel_report --out results/kernels`）

INT8 入力（α=8）・INT8 出力（刻み 0.063）の Swish:

| カーネル | 近似自体の最大誤差 | 整数カーネルの最大誤差 |
|---|---|---|
| poly:linf_swish | 0.0387 | 0.0643 |
| poly:l2_swish | 0.0448 | 0.0671 |
| poly:linf_tanh | 0.0675 | 0.0964 |
| poly:l2_tanh | 0.0733 | 0.0964 |
| hardswish | 0.1423 | 0.1458 |
| lut（実機相当） | – | 0.0313 |

整数カーネルでも、最大誤差の順位は Table 3 の順位と一致する。LUT が最も誤差が小さく、実機の WER（4.97%）がシミュレータ（5.32%）より良いという論文の説明と整合する。

Softmax（INT8 スコア、確率の最大絶対誤差）:

| スコアの範囲 α | I-BERT 方式 | log2 方式 |
|---|---|---|
| 8 | 0.0110 | 0.0040 |
| 16 | 0.0218 | 0.0041 |
| 32 | 0.0453 | 0.0041 |
| 32（INT16 スコア） | 0.0044 | 0.0044 |

## 4. 残りを実行する手順（モデル・データ・GPU・実機がある環境で）

```bash
uv sync --extra deploy
# M1: データとモデル（huggingface.co / openslr.org へのアクセスが必要）
uv run python -m scripts.prepare_data --download --data-dir data --splits dev-other test-clean test-other
uv run python -m scripts.eval_fp32 --nemo data/parakeet-ctc-0.6b.nemo --manifest data/manifests/test-other.jsonl --out results/fp32/test-other
uv run python -m scripts.eval_fp32 --nemo data/parakeet-ctc-0.6b.nemo --manifest data/manifests/test-clean.jsonl --out results/fp32/test-clean
# M2: バケットの刻み（パディング率 23% に近いもの）
uv run python -m scripts.choose_buckets --manifest data/manifests/test-other.jsonl --target 0.23
# M3: レンジ解析とキャリブレーション統計（dev-other）
uv run python -m scripts.range_analysis --nemo data/parakeet-ctc-0.6b.nemo --manifest data/manifests/dev-other.jsonl --out results/range
# M5: Table 2 / Table 3（GPU 推奨）。--fp32 に {"test-clean": x, "test-other": y} の JSON を渡すと FP32 行も判定される
uv run python -m scripts.run_ablation --nemo data/parakeet-ctc-0.6b.nemo --stats results/range/calibration_stats.json \
    --manifests data/manifests/test-clean.jsonl data/manifests/test-other.jsonl --out results/ablation
# M6: NPU 用のグラフとエンコーディング（summary.json の missing_encodings / unquantized_inputs が 0 であること）
uv run python -m scripts.export_npu --nemo data/parakeet-ctc-0.6b.nemo --stats results/range/calibration_stats.json \
    --recipe iparakeet_lut --bucket-step <M2 の結果> --out results/npu
#   → results/npu/qnn_build.sh を QAIRT 2.47 の環境で実行（ツール名・オプションは SDK のドキュメントで確認）
# M7: 実機の入力 → qnn-net-run → 評価
uv run python -m scripts.prepare_device_inputs --nemo data/parakeet-ctc-0.6b.nemo --manifest data/manifests/test-other.jsonl \
    --bucket-step <M2 の結果> --out results/device_inputs
#   adb push し、バケットごとに qnn-net-run --retrieve_context ... --input_list <bucket>/input_list.txt
uv run python -m scripts.eval_device --nemo data/parakeet-ctc-0.6b.nemo --manifest data/manifests/test-other.jsonl \
    --mapping results/device_inputs/mapping.json --results results/device_outputs --latency-csv results/device_latency.csv \
    --out results/device.json
```

実機がない場合でも、次の2通りでシミュレータとの一致（M6）を実データで測れる。
- `iparakeet.deploy.emulate.run_input_lists_with_ort` で QDQ グラフを onnxruntime 上で実行する
- `--qdq` 付きで出力したグラフを使う

## 5. このクラウド環境で残りを実行するための設定

環境の Network access で、次のドメインを許可すれば M1〜M5 はこの環境でも実行できる（ただし GPU がないため M5 には時間がかかる）。
- `huggingface.co`（とダウンロード用の CDN）
- `www.openslr.org` / `us.openslr.org`
- Common Voice の配布元

M6・M7 には、QAIRT SDK と Snapdragon 端末（または Qualcomm AI Hub のアカウント）が必要。
