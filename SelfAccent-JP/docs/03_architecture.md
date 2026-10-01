# 03. アーキテクチャ

## 1. 全体像

```
                  ┌───────────── 一般語を含むテキスト（録音なし） ─────────────┐
                  │  "昨日、橋を見ました。"                                  │
                  ▼                                                          │
  [mining] candidate_words → DistillPair                                     │
      teacher: "昨日、橋を見ました。"                                        │
      student: "昨日、<PHON_START>ハシ'<PHON_END>を見ました。"               │
                  │                                                          │
                  ▼                                                          │
  [distill] generate_teacher_targets ── 凍結した骨格で teacher を合成 ──▶ 目標（mel / 音声トークン）
                  │                                                          │
                  ▼                                                          ▼
  [distill] prepare_student: タグ記号の埋め込み行を追加 ＋ LoRA(q,k,v,o) 注入、他はすべて凍結
                  │
                  ▼
  [distill] train_student: 骨格自身の損失(student テキスト, teacher 目標) で adapter だけ更新
                  │
                  ▼
  adapter.pt（LoRA ＋ 新規の埋め込み行 ＋ 骨格の語彙ハッシュ）── 推論時: 任意の語をタグで書く
```

本手法が骨格に求めるものは次の 3 つだけである。

1. **自分で生成できる**（教師の出力）
2. **自分の学習損失を計算できる**（生徒の学習）
3. **テキストのトークンを追加でき、LoRA を差し込める線形層がある**

## 2. モジュール構成

| モジュール | 役割 | 主な API |
|---|---|---|
| `selfaccent/tags.py` | タグ記法 `<PHON_START>チ'ミ/モーリョー<PHON_END>` の解析・生成・検証。モーラ分割、東京式の高低パターン | `parse_tag_body`, `format_tag_body`, `make_tag`, `find_tags`, `strip_tags`, `AccentPhrase.pitch` |
| `selfaccent/frontend.py` | pyopenjtalk による語の解析（読み・アクセント・連結フラグ・文字位置）と HTS オラクル | `analyze`, `word_phrases`, `OracleRenderer.render_with_override` |
| `selfaccent/mining.py` | 自己蒸留ペアの採掘 | `candidate_words`, `mine_pairs`, `tag_words` |
| `selfaccent/lora.py` | LoRA と、凍結した埋め込みの後ろに学習可能な行を足す `ExtendedEmbedding`。adapter の state dict | `inject_lora`, `ExtendedEmbedding`, `adapter_state_dict`, `merge_lora` |
| `selfaccent/tokenizer.py` | raw text 用の文字トークナイザ。タグ記号を 1 トークンとして扱う | `CharTokenizer.build/extend/encode` |
| `selfaccent/distill.py` | 教師の生成、生徒の準備と学習、adapter の保存・読み込み | `generate_teacher_targets`, `prepare_student`, `train_student`, `save_adapter`, `load_adapter` |
| `selfaccent/backbones/base.py` | 骨格のプロトコル | `Backbone` |
| `selfaccent/backbones/tiny_matcha.py` | CPU 検証用の raw text CFM-TTS と、その `Backbone` 実装 | `TinyMatcha`, `TinyMatchaBackbone` |
| `selfaccent/backbones/mas.py` | Monotonic Alignment Search | `maximum_path`, `monotonic_alignment` |
| `selfaccent/audio.py` | Slaney mel（HiFi-GAN 互換）、Griffin-Lim、WORLD Harvest による F0 | `MelExtractor`, `griffin_lim`, `extract_f0` |
| `selfaccent/metrics.py` | DTW とアクセントの強制選択判定 | `dtw_path`, `ReferenceSet`, `forced_choice`, `mel_distance` |
| `selfaccent/benchmark.py` | 評価条件の生成、実行、集計 | `condition_inputs`, `run_benchmark`, `summarize` |
| `selfaccent/data/` | トイ実験の語彙・テンプレートとコーパス分割 | `build_splits` |
| `selfaccent/training.py` | トイ骨格の学習ループ | `train_backbone` |

## 3. 各段の詳細

### 3.1 タグ記法（UtterTune と互換）

- 中身はカタカナの読み。発音形で書くので、長音は「ー」で表す（例: トーキョー）。
- `'` はアクセント核（下がり目の直前の高いモーラ）の直後に置く。`'` がなければ平板型（0 型）。
- `/` はタグ内のアクセント句の区切り（例: `チ'ミ/モーリョー`）。
- モーラ分割では、小書きのャュョァィゥェォヮを直前の仮名に結合する。ッ・ン・ーはそれぞれ 1 モーラとする。
- 文字列としての検証に加え、`AccentPhrase(morae, accent)` で 0 ≤ accent ≤ モーラ数 を保証する。

### 3.2 ペアの採掘

- 文を pyopenjtalk で解析し、次をすべて満たす語を候補にする。
  1. 品詞が名詞（変更可）
  2. アクセント句の先頭（chain_flag ≠ 1）
  3. 直後のノードが「連結した内容語」ではない。これで複合語の先頭は除外される。例えば「東京駅」の「東京」は、複合語のアクセント規則で acc が 3 に書き換わり、単独の「東京（0 型）」と食い違う。
  4. 読みのモーラ数が NJD の mora_size と一致し、acc がモーラ数以下
- 「一般語」は、骨格の学習テキスト（または同じ領域の大規模テキスト）での出現数が `min_count` 以上の語とする。論文の「骨格が既に正しく読める語」の代用である。
- `identity_ratio` の割合で、タグなしの恒等ペア（生徒 = 教師）を混ぜる。adapter がタグのない入力の挙動を変えないようにするための**本実装の追加**。

### 3.3 教師の生成

- 凍結した骨格で teacher テキストを合成する。ペアごとに乱数シード（seed + index）を固定し、バッチ構成によらず再現できるようにした。
- TinyMatcha では、ODE の 10 ステップ、温度 0.667（Matcha の既定）で合成した mel が目標になる。

### 3.4 生徒（adapter）

- **新しいトークン**: `<PHON_START>`、`<PHON_END>`、`'`、`/` と、語彙にないカタカナ。`ExtendedEmbedding` で既存の埋め込み表の後ろに行を足す。既存の行は凍結したままなので、weight decay でも変わらない。
- **LoRA**: 既定は UtterTune と同じ r = 16、α = 64、dropout 0.05、対象は q / k / v / o_proj。
- **損失**: 骨格自身の学習損失をそのまま使う。TinyMatcha の場合は prior（MAS で整列した Gaussian NLL）＋ 持続長 ＋ CFM。
- **TinyMatcha 固有の扱い**: Matcha は持続長予測器の入力を detach する。そのままだとタグのトークンの持続長が学習されないので、adapter の学習中だけ detach を外す（`prepare_for_adaptation`）。持続長予測器そのものは凍結したまま。

### 3.5 adapter ファイル

`adapter.pt` に次を保存する。

- `adapter_config`（r、α、dropout、対象層）
- `info`
  - 骨格の語彙サイズと SHA1（別の骨格への誤適用を防ぐ）
  - 追加した記号の並び（トークン ID を再現するため）
- `state_dict`（`lora_A/B` と `extra` だけ）
- `extra`（学習時の要約）

TinyMatcha（5M）では数百 kB になる。CosyVoice2 規模なら UtterTune と同じ 10 MB 未満になる見込み。

## 4. 実在の骨格への移植

`Backbone` プロトコルを実装すればよい。

| メソッド | CosyVoice2 系（テキスト → 音声トークン LM → flow → HiFT） | Sarashina2.2-TTS / F5 系（CFM） |
|---|---|---|
| `synthesize(texts, seeds)` | LLM で音声トークン列を生成（top-k / top-p のサンプリング。シードを固定）。**目標は音声トークン列** | mel を生成。目標は mel（または VAE 潜在） |
| `training_loss(texts, targets)` | LLM の次トークン交差エントロピー（テキスト条件 → 教師トークン列） | flow-matching 損失（テキスト条件 → 教師 mel） |
| `add_tag_tokens(symbols)` | BPE トークナイザに `<PHON_START>` / `<PHON_END>` を追加し、`resize_token_embeddings` で行を足す。カナは既存のトークンを使う | 文字語彙に追加 |
| `adapter_root()` / `default_lora_targets()` | `llm.model` / `q_proj,k_proj,v_proj,o_proj` | テキスト側の Transformer |

- 声質は flow と ボコーダが担うので、凍結したままでよい。LoRA は読みを決める部分（テキスト → 音声トークン）にだけ入れる。
- 教師のトークン列は、骨格の音声トークナイザで再符号化する必要がない。生成したトークンをそのまま目標にする。教師のサンプリングを複数回行えば、データ拡張にもなる。
- **一般語の判定**: 骨格の学習テキストが手に入らないので、次のどちらかで決める。
  - (a) 日本語 Wikipedia などの頻度上位語
  - (b) 教師の出力を ASR（kotoba-whisper、ReazonSpeech など）で認識し、読みが辞書と一致した語だけを採用する
  
  (b) が論文の「骨格が正しく読める」に最も近い。

## 5. 論文に書かれていない点と本実装での決定

| 項目 | 論文（アブストラクト）から分かること | 本実装での決定 | 理由 |
|---|---|---|---|
| 学習するパラメータ | LoRA 系であることが示唆される（前身 UtterTune は LoRA） | LoRA(q,k,v,o) ＋ タグ記号の埋め込み行 | UtterTune と同じ |
| 損失 | 骨格自身の損失（4 種の骨格に適用） | 骨格の学習損失をそのまま使う | 骨格を選ばないため |
| 損失を語の区間に限るか | 不明 | 文全体（整列は MAS 任せ） | 教師と生徒で文脈が同じなので、区間を限る必要が薄い |
| 教師のサンプリング | 不明 | ペアごとに 1 サンプル、シード固定、温度 0.667 | 再現性のため |
| 恒等ペア | 不明 | 10% を混ぜる | タグのない入力への副作用を防ぐ |
| 一般語の基準 | 「骨格が正しく読める一般語」 | 学習テキストでの出現数 ≥ 3 | トイ実験では学習テキストが既知 |
| 複数の句にまたがるタグ | UtterTune の記法は対応 | 記法・オラクルは対応、ペア採掘は単一句の語だけ | 単語単位の採掘では単一句で足りる |
