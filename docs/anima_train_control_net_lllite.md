# ControlNet-LLLite Training Guide for Anima using `anima_train_control_net_lllite.py` / `anima_train_control_net_lllite.py` を用いた Anima モデルの ControlNet-LLLite 学習ガイド

This document explains how to train a **ControlNet-LLLite** for Anima using `anima_train_control_net_lllite.py`, and how to run minimal inference with the trained weights via `anima_minimal_inference_control_net_lllite.py`.

ControlNet-LLLite is a lightweight, LoRA-like conditional control module originally introduced for SDXL (see [`train_lllite_README.md`](./train_lllite_README.md)). This Anima port retargets it to the DiT (MiniTrainDIT) architecture used by Anima: small adapter modules are attached to the attention `Linear` layers of each transformer block, and a shared conditioning-image embedding (`conditioning1`) is broadcast to all of them.

> **Status:** experimental. Currently supports image generation only (`T=1`). `--blocks_to_swap`, `--cpu_offload_checkpointing`, `--unsloth_offload_checkpointing`, `--deepspeed`, and `--fused_backward_pass` are not yet supported and the training script will assert if any of them is enabled.

<details>
<summary>日本語</summary>

このドキュメントでは、`sd-scripts` リポジトリに含まれる `anima_train_control_net_lllite.py` を用いて Anima モデル向けの **ControlNet-LLLite** を学習する手順、および学習した重みを `anima_minimal_inference_control_net_lllite.py` で推論する基本的な手順について解説します。

ControlNet-LLLite は SDXL 向けに導入された LoRA ライクな軽量条件付け手法です（オリジナルの解説は [`train_lllite_README-ja.md`](./train_lllite_README-ja.md) を参照）。Anima 版では、Anima が採用する DiT (MiniTrainDIT) アーキテクチャに移植してあり、各 Transformer ブロックの attention の `Linear` レイヤに小さな adapter を貼り、conditioning 画像を埋め込んだ単一の `conditioning1` を全モジュールに配布する構成になっています。

> **ステータス:** 実験的実装です。現状は画像生成（`T=1`）のみ対応しています。`--blocks_to_swap` / `--cpu_offload_checkpointing` / `--unsloth_offload_checkpointing` / `--deepspeed` / `--fused_backward_pass` には未対応で、指定すると学習スクリプトが assert で停止します。

</details>

## 1. How it Differs from the Standard Anima LoRA Script / 通常の Anima LoRA 学習との違い

`anima_train_control_net_lllite.py` is derived from `anima_train.py` but trains **only** the ControlNet-LLLite adapter; the DiT itself is fully frozen.

| | `anima_train_network.py` (LoRA) | `anima_train_control_net_lllite.py` |
|---|---|---|
| Target | DiT LoRA | ControlNet-LLLite adapter only (DiT frozen) |
| Dataset | DreamBooth / fine-tuning | **ControlNet format** (image + conditioning image) |
| Network module | `--network_module=networks.lora_anima` | (none — built-in `ControlNetLLLiteDiT`) |
| Extra inputs at train step | — | `conditioning_images` from each batch |
| Saved weights | LoRA `.safetensors` | LLLite `.safetensors` (`conditioning1.*` + `lllite_modules.*`) |

The dataset format is the same as the existing SDXL ControlNet-LLLite script. See the **Preparing the dataset** section of [`train_lllite_README.md`](./train_lllite_README.md#preparing-the-dataset) ([日本語](./train_lllite_README-ja.md#データセットの準備)) for the directory layout, `conditioning_data_dir`, and dataset synthesis tips.

<details>
<summary>日本語</summary>

`anima_train_control_net_lllite.py` は `anima_train.py` の派生で、DiT 本体は完全に凍結し **ControlNet-LLLite adapter のみ**を学習します。

| | `anima_train_network.py`（LoRA） | `anima_train_control_net_lllite.py` |
|---|---|---|
| 学習対象 | DiT の LoRA | ControlNet-LLLite adapter のみ（DiT は凍結） |
| データセット形式 | DreamBooth / fine-tuning | **ControlNet 形式**（教師画像 + conditioning 画像） |
| Network module | `--network_module=networks.lora_anima` | （不要、`ControlNetLLLiteDiT` 内蔵） |
| 学習ステップの追加入力 | — | バッチ内の `conditioning_images` |
| 保存される重み | LoRA `.safetensors` | LLLite `.safetensors`（`conditioning1.*` と `lllite_modules.*`） |

データセット形式は既存の SDXL 向け ControlNet-LLLite と同一です。ディレクトリ構成、`conditioning_data_dir` の指定、データセット合成のヒントなどは [`train_lllite_README-ja.md`](./train_lllite_README-ja.md#データセットの準備) を参照してください。

</details>

## 2. Preparation / 準備

The same model files as ordinary Anima training are required. See [`anima_train_network.md` Section 3](./anima_train_network.md#3-preparation--準備) for details.

In addition you need:

* A **paired dataset** of training images and conditioning images (e.g. lineart, canny, depth) saved with matching basenames. Either a TOML `dataset_config` describing `conditioning_data_dir`, or the CLI form `--train_data_dir <dir> --conditioning_data_dir <dir>` (subset-by-subdir layout) is supported.
* Optionally, a `prompts.txt` with `--cn <path>` (and `--am <float>`) entries for sample-image generation during training.

<details>
<summary>日本語</summary>

通常の Anima 学習で必要なモデルファイル群（DiT、Qwen3、Qwen-Image VAE、LLM Adapter、T5 トークナイザ）が同様に必要です。詳細は [`anima_train_network.md` セクション 3](./anima_train_network.md#3-preparation--準備) を参照してください。

加えて以下が必要です：

* 教師画像と conditioning 画像（例: lineart、canny、depth）を同じベース名でペア化した**データセット**。TOML の `dataset_config` で `conditioning_data_dir` を指定する方法、もしくは CLI で `--train_data_dir <dir> --conditioning_data_dir <dir>`（サブディレクトリ単位の自動 subset 生成）を指定する方法、どちらも使えます。
* 学習中のサンプル画像生成を行いたい場合は、`--cn <path>`（任意で `--am <float>`）を含む `prompts.txt`。

</details>

## 3. Running the Training / 学習の実行

Example command (one line in practice — line breaks shown for readability; use `\` on Linux/macOS or `^` on Windows to wrap):

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_control_net_lllite.py \
  --pretrained_model_name_or_path="<path to Anima DiT model>" \
  --qwen3="<path to Qwen3-0.6B model or directory>" \
  --vae="<path to Qwen-Image VAE model>" \
  --dataset_config="my_anima_lllite_dataset.toml" \
  --output_dir="<output directory>" \
  --output_name="my_anima_lllite" \
  --save_model_as=safetensors \
  --cond_emb_dim=32 \
  --lllite_mlp_dim=64 \
  --lllite_target_layers=self_attn_q \
  --learning_rate=5e-5 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --timestep_sampling="sigmoid" \
  --discrete_flow_shift=1.0 \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --vae_chunk_size=64 \
  --vae_disable_cache
```

A minimal dataset TOML for the ControlNet format looks like:

```toml
[general]
caption_extension = ".txt"
shuffle_caption = false

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "/path/to/training_images"
  conditioning_data_dir = "/path/to/conditioning_images"
  num_repeats = 1
```

For a fuller description of dataset options, see the SDXL LLLite guide ([English](./train_lllite_README.md#preparing-the-dataset) / [日本語](./train_lllite_README-ja.md#データセットの準備)) and the [Dataset Configuration Guide](./config_README-en.md). The dataset format and the meaning of `conditioning_data_dir` are identical to the SDXL version.

<details>
<summary>日本語</summary>

学習の実行コマンド例は英語側を参照してください（実際は1行で書くか、Linux/macOS では `\`、Windows では `^` で改行してください）。

ControlNet 形式のデータセット TOML の最小例も英語側にある通りで、`conditioning_data_dir` の指定が SDXL LLLite と同一です。データセット設定の詳細については SDXL LLLite ガイド（[`train_lllite_README-ja.md`](./train_lllite_README-ja.md#データセットの準備)）と [データセット設定ガイド](./config_README-ja.md) を参照してください。

</details>

### 3.1. LLLite-Specific Arguments / LLLite 固有の引数

The following options are unique to this script. Anima-related arguments (`--qwen3`, `--vae`, `--llm_adapter_path`, `--t5_tokenizer_path`, `--timestep_sampling`, `--discrete_flow_shift`, etc.) and common training arguments (`--learning_rate`, `--optimizer_type`, `--gradient_checkpointing`, `--cache_latents`, `--cache_text_encoder_outputs`, ...) behave the same as in [`anima_train_network.md`](./anima_train_network.md).

* `--cond_emb_dim=<int>` (default `32`)
  * Channel dimension of the conditioning-image embedding produced by `conditioning1` (a small Conv2d /4 → /4 stack with overall stride 16). Larger values give more capacity to the conditioning representation.

* `--lllite_mlp_dim=<int>` (default `64`)
  * Hidden dimension of the LoRA-like down/mid/up MLP inside each LLLite module. Analogous to `network_dim` in standard LoRA.

* `--lllite_target_layers=<choice>` (default `self_attn_q`)
  * Which attention `Linear` layers receive an LLLite module:
    * `self_attn_q` — only `self_attn.q_proj` of each block (the lightest setting; ~1 module per block).
    * `self_attn_qkv` — adds `self_attn.k_proj` and `self_attn.v_proj`.
    * `self_attn_qkv_cross_q` — additionally adds `cross_attn.q_proj`.
  * `cross_attn.{k,v}_proj` and any `output_proj` are always skipped (they are incompatible with the conditioning sequence shape, or empirically reduce the additive effect).

* `--lllite_dropout=<float>` (default `None`)
  * Dropout applied to the LLLite mid output during training.

* `--lllite_multiplier=<float>` (default `1.0`)
  * Multiplier applied to the LLLite output during training. This same value is used for sample image generation unless overridden per-prompt with `--am`. **Setting this to `0.0` would disable LLLite at training time as well**, so do not use `0` as the global default — use the per-prompt `--am 0` only for inspection (see Section 4).

* `--network_weights=<path>`
  * Path to a pre-trained LLLite `.safetensors` file to resume from. The file is loaded with `strict=False`. The script does not currently enforce that `--lllite_target_layers` matches the metadata of the loaded file, so make sure they agree.

* `--conditioning_data_dir=<dir>`
  * Used only when **not** specifying `--dataset_config`. Together with `--train_data_dir` it produces a single subset-by-subdir dataset.

<details>
<summary>日本語</summary>

本スクリプト固有の引数は以下の通りです。Anima 関連の引数（`--qwen3`、`--vae`、`--llm_adapter_path`、`--t5_tokenizer_path`、`--timestep_sampling`、`--discrete_flow_shift` など）や共通の学習引数（`--learning_rate`、`--optimizer_type`、`--gradient_checkpointing`、`--cache_latents`、`--cache_text_encoder_outputs` ...）の挙動は [`anima_train_network.md`](./anima_train_network.md) と同一です。

* `--cond_emb_dim=<int>`（デフォルト `32`）— `conditioning1`（stride 4 の Conv2d を 2 段重ねた合計 stride 16 のエンコーダ）が出力する conditioning 埋め込みのチャンネル数。大きくすると表現力が増します。
* `--lllite_mlp_dim=<int>`（デフォルト `64`）— 各 LLLite モジュール内の down/mid/up MLP の中間次元。標準 LoRA の `network_dim` 相当です。
* `--lllite_target_layers=<choice>`（デフォルト `self_attn_q`）— LLLite モジュールを貼る attention の `Linear` レイヤを選択します：
  * `self_attn_q` — 各ブロックの `self_attn.q_proj` のみ（最軽量、1 ブロックあたり 1 モジュール）。
  * `self_attn_qkv` — `self_attn.k_proj` と `self_attn.v_proj` を追加。
  * `self_attn_qkv_cross_q` — さらに `cross_attn.q_proj` を追加。
  * なお `cross_attn.{k,v}_proj` と `output_proj` は常時スキップされます（前者は context 側との shape 不整合、後者は加算成分を弱める性質のため）。
* `--lllite_dropout=<float>`（デフォルト `None`）— LLLite の mid 出力に対する学習時 dropout。
* `--lllite_multiplier=<float>`（デフォルト `1.0`）— 学習中の LLLite 出力倍率。サンプル画像生成時にも、prompt 行で `--am` による上書きが無ければこの値が使われます。**`0.0` を指定すると学習時にも LLLite が完全 bypass され grad が乗らず学習が壊れる**ため、グローバルなデフォルトには `0` を使わないでください（観察用途であれば prompt 行で `--am 0` を指定する方法を推奨。第 4 節参照）。
* `--network_weights=<path>` — 続きから学習する場合の LLLite 重み（`.safetensors`）のパス。`strict=False` でロードします。`--lllite_target_layers` と保存時の値が一致しているか確認してください（現状自動チェックはしていません）。
* `--conditioning_data_dir=<dir>` — `--dataset_config` を使わない場合のみ用います。`--train_data_dir` と組み合わせ、サブディレクトリ単位の subset を自動生成します。

</details>

### 3.2. Recommended Starting Settings / 推奨される開始設定

The Phase D training that produced the first working weights used the following lightweight setup (only ~3.7 M trainable parameters), and even with that very small budget produced clearly recognizable line-following behavior:

* `--cond_emb_dim=32`
* `--lllite_mlp_dim=32`
* `--lllite_target_layers=self_attn_q`
* `--learning_rate=5e-5` (roughly half of the SDXL LLLite default; AdaLN-conditioned DiTs tend to be more sensitive to additive bias)
* `--optimizer_type=AdamW8bit`
* `--mixed_precision=bf16`
* `--gradient_checkpointing`, `--cache_latents`, `--cache_text_encoder_outputs`
* ~2,000 image / conditioning pairs at 1024² for ~10 epochs

For more demanding control signals (e.g. depth, segmentation), increase `--lllite_mlp_dim` and/or move to `--lllite_target_layers=self_attn_qkv` or `self_attn_qkv_cross_q`.

<details>
<summary>日本語</summary>

最初の実用重みを得た Phase D の学習では、わずか約 3.7M のパラメータ（極小設定）で線画追従が明確に再現されました。推奨の出発点は以下の通りです：

* `--cond_emb_dim=32`
* `--lllite_mlp_dim=32`
* `--lllite_target_layers=self_attn_q`
* `--learning_rate=5e-5`（SDXL LLLite のデフォルトのおよそ半分。AdaLN ベースの DiT は加算成分に対する感度が高めのため）
* `--optimizer_type=AdamW8bit`
* `--mixed_precision=bf16`
* `--gradient_checkpointing`、`--cache_latents`、`--cache_text_encoder_outputs`
* 1024² の画像/条件画像ペアを約 2,000 組、約 10 epoch

depth や segmentation などより難しい条件付けでは、`--lllite_mlp_dim` を増やす、`--lllite_target_layers` を `self_attn_qkv` や `self_attn_qkv_cross_q` に切り替える、といった調整を検討してください。

</details>

## 4. Sample Image Generation During Training / 学習中のサンプル画像生成

`--sample_prompts`, `--sample_every_n_epochs`, `--sample_every_n_steps`, `--sample_at_first` work the same as the LoRA training script. To pass a per-prompt control image and (optionally) a per-prompt LLLite multiplier, use the following extras in each `prompts.txt` line:

* `--cn <path>` — control image to feed into LLLite for this prompt.
* `--am <float>` — LLLite multiplier override for this prompt (`additional_network_multiplier`). The first value is used.

Example `prompts.txt`:

```
a cat sitting on a chair --w 1024 --h 1024 --cn lineart_a.png --am 0.8 --d 42
a dog --w 1024 --h 1024 --cn lineart_b.png
inspect base model output --w 1024 --h 1024 --cn lineart_c.png --am 0
```

If `--cn` is omitted (or the file does not exist), the prompt is rendered with the base DiT (LLLite cond cleared) and a warning is logged. The pre-prompt multiplier is saved before each sample and restored afterwards, so an `--am 0` line will not bleed into the next training step.

<details>
<summary>日本語</summary>

`--sample_prompts`、`--sample_every_n_epochs`、`--sample_every_n_steps`、`--sample_at_first` は LoRA 学習スクリプトと同様に動作します。プロンプト毎に control 画像（および LLLite multiplier）を指定するため、`prompts.txt` の各行で以下の追加オプションを使えます：

* `--cn <path>` — このプロンプトで LLLite に与える control 画像。
* `--am <float>` — このプロンプトでの LLLite multiplier 上書き値（`additional_network_multiplier`）。リスト形式で先頭値が使われます。

`--cn` が指定されない、もしくはファイルが存在しない場合は、LLLite の cond を解除した上で素の DiT で生成し、warning を出します。各プロンプトの直前に現在の multiplier を退避し、終了時に復元する仕組みが入っているため、`--am 0` を指定したプロンプトの影響が直後の学習ステップに漏れることはありません。

</details>

## 5. Saved Weights / 保存される重み

The saved `.safetensors` contains only the LLLite-side parameters:

```
conditioning1.0.weight, conditioning1.0.bias
conditioning1.2.weight, conditioning1.2.bias
lllite_modules.{i}.down.0.weight / .bias
lllite_modules.{i}.mid.0.weight  / .bias
lllite_modules.{i}.up.weight     / .bias
```

The metadata records `modelspec.architecture = "anima-preview/control-net-lllite"` together with `lllite.cond_emb_dim`, `lllite.mlp_dim`, and `lllite.target_layers`. These are read back by the inference script (Section 6) so you normally do not need to specify those three on the command line again.

Save cadence options (`--save_every_n_epochs`, `--save_every_n_steps`, `--save_state`, `--save_last_n_epochs`, `--save_last_n_steps`, ...) work the same as in standard training scripts.

<details>
<summary>日本語</summary>

保存される `.safetensors` には LLLite 側のパラメータのみが含まれます（state_dict のキー構造は英語側参照）。

メタデータには `modelspec.architecture = "anima-preview/control-net-lllite"` のほか、`lllite.cond_emb_dim`、`lllite.mlp_dim`、`lllite.target_layers` も書き込まれます。これらは推論スクリプト（第 6 節）で自動的に読み出されるため、通常はコマンドラインで再指定する必要はありません。

保存頻度の各オプション（`--save_every_n_epochs`、`--save_every_n_steps`、`--save_state`、`--save_last_n_epochs`、`--save_last_n_steps` など）は通常の学習スクリプトと同様に使えます。

</details>

## 6. Minimal Inference / 最低限の推論

`anima_minimal_inference_control_net_lllite.py` extends `anima_minimal_inference.py` (see its docstring for shared behavior — VAE / TE / DiT loading, `--from_file` batch mode, `--interactive`, `--latent_path` decode mode, prompt-line `--w/--h/--d/--s/--g/--fs/--n` overrides, etc.) and adds LLLite attachment. All standard inference options are inherited.

### 6.1. Single-Prompt Example / 単発プロンプトの例

```bash
python anima_minimal_inference_control_net_lllite.py \
  --dit "<path to Anima DiT>" \
  --vae "<path to Qwen-Image VAE>" \
  --text_encoder "<path to Qwen3-0.6B>" \
  --lllite_weights "out/my_anima_lllite-last.safetensors" \
  --control_image "lineart.png" \
  --prompt "a cat sitting on a chair" \
  --image_size 1024 1024 \
  --infer_steps 50 \
  --guidance_scale 3.5 \
  --save_path "out/"
```

### 6.2. Batch Mode / バッチモード

```bash
python anima_minimal_inference_control_net_lllite.py \
  --dit "<...>" --vae "<...>" --text_encoder "<...>" \
  --lllite_weights "out/my_anima_lllite-last.safetensors" \
  --control_image "default.png" \
  --from_file "infer_prompts.txt" \
  --save_path "out/"
```

`infer_prompts.txt` lines may include the standard prompt-line overrides plus two LLLite-specific ones:

* `--cn <path>` — per-prompt control image (overrides `--control_image`).
* `--am <float>` — per-prompt LLLite multiplier (overrides `--lllite_multiplier`).

Example:

```
a cat sitting on a chair --w 1024 --h 1024 --d 42 --cn lineart_a.png --am 0.8
a dog --w 1024 --h 1024 --d 0 --cn lineart_b.png
```

### 6.3. Inference-Only Arguments / 推論専用の引数

* `--lllite_weights <path>` **[required, unless `--latent_path` is given]** — trained LLLite weights.
* `--control_image <path>` — global control image. Required for single-prompt mode; optional in `--from_file` / `--interactive` mode if every prompt provides `--cn`.
* `--lllite_multiplier <float>` (default `1.0`) — global LLLite multiplier.
* `--lllite_cond_emb_dim`, `--lllite_mlp_dim`, `--lllite_target_layers` — manual overrides. Normally unnecessary because the values are read from the weights metadata.

CFG inference (cond / uncond passes) is handled by simply broadcasting the same `cond_emb` to both passes, so control is applied symmetrically.

<details>
<summary>日本語</summary>

`anima_minimal_inference_control_net_lllite.py` は `anima_minimal_inference.py` を拡張したスクリプトで、VAE / TE / DiT のロード、`--from_file`（バッチ）、`--interactive`、`--latent_path`（latent からの再デコード）、prompt 行での `--w/--h/--d/--s/--g/--fs/--n` オーバーライドなど、既存の推論機能はそのまま継承します。

* 単発推論のコマンド例、バッチ推論のコマンド例、`infer_prompts.txt` の書式は英語側を参照してください。
* バッチ用の追加 prompt 行オプション：
  * `--cn <path>` — このプロンプトでの control 画像（`--control_image` を上書き）。
  * `--am <float>` — このプロンプトでの LLLite 倍率（`--lllite_multiplier` を上書き）。
* 主要な推論専用引数：
  * `--lllite_weights <path>` **[必須、ただし `--latent_path` 指定時を除く]** — 学習済み LLLite 重み。
  * `--control_image <path>` — グローバル control 画像。単発推論では必須。`--from_file` / `--interactive` で全プロンプトが `--cn` を持つ場合は省略可。
  * `--lllite_multiplier <float>`（デフォルト `1.0`）— グローバル LLLite 倍率。
  * `--lllite_cond_emb_dim` / `--lllite_mlp_dim` / `--lllite_target_layers` — 通常はメタデータから自動読み込みされるため指定不要。形式変換などで必要な場合のみ手動上書きに用います。

CFG 推論（cond / uncond の 2 pass）は両 pass に同じ `cond_emb` を配布する形になっており、control は両側に対称に作用します。

</details>

## 7. Tips & Limitations / 補足と制限

* **Resolution alignment.** The conditioning encoder uses fixed stride 16, so `cond_image` HW must equal `latent HW × 8` (i.e. the original training image size) (the latent is patchified with patch size=2, so stride is 8*2=16). The DataLoader for the ControlNet dataset already resizes the conditioning image to match the training image, so in practice you only need to make sure the control image you pass at inference time matches the requested `--image_size`.
* **`T=1` only.** Video-style multi-frame inputs are not supported — the wrapper asserts `T==1` at forward time.
* **Bucket size.** The training script enforces a bucket resolution step of 16 (Qwen-Image VAE /8 × patch /2).
* **Memory.** `--blocks_to_swap`, `--cpu_offload_checkpointing`, `--unsloth_offload_checkpointing` are not yet supported. If VRAM is tight, prefer `--full_bf16`, smaller `--lllite_mlp_dim`, lower `--cond_emb_dim`, and `--gradient_checkpointing`.
* **Save format.** The saved `.safetensors` is **not** compatible with the SDXL LLLite format and **not** loadable by `sdxl_gen_img.py`. Use the dedicated inference script in Section 6.

<details>
<summary>日本語</summary>

* **解像度の整合性.** `conditioning1` の stride は 16 固定なので、`cond_image` の縦横は `latent HW × 8`（つまり元の学習画像サイズ）に一致している必要があります（latent はモデル内で patch size=2 で patchfy されるため、stride は 8*2=16 となる）。ControlNet 形式のデータローダ側で conditioning 画像は教師画像と同じサイズにリサイズされるため、実用上は推論時に渡す control 画像のサイズを `--image_size` と合わせれば OK です。
* **`T=1` のみ.** 動画的な多フレーム入力はサポートしていません（wrapper の forward 冒頭で assert）。
* **bucket サイズ.** 学習スクリプトは bucket 解像度ステップを 16（Qwen-Image VAE /8 × patch /2）として検証します。
* **メモリ.** `--blocks_to_swap`、`--cpu_offload_checkpointing`、`--unsloth_offload_checkpointing` は未対応です。VRAM が厳しい場合は `--full_bf16`、`--lllite_mlp_dim` を下げる、`--cond_emb_dim` を下げる、`--gradient_checkpointing` を有効にする、などで対応してください。
* **保存形式.** 保存される `.safetensors` は SDXL LLLite フォーマットとは**互換性がなく**、`sdxl_gen_img.py` ではロードできません。推論には第 6 節の専用スクリプトを使用してください。

</details>
