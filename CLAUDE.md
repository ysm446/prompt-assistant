# CLAUDE.md

## プロジェクト概要

SD WebUI Forge × Qwen3-VL を組み合わせた画像・動画生成プロンプトアシスタント。
Qwen3-VL との会話でプロンプトを磨き、WebUI Forge または ComfyUI で画像を生成する Gradio アプリ。
動画生成は ComfyUI + WAN 2.2 ワークフローに対応。

## 環境・実行方法

- **Python 環境**: conda `main` 環境
- **起動**: `start.bat` または `python app.py`
- **WebUI Forge**: `http://127.0.0.1:7861` で起動済みであること（`--api` フラグ不要、Gradio API を使用）
- **ComfyUI**: `http://127.0.0.1:8188` で起動済みであること（REST API 使用）
- **モデルキャッシュ**: `HF_HOME` を `models/` に設定済み

```bat
call conda activate main
python app.py
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| `app.py` | Gradio UI 定義・イベントハンドラ |
| `a1111_client.py` | WebUI Forge Gradio API クライアント（`/txt2img`、152 パラメータ構成）、`free_vram()` |
| `comfyui_client.py` | ComfyUI REST API クライアント（`/prompt`・`/history`・`/view`・`/free`）、ワークフロー差し替え |
| `qwen_client.py` | Qwen3-VL 推論クライアント（transformers + qwen_vl_utils）、`unload_model()` |
| `prompt_parser.py` | `[PROMPT_UPDATE]` タグパース・A1111/ComfyUI PNG メタデータ読み取り |
| `settings_manager.py` | 設定の JSON 保存・読み込み（`settings.json`） |
| `discover_forge_api.py` | Forge API エンドポイント調査ツール（開発用） |
| `workflows/` | ComfyUI ワークフロー JSON 置き場（起動時に自動スキャン） |

## 重要な注意点

### WebUI Forge API
- Forge (f2.0.1+) は `/sdapi/v1/txt2img` REST API を**廃止**。`gradio_client` 経由の Gradio API のみ。
- `a1111_client.py` の `generate_image()` は 152 個の引数を位置引数で渡す。
- パラメータ順は `forge_api_info.txt`（`discover_forge_api.py` の出力）を参照。
- **Sampling steps は `int` で渡すこと**（`float` だと `torch.linspace` がエラー）。
- `_get_client()` は `requests.get(FORGE_URL, timeout=5)` で事前疎通確認を行う（未起動時の無限ハング防止）。
- VRAM解放: `POST /sdapi/v1/unload-checkpoint`（対応していない場合はエラーメッセージを返す）。

### ComfyUI API
- REST API ベース（WebSocket は使用しない）。
- 生成完了は `/history/{prompt_id}` をポーリング（0.5 秒間隔、最大 300 秒）で検出。
- ワークフロー JSON は `_patch_workflow()` で CLIPTextEncode・KSampler・EmptyLatentImage 系・WAN 動画系ノードを差し替え。
- ネガティブ判定: `CLIPTextEncode` ノードのタイトルに `negative` / `ネガティブ` / `neg` が含まれるか。
- seed=-1 の場合は `random.randint(0, 2**32 - 1)` でランダム値を生成してから書き込む。
- VRAM解放: `POST /free` に `{"unload_models": true, "free_memory": true}` を送信。
- サイズ差し替え対象ノード（`_LATENT_IMAGE_NODES`）: `EmptyLatentImage`・`EmptySD3LatentImage`・`EmptyLatentImageSD3`・`EmptyHunyuanLatentVideo`・`WanImageToVideo`・`WanVideoToVideo`・`EmptyWanLatentVideo`。
- 動画ワークフローは `workflows/video/` に配置（`VIDEO_WORKFLOW_PRESETS` として自動スキャン）。
- 動画生成は `threading.Thread` でバックグラウンド実行し、メインループで経過時間をポーリング。

### Qwen3-VL
- `process_vision_info(messages)` を使って画像を抽出すること（直接 PIL を渡すと不一致エラー）。
- `AutoModelForImageTextToText` を使用（`AutoModelForCausalLM` ではない）。
- `dtype` 引数を使うこと（`torch_dtype` は deprecated）。
- モデルアンロード: `_model = _processor = _loaded_model_id = None` + `torch.cuda.empty_cache()`。
- 動画プロンプト生成: `generate_video_prompt_stream(image, positive_prompt, extra_instruction, sections)` でストリーミング生成。
  - `sections` に生成するセクション（`scene`・`action`・`camera`・`style`・`prompt`）のリストを渡す。
  - `_SECTION_TEMPLATES` に各セクションのテンプレート文字列を定義。`_ALL_SECTIONS` は全セクション順序を保持。

### Gradio 6.5.1
- `gr.Chatbot` はデフォルトで messages 形式を使用。
  - `[{"role": "user", "content": "..."}]` 形式で渡すこと。
  - タプル形式 `[[user, assistant]]` は不可。
- `gr.Chatbot(type=..., bubble_full_width=...)` 引数は存在しない。
- `gr.Image` に `show_fullscreen_button` 引数は存在しない（CSS で非表示にする）。
- ジェネレーター関数（`yield`）を使う場合は `app.queue()` が必須。
- 連続生成の停止: `gen_event = btn.click(...)` + `stop_btn.click(fn=None, cancels=[gen_event])`。

### UI レイアウト
- トップレベルは `gr.Tabs()` で「画像生成」「動画生成」タブに分割。
- **画像生成タブ**: 左（画像表示）・中（プロンプト・パラメータ）・右（Qwen3-VL チャット）の3列。
- **動画生成タブ**: 左（生成動画・ステータス）・中（動画プロンプト・生成/停止ボタン・VRAM）・右（動画生成パラメータ・追加指示・セクション選択・プロンプト生成）の3列。
- VRAM アコーディオンは動画タブの中列（生成/停止ボタン下）に配置。

## 設定ファイル

`settings.json`（gitignore 済み）に以下を保存：
- `model`, `positive_prompt`, `negative_prompt`
- `steps`, `cfg`, `sampler`, `width`, `height`, `seed`
- `backend` (`"WebUI Forge"` or `"ComfyUI"`)
- `comfyui_workflow`, `comfyui_url`, `comfyui_seed`, `comfyui_width`, `comfyui_height`
- `generate_count`
- `video_sections`（リスト: `scene` / `action` / `camera` / `style` / `prompt`）
- `video_width`, `video_height`（動画サイズ、スライダー 240–1920 step16）
- `video_seed`（-1 でランダム）

各 UI コンポーネントの `.change` イベントで自動保存。

## 開発コマンド

```bash
# Forge API のパラメータ一覧を調査
python discover_forge_api.py
# → forge_api_info.txt に出力（cp932 問題を回避するためファイル出力）
```
