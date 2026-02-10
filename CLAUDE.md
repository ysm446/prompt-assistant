# CLAUDE.md

## プロジェクト概要

SD WebUI Forge 2 × Qwen3-VL を組み合わせた画像生成プロンプトアシスタント。
Qwen3-VL との会話でプロンプトを磨き、Forge 2 で画像を生成する Gradio アプリ。

## 環境・実行方法

- **Python 環境**: conda `main` 環境
- **起動**: `start.bat` または `python app.py`
- **Forge 2**: `http://127.0.0.1:7861` で起動済みであること（`--api` フラグ不要、Gradio API を使用）
- **モデルキャッシュ**: `HF_HOME` を `models/` に設定済み

```bat
call conda activate main
python app.py
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| `app.py` | Gradio UI 定義・イベントハンドラ |
| `a1111_client.py` | Forge 2 Gradio API クライアント（`/txt2img`、152 パラメータ構成） |
| `qwen_client.py` | Qwen3-VL 推論クライアント（transformers + qwen_vl_utils） |
| `prompt_parser.py` | `[PROMPT_UPDATE]` タグパース・PNG メタデータ読み取り |
| `settings_manager.py` | 設定の JSON 保存・読み込み（`settings.json`） |
| `discover_forge_api.py` | Forge 2 API エンドポイント調査ツール（開発用） |

## 重要な注意点

### Forge 2 API
- Forge 2 (f2.0.1+) は `/sdapi/v1/txt2img` REST API を**廃止**。`gradio_client` 経由の Gradio API のみ。
- `a1111_client.py` の `generate_image()` は 152 個の引数を位置引数で渡す。
- パラメータ順は `forge_api_info.txt`（`discover_forge_api.py` の出力）を参照。
- **Sampling steps は `int` で渡すこと**（`float` だと `torch.linspace` がエラー）。

### Qwen3-VL
- `process_vision_info(messages)` を使って画像を抽出すること（直接 PIL を渡すと不一致エラー）。
- `AutoModelForImageTextToText` を使用（`AutoModelForCausalLM` ではない）。
- `dtype` 引数を使うこと（`torch_dtype` は deprecated）。

### Gradio 6.5.1
- `gr.Chatbot` はデフォルトで messages 形式を使用。
  - `[{"role": "user", "content": "..."}]` 形式で渡すこと。
  - タプル形式 `[[user, assistant]]` は不可。
- `gr.Chatbot(type=..., bubble_full_width=...)` 引数は存在しない。

## 設定ファイル

`settings.json`（gitignore 済み）に以下を保存：
- model, positive_prompt, negative_prompt
- steps, cfg, sampler, width, height, seed

各 UI コンポーネントの `.change` イベントで自動保存。

## 開発コマンド

```bash
# Forge 2 API のパラメータ一覧を調査
python discover_forge_api.py
# → forge_api_info.txt に出力（cp932 問題を回避するためファイル出力）
```
