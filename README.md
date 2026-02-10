# Prompt Assistant

Stable Diffusion × Qwen3-VL を組み合わせた画像生成プロンプトアシスタント。

Qwen3-VL との会話を通じてプロンプトを磨き、SD WebUI Forge 2 で画像を生成する Gradio アプリです。

## 機能

- **プロンプト提案**: 画像や意図を伝えると Qwen3-VL が Stable Diffusion 向けのプロンプトを提案
- **プロンプト自動反映**: `[PROMPT_UPDATE]` タグを使って会話からプロンプトエリアを自動更新
- **画像生成**: Forge 2 の Gradio API 経由で txt2img 生成
- **画像ドロップ**: 既存の PNG をドロップすると A1111 メタデータを自動パース・反映
- **設定の自動保存**: プロンプト・パラメータ・モデル選択を `settings.json` に保存

## 必要環境

| 項目 | 要件 |
|---|---|
| Python | 3.10 以上 |
| GPU | CUDA 対応 GPU（Qwen3-VL-4B で VRAM ~6GB） |
| SD WebUI | [Stable Diffusion WebUI Forge 2](https://github.com/lllyasviel/stable-diffusion-webui-forge) |

## セットアップ

### 1. Forge 2 の準備

Forge 2 を起動してください（`--api` フラグは不要です）。

```
http://127.0.0.1:7861
```

### 2. 依存パッケージのインストール

```bash
conda activate main
pip install -r requirements.txt
```

### 3. 起動

```bat
start.bat
```

または

```bash
conda activate main
python app.py
```

ブラウザが自動で開きます（`http://127.0.0.1:7860`）。

## 使い方

1. **モデルをロード**: 上部のドロップダウンでモデルを選択し「モデルをロード」をクリック
2. **プロンプトを入力**: Positive / Negative Prompt を手入力、または画像をドロップして自動読み込み
3. **Qwen3-VL と会話**: 右カラムの入力欄にメッセージを入力して「送信」
   - プロンプト更新を提案された場合は自動でプロンプトエリアに反映されます
4. **画像を生成**: 「画像生成」ボタンをクリック

### プロンプト自動更新

Qwen3-VL の返答に以下のタグが含まれると、プロンプトエリアが自動更新されます。

```
[PROMPT_UPDATE]
Positive: 1girl, sunset, orange sky, dramatic lighting, ...
Negative: bad quality, worst quality, blurry, ...
[/PROMPT_UPDATE]
```

### 画像のドロップ

- **A1111 生成画像（PNG）**: メタデータを自動パースし、プロンプト・Steps・CFG・Sampler・サイズ・Seed を反映
- **メタデータなし画像**: 画像のみ Qwen3-VL のコンテキストに渡し、内容を分析させることができます

## モデルプリセット

| ラベル | モデル ID | 目安 VRAM |
|---|---|---|
| qwen3-vl-4b (推奨) | `Qwen/Qwen3-VL-4B-Instruct` | ~6 GB |
| qwen3-vl-8b (高性能) | `Qwen/Qwen3-VL-8B-Instruct` | ~10-12 GB |
| huihui-qwen3-vl-4b-abliterated | `huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated` | ~6 GB |
| huihui-qwen3-vl-8b-abliterated | `huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated` | ~10-12 GB |

モデルは初回ロード時に `models/` フォルダにダウンロードされます。

## ファイル構成

```
prompt-assistant/
├── app.py                  # Gradio アプリ本体
├── a1111_client.py         # Forge 2 Gradio API クライアント
├── qwen_client.py          # Qwen3-VL 推論クライアント
├── prompt_parser.py        # プロンプトパース・PNG メタデータ読み取り
├── settings_manager.py     # 設定の保存・読み込み
├── discover_forge_api.py   # Forge 2 API 調査ツール（開発用）
├── requirements.txt
├── start.bat
└── models/                 # モデルキャッシュ（HF_HOME）
```
