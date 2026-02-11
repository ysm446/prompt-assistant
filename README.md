# Prompt Assistant

Stable Diffusion × Qwen3-VL を組み合わせた画像・動画生成プロンプトアシスタント。

Qwen3-VL との会話を通じてプロンプトを磨き、SD WebUI Forge または ComfyUI で画像・動画を生成する Gradio アプリです。

## 機能

- **プロンプト提案**: 画像や意図を伝えると Qwen3-VL が Stable Diffusion 向けのプロンプトを提案
- **プロンプト自動反映**: `[PROMPT_UPDATE]` タグを使って会話からプロンプトエリアを自動更新
- **画像生成**: WebUI Forge（Gradio API）または ComfyUI（REST API）で txt2img 生成
- **連続生成**: 生成枚数を指定して連続生成。seed が指定値の場合は +1 ずつインクリメント
- **画像ドロップ**: A1111 / ComfyUI 生成 PNG をドロップするとメタデータを自動パース・反映
- **動画生成**: ComfyUI + WAN 2.2 ワークフローで画像から動画を生成
- **動画プロンプト生成**: Qwen3-VL が Scene / Action / Camera / Style / Final Prompt の各セクションを自動生成（セクションをチェックボックスで選択可）
- **VRAM解放**: LLM・WebUI Forge・ComfyUI のモデルをアプリから個別にアンロード
- **設定の自動保存**: プロンプト・パラメータ・モデル選択を `settings.json` に保存

## 必要環境

| 項目 | 要件 |
|---|---|
| Python | 3.10 以上 |
| GPU | CUDA 対応 GPU（Qwen3-VL-4B で VRAM ~6GB） |
| 画像生成バックエンド | [SD WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) または [ComfyUI](https://github.com/comfyanonymous/ComfyUI) |

## セットアップ

### 1. 画像生成バックエンドの準備

**WebUI Forge** を使う場合:
```
http://127.0.0.1:7861 で起動（--api フラグ不要）
```

**ComfyUI** を使う場合:
```
http://127.0.0.1:8188 で起動
workflows/ フォルダにワークフロー JSON を配置
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

### 画像生成

1. **モデルをロード**: モデル設定アコーディオンでモデルを選択し「モデルをロード」をクリック
2. **プロンプトを入力**: Positive / Negative Prompt を手入力、または画像をドロップして自動読み込み
3. **Qwen3-VL と会話**: 右カラムの入力欄にメッセージを入力して「送信」
   - プロンプト更新を提案された場合は自動でプロンプトエリアに反映されます
4. **画像を生成**: 「画像生成」ボタンをクリック（枚数指定・停止ボタンあり）
5. **バックエンド切り替え**: 「画像生成パラメータ」アコーディオンからバックエンドを選択

### 動画生成

1. **画像生成タブで画像を用意**: 生成または画像をドロップして current_image にセット
2. **動画生成タブに切り替え**
3. **（任意）動画プロンプトを生成**:
   - 生成するセクション（Scene / Action / Camera / Style / Final Prompt）をチェックで選択
   - 追加指示を入力（任意）
   - 「動画プロンプト生成」ボタンをクリック → Qwen3-VL が各セクションを生成
4. **動画生成パラメータを設定**: ワークフロー・Width・Height・Seed をアコーディオンで設定
5. **「動画生成」ボタンをクリック**: ComfyUI にジョブを送信、経過時間をステータスに表示

### プロンプト自動更新

Qwen3-VL の返答に以下のタグが含まれると、プロンプトエリアが自動更新されます。

```
[PROMPT_UPDATE]
Positive: 1girl, sunset, orange sky, dramatic lighting, ...
Negative: bad quality, worst quality, blurry, ...
[/PROMPT_UPDATE]
```

### 画像のドロップ

- **A1111 / WebUI Forge 生成画像（PNG）**: プロンプト・Steps・CFG・Sampler・サイズ・Seed を反映
- **ComfyUI 生成画像（PNG）**: CLIPTextEncode ノードからポジティブ・ネガティブプロンプトを読み取り反映
- **メタデータなし画像**: 画像のみ Qwen3-VL のコンテキストに渡し、内容を分析させることができます

### VRAM 解放

動画生成タブの「VRAM」アコーディオンに3つのボタンがあります:
- **LLM 解放**: Qwen3-VL モデルをアンロード（再ロードは「モデルをロード」ボタン）
- **WebUI Forge 解放**: Forge のチェックポイントをアンロード
- **ComfyUI 解放**: ComfyUI のモデルキャッシュを解放（`/free` エンドポイント）

### ComfyUI ワークフロー

- **画像ワークフロー**: `workflows/image/` に配置 → 画像生成パラメータのドロップダウンに表示
- **動画ワークフロー**: `workflows/video/` に配置 → 動画生成パラメータのドロップダウンに表示

ワークフロー内の `CLIPTextEncode` ノードのタイトルに `negative` / `ネガティブ` / `neg` が含まれるとネガティブプロンプトとして扱われます。

サイズ（Width / Height）は以下のノードに自動で書き込まれます:
`EmptyLatentImage`・`EmptySD3LatentImage`・`WanImageToVideo`・`WanVideoToVideo`・`EmptyWanLatentVideo` 他

## LLM モデルプリセット

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
├── a1111_client.py         # WebUI Forge Gradio API クライアント
├── comfyui_client.py       # ComfyUI REST API クライアント
├── qwen_client.py          # Qwen3-VL 推論クライアント
├── prompt_parser.py        # プロンプトパース・PNG メタデータ読み取り
├── settings_manager.py     # 設定の保存・読み込み
├── discover_forge_api.py   # Forge API 調査ツール（開発用）
├── requirements.txt
├── start.bat
├── workflows/
│   ├── image/              # 画像生成用 ComfyUI ワークフロー JSON
│   └── video/              # 動画生成用 ComfyUI ワークフロー JSON（WAN 2.2 等）
└── models/                 # モデルキャッシュ（HF_HOME）
```
