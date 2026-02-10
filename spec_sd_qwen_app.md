# Prompt Assistant 画像生成アシスタント 仕様書

## 概要

Stable Diffusion（Automatic1111）と Qwen3-VL を組み合わせた画像生成アシスタントアプリ。
Qwen3-VL との会話を通じてプロンプトを整え、画像の完成度を反復的に高めていくことを目的とする。

---

## 技術スタック

| コンポーネント | 採用技術 |
|---|---|
| UI フレームワーク | Gradio (`gr.Blocks`) |
| 画像生成バックエンド | Automatic1111 REST API |
| VLM | Qwen3-VL（transformers ライブラリ経由） |
| 言語 | Python 3.10+ |

---

### プリセット一覧

**Qwen3-VL シリーズ（最新世代）**
- **qwen3-vl-4b (推奨)**: `Qwen/Qwen3-VL-4B-Instruct`（VRAM ~6GB）
- **qwen3-vl-8b (高性能)**: `Qwen/Qwen3-VL-8B-Instruct`（VRAM ~10-12GB）

**試験用（フィルタ除去版）**
- **huihui-qwen3-vl-4b-abliterated**: `huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated`
- **huihui-qwen3-vl-8b-abliterated**: `huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated`


## UI レイアウト

```
┌─────────────────────────┬──────────────────────────┐
│                         │  【プロンプトエリア】      │
│   【画像表示エリア】      │  Positive Prompt         │
│                         │  [テキストボックス]        │
│   gr.Image              │                          │
│   (生成画像を表示)        │  Negative Prompt         │
│                         │  [テキストボックス]        │
│                         ├──────────────────────────┤
│                         │  【生成パラメータ】        │
│                         │  Steps / CFG / Sampler   │
│                         │  Width / Height / Seed   │
├─────────────────────────┴──────────────────────────┤
│  【会話エリア】                                      │
│  gr.Chatbot（Qwen3-VL との会話履歴）                 │
├─────────────────────────────────────────────────────┤
│  【入力ボックス】  [テキスト入力]  [送信]  [画像生成] │
└─────────────────────────────────────────────────────┘
```

---

## 各エリアの詳細仕様

### 画像表示エリア
- `gr.Image` コンポーネントを使用（`interactive=True`）
- 生成完了後に最新画像を表示する
- 画像をクリックして拡大表示できることが望ましい
- **画像ドロップ／アップロード対応**
  - ユーザーが既存の画像をドロップ・アップロードできる
  - ドロップした画像は `current_image` として状態に保存され、次回 Qwen3-VL へのリクエスト時に会話コンテキストに含まれる
  - A1111 生成画像の場合、PNG メタデータ（`parameters` チャンク）を自動パースしてプロンプト・生成パラメータを各エリアに反映する
  - メタデータがない画像（スクリーンショット・外部画像など）はプロンプト・パラメータエリアを変更せずに画像のみ `current_image` に保持し、Qwen3-VL に内容を分析させることができる

### プロンプトエリア
- Positive Prompt と Negative Prompt の 2 つのテキストボックス（Negative Promptは小さめ）
- Qwen3-VL の提案を受けて自動更新される
- ユーザーが手動で直接編集することも可能
- 更新時は上書きではなく差分が分かるよう、変更箇所をハイライト表示することが望ましい（初期実装では上書きで可）（巻き戻しも簡単にできるようにしたい）

### 生成パラメータエリア
- Steps（スライダー、デフォルト 28、範囲 1〜150）
- CFG Scale（スライダー、デフォルト 7.0、範囲 1〜30）
- Sampler（ドロップダウン、A1111 から取得した一覧を表示）
- Width / Height（ドロップダウン、512 / 768 / 832 / 1024 / 1216）
- Seed（数値入力、-1 でランダム）

### 会話エリア
- gr.Chatbot コンポーネントを使用、テキスト式（LM Studio風）スタイルに設定
- 発言者名（You / Assistant）をラベルとして表示し、本文はその下にフラットなテキストで表示
- 吹き出しなし・背景色なしでコンパクトに表示、長文レスポンスが読みやすい
- スクロール可能、十分な高さを確保する（400px 以上推奨）
- Qwen3-VL 発言にプロンプト更新が含まれる場合、[PROMPT_UPDATE] タグ部分は非表示にしてプロンプトエリアの更新のみ行う

### 入力ボックス
- `gr.Textbox` で 1 行入力
- 【送信】ボタン：Qwen3-VL に送信して会話を継続
- 【画像生成】ボタン：現在のプロンプトで A1111 に生成リクエスト

---

## Qwen3-VL へ渡すコンテキスト

毎回のリクエストで以下の 3 つをコンテキストとして含める。

```python
system_prompt = """
あなたは画像生成のプロンプトエンジニアリングの専門家です。
ユーザーの意図を理解し、Stable Diffusion（Illustrious チェックポイント）向けの
高品質なプロンプトを提案してください。

現在のプロンプト:
Positive: {current_positive_prompt}
Negative: {current_negative_prompt}
"""

# メッセージに画像を含める（multimodal）
messages = [
    {"role": "system", "content": system_prompt},
    *conversation_history,  # 過去の会話履歴
    {
        "role": "user",
        "content": [
            {"type": "image", "image": current_image},  # 最新の生成画像
            {"type": "text",  "text": user_input},
        ]
    }
]
```

画像が未生成の場合（初回）は image フィールドを省略する。

---

## Automatic1111 API 連携

### エンドポイント

| 用途 | メソッド | パス |
|---|---|---|
| テキストから画像生成 | POST | `/sdapi/v1/txt2img` |
| サンプラー一覧取得 | GET | `/sdapi/v1/samplers` |
| 進捗確認 | GET | `/sdapi/v1/progress` |

### リクエスト例（txt2img）

```python
import requests
import base64
from io import BytesIO
from PIL import Image

A1111_URL = "http://127.0.0.1:7860"

def generate_image(positive, negative, steps, cfg, sampler, width, height, seed):
    payload = {
        "prompt": positive,
        "negative_prompt": negative,
        "steps": steps,
        "cfg_scale": cfg,
        "sampler_name": sampler,
        "width": width,
        "height": height,
        "seed": seed,
    }
    response = requests.post(f"{A1111_URL}/sdapi/v1/txt2img", json=payload)
    response.raise_for_status()

    image_b64 = response.json()["images"][0]
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    return image
```

### 起動確認

アプリ起動時に `/sdapi/v1/samplers` を GET して A1111 が動いているか確認する。
接続できない場合はエラーメッセージを表示する。

---

## Qwen3-VL 連携

### モデルのロード

transformers ライブラリで直接ロードする。RTX PRO 5000（48GB VRAM）であれば
7B〜72B クラスのモデルも動作可能。

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

# Qwen3-VL のモデル ID（公開後に確定）
# 現時点では Qwen2.5-VL-7B-Instruct または 72B-Instruct を想定
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # bfloat16 で VRAM を節約
    device_map="auto",           # GPU に自動配置
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# アプリ起動時に 1 回だけロードし、グローバル変数として保持する
# 推論ごとにロードし直すとオーバーヘッドが大きいため
```

> **Note:** A1111（SD）と Qwen3-VL を同時に VRAM に乗せる場合、
> SD モデルが約 4〜8GB、Qwen3-VL-7B が約 16GB を消費する。
> 48GB VRAM であれば両方の同時実行に余裕がある。

### 推論

```python
def query_qwen(messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=..., return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response
```

### プロンプト更新ロジック

Qwen3-VL の返答にプロンプト提案が含まれる場合、以下のフォーマットを返すよう
システムプロンプトで指示する。

```
[PROMPT_UPDATE]
Positive: 1girl, sunset, orange sky, dramatic lighting, ...
Negative: bad quality, worst quality, ...
[/PROMPT_UPDATE]
```

アプリ側でこのタグをパースして、プロンプトエリアを自動更新する。
タグが含まれない場合は通常の会話として処理する。

---

## PNG メタデータの読み取り

### A1111 生成画像のフォーマット

A1111 が PNG に埋め込む `parameters` チャンクのフォーマット：

```
1girl, sunset, orange sky, dramatic lighting, ...
Negative prompt: bad quality, worst quality, ...
Steps: 28, Sampler: Euler a, CFG scale: 7, Seed: 12345, Size: 512x768, Model: ..., ...
```

### パースロジック（prompt_parser.py）

```python
from PIL import Image

def read_a1111_metadata(image: Image.Image) -> dict | None:
    """
    A1111 生成画像から生成パラメータを読み取る。
    メタデータがない場合は None を返す。
    """
    params_text = image.info.get("parameters", "")
    if not params_text:
        return None

    lines = params_text.strip().splitlines()

    # Negative prompt 行を境界として positive / negative を分割
    neg_idx = next((i for i, l in enumerate(lines) if l.startswith("Negative prompt:")), None)
    param_idx = next((i for i, l in enumerate(lines) if l.startswith("Steps:")), None)

    positive = "\n".join(lines[:neg_idx or param_idx or len(lines)]).strip()
    negative = ""
    if neg_idx is not None:
        end = param_idx if param_idx else len(lines)
        negative = "\n".join(lines[neg_idx:end]).removeprefix("Negative prompt:").strip()

    result = {"positive": positive, "negative": negative}

    # Steps 行からキー値ペアを取得
    if param_idx is not None:
        param_line = ", ".join(lines[param_idx:])
        for token in param_line.split(","):
            token = token.strip()
            if ":" in token:
                k, _, v = token.partition(":")
                result[k.strip().lower().replace(" ", "_")] = v.strip()

    return result
```

### 読み取り結果の UI への反映

| メタデータキー | 反映先 |
|---|---|
| `positive` | Positive Prompt テキストボックス |
| `negative` | Negative Prompt テキストボックス |
| `steps` | Steps スライダー |
| `cfg_scale` | CFG Scale スライダー |
| `sampler` | Sampler ドロップダウン（一覧に存在すれば選択、なければそのまま表示） |
| `size`（`512x768` 形式） | Width / Height ドロップダウン |
| `seed` | Seed 数値入力 |

---

## 状態管理（Gradio State）

```python
state = {
    "conversation_history": [],   # Qwen3-VL との会話履歴（messages 形式）
    "current_image": None,        # 最新の生成画像（PIL.Image）
    "positive_prompt": "",        # 現在の Positive プロンプト
    "negative_prompt": "",        # 現在の Negative プロンプト
}
```

Gradio の `gr.State` コンポーネントを使用してセッションをまたいで保持する。

---

## ファイル構成

```
project/
├── app.py                  # Gradio アプリ本体・エントリーポイント
├── a1111_client.py         # Automatic1111 API クライアント
├── qwen_client.py          # Qwen3-VL 推論クライアント
├── prompt_parser.py        # Qwen3-VL の返答からプロンプトをパース／PNG メタデータの読み取り
├── requirements.txt        # 依存パッケージ
└── README.md
```

---

## requirements.txt

```
gradio>=4.0.0
requests
Pillow
transformers>=4.40.0
torch
accelerate
qwen-vl-utils
```

---

## 起動方法

```bash
# A1111 を API モードで起動（別ターミナル）
# webui.bat に --api オプションを追加して起動

# アプリを起動
python app.py
```

---

## 将来的な拡張候補

- ComfyUI バックエンドの追加対応（Z Image Turbo / Qwen Image 向け）
- バックエンド切り替えドロップダウンの追加
- 生成履歴の保存・ブラウズ機能
- プロンプト差分ハイライト表示
- A1111 の生成中プレビュー（`/sdapi/v1/progress` のポーリング）
- LoRA・VAE の切り替え UI
