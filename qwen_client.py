"""
qwen_client.py
Qwen3-VL 推論クライアント（transformers ライブラリ経由）
アプリ起動時に 1 回だけモデルをロードし、グローバルに保持する。
"""

import threading

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info

# 利用可能なモデルプリセット
MODEL_PRESETS = {
    "qwen3-vl-4b (推奨)": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b (高性能)": "Qwen/Qwen3-VL-8B-Instruct",
    "huihui-qwen3-vl-4b-abliterated": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
    "huihui-qwen3-vl-8b-abliterated": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
}

_model = None
_processor = None
_loaded_model_id = None


def load_model(model_id: str) -> str:
    """
    モデルをロードする。既に同じモデルがロード済みの場合はスキップ。
    Returns: ステータスメッセージ
    """
    global _model, _processor, _loaded_model_id

    if _loaded_model_id == model_id and _model is not None:
        return f"モデル {model_id} は既にロード済みです。"

    try:
        _model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _processor = AutoProcessor.from_pretrained(model_id)
        _loaded_model_id = model_id
        return f"モデル {model_id} のロードが完了しました。"
    except Exception as e:
        _model = None
        _processor = None
        _loaded_model_id = None
        raise RuntimeError(f"モデルのロードに失敗しました: {e}") from e


def is_loaded() -> bool:
    return _model is not None and _processor is not None


def _build_inputs(
    conversation_history: list[dict],
    user_input: str,
    current_image: "Image.Image | None",
    positive_prompt: str,
    negative_prompt: str,
):
    """メッセージを組み立て、processor に通してテンソルを返す。"""
    system_prompt = (
        "あなたは画像生成のプロンプトエンジニアリングの専門家です。\n"
        "ユーザーの意図を理解し、Stable Diffusion（Illustrious チェックポイント）向けの\n"
        "高品質なプロンプトを提案してください。\n\n"
        f"現在のプロンプト:\n"
        f"Positive: {positive_prompt}\n"
        f"Negative: {negative_prompt}\n\n"
        "【修正する場合の方針】\n"
        "プロンプトの構成をなるべく変更せず、単語だけ置き換えること。\n"
        "【ネガティブプロンプトの方針】\n"
        "- ネガティブプロンプトは原則として空のままにすること。\n"
        "- ユーザーが「〜を除外したい」「〜を出したくない」と明示的に求めた場合のみ追加すること。\n"
        "- 追加する場合も 10 タグ以内に抑えること。\n\n"
        "プロンプトを更新する場合は、返答の中に以下のフォーマットで含めてください:\n"
        "[PROMPT_UPDATE]\n"
        "Positive: <新しい positive プロンプト>\n"
        "Negative: <新しい negative プロンプト>\n"
        "[/PROMPT_UPDATE]"
    )

    user_content: list[dict] = []
    if current_image is not None:
        user_content.append({"type": "image", "image": current_image})
    user_content.append({"type": "text", "text": user_input})

    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": user_content},
    ]

    try:
        # Qwen3 系は thinking モードがデフォルト ON なので明示的に無効化
        text = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    image_inputs, video_inputs = process_vision_info(messages)

    return _processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)


def query_stream(
    conversation_history: list[dict],
    user_input: str,
    current_image: "Image.Image | None",
    positive_prompt: str,
    negative_prompt: str,
):
    """
    Qwen3-VL にストリーミングで問い合わせ、トークンを逐次 yield する。
    """
    if not is_loaded():
        raise RuntimeError("モデルがロードされていません。先にモデルをロードしてください。")

    inputs = _build_inputs(
        conversation_history, user_input, current_image, positive_prompt, negative_prompt
    )

    streamer = TextIteratorStreamer(
        _processor, skip_prompt=True, skip_special_tokens=True
    )

    def _generate():
        with torch.inference_mode():
            _model.generate(**inputs, max_new_tokens=512, streamer=streamer)

    thread = threading.Thread(target=_generate)
    thread.start()

    try:
        for token in streamer:
            yield token
    finally:
        thread.join()
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
