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


def unload_model() -> str:
    """モデルをアンロードして VRAM を解放する。"""
    global _model, _processor, _loaded_model_id
    if _model is None:
        return "Qwen3-VL モデルはロードされていません。"
    _model = None
    _processor = None
    _loaded_model_id = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Qwen3-VL モデルをアンロードしました。"


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


_SECTION_TEMPLATES = {
    "scene":  "**Scene**: [Describe the visual scene in detail based on the image]",
    "action": "**Action**: [Describe the motion/movement to add - be specific about what moves and how]",
    "camera": "**Camera**: [Describe camera movement: static, slow pan, zoom in/out, dolly, tracking, etc.]",
    "style":  "**Style**: [Describe the visual style and mood]",
    "prompt": "---\n**Final Prompt for WAN 2.2**:\n[Write a single paragraph combining all elements. This should be copy-paste ready for WAN 2.2. Write in English, be concise but descriptive. Focus on motion and cinematic qualities.]",
}
_ALL_SECTIONS = list(_SECTION_TEMPLATES.keys())

_VIDEO_SYSTEM_PROMPT_TEMPLATE = """\
あなたはWan2.2動画生成のプロンプトエンジニアリングの専門家です。
提供された画像・画像プロンプト・追加指示を元に、以下のセクションを順番に出力してください。

現在の画像プロンプト: {positive_prompt}

出力するセクション（指定されたものだけ出力してください）:
{sections_text}

ルール:
- 指定されたセクションのみを出力してください（他のセクション・前置き・コメント不要）
- 英語で記述してください
- 動きや変化・雰囲気を具体的に記述してください
"""


def generate_video_prompt_stream(
    image: "Image.Image | None",
    positive_prompt: str,
    extra_instruction: str,
    sections: "list[str] | None" = None,
):
    """
    画像・画像プロンプト・追加指示から動画用プロンプトをストリーミング生成する。
    チャット履歴は使わないワンショット呼び出し。プロンプトテキストのみを yield する。
    """
    if not is_loaded():
        raise RuntimeError("モデルがロードされていません。先にモデルをロードしてください。")

    active_sections = sections if sections else _ALL_SECTIONS
    sections_text = "\n".join(
        _SECTION_TEMPLATES[s] for s in _ALL_SECTIONS if s in active_sections
    )
    system_content = _VIDEO_SYSTEM_PROMPT_TEMPLATE.format(
        positive_prompt=positive_prompt,
        sections_text=sections_text,
    )
    user_content: list[dict] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({
        "type": "text",
        "text": f"追加指示: {extra_instruction}\n\n動画プロンプトを生成してください。",
    })
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    try:
        text = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    streamer = TextIteratorStreamer(
        _processor, skip_prompt=True, skip_special_tokens=True
    )

    def _generate():
        with torch.inference_mode():
            _model.generate(**inputs, max_new_tokens=256, streamer=streamer)

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
