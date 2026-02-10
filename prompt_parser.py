"""
prompt_parser.py
- Qwen3-VL の返答から [PROMPT_UPDATE] タグをパース
- A1111 生成 PNG のメタデータからプロンプト・生成パラメータを読み取る
"""

import re
from PIL import Image


def parse_prompt_update(text: str) -> tuple[str | None, str | None, str]:
    """
    Qwen3-VL の返答から [PROMPT_UPDATE] ブロックを抽出する。

    Returns:
        (positive, negative, display_text)
        - positive / negative: 更新があれば文字列、なければ None
        - display_text: タグブロックを除いたユーザー表示用テキスト
    """
    pattern = re.compile(
        r"\[PROMPT_UPDATE\](.*?)\[/PROMPT_UPDATE\]", re.DOTALL
    )
    match = pattern.search(text)
    if not match:
        return None, None, text

    block = match.group(1).strip()
    display_text = pattern.sub("", text).strip()

    positive = None
    negative = None
    current_field = None
    current_lines: list[str] = []

    def _flush():
        nonlocal positive, negative
        value = "\n".join(current_lines).strip() or None
        if current_field == "positive":
            positive = value
        elif current_field == "negative":
            negative = value

    for line in block.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("positive:"):
            _flush()
            current_field = "positive"
            current_lines = [stripped[len("positive:"):].strip()]
        elif stripped.lower().startswith("negative:"):
            _flush()
            current_field = "negative"
            current_lines = [stripped[len("negative:"):].strip()]
        elif current_field is not None:
            current_lines.append(stripped)

    _flush()

    return positive, negative, display_text


def read_a1111_metadata(image: Image.Image) -> dict | None:
    """
    A1111 生成画像の PNG メタデータからプロンプト・生成パラメータを読み取る。
    メタデータがない場合は None を返す。
    """
    params_text = image.info.get("parameters", "")
    if not params_text:
        return None

    lines = params_text.strip().splitlines()

    # "Negative prompt:" 行と "Steps:" 行の位置を特定
    neg_idx = next(
        (i for i, l in enumerate(lines) if l.startswith("Negative prompt:")), None
    )
    param_idx = next(
        (i for i, l in enumerate(lines) if l.startswith("Steps:")), None
    )

    # Positive prompt: 先頭〜 Negative / Steps 行の手前
    pos_end = neg_idx if neg_idx is not None else (param_idx if param_idx is not None else len(lines))
    positive = "\n".join(lines[:pos_end]).strip()

    # Negative prompt
    negative = ""
    if neg_idx is not None:
        neg_end = param_idx if param_idx is not None else len(lines)
        neg_lines = lines[neg_idx:neg_end]
        neg_lines[0] = neg_lines[0].removeprefix("Negative prompt:").strip()
        negative = "\n".join(neg_lines).strip()

    result: dict = {"positive": positive, "negative": negative}

    # Steps 行以降をカンマ区切りでパースしてキー値取得
    if param_idx is not None:
        param_line = ", ".join(lines[param_idx:])
        for token in param_line.split(","):
            token = token.strip()
            if ":" in token:
                k, _, v = token.partition(":")
                key = k.strip().lower().replace(" ", "_")
                result[key] = v.strip()

    # Size を width / height に分解（例: "512x768"）
    if "size" in result:
        try:
            w, h = result["size"].lower().split("x")
            result["width"] = int(w)
            result["height"] = int(h)
        except ValueError:
            pass

    # 数値フィールドを適切な型に変換
    for key, typ in [("steps", int), ("cfg_scale", float), ("seed", int)]:
        if key in result:
            try:
                result[key] = typ(result[key])
            except (ValueError, TypeError):
                pass

    return result
