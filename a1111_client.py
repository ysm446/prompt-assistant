"""
a1111_client.py
Automatic1111 REST API クライアント
"""

import base64
from io import BytesIO

import requests
from PIL import Image

A1111_URL = "http://127.0.0.1:7861"
TIMEOUT = 300  # 生成は時間がかかるので長めに設定


def check_connection() -> tuple[bool, str]:
    """
    A1111 が起動しているか確認する。
    Returns: (ok, message)
    """
    try:
        resp = requests.get(f"{A1111_URL}/sdapi/v1/samplers", timeout=5)
        resp.raise_for_status()
        return True, "A1111 に接続しました。"
    except requests.exceptions.ConnectionError:
        return False, f"A1111 に接続できません。{A1111_URL} で起動しているか確認してください。"
    except Exception as e:
        return False, f"A1111 接続エラー: {e}"


def get_samplers() -> list[str]:
    """
    利用可能なサンプラー一覧を取得する。
    失敗時は空リストを返す。
    """
    try:
        resp = requests.get(f"{A1111_URL}/sdapi/v1/samplers", timeout=5)
        resp.raise_for_status()
        return [s["name"] for s in resp.json()]
    except Exception:
        return []


def generate_image(
    positive: str,
    negative: str,
    steps: int,
    cfg: float,
    sampler: str,
    width: int,
    height: int,
    seed: int,
) -> Image.Image:
    """
    txt2img でテキストから画像を生成する。
    生成された PIL.Image を返す。
    """
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
    resp = requests.post(
        f"{A1111_URL}/sdapi/v1/txt2img", json=payload, timeout=TIMEOUT
    )
    resp.raise_for_status()

    image_b64 = resp.json()["images"][0]
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    return image
