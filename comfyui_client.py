"""
comfyui_client.py
ComfyUI API クライアント（REST + WebSocket）

ワークフロー JSON テンプレートをロードし、プロンプト・パラメータを差し替えて
/prompt エンドポイントに送信する。
"""

import json
import os
import random
import uuid
from io import BytesIO

import requests
from PIL import Image

COMFYUI_URL = "http://127.0.0.1:8188"

# A1111 サンプラー名 → ComfyUI サンプラー名 のマッピング
_SAMPLER_MAP = {
    "Euler": "euler",
    "Euler a": "euler_ancestral",
    "Euler CFG++": "euler_cfg_pp",
    "Euler a CFG++": "euler_ancestral_cfg_pp",
    "LMS": "lms",
    "Heun": "heun",
    "DPM2": "dpm_2",
    "DPM2 a": "dpm_2_ancestral",
    "DPM++ 2S a": "dpmpp_2s_ancestral",
    "DPM++ SDE": "dpmpp_sde",
    "DPM++ 2M": "dpmpp_2m",
    "DPM++ 2M SDE": "dpmpp_2m_sde",
    "DPM++ 3M SDE": "dpmpp_3m_sde",
    "DDIM": "ddim",
    "UniPC": "uni_pc",
    "LCM": "lcm",
}

# workflows/ ディレクトリを自動スキャンして {ラベル: ファイルパス} を生成
_WORKFLOWS_DIR = os.path.join(os.path.dirname(__file__), "workflows")


def _scan_workflows() -> dict[str, str]:
    if not os.path.isdir(_WORKFLOWS_DIR):
        return {}
    result = {}
    for fname in sorted(os.listdir(_WORKFLOWS_DIR)):
        if fname.endswith(".json"):
            label = os.path.splitext(fname)[0]
            result[label] = os.path.join(_WORKFLOWS_DIR, fname)
    return result


WORKFLOW_PRESETS: dict[str, str] = _scan_workflows()


def _get_url() -> str:
    return COMFYUI_URL


def check_connection() -> tuple[bool, str]:
    """ComfyUI の /system_stats に GET して疎通確認。"""
    try:
        resp = requests.get(f"{_get_url()}/system_stats", timeout=5)
        resp.raise_for_status()
        return True, f"ComfyUI 接続OK ({_get_url()})"
    except Exception as e:
        return False, f"ComfyUI に接続できません: {e}"


def get_samplers() -> list[str]:
    """よく使われるサンプラー名の固定リスト（ComfyUI 標準）。"""
    return [
        "euler",
        "euler_cfg_pp",
        "euler_ancestral",
        "euler_ancestral_cfg_pp",
        "dpm_2",
        "dpm_2_ancestral",
        "dpmpp_2s_ancestral",
        "dpmpp_sde",
        "dpmpp_2m",
        "dpmpp_2m_sde",
        "dpmpp_3m_sde",
        "ddim",
        "uni_pc",
        "lcm",
    ]


def reload_workflows():
    """workflows/ ディレクトリを再スキャンして WORKFLOW_PRESETS を更新する。"""
    global WORKFLOW_PRESETS
    WORKFLOW_PRESETS = _scan_workflows()


_LATENT_IMAGE_NODES = (
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyLatentImageSD3",
    "EmptyHunyuanLatentVideo",
)


def _patch_workflow(
    workflow: dict,
    positive: str,
    negative: str,
    seed: int = -1,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """
    ワークフロー JSON のプロンプト・seed・サイズを差し替える。

    CLIPTextEncode:
      - タイトルに "negative"/"ネガティブ"/"neg" → negative
      - それ以外 → positive
    KSampler / KSamplerAdvanced:
      - seed != -1 なら上書き（-1 はワークフロー側の値をそのまま使う）
    EmptyLatentImage 系:
      - width / height が None でなければ上書き
    """
    import copy
    patched = copy.deepcopy(workflow)

    actual_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed

    _neg_keywords = ("negative", "ネガティブ", "neg")
    for node in patched.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type == "CLIPTextEncode":
            title = (node.get("_meta", {}).get("title", "") or "").lower()
            if any(kw in title for kw in _neg_keywords):
                inputs["text"] = negative
            else:
                inputs["text"] = positive

        elif class_type in ("KSampler", "KSamplerAdvanced"):
            if "seed" in inputs:
                inputs["seed"] = actual_seed
            if "noise_seed" in inputs:
                inputs["noise_seed"] = actual_seed

        elif class_type in _LATENT_IMAGE_NODES:
            if width is not None:
                inputs["width"] = width
            if height is not None:
                inputs["height"] = height

    return patched


def generate_image(
    workflow_path: str,
    positive: str,
    negative: str,
    seed: int = -1,
    width: int | None = None,
    height: int | None = None,
) -> Image.Image:
    """
    ワークフロー JSON のプロンプト・seed・サイズを差し替えて ComfyUI で生成し、PIL.Image を返す。
    seed=-1 はランダム、width/height=None はワークフロー側の値をそのまま使う。
    """
    if not workflow_path or not os.path.isfile(workflow_path):
        raise FileNotFoundError(f"ワークフローファイルが見つかりません: {workflow_path}")

    with open(workflow_path, encoding="utf-8") as f:
        workflow = json.load(f)

    patched = _patch_workflow(workflow, positive, negative, seed=seed, width=width, height=height)

    client_id = str(uuid.uuid4())
    base_url = _get_url()

    # ジョブを投入
    resp = requests.post(
        f"{base_url}/prompt",
        json={"prompt": patched, "client_id": client_id},
        timeout=30,
    )
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"ComfyUI /prompt エラー {resp.status_code}: {detail}")
    prompt_id = resp.json()["prompt_id"]

    # /history をポーリングして完了を待機（最大300秒）
    import time
    outputs = {}
    for _ in range(600):
        history = requests.get(f"{base_url}/history/{prompt_id}", timeout=10).json()
        entry = history.get(prompt_id, {})
        # status.status_str が "error" ならエラーとして扱う
        status_str = entry.get("status", {}).get("status_str", "")
        if status_str == "error":
            raise RuntimeError(f"ComfyUI 実行エラー（history より）: {entry.get('status')}")
        outputs = entry.get("outputs", {})
        if outputs:
            break
        time.sleep(0.5)

    image_info = None
    for node_output in outputs.values():
        images = node_output.get("images", [])
        if images:
            image_info = images[0]
            break

    if image_info is None:
        raise RuntimeError(
            f"ComfyUI から画像出力が得られませんでした。"
            f" outputs のキー: {list(outputs.keys()) or '空'}"
            f"（ワークフローに SaveImage または PreviewImage ノードがあるか確認してください）"
        )

    params = {
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output"),
    }
    img_resp = requests.get(f"{base_url}/view", params=params, timeout=60)
    img_resp.raise_for_status()

    return Image.open(BytesIO(img_resp.content)).convert("RGB")
