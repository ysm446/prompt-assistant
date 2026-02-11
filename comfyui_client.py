"""
comfyui_client.py
ComfyUI API クライアント（REST + WebSocket）

ワークフロー JSON テンプレートをロードし、プロンプト・パラメータを差し替えて
/prompt エンドポイントに送信する。
"""

import json
import os
import random
import tempfile
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
_IMAGE_WORKFLOWS_DIR = os.path.join(_WORKFLOWS_DIR, "image")
_VIDEO_WORKFLOWS_DIR = os.path.join(_WORKFLOWS_DIR, "video")


def _scan_workflows(directory: str | None = None) -> dict[str, str]:
    """指定ディレクトリの .json ファイルをスキャンして {ラベル: パス} を返す。"""
    target = directory if directory is not None else _WORKFLOWS_DIR
    if not os.path.isdir(target):
        return {}
    result = {}
    for fname in sorted(os.listdir(target)):
        if fname.endswith(".json"):
            label = os.path.splitext(fname)[0]
            result[label] = os.path.join(target, fname)
    return result


# 全ワークフロー（後方互換）
WORKFLOW_PRESETS: dict[str, str] = {
    **_scan_workflows(_IMAGE_WORKFLOWS_DIR),
    **_scan_workflows(_VIDEO_WORKFLOWS_DIR),
    **_scan_workflows(),  # ルート直下の .json もフォールバックとして含める
}
# 画像ワークフロー専用
IMAGE_WORKFLOW_PRESETS: dict[str, str] = _scan_workflows(_IMAGE_WORKFLOWS_DIR) or _scan_workflows()
# 動画ワークフロー専用
VIDEO_WORKFLOW_PRESETS: dict[str, str] = _scan_workflows(_VIDEO_WORKFLOWS_DIR) or _scan_workflows()


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
    """workflows/ 以下を再スキャンして各 PRESETS を更新する。"""
    global WORKFLOW_PRESETS, IMAGE_WORKFLOW_PRESETS, VIDEO_WORKFLOW_PRESETS
    IMAGE_WORKFLOW_PRESETS = _scan_workflows(_IMAGE_WORKFLOWS_DIR) or _scan_workflows()
    VIDEO_WORKFLOW_PRESETS = _scan_workflows(_VIDEO_WORKFLOWS_DIR) or _scan_workflows()
    WORKFLOW_PRESETS = {
        **_scan_workflows(_IMAGE_WORKFLOWS_DIR),
        **_scan_workflows(_VIDEO_WORKFLOWS_DIR),
        **_scan_workflows(),
    }


def upload_image(pil_image: Image.Image, filename: str = "video_input.png") -> str:
    """PIL Image を ComfyUI の input ディレクトリにアップロードしてファイル名を返す。"""
    base_url = _get_url()
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post(
        f"{base_url}/upload/image",
        files={"image": (filename, buf, "image/png")},
        data={"overwrite": "true"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["name"]


def free_vram() -> str:
    """ComfyUI のモデルをアンロードして VRAM を解放する。"""
    try:
        resp = requests.post(
            f"{_get_url()}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
        resp.raise_for_status()
        return f"ComfyUI のモデルをアンロードしました。({_get_url()})"
    except Exception as e:
        return f"ComfyUI VRAM 解放エラー: {e}"


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
    input_image_name: str | None = None,
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

        elif class_type == "LoadImage" and input_image_name is not None:
            inputs["image"] = input_image_name

    return patched


def generate_image(
    workflow_path: str,
    positive: str,
    negative: str,
    seed: int = -1,
    width: int | None = None,
    height: int | None = None,
    input_image: Image.Image | None = None,
) -> "Image.Image | str":
    """
    ワークフロー JSON のプロンプト・seed・サイズを差し替えて ComfyUI で生成する。
    画像ワークフローは PIL.Image を返す。動画ワークフローはファイルパス (str) を返す。
    seed=-1 はランダム、width/height=None はワークフロー側の値をそのまま使う。
    input_image が指定された場合は ComfyUI にアップロードして LoadImage ノードに差し替える。
    """
    if not workflow_path or not os.path.isfile(workflow_path):
        raise FileNotFoundError(f"ワークフローファイルが見つかりません: {workflow_path}")

    with open(workflow_path, encoding="utf-8") as f:
        workflow = json.load(f)

    input_image_name = None
    if input_image is not None:
        input_image_name = upload_image(input_image, "video_input.png")

    patched = _patch_workflow(
        workflow, positive, negative,
        seed=seed, width=width, height=height,
        input_image_name=input_image_name,
    )

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

    # /history をポーリングして完了を待機（最大600秒）
    # status.completed == True になるまで待つ（動画生成では中間ノードが先に outputs に出るため）
    import time
    outputs = {}
    for _ in range(1200):
        history = requests.get(f"{base_url}/history/{prompt_id}", timeout=10).json()
        entry = history.get(prompt_id, {})
        status = entry.get("status", {})
        status_str = status.get("status_str", "")
        if status_str == "error":
            raise RuntimeError(f"ComfyUI 実行エラー（history より）: {status}")
        if status.get("completed", False):
            outputs = entry.get("outputs", {})
            break
        time.sleep(0.5)

    _VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov"}
    image_info = None
    video_info = None
    for node_output in outputs.values():
        # "videos" キーを最優先
        if not video_info:
            for v in node_output.get("videos", []):
                video_info = v
                break
        # "images" キーでも動画拡張子のファイルは video_info として扱う
        if not video_info:
            for img in node_output.get("images", []):
                ext = os.path.splitext(img.get("filename", ""))[1].lower()
                if ext in _VIDEO_EXTENSIONS:
                    video_info = img
                    break
        # 通常の画像
        if not image_info:
            for img in node_output.get("images", []):
                ext = os.path.splitext(img.get("filename", ""))[1].lower()
                if ext not in _VIDEO_EXTENSIONS:
                    image_info = img
                    break

    if image_info is None and video_info is None:
        raise RuntimeError(
            f"ComfyUI から出力が得られませんでした。"
            f" outputs のキー: {list(outputs.keys()) or '空'}"
            f"（ワークフローに SaveImage/PreviewImage または SaveVideo ノードがあるか確認してください）"
        )

    # 動画出力を優先
    if video_info is not None:
        params = {
            "filename": video_info["filename"],
            "subfolder": video_info.get("subfolder", ""),
            "type": video_info.get("type", "output"),
        }
        video_resp = requests.get(f"{base_url}/view", params=params, timeout=300)
        video_resp.raise_for_status()
        suffix = os.path.splitext(video_info["filename"])[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(video_resp.content)
        tmp.close()
        return tmp.name

    params = {
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output"),
    }
    img_resp = requests.get(f"{base_url}/view", params=params, timeout=60)
    img_resp.raise_for_status()
    return Image.open(BytesIO(img_resp.content)).convert("RGB")
