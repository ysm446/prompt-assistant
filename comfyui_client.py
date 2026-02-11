"""
comfyui_client.py
ComfyUI API クライアント（REST + WebSocket）

ワークフロー JSON テンプレートをロードし、プロンプト・パラメータを差し替えて
/prompt エンドポイントに送信する。
"""

import json
import os
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


def _patch_workflow(workflow: dict, positive: str, negative: str) -> dict:
    """
    ワークフロー JSON のプロンプトテキストだけを差し替える。
    steps / cfg / sampler / size はワークフロー側の値をそのまま使う。

    CLIPTextEncode:
      - メタタイトルに "negative"/"ネガティブ"/"neg" が含まれる → negative
      - それ以外 → positive
    """
    import copy
    patched = copy.deepcopy(workflow)

    _neg_keywords = ("negative", "ネガティブ", "neg")
    for node in patched.values():
        if not isinstance(node, dict) or node.get("class_type") != "CLIPTextEncode":
            continue
        title = (node.get("_meta", {}).get("title", "") or "").lower()
        if any(kw in title for kw in _neg_keywords):
            node["inputs"]["text"] = negative
        else:
            node["inputs"]["text"] = positive

    return patched


def generate_image(
    workflow_path: str,
    positive: str,
    negative: str,
) -> Image.Image:
    """
    ワークフロー JSON のプロンプトだけを差し替えて ComfyUI で生成し、PIL.Image を返す。
    steps / cfg / sampler / size はワークフロー側の設定を使用する。
    """
    if not workflow_path or not os.path.isfile(workflow_path):
        raise FileNotFoundError(f"ワークフローファイルが見つかりません: {workflow_path}")

    with open(workflow_path, encoding="utf-8") as f:
        workflow = json.load(f)

    patched = _patch_workflow(workflow, positive, negative)

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
