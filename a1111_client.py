"""
a1111_client.py
SD WebUI Forge 2 gradio_client ベースクライアント。

Forge 2 は /sdapi/v1/txt2img REST API を廃止し Gradio API のみ提供。
パラメータ構成は discover_forge_api.py で確認済み（152 パラメータ）。
"""

import os

import requests
from PIL import Image
from gradio_client import Client

_FORGE_HOST = "http://127.0.0.1"
_FORGE_PORT_START = 7860
_FORGE_PORT_END = 7880  # 7860〜7879 を探索
_FORGE_PROBE_TIMEOUT = 0.25

# Forge 2 は /sdapi/v1/samplers を持たないため固定リストを使用
FALLBACK_SAMPLERS = [
    "DPM++ 2M", "DPM++ 2M SDE", "DPM++ SDE",
    "DPM++ 3M SDE", "Euler a", "Euler",
    "DDIM", "PLMS", "UniPC", "LCM",
]

# 接続確認済みの URL（None = 未接続）
FORGE_URL: str | None = None

# Gradio クライアントをキャッシュ（接続は一度だけ確立する）
_client: Client | None = None


def _is_forge_gradio(url: str) -> bool:
    """候補 URL が Forge 2 の Gradio API（/txt2img）を持つか判定する。"""
    try:
        client = Client(url, verbose=False)
    except Exception:
        return False

    try:
        config = getattr(client, "config", None)
        if isinstance(config, dict):
            for dep in config.get("dependencies", []) or []:
                if dep.get("api_name") in ("/txt2img", "txt2img"):
                    return True
    except Exception:
        pass

    try:
        api_info = client.view_api(return_format="dict")
        if isinstance(api_info, dict):
            named = api_info.get("named_endpoints", {})
            if isinstance(named, dict) and "/txt2img" in named:
                return True
    except Exception:
        pass

    return False


def _find_forge_url() -> str:
    """ポート 7860〜7879 を順番に試し、最初に応答した URL を返す。"""
    for port in range(_FORGE_PORT_START, _FORGE_PORT_END):
        url = f"{_FORGE_HOST}:{port}"
        try:
            requests.get(url, timeout=_FORGE_PROBE_TIMEOUT)
            if _is_forge_gradio(url):
                return url
        except Exception:
            continue
    raise RuntimeError(
        f"WebUI Forge に接続できません。"
        f"ポート {_FORGE_PORT_START}〜{_FORGE_PORT_END - 1} を確認しましたが応答がありませんでした。"
    )


def _get_client() -> Client:
    global _client, FORGE_URL
    if _client is None:
        FORGE_URL = _find_forge_url()
        _client = Client(FORGE_URL, verbose=False)
    return _client


def _reset_client() -> None:
    global _client, FORGE_URL
    _client = None
    FORGE_URL = None


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def check_connection() -> tuple[bool, str]:
    """Forge 2 が起動しているか確認する。"""
    try:
        _reset_client()
        _get_client()
        return True, f"SD WebUI Forge に接続しました。({FORGE_URL})"
    except Exception as e:
        _reset_client()
        return False, (
            f"SD WebUI Forge に接続できません。"
            f"ポート {_FORGE_PORT_START}〜{_FORGE_PORT_END - 1} で起動しているか確認してください。({e})"
        )


def get_samplers() -> list[str]:
    """利用可能なサンプラー一覧を返す（Forge 2 は固定リスト）。"""
    return FALLBACK_SAMPLERS



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
    Forge 2 Gradio API (/txt2img) で画像を生成する。

    パラメータ順は discover_forge_api.py で確認済みの 152 引数構成に従う。
    """
    client = _get_client()

    # fmt: off
    args = [
        # [00] parameter_47 (不明、おそらく内部タブ ID か override_settings)
        "",
        # [01] Prompt
        positive,
        # [02] Negative prompt
        negative,
        # [03] Styles
        [],
        # [04] Batch count
        1,
        # [05] Batch size
        1,
        # [06] CFG Scale
        cfg,
        # [07] Distilled CFG Scale (Forge 2 独自。通常モデルは影響なし)
        3.5,
        # [08] Height
        int(height),
        # [09] Width
        int(width),
        # [10] Hires. fix
        False,
        # [11] Denoising strength
        0.7,
        # [12] Upscale by
        2.0,
        # [13] Upscaler
        "Latent",
        # [14] Hires steps
        0.0,
        # [15] Resize width to
        0.0,
        # [16] Resize height to
        0.0,
        # [17] Hires Checkpoint
        "Use same checkpoint",
        # [18] Hires VAE / Text Encoder
        [],
        # [19] Hires sampling method
        "Use same sampler",
        # [20] Hires schedule type
        "Use same scheduler",
        # [21] Hires prompt
        "",
        # [22] Hires negative prompt
        "",
        # [23] Hires CFG Scale
        0.0,
        # [24] Hires Distilled CFG Scale
        0.0,
        # [25] Override settings
        [],
        # [26] Script
        "None",
        # [27] Sampling steps (int で渡す必要がある。float だと torch.linspace がエラー)
        int(steps),
        # [28] Sampling method
        sampler,
        # [29] Schedule type
        "Automatic",
        # [30] Refiner
        False,
        # [31] Checkpoint
        "",
        # [32] Switch at
        0.8,
        # [33] Seed
        float(seed),
        # [34] Extra (extra seed)
        False,
        # [35] Variation seed
        -1.0,
        # [36] Variation strength
        0.0,
        # [37] Resize seed from width
        0.0,
        # [38] Resize seed from height
        0.0,
        # ---- Dynamic Prompts extension [39-56] ----
        # [39] Dynamic Prompts enabled
        False,
        # [40] Combinatorial generation
        False,
        # [41] Combinatorial batches
        1.0,
        # [42] Magic prompt
        False,
        # [43] I'm feeling lucky
        False,
        # [44] Attention grabber
        False,
        # [45] Minimum attention
        1.1,
        # [46] Maximum attention
        1.5,
        # [47] Max magic prompt length
        100.0,
        # [48] Magic prompt creativity
        0.7,
        # [49] Fixed seed
        False,
        # [50] Unlink seed from prompt
        False,
        # [51] Don't apply to negative prompts
        False,
        # [52] Enable Jinja2 templates
        False,
        # [53] Don't generate images
        False,
        # [54] Max generations
        0.0,
        # [55] Magic prompt model
        "Gustavosta/MagicPrompt-Stable-Diffusion",
        # [56] Magic prompt blocklist regex
        "",
        # ---- CFG Combinator [57-68] ----
        # [57] Enabled
        False,
        # [58] Mimic Scale
        7.0,
        # [59] Threshold Percentile
        1.0,
        # [60] Mimic Mode
        "Constant",
        # [61] Mimic Scale Min
        0.0,
        # [62] Cfg Mode
        "Constant",
        # [63] Cfg Scale Min
        0.0,
        # [64] Sched Val
        4.0,
        # [65] Separate Feature Channels
        "enable",
        # [66] Scaling Startpoint
        "MEAN",
        # [67] Variability Measure
        "AD",
        # [68] Interpolate Phi
        1.0,
        # ---- FreeU [69-75] ----
        # [69] FreeU Integrated
        False,
        # [70] B1
        1.3,
        # [71] B2
        1.4,
        # [72] S1
        0.9,
        # [73] S2
        0.2,
        # [74] Start step
        0.0,
        # [75] End step
        1.0,
        # ---- SelfAttentionGuidance [76-79] ----
        # [76] SAG Integrated
        False,
        # [77] Scale
        0.75,
        # [78] Blur Sigma
        2.0,
        # [79] Blur mask threshold
        0.0,
        # ---- PerturbedAttentionGuidance [80-84] ----
        # [80] PAG Integrated
        False,
        # [81] Scale
        3.0,
        # [82] Attenuation
        0.0,
        # [83] Start step
        0.0,
        # [84] End step
        1.0,
        # ---- Kohya HRFix [85-92] ----
        # [85] Kohya HRFix Integrated
        False,
        # [86] Block Number
        3.0,
        # [87] Downscale Factor
        2.0,
        # [88] Start Percent
        0.0,
        # [89] End Percent
        0.35,
        # [90] Downscale After Skip
        True,
        # [91] Downscale Method
        "bicubic",
        # [92] Upscale Method
        "bicubic",
        # ---- Sharpness / Tonemap / CFG Combat [93-113] ----
        # [93] Enabled
        False,
        # [94] Sharpness Multiplier
        2.0,
        # [95] Sharpness Method
        "anisotropic",
        # [96] Tonemap Multiplier
        1.0,
        # [97] Tonemap Method
        "reinhard",
        # [98] Tonemap Percentile
        100.0,
        # [99] Contrast Multiplier
        0.0,
        # [100] Combat Method
        "subtract",
        # [101] Combat Cfg Drift
        0.0,
        # [102] Rescale Cfg Phi
        0.0,
        # [103] Extra Noise Type
        "gaussian",
        # [104] Extra Noise Method
        "add",
        # [105] Extra Noise Multiplier
        0.0,
        # [106] Extra Noise Lowpass
        100.0,
        # [107] Divisive Norm Size
        127.0,
        # [108] Divisive Norm Multiplier
        0.0,
        # [109] Spectral Mod Mode
        "hard_clamp",
        # [110] Spectral Mod Percentile
        5.0,
        # [111] Spectral Mod Multiplier
        0.0,
        # [112] Affect Uncond
        "None",
        # [113] Dyn Cfg Augmentation
        "None",
        # ---- MultiDiffusion [114-121] ----
        # [114] Enabled
        False,
        # [115] Method
        "MultiDiffusion",
        # [116] Tile Width
        96.0,
        # [117] Tile Height
        96.0,
        # [118] Tile Overlap
        48.0,
        # [119] Tile Batch Size
        4.0,
        # [120] Share attention in batch
        False,
        # [121] Strength
        1.0,
        # ---- Memory management [122-123] ----
        # [122] Enabled for UNet
        False,
        # [123] Enabled for VAE
        False,
        # ---- Script args (Script="None" のため無効) [124-151] ----
        # [124] Put variable parts at start of prompt (Prompt matrix)
        False,
        # [125] Use different seed for each picture
        False,
        # [126] Select prompt
        "positive",
        # [127] Select joining char
        "comma",
        # [128] Grid margins (px)
        0.0,
        # [129] Iterate seed every line (Prompts from file)
        False,
        # [130] Use same random seed for all lines
        False,
        # [131] Insert prompts at the
        "start",
        # [132] List of prompt inputs
        "",
        # [133] Make a combined image
        False,
        # [134] X type (X/Y/Z plot)
        "Nothing",
        # [135] X values (str)
        "",
        # [136] X values (list)
        [],
        # [137] Y type
        "Nothing",
        # [138] Y values (str)
        "",
        # [139] Y values (list)
        [],
        # [140] Z type
        "Nothing",
        # [141] Z values (str)
        "",
        # [142] Z values (list)
        [],
        # [143] Draw legend
        True,
        # [144] Include Sub Images
        False,
        # [145] Include Sub Grids
        False,
        # [146] Keep -1 for seeds
        False,
        # [147] Vary seeds for X
        False,
        # [148] Vary seeds for Y
        False,
        # [149] Vary seeds for Z
        False,
        # [150] Grid margins (px)
        0.0,
        # [151] Use text inputs instead of dropdowns
        False,
    ]
    # fmt: on

    try:
        result = client.predict(*args, api_name="/txt2img")
    except Exception:
        # クライアントが古い/切断済みの場合に一度だけ再接続して再試行
        _reset_client()
        client = _get_client()
        result = client.predict(*args, api_name="/txt2img")
    return _result_to_image(result)


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def _result_to_image(result) -> Image.Image:
    """
    gradio_client の predict 戻り値から PIL.Image を取得する。
    Forge 2 は (gallery_data, generation_info, html_info) のタプルを返す。
    """
    if isinstance(result, (list, tuple)):
        gallery = result[0] if result else None
    else:
        gallery = result

    if gallery is None:
        raise RuntimeError("Forge 2 から画像が返されませんでした。")

    if isinstance(gallery, (list, tuple)) and len(gallery) > 0:
        first = gallery[0]
    else:
        first = gallery

    return _load_image_from_entry(first)


def _load_image_from_entry(entry) -> Image.Image:
    """単一の画像エントリを PIL.Image に変換する。"""

    # gradio_client.utils.FileData オブジェクト（.path 属性）
    if hasattr(entry, "path") and entry.path:
        return Image.open(entry.path).copy()

    # dict 形式: {"image": ..., "name": ..., "path": ...}
    if isinstance(entry, dict):
        for key in ("image", "name", "path", "url"):
            val = entry.get(key)
            if val is not None:
                return _load_image_from_entry(val)

    # ファイルパス文字列
    if isinstance(entry, str) and os.path.isfile(entry):
        return Image.open(entry).copy()

    raise RuntimeError(
        f"画像データの形式を認識できませんでした: type={type(entry)}, value={entry!r}"
    )
