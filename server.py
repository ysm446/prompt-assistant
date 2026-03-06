"""
server.py
Electron アプリ向け FastAPI バックエンドサーバー

Gradio の代替として全 UI ロジックを REST API + SSE で提供する。
"""

import argparse
import asyncio
import base64
import functools
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import a1111_client
import comfyui_client
import qwen_client
import settings_manager
from prompt_parser import parse_prompt_update, read_a1111_metadata, read_comfyui_metadata

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE_DIR = Path(__file__).parent
_FRONTEND_DIR = _BASE_DIR / "frontend"

app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")


@app.middleware("http")
async def no_cache_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/frontend/"):
        response.headers["Cache-Control"] = "no-store"
    return response

# ---------------------------------------------------------------------------
# Global state (single-user desktop app)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "conversation_history": [],
    "current_image": None,        # PIL Image
    "current_image_stem": "",
    "current_image_path": "",     # 元ファイルの絶対パス
    "current_video_path": "",     # 直近の動画ファイルの絶対パス
}
_video_gen_id = 0

# Temp files: token -> file path (for serving generated videos)
_temp_files: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _get_next_seq(save_dir: str, exts: tuple) -> int:
    pattern = re.compile(r"^(\d{5})-")
    max_seq = 0
    try:
        for name in os.listdir(save_dir):
            if not any(name.lower().endswith(e) for e in exts):
                continue
            m = pattern.match(name)
            if m:
                max_seq = max(max_seq, int(m.group(1)))
    except Exception:
        return 1
    return max_seq + 1


def _build_pnginfo(image: Image.Image) -> PngInfo | None:
    info = getattr(image, "info", {}) or {}
    pnginfo = PngInfo()
    added = False
    for k, v in info.items():
        if v is None:
            continue
        if isinstance(v, str):
            pnginfo.add_text(str(k), v)
            added = True
        elif isinstance(v, (dict, list)):
            try:
                pnginfo.add_text(str(k), json.dumps(v, ensure_ascii=False))
                added = True
            except Exception:
                pass
    return pnginfo if added else None


async def _run_in_thread(func, *args, **kwargs):
    """ブロッキング関数をスレッドプールで実行する。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Routes - Static
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/api/file/{token}")
def serve_file(token: str):
    path = _temp_files.get(token)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Routes - Settings
# ---------------------------------------------------------------------------

@app.get("/api/settings")
def get_settings():
    return settings_manager.load()


@app.post("/api/settings")
async def save_settings(request: Request):
    body = await request.json()
    settings_manager.save(body)
    if "comfyui_url" in body:
        comfyui_client.COMFYUI_URL = body["comfyui_url"]
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes - Model management
# ---------------------------------------------------------------------------

@app.get("/api/model_presets")
def get_model_presets():
    return {"presets": list(qwen_client.MODEL_PRESETS.keys())}


@app.post("/api/load_model")
async def load_model(request: Request):
    body = await request.json()
    model_label = body.get("model_label", "")
    model_id = qwen_client.MODEL_PRESETS.get(model_label, model_label)
    try:
        msg = await _run_in_thread(qwen_client.load_model, model_id)
        return {"ok": True, "message": msg}
    except RuntimeError as e:
        return {"ok": False, "message": str(e)}


@app.post("/api/unload_qwen")
def unload_qwen():
    return {"ok": True, "message": qwen_client.unload_model()}


@app.post("/api/free_comfyui")
def free_comfyui():
    return {"ok": True, "message": comfyui_client.free_vram()}


# ---------------------------------------------------------------------------
# Routes - Backend / workflow discovery
# ---------------------------------------------------------------------------

@app.get("/api/samplers")
def get_samplers():
    return {"samplers": a1111_client.get_samplers()}


@app.get("/api/workflows")
def get_workflows():
    comfyui_client.reload_workflows()
    return {"workflows": list(comfyui_client.IMAGE_WORKFLOW_PRESETS.keys())}


@app.get("/api/video_workflows")
def get_video_workflows():
    comfyui_client.reload_workflows()
    return {"workflows": list(comfyui_client.VIDEO_WORKFLOW_PRESETS.keys())}


@app.post("/api/open_path")
async def open_path(request: Request):
    body = await request.json()
    target = (body.get("path", "") or "").strip()
    if not target:
        return {"ok": False, "message": "パスが指定されていません"}
    # ファイルパスの場合は親フォルダを開く
    p = Path(target)
    folder = str(p.parent) if p.is_file() else str(p)
    os.makedirs(folder, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(folder)
        else:
            import subprocess
            subprocess.Popen(["xdg-open", folder])
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "message": str(e)}


@app.post("/api/open_workflow_folder")
async def open_workflow_folder(request: Request):
    body = await request.json()
    kind = body.get("kind", "image")
    if kind == "video":
        folder = comfyui_client._VIDEO_WORKFLOWS_DIR
    else:
        folder = comfyui_client._IMAGE_WORKFLOWS_DIR
    os.makedirs(folder, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(folder)
        else:
            import subprocess
            subprocess.Popen(["xdg-open", folder])
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "message": str(e)}


# ---------------------------------------------------------------------------
# Routes - Image
# ---------------------------------------------------------------------------

@app.post("/api/image/upload")
async def upload_image(file: UploadFile = File(...), image_path: str = Form("")):
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).copy()
    _state["current_image"] = image
    _state["current_image_path"] = image_path
    b64 = _image_to_b64(image)
    meta = read_a1111_metadata(image) or read_comfyui_metadata(image)
    if meta is None:
        return {
            "image": b64,
            "status": "画像を読み込みました。メタデータが見つかりません。",
            "meta": None,
        }
    source = "A1111" if image.info.get("parameters") else "ComfyUI"
    return {
        "image": b64,
        "status": f"メタデータを読み込みました。({source})",
        "meta": {
            "positive": meta.get("positive", ""),
            "negative": meta.get("negative", ""),
            "steps": meta.get("steps"),
            "cfg_scale": meta.get("cfg_scale"),
            "sampler": meta.get("sampler"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "seed": meta.get("seed"),
        },
    }


@app.post("/api/seed_from_image")
def seed_from_image():
    image = _state.get("current_image")
    if image is None:
        return {"ok": False, "message": "現在の画像がありません"}
    meta = read_a1111_metadata(image) or read_comfyui_metadata(image)
    if meta is None or "seed" not in meta:
        return {"ok": False, "message": "メタデータに Seed がありません"}
    seed = int(meta["seed"])
    return {"ok": True, "seed": seed, "message": f"Seed を反映しました: {seed}"}


@app.post("/api/seed_from_video")
def seed_from_video():
    video_path = _state.get("current_video_path", "")
    if not video_path:
        return {"ok": False, "message": "現在の動画がありません"}
    # ファイル名から seed を抽出（例: 00001-1234567890.mp4）
    stem = os.path.splitext(os.path.basename(video_path))[0]
    parts = stem.split("-", 1)
    if len(parts) == 2 and parts[1].lstrip("-").isdigit():
        seed = int(parts[1])
        return {"ok": True, "seed": seed, "message": f"Seed を反映しました: {seed}"}
    return {"ok": False, "message": "ファイル名から Seed を読み取れませんでした"}


@app.post("/api/generate_image/stream")
async def generate_image_stream(request: Request):
    body = await request.json()
    positive = body.get("positive", "")
    negative = body.get("negative", "")
    steps = int(body.get("steps", 28))
    cfg = float(body.get("cfg", 7.0))
    sampler = body.get("sampler", "Euler a")
    width = int(body.get("width", 512))
    height = int(body.get("height", 768))
    seed = int(body.get("seed", -1))
    backend = body.get("backend", "WebUI Forge")
    comfyui_workflow = body.get("comfyui_workflow", "")
    comfyui_width = int(body.get("comfyui_width", 1024))
    comfyui_height = int(body.get("comfyui_height", 1024))
    comfyui_seed = int(body.get("comfyui_seed", -1))
    save_dir = (body.get("image_save_path", "") or "").strip() or "./outputs/images"

    _state["positive_prompt"] = positive
    _state["negative_prompt"] = negative

    async def event_gen():
        seed_i = seed
        cseed_i = comfyui_seed
        yield _sse({"type": "status", "content": "生成中..."})
        try:
            start = time.time()
            if backend == "ComfyUI":
                wf_path = comfyui_client.IMAGE_WORKFLOW_PRESETS.get(comfyui_workflow, comfyui_workflow)
                image = await _run_in_thread(
                    comfyui_client.generate_image,
                    workflow_path=wf_path,
                    positive=positive,
                    negative=negative,
                    seed=cseed_i,
                    width=comfyui_width,
                    height=comfyui_height,
                )
            else:
                image = await _run_in_thread(
                    a1111_client.generate_image,
                    positive=positive,
                    negative=negative,
                    steps=steps,
                    cfg=cfg,
                    sampler=sampler,
                    width=width,
                    height=height,
                    seed=seed_i,
                )
            elapsed = time.time() - start

            if not isinstance(image, Image.Image):
                yield _sse({"type": "error", "content": "画像の取得に失敗しました"})
            else:
                _state["current_image"] = image
                _state["current_image_stem"] = (
                    comfyui_client.get_last_output_filename()
                    if backend == "ComfyUI"
                    else time.strftime("forge_%Y%m%d_%H%M%S")
                )

                saved_path = ""
                save_status = ""
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    meta = read_a1111_metadata(image) or read_comfyui_metadata(image) or {}
                    sv = int(meta.get("seed", cseed_i if backend == "ComfyUI" else seed_i) or 0)
                    fname = f"{_get_next_seq(save_dir, ('.png',)):05d}-{sv}.png"
                    sp = os.path.join(save_dir, fname)
                    sp = os.path.abspath(sp)
                    pnginfo = _build_pnginfo(image)
                    image.save(sp, pnginfo=pnginfo) if pnginfo else image.save(sp)
                    saved_path = sp
                    _state["current_image_path"] = sp
                except Exception as se:
                    save_status = f" / 保存失敗: {se}"

                yield _sse({
                    "type": "image",
                    "image": _image_to_b64(image),
                    "saved_path": saved_path,
                    "status": f"画像を生成しました。（{elapsed:.1f}秒）{save_status}",
                })
        except Exception as e:
            yield _sse({"type": "error", "content": f"画像生成エラー: {e}"})

        yield _sse({"type": "done"})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/interrupt_image")
def interrupt_image():
    comfyui_client.interrupt()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes - Chat
# ---------------------------------------------------------------------------

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    user_input = body.get("user_input", "").strip()
    model_label = body.get("model_label", "")
    positive = body.get("positive", "")
    negative = body.get("negative", "")

    if not user_input:
        async def empty():
            yield _sse({"type": "done"})
        return StreamingResponse(empty(), media_type="text/event-stream")

    _state["positive_prompt"] = positive
    _state["negative_prompt"] = negative

    loop = asyncio.get_running_loop()

    async def event_gen():
        queue: asyncio.Queue = asyncio.Queue()

        def run():
            try:
                if not qwen_client.is_loaded():
                    model_id = qwen_client.MODEL_PRESETS.get(model_label, model_label)
                    try:
                        msg = qwen_client.load_model(model_id)
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"type": "model_loaded", "message": msg}), loop
                        )
                    except RuntimeError as e:
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"type": "error", "content": f"モデルロード失敗: {e}"}), loop
                        )
                        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                        return

                partial = ""
                for token in qwen_client.query_stream(
                    conversation_history=_state["conversation_history"],
                    user_input=user_input,
                    current_image=_state.get("current_image"),
                    positive_prompt=positive,
                    negative_prompt=negative,
                ):
                    partial += token
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"type": "token", "content": token}), loop
                    )

                positive_new, negative_new, display_text = parse_prompt_update(partial)

                _state["conversation_history"].append({
                    "role": "user",
                    "content": [{"type": "text", "text": user_input}],
                })
                _state["conversation_history"].append({
                    "role": "assistant",
                    "content": partial,
                })

                asyncio.run_coroutine_threadsafe(
                    queue.put({
                        "type": "done",
                        "positive": positive_new,
                        "negative": negative_new,
                        "display_text": display_text,
                    }),
                    loop,
                )
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error", "content": str(e)}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/chat/clear")
def chat_clear():
    _state["conversation_history"] = []
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes - Video prompt
# ---------------------------------------------------------------------------

@app.post("/api/video_prompt/stream")
async def video_prompt_stream(request: Request):
    body = await request.json()
    positive = body.get("positive", "")
    extra_instruction = body.get("extra_instruction", "")
    sections = body.get("sections", [])
    model_label = body.get("model_label", "")

    if not sections:
        async def err():
            yield _sse({"type": "error", "content": "セクションを1つ以上選択してください"})
        return StreamingResponse(err(), media_type="text/event-stream")

    loop = asyncio.get_running_loop()

    async def event_gen():
        queue: asyncio.Queue = asyncio.Queue()

        def run():
            try:
                if not qwen_client.is_loaded():
                    model_id = qwen_client.MODEL_PRESETS.get(model_label, model_label)
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"type": "status", "content": "モデルをロード中..."}), loop
                    )
                    try:
                        qwen_client.load_model(model_id)
                    except RuntimeError as e:
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"type": "error", "content": f"モデルロード失敗: {e}"}), loop
                        )
                        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                        return

                for token in qwen_client.generate_video_prompt_stream(
                    image=_state.get("current_image"),
                    positive_prompt=positive,
                    extra_instruction=extra_instruction,
                    sections=sections,
                ):
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"type": "token", "content": token}), loop
                    )

                asyncio.run_coroutine_threadsafe(queue.put({"type": "done"}), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error", "content": str(e)}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Routes - Video generation
# ---------------------------------------------------------------------------

@app.post("/api/stop_video")
def stop_video():
    global _video_gen_id
    _video_gen_id += 1
    comfyui_client.interrupt()
    return {"ok": True, "message": "動画生成をキャンセルしました"}


@app.post("/api/generate_video/stream")
async def generate_video_stream(request: Request):
    global _video_gen_id
    _video_gen_id += 1
    my_id = _video_gen_id

    body = await request.json()
    video_prompt_text = body.get("video_prompt", "")
    workflow_name = body.get("workflow", "")
    seed = int(body.get("seed", -1))
    width = int(body.get("width", 848)) if body.get("width") else None
    height = int(body.get("height", 480)) if body.get("height") else None
    save_dir = (body.get("video_save_path", "") or "").strip() or "./outputs/videos"
    unload_llm = bool(body.get("unload_llm_before_video", False))

    if _state.get("current_image") is None:
        async def no_img():
            yield _sse({"type": "error", "content": "画像がありません。画像エリアに画像をセットしてください。"})
        return StreamingResponse(no_img(), media_type="text/event-stream")

    if unload_llm:
        qwen_client.unload_model()

    result_box: list = []
    error_box: list = []

    def _run():
        if my_id != _video_gen_id:
            return
        try:
            wf_path = comfyui_client.VIDEO_WORKFLOW_PRESETS.get(workflow_name, workflow_name)
            result_box.append(comfyui_client.generate_image(
                workflow_path=wf_path,
                positive=video_prompt_text,
                negative="",
                seed=seed,
                width=width,
                height=height,
                input_image=_state["current_image"],
            ))
        except Exception as e:
            error_box.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    start = time.time()

    async def event_gen():
        while t.is_alive():
            if my_id != _video_gen_id:
                return
            yield _sse({
                "type": "status",
                "content": f"動画生成中... {time.time() - start:.0f}秒",
            })
            await asyncio.sleep(1)

        if my_id != _video_gen_id:
            return

        elapsed = time.time() - start
        if error_box:
            yield _sse({"type": "error", "content": f"動画生成エラー: {error_box[0]}"})
        elif result_box:
            result = result_box[0]
            if isinstance(result, str):
                saved_path = ""
                save_status = ""
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    ext = os.path.splitext(result)[1].lower() or ".mp4"
                    if ext not in {".mp4", ".webm", ".avi", ".mov"}:
                        ext = ".mp4"
                    used_seed = comfyui_client.get_last_actual_seed()
                    fname = (
                        f"{_get_next_seq(save_dir, ('.mp4', '.webm', '.avi', '.mov')):05d}"
                        f"-{int(used_seed)}{ext}"
                    )
                    sp = os.path.join(save_dir, fname)
                    sp = os.path.abspath(sp)
                    shutil.copy2(result, sp)
                    saved_path = sp
                    _state["current_video_path"] = sp
                except Exception as se:
                    save_status = f" / 保存失敗: {se}"
                token = str(uuid.uuid4())
                _temp_files[token] = result
                yield _sse({
                    "type": "video",
                    "url": f"/api/file/{token}",
                    "saved_path": saved_path,
                    "status": f"動画生成完了（{elapsed:.0f}秒）{save_status}",
                })
            else:
                yield _sse({"type": "error", "content": "動画ではなく画像として出力されました"})
        else:
            yield _sse({"type": "error", "content": "結果が取得できませんでした"})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Routes - Shutdown (Electron からの終了通知)
# ---------------------------------------------------------------------------

@app.post("/api/shutdown")
def shutdown():
    """Electron ウィンドウが閉じたときにサーバーを終了する。"""
    def _exit():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes - Save JSON
# ---------------------------------------------------------------------------

@app.post("/api/save_json")
async def save_json_endpoint(request: Request):
    body = await request.json()
    video_prompt = body.get("video_prompt", "")
    additional_instruction = body.get("additional_instruction", "")

    image_path = _state.get("current_image_path", "")
    if not image_path:
        return {"ok": False, "message": "画像が読み込まれていません"}

    image = _state.get("current_image")
    meta_raw = {}
    if image:
        meta_raw = read_a1111_metadata(image) or read_comfyui_metadata(image) or {}

    p = Path(image_path)
    size_list = list(image.size) if image else []
    _exclude = {"positive", "negative", "width", "height"}
    settings = {k: v for k, v in meta_raw.items() if k not in _exclude}

    data = {
        "image_filename": p.name,
        "image_path": str(p),
        "metadata": {
            "path": str(p),
            "filename": p.name,
            "size": size_list,
            "prompt": meta_raw.get("positive", ""),
            "negative_prompt": meta_raw.get("negative", ""),
            "settings": settings,
        },
        "prompt": video_prompt,
        "additional_instruction": additional_instruction,
    }

    json_path = p.parent / (p.stem + ".json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"ok": True, "message": f"保存しました: {json_path}"}
    except Exception as e:
        return {"ok": False, "message": f"保存エラー: {e}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # ComfyUI 設定を初期化
    settings = settings_manager.load()
    comfyui_client.COMFYUI_URL = settings.get("comfyui_url", "http://127.0.0.1:8188")
    comfyui_client.reload_workflows()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
