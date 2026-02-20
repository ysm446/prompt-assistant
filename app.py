"""
app.py
SD × Qwen3-VL 画像生成アシスタント - Gradio アプリ本体
"""

import os
import re
import shutil
import threading
import time
import json

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import a1111_client
import comfyui_client
import qwen_client
import settings_manager
from prompt_parser import parse_prompt_update, read_a1111_metadata, read_comfyui_metadata

DEFAULT_SAMPLER = "Euler a"


# ---------------------------------------------------------------------------
# イベントハンドラ
# ---------------------------------------------------------------------------

def on_load_model(model_label: str) -> str:
    model_id = qwen_client.MODEL_PRESETS.get(model_label, model_label)
    try:
        return qwen_client.load_model(model_id)
    except RuntimeError as e:
        return str(e)


def on_image_drop(
    image: Image.Image,
    state: dict,
    sampler_choices: list[str],
) -> tuple:
    """
    画像がドロップ／アップロードされたときの処理。
    A1111 メタデータがあれば各フィールドに反映する。
    """
    if image is None:
        return (
            state,
            gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            "画像が読み込まれました（メタデータなし）。",
        )

    state["current_image"] = image
    meta = read_a1111_metadata(image) or read_comfyui_metadata(image)

    if meta is None:
        msg = "画像を読み込みました。メタデータが見つからないため、Qwen3-VL に内容を分析させることができます。"
        return (
            state,
            gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            msg,
        )

    # メタデータが読み取れた場合は各フィールドを更新
    positive = meta.get("positive", "")
    negative = meta.get("negative", "")
    state["positive_prompt"] = positive
    state["negative_prompt"] = negative

    steps_update = gr.update(value=meta["steps"]) if "steps" in meta else gr.update()
    cfg_update = gr.update(value=meta["cfg_scale"]) if "cfg_scale" in meta else gr.update()

    sampler_val = meta.get("sampler")
    sampler_update = (
        gr.update(value=sampler_val)
        if sampler_val and sampler_val in sampler_choices
        else gr.update()
    )

    width_update = gr.update(value=meta["width"]) if "width" in meta else gr.update()
    height_update = gr.update(value=meta["height"]) if "height" in meta else gr.update()
    seed_update = gr.update(value=meta["seed"]) if "seed" in meta else gr.update()
    comfyui_seed_update = gr.update(value=meta["seed"]) if "seed" in meta else gr.update()

    source = "A1111" if image.info.get("parameters") else "ComfyUI"
    msg = f"メタデータを読み込み、プロンプトを反映しました。（{source}）"
    return (
        state,
        gr.update(value=positive),
        gr.update(value=negative),
        steps_update,
        cfg_update,
        sampler_update,
        width_update,
        height_update,
        seed_update,
        comfyui_seed_update,
        msg,
    )


def on_clear(state: dict) -> tuple:
    """会話履歴をリセットする。"""
    state["conversation_history"] = []
    return state, []


def on_set_seed_random() -> int:
    return -1


def on_set_seed_from_current_image(state: dict) -> tuple:
    image = state.get("current_image")
    if image is None:
        return gr.update(), "現在の画像がないため Seed を取得できません。"

    meta = read_a1111_metadata(image) or read_comfyui_metadata(image)
    if meta is None or "seed" not in meta:
        return gr.update(), "現在の画像のメタデータに Seed がありません。"

    seed = int(meta["seed"])
    return gr.update(value=seed), f"現在の画像の Seed を反映しました: {seed}"


def _get_next_image_sequence(save_dir: str) -> int:
    pattern = re.compile(r"^(\d{5})-(-?\d+)\.png$")
    max_seq = 0
    try:
        for name in os.listdir(save_dir):
            match = pattern.match(name)
            if not match:
                continue
            seq = int(match.group(1))
            if seq > max_seq:
                max_seq = seq
    except Exception:
        return 1
    return max_seq + 1


def _get_next_video_sequence(save_dir: str) -> int:
    pattern = re.compile(r"^(\d{5})-(-?\d+)\.(mp4|webm|avi|mov)$", re.IGNORECASE)
    max_seq = 0
    try:
        for name in os.listdir(save_dir):
            match = pattern.match(name)
            if not match:
                continue
            seq = int(match.group(1))
            if seq > max_seq:
                max_seq = seq
    except Exception:
        return 1
    return max_seq + 1


def _build_pnginfo_from_image(image: Image.Image) -> PngInfo | None:
    info = getattr(image, "info", {}) or {}
    pnginfo = PngInfo()
    added = False
    for key, value in info.items():
        if value is None:
            continue
        if isinstance(value, str):
            pnginfo.add_text(str(key), value)
            added = True
            continue
        if isinstance(value, (dict, list)):
            try:
                pnginfo.add_text(str(key), json.dumps(value, ensure_ascii=False))
                added = True
            except Exception:
                continue
    return pnginfo if added else None


def on_send(
    user_input: str,
    state: dict,
    chatbot: list,
    model_label: str,
    positive: str,
    negative: str,
):
    """
    「送信」ボタン：Qwen3-VL にストリーミングで問い合わせる。
    モデル未ロードの場合は自動的にロードする。
    """
    if not user_input.strip():
        yield state, chatbot, gr.update(), gr.update(), "", gr.update()
        return

    # UI上の現在値で state を更新
    state["positive_prompt"] = positive
    state["negative_prompt"] = negative

    auto_load_status = None
    if not qwen_client.is_loaded():
        # ロード中であることを表示
        yield (
            state,
            chatbot + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": "モデルをロード中..."},
            ],
            gr.update(), gr.update(), "", gr.update(value="ロード中..."),
        )
        model_id = qwen_client.MODEL_PRESETS.get(model_label, model_label)
        try:
            auto_load_status = qwen_client.load_model(model_id)
        except RuntimeError as e:
            auto_load_status = str(e)
        if not qwen_client.is_loaded():
            yield (
                state,
                chatbot + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": f"モデルのロードに失敗しました: {auto_load_status}"},
                ],
                gr.update(), gr.update(), "", gr.update(value=auto_load_status),
            )
            return

    # ストリーミング生成
    partial_response = ""
    streaming_chatbot = chatbot + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": ""},
    ]

    try:
        for token in qwen_client.query_stream(
            conversation_history=state["conversation_history"],
            user_input=user_input,
            current_image=state.get("current_image"),
            positive_prompt=positive,
            negative_prompt=negative,
        ):
            partial_response += token
            streaming_chatbot[-1]["content"] = partial_response
            yield state, streaming_chatbot, gr.update(), gr.update(), "", gr.update()
    except Exception as e:
        yield (
            state,
            chatbot + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"エラーが発生しました: {e}"},
            ],
            gr.update(), gr.update(), "", gr.update(),
        )
        return

    positive_new, negative_new, display_text = parse_prompt_update(partial_response)

    # 会話履歴を更新（画像は current_image として別管理）
    state["conversation_history"].append({"role": "user", "content": [{"type": "text", "text": user_input}]})
    state["conversation_history"].append({"role": "assistant", "content": partial_response})

    # プロンプト更新があれば state に反映
    positive_update = gr.update()
    negative_update = gr.update()
    if positive_new is not None:
        state["positive_prompt"] = positive_new
        positive_update = gr.update(value=positive_new)
    if negative_new is not None:
        state["negative_prompt"] = negative_new
        negative_update = gr.update(value=negative_new)

    # 最終 yield: [PROMPT_UPDATE] ブロックを除いた表示テキストに差し替え
    final_chatbot = chatbot + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": display_text},
    ]
    status_update = gr.update(value=auto_load_status) if auto_load_status else gr.update()
    yield state, final_chatbot, positive_update, negative_update, "", status_update


def on_free_qwen() -> str:
    return qwen_client.unload_model()


def on_free_forge() -> str:
    return a1111_client.free_vram()


def on_free_comfyui() -> str:
    return comfyui_client.free_vram()


def on_generate(
    state: dict,
    positive: str,
    negative: str,
    steps: int,
    cfg: float,
    sampler: str,
    width: int,
    height: int,
    seed: int,
    backend: str,
    comfyui_workflow: str,
    comfyui_width: int,
    comfyui_height: int,
    comfyui_seed: int,
    count: int,
    save_generated_image: bool,
    image_save_path: str,
):
    """
    「画像生成」ボタン：選択中のバックエンドに生成リクエストを送る。
    count=1 で1枚、count>1 で指定枚数、count=0 で停止ボタンまで無限生成。
    """
    state["positive_prompt"] = positive
    state["negative_prompt"] = negative
    count = max(1, int(count) if count is not None else 1)
    save_enabled = bool(save_generated_image)
    save_dir = (image_save_path or "").strip() or "./outputs/images"
    _comfyui_seed_base = int(comfyui_seed) if comfyui_seed is not None else -1
    _forge_seed_base = int(seed) if seed is not None else -1
    for i in range(1, count + 1):
        _comfyui_seed_i = (_comfyui_seed_base + (i - 1)) if _comfyui_seed_base != -1 else -1
        _forge_seed_i = (_forge_seed_base + (i - 1)) if _forge_seed_base != -1 else -1
        try:
            _start = time.time()
            if backend == "ComfyUI":
                workflow_path = comfyui_client.IMAGE_WORKFLOW_PRESETS.get(comfyui_workflow, comfyui_workflow)
                image = comfyui_client.generate_image(
                    workflow_path=workflow_path,
                    positive=positive,
                    negative=negative,
                    seed=_comfyui_seed_i,
                    width=int(comfyui_width) if comfyui_width else None,
                    height=int(comfyui_height) if comfyui_height else None,
                )
            else:
                image = a1111_client.generate_image(
                    positive=positive,
                    negative=negative,
                    steps=steps,
                    cfg=cfg,
                    sampler=sampler,
                    width=width,
                    height=height,
                    seed=_forge_seed_i,
                )
            elapsed = time.time() - _start
            state["current_image"] = image
            if backend == "ComfyUI":
                state["current_image_stem"] = comfyui_client.get_last_output_filename()
            else:
                state["current_image_stem"] = time.strftime("forge_%Y%m%d_%H%M%S")
            save_status = ""
            if save_enabled:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    meta = read_a1111_metadata(image) or read_comfyui_metadata(image) or {}
                    seed_value = meta.get("seed")
                    if seed_value is None:
                        seed_value = _comfyui_seed_i if backend == "ComfyUI" else _forge_seed_i
                    seed_value = int(seed_value)
                    next_seq = _get_next_image_sequence(save_dir)
                    filename = f"{next_seq:05d}-{seed_value}.png"
                    save_path = os.path.join(save_dir, filename)
                    pnginfo = _build_pnginfo_from_image(image)
                    if pnginfo is not None:
                        image.save(save_path, pnginfo=pnginfo)
                    else:
                        image.save(save_path)
                    save_status = f" / 保存: {save_path}"
                except Exception as save_error:
                    save_status = f" / 保存失敗: {save_error}"
            suffix = f"（{i}/{count}枚）" if count > 1 else ""
            yield state, image, f"画像を生成しました。{suffix}（{elapsed:.1f}秒）{save_status}"
        except Exception as e:
            yield state, None, f"画像生成エラー: {e}"
            break


def on_generate_video_prompt(state: dict, positive: str, extra_instruction: str, sections: list):
    """
    「動画プロンプト生成」ボタン：画像・画像プロンプト・追加指示から動画プロンプトをワンショット生成する。
    チャット履歴は使わず、video_prompt テキストボックスにストリーミング出力する。
    """
    if not sections:
        yield "生成するセクションを1つ以上選択してください。"
        return

    if not extra_instruction.strip() and not positive.strip():
        yield "追加指示または Positive Prompt を入力してください。"
        return

    if not qwen_client.is_loaded():
        yield "モデルをロード中..."
        model_id = qwen_client.MODEL_PRESETS.get(
            state.get("model", list(qwen_client.MODEL_PRESETS.keys())[0]),
            list(qwen_client.MODEL_PRESETS.keys())[0],
        )
        try:
            qwen_client.load_model(model_id)
        except RuntimeError as e:
            yield f"モデルのロードに失敗しました: {e}"
            return

    accumulated = ""
    try:
        for token in qwen_client.generate_video_prompt_stream(
            image=state.get("current_image"),
            positive_prompt=positive,
            extra_instruction=extra_instruction,
            sections=sections,
        ):
            accumulated += token
            yield accumulated
    except Exception as e:
        yield f"動画プロンプト生成エラー: {e}"


# 動画生成の世代カウンタ：新しい生成または停止のたびにインクリメントし、
# 古いスレッドの結果が後から表示されるのを防ぐ。
_video_gen_id = 0


def on_stop_video():
    """停止ボタン：ComfyUI の生成を中断し、世代カウンタをインクリメントして結果を破棄する。"""
    global _video_gen_id
    _video_gen_id += 1
    comfyui_client.interrupt()
    return "動画生成をキャンセルしました"


def on_generate_video(
    state: dict,
    video_prompt_text: str,
    workflow_name: str,
    seed,
    width,
    height,
    save_generated_video: bool,
    video_save_path: str,
):
    """
    「動画生成」ボタン：ComfyUI に画像 + 動画プロンプトを送って動画を生成する。
    """
    global _video_gen_id
    _video_gen_id += 1
    my_id = _video_gen_id

    image = state.get("current_image")
    if image is None:
        yield state, gr.update(), "画像がありません。上の画像エリアに画像をセットしてください。"
        return
    actual_seed = int(seed) if seed is not None else -1
    actual_width  = int(width)  if width  else None
    actual_height = int(height) if height else None
    save_enabled = bool(save_generated_video)
    save_dir = (video_save_path or "").strip() or "./outputs/videos"

    result_box: list = []
    error_box: list = []

    def _run():
        # 実行開始前にすでに新しい世代が始まっていたら即終了
        if my_id != _video_gen_id:
            return
        try:
            workflow_path = comfyui_client.VIDEO_WORKFLOW_PRESETS.get(workflow_name, workflow_name)
            result_box.append(comfyui_client.generate_image(
                workflow_path=workflow_path,
                positive=video_prompt_text,
                negative="",
                seed=actual_seed,
                width=actual_width,
                height=actual_height,
                input_image=image,
            ))
        except Exception as e:
            error_box.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    start = time.time()

    while t.is_alive():
        # Gradio がこのジェネレータをキャンセルした場合、次の yield で停止する
        if my_id != _video_gen_id:
            return
        elapsed = time.time() - start
        yield state, gr.update(), f"動画生成中... {elapsed:.0f}秒"
        time.sleep(1)

    # スレッド完了後、すでに新しい世代が始まっていたら結果を捨てる
    if my_id != _video_gen_id:
        return

    elapsed = time.time() - start
    if error_box:
        yield state, gr.update(), f"動画生成エラー: {error_box[0]}"
    elif result_box:
        result = result_box[0]
        if isinstance(result, str):
            save_status = ""
            if save_enabled:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    next_seq = _get_next_video_sequence(save_dir)
                    ext = os.path.splitext(result)[1].lower() or ".mp4"
                    if ext not in {".mp4", ".webm", ".avi", ".mov"}:
                        ext = ".mp4"
                    used_seed = comfyui_client.get_last_actual_seed()
                    if used_seed is None or int(used_seed) < 0:
                        used_seed = int(actual_seed)
                    filename = f"{next_seq:05d}-{int(used_seed)}{ext}"
                    save_path = os.path.join(save_dir, filename)
                    shutil.copy2(result, save_path)
                    save_status = f" / 保存: {save_path}"
                except Exception as save_error:
                    save_status = f" / 保存失敗: {save_error}"
            yield state, gr.update(value=result), f"動画生成完了（{elapsed:.0f}秒）{save_status}"
        else:
            yield state, gr.update(), f"完了（動画ではなく画像として出力されました）（{elapsed:.0f}秒）"
    else:
        yield state, gr.update(), f"動画生成エラー: 結果が取得できませんでした"


def on_save_text(
    state: dict,
    save_folder: str,
    positive: str,
    extra_instruction: str,
    video_prompt_text: str,
) -> str:
    """
    「テキスト保存」ボタン：画像プロンプト・追加指示・動画プロンプトをテキストファイルに保存する。
    ファイル名は現在の画像と同じ stem を使用する。
    """
    stem = state.get("current_image_stem", "")
    if not stem:
        stem = time.strftime("prompt_%Y%m%d_%H%M%S")

    folder = save_folder.strip() if save_folder.strip() else "."
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        return f"フォルダの作成に失敗しました: {e}"

    filepath = os.path.join(folder, stem + ".txt")
    lines = []
    lines.append("=== Positive Prompt ===")
    lines.append(positive)
    lines.append("")
    lines.append("=== 追加指示 ===")
    lines.append(extra_instruction)
    lines.append("")
    lines.append("=== 動画プロンプト ===")
    lines.append(video_prompt_text)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return f"保存しました: {filepath}"
    except Exception as e:
        return f"保存エラー: {e}"


# ---------------------------------------------------------------------------
# Gradio UI 定義
# ---------------------------------------------------------------------------

def build_ui():
    # Forge 未起動時でも UI 起動をブロックしないため、起動時チェックは行わない
    # 接続確認は実際に Forge を使う操作時に行う
    a1111_msg = "未確認（起動時チェックをスキップ）"
    sampler_list = a1111_client.get_samplers()

    # 保存済み設定をロード
    cfg_saved = settings_manager.load()
    saved_model = cfg_saved.get("model", list(qwen_client.MODEL_PRESETS.keys())[0])
    saved_positive = cfg_saved.get("positive_prompt", "")
    saved_negative = cfg_saved.get("negative_prompt", "")
    saved_steps = cfg_saved.get("steps", 28)
    saved_cfg = cfg_saved.get("cfg", 7.0)
    saved_sampler = cfg_saved.get("sampler", DEFAULT_SAMPLER)
    saved_width = cfg_saved.get("width", 512)
    saved_height = cfg_saved.get("height", 768)
    saved_seed = cfg_saved.get("seed", -1)
    saved_backend = cfg_saved.get("backend", "WebUI Forge")
    if saved_backend == "Forge 2":  # 旧名称の移行
        saved_backend = "WebUI Forge"
    saved_comfyui_workflow = cfg_saved.get("comfyui_workflow", "")
    saved_comfyui_url = cfg_saved.get("comfyui_url", "http://127.0.0.1:8188")
    saved_comfyui_seed = cfg_saved.get("comfyui_seed", -1)
    saved_comfyui_width = cfg_saved.get("comfyui_width", 1024)
    saved_comfyui_height = cfg_saved.get("comfyui_height", 1024)
    saved_generate_count = cfg_saved.get("generate_count", 1)
    saved_save_generated_image = cfg_saved.get("save_generated_image", False)
    saved_image_save_path = cfg_saved.get("image_save_path", "./outputs/images")
    saved_save_generated_video = cfg_saved.get("save_generated_video", False)
    saved_video_save_path = cfg_saved.get("video_save_path", "./outputs/videos")
    saved_video_sections = cfg_saved.get("video_sections", ["scene", "action", "camera", "style", "prompt"])
    saved_video_width    = cfg_saved.get("video_width", None)
    saved_video_height   = cfg_saved.get("video_height", None)
    saved_video_seed     = cfg_saved.get("video_seed", -1)

    # ComfyUI URL を comfyui_client に反映・接続確認
    comfyui_client.COMFYUI_URL = saved_comfyui_url
    comfyui_client.reload_workflows()
    workflow_list = list(comfyui_client.IMAGE_WORKFLOW_PRESETS.keys())
    video_workflow_list = list(comfyui_client.VIDEO_WORKFLOW_PRESETS.keys())
    comfyui_sampler_list = comfyui_client.get_samplers()
    _, comfyui_msg = comfyui_client.check_connection()

    # 保存されたサンプラーが一覧にない場合はフォールバック
    if saved_sampler not in sampler_list:
        saved_sampler = sampler_list[0] if sampler_list else DEFAULT_SAMPLER

    initial_state = {
        "conversation_history": [],
        "current_image": None,
        "current_image_stem": "",
        "positive_prompt": saved_positive,
        "negative_prompt": saved_negative,
    }

    with gr.Blocks(title="Prompt Assistant", css=".fullscreen-button { display: none !important; }") as demo:
        state = gr.State(value=initial_state)
        sampler_choices = gr.State(value=sampler_list)

        gr.Markdown("# Prompt Assistant")

        # ---- メインエリア（タブ切り替え）----
        with gr.Tabs():

            # タブ1: 画像生成
            with gr.Tab("画像生成"):
                with gr.Row(equal_height=False):

                    # 左: 画像表示エリア（ドロップ対応）
                    with gr.Column(scale=1):
                        image_display = gr.Image(
                            label="生成画像 / 画像をドロップしてロード",
                            type="pil",
                            interactive=True,
                            height=480,
                            sources=["upload", "clipboard"],
                        )
                        image_status = gr.Textbox(
                            label="画像ロードステータス",
                            interactive=False,
                            visible=True,
                            max_lines=2,
                        )

                    # 中: プロンプト ＋ パラメータ ＋ 生成ボタン
                    with gr.Column(scale=1):
                        positive_prompt = gr.Textbox(
                            label="Positive Prompt",
                            value=saved_positive,
                            lines=6,
                            placeholder="例: 1girl, sunset, orange sky, dramatic lighting, ...",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value=saved_negative,
                            lines=3,
                            placeholder="例: bad quality, worst quality, ...",
                        )

                        with gr.Row():
                            generate_btn = gr.Button("画像生成", variant="primary", scale=3)
                            stop_btn = gr.Button("停止", variant="secondary", scale=1)
                        count_input = gr.Number(
                            value=saved_generate_count,
                            label="生成枚数",
                            precision=0,
                            minimum=1,
                        )
                        save_generated_image_checkbox = gr.Checkbox(
                            value=saved_save_generated_image,
                            label="画像を保存する",
                        )
                        image_save_path_input = gr.Textbox(
                            value=saved_image_save_path,
                            label="保存先パス",
                            placeholder="例: ./outputs/images",
                            interactive=bool(saved_save_generated_image),
                        )

                        with gr.Accordion("画像生成パラメータ", open=False):
                            with gr.Row():
                                backend_radio = gr.Radio(
                                    choices=["WebUI Forge", "ComfyUI"],
                                    value=saved_backend,
                                    label="バックエンド",
                                )
                            _initial_conn_msg = (
                                f"ComfyUI: {comfyui_msg}" if saved_backend == "ComfyUI"
                                else f"WebUI Forge: {a1111_msg}"
                            )
                            connection_status = gr.Markdown(_initial_conn_msg)
                            comfyui_workflow_dropdown = gr.Dropdown(
                                choices=workflow_list,
                                value=saved_comfyui_workflow if saved_comfyui_workflow in workflow_list else (workflow_list[0] if workflow_list else None),
                                label="ComfyUI ワークフロー",
                                visible=(saved_backend == "ComfyUI"),
                            )
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=1, maximum=150, value=saved_steps, step=1, label="Steps",
                                    visible=(saved_backend != "ComfyUI"),
                                )
                                cfg_slider = gr.Slider(
                                    minimum=1.0, maximum=30.0, value=saved_cfg, step=0.5, label="CFG Scale",
                                    visible=(saved_backend != "ComfyUI"),
                                )
                            sampler_dropdown = gr.Dropdown(
                                choices=sampler_list,
                                value=saved_sampler,
                                label="Sampler",
                                visible=(saved_backend != "ComfyUI"),
                            )
                            width_input = gr.Slider(
                                minimum=64, maximum=2048, value=saved_width, step=8, label="Width",
                                visible=(saved_backend != "ComfyUI"),
                            )
                            height_input = gr.Slider(
                                minimum=64, maximum=2048, value=saved_height, step=8, label="Height",
                                visible=(saved_backend != "ComfyUI"),
                            )
                            with gr.Row():
                                seed_input = gr.Number(
                                    value=saved_seed, label="Seed（-1 でランダム）", precision=0,
                                    visible=(saved_backend != "ComfyUI"),
                                    scale=8,
                                )
                                seed_random_btn = gr.Button("🎲", size="sm", visible=(saved_backend != "ComfyUI"), scale=1, min_width=48)
                                seed_from_image_btn = gr.Button("♻️", size="sm", visible=(saved_backend != "ComfyUI"), scale=1, min_width=48)
                            with gr.Row():
                                comfyui_width_input = gr.Slider(
                                    minimum=64, maximum=2048, value=saved_comfyui_width, step=8, label="Width",
                                    visible=(saved_backend == "ComfyUI"),
                                )
                                comfyui_height_input = gr.Slider(
                                    minimum=64, maximum=2048, value=saved_comfyui_height, step=8, label="Height",
                                    visible=(saved_backend == "ComfyUI"),
                                )
                            with gr.Row():
                                comfyui_seed_input = gr.Number(
                                    value=saved_comfyui_seed, label="Seed（-1 でランダム）", precision=0,
                                    visible=(saved_backend == "ComfyUI"),
                                    scale=8,
                                )
                                comfyui_seed_random_btn = gr.Button("🎲", size="sm", visible=(saved_backend == "ComfyUI"), scale=1, min_width=48)
                                comfyui_seed_from_image_btn = gr.Button("♻️", size="sm", visible=(saved_backend == "ComfyUI"), scale=1, min_width=48)

                    # 右: Qwen3-VL 会話エリア
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(
                            label="会話 (Qwen3-VL)",
                            height=480,
                        )
                        user_input = gr.Textbox(
                            placeholder="Qwen3-VL へのメッセージを入力...",
                            label="",
                            lines=1,
                        )
                        with gr.Row():
                            send_btn = gr.Button("送信", scale=1, variant="primary")
                            clear_btn = gr.Button("クリア", scale=1, variant="secondary")

                        # ---- モデル設定 ----
                        with gr.Accordion("モデル設定", open=False):
                            with gr.Row():
                                model_dropdown = gr.Dropdown(
                                    choices=list(qwen_client.MODEL_PRESETS.keys()),
                                    value=saved_model if saved_model in qwen_client.MODEL_PRESETS else list(qwen_client.MODEL_PRESETS.keys())[0],
                                    label="Qwen3-VL モデル",
                                    scale=3,
                                )
                                load_btn = gr.Button("モデルをロード", scale=1)
                            model_status = gr.Textbox(
                                value="モデル未ロード",
                                label="モデルステータス",
                                interactive=False,
                            )

            # タブ2: 動画生成
            with gr.Tab("動画生成"):
                with gr.Row(equal_height=False):

                    # 列1: 生成動画 + ステータス
                    with gr.Column(scale=1):
                        video_display = gr.Video(
                            label="生成動画",
                            height=480,
                        )
                        video_status = gr.Textbox(
                            label="動画生成ステータス",
                            interactive=False,
                            max_lines=2,
                        )

                    # 列2: 動画プロンプト + 生成/停止ボタン + VRAM
                    with gr.Column(scale=1):
                        video_prompt = gr.Textbox(
                            label="動画プロンプト",
                            lines=10,
                            interactive=True,
                            placeholder="「動画プロンプト生成」ボタンで自動生成、または直接入力",
                        )
                        with gr.Row():
                            generate_video_btn = gr.Button("動画生成", variant="primary", scale=3)
                            video_stop_btn = gr.Button("停止", variant="secondary", scale=1)

                        with gr.Accordion("VRAM", open=False):
                            free_vram_status = gr.Textbox(
                                label="VRAM解放ステータス", interactive=False, lines=2,
                            )
                            with gr.Row():
                                free_qwen_btn  = gr.Button("LLM 解放",          variant="secondary", scale=1)
                                free_forge_btn = gr.Button("WebUI Forge 解放", variant="secondary", scale=1)
                                free_comfy_btn = gr.Button("ComfyUI 解放",     variant="secondary", scale=1)

                        with gr.Accordion("動画生成パラメータ", open=False):
                            video_workflow_dropdown = gr.Dropdown(
                                choices=video_workflow_list,
                                value=video_workflow_list[0] if video_workflow_list else None,
                                label="動画ワークフロー",
                            )
                            video_width_input = gr.Slider(
                                minimum=240, maximum=1920, step=16,
                                value=saved_video_width if saved_video_width else 848,
                                label="Width",
                            )
                            video_height_input = gr.Slider(
                                minimum=240, maximum=1920, step=16,
                                value=saved_video_height if saved_video_height else 480,
                                label="Height",
                            )
                            with gr.Row():
                                video_res_640_btn  = gr.Button("640 × 480",  variant="secondary", size="sm")
                                video_res_1280_btn = gr.Button("1280 × 720", variant="secondary", size="sm")
                            video_seed_input = gr.Number(
                                value=saved_video_seed,
                                label="Seed（-1 でランダム）",
                                precision=0,
                            )
                            save_generated_video_checkbox = gr.Checkbox(
                                value=saved_save_generated_video,
                                label="動画を保存する",
                            )
                            video_save_path_input = gr.Textbox(
                                value=saved_video_save_path,
                                label="保存先URL",
                                placeholder="例: ./outputs/videos",
                                interactive=bool(saved_save_generated_video),
                            )

                    # 列3: 追加指示 + セクション選択 + プロンプト生成
                    with gr.Column(scale=1):
                        video_extra_instruction = gr.Textbox(
                            label="追加指示",
                            lines=3,
                            placeholder="例: slowly zoom in, hair flowing in the wind, cinematic lighting",
                        )
                        video_section_checkboxes = gr.CheckboxGroup(
                            choices=["scene", "action", "camera", "style", "prompt"],
                            value=saved_video_sections,
                            label="生成するセクション",
                        )
                        generate_video_prompt_btn = gr.Button("動画プロンプト生成", variant="primary")

                        with gr.Accordion("テキスト保存", open=False):
                            save_folder_input = gr.Textbox(
                                label="保存先フォルダ",
                                value="./outputs/text",
                                placeholder="保存先フォルダのパスを入力...",
                            )
                            save_text_btn = gr.Button("テキスト保存", variant="secondary")
                            save_text_status = gr.Textbox(
                                label="保存ステータス",
                                interactive=False,
                                max_lines=2,
                            )

        # ---- 設定自動保存 ----

        _save_inputs = [
            model_dropdown,
            positive_prompt, negative_prompt,
            steps_slider, cfg_slider, sampler_dropdown,
            width_input, height_input, seed_input,
            backend_radio, comfyui_workflow_dropdown,
            comfyui_width_input, comfyui_height_input, comfyui_seed_input,
            count_input,
            save_generated_image_checkbox, image_save_path_input,
            save_generated_video_checkbox, video_save_path_input,
            video_section_checkboxes,
            video_width_input, video_height_input, video_seed_input,
        ]

        def _save_settings(model, positive, negative, steps, cfg, sampler, width, height, seed, backend, comfyui_workflow, comfyui_width, comfyui_height, comfyui_seed, generate_count, save_generated_image, image_save_path, save_generated_video, video_save_path, video_sections, video_width, video_height, video_seed):
            settings_manager.save({
                "model": model,
                "positive_prompt": positive,
                "negative_prompt": negative,
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler": sampler,
                "width": int(width) if width else 512,
                "height": int(height) if height else 768,
                "seed": int(seed) if seed is not None else -1,
                "backend": backend,
                "comfyui_workflow": comfyui_workflow or "",
                "comfyui_width": int(comfyui_width) if comfyui_width else 1024,
                "comfyui_height": int(comfyui_height) if comfyui_height else 1024,
                "comfyui_seed": int(comfyui_seed) if comfyui_seed is not None else -1,
                "generate_count": int(generate_count) if generate_count is not None else 1,
                "save_generated_image": bool(save_generated_image),
                "image_save_path": (image_save_path or "").strip() or "./outputs/images",
                "save_generated_video": bool(save_generated_video),
                "video_save_path": (video_save_path or "").strip() or "./outputs/videos",
                "video_sections": video_sections or [],
                "video_width": int(video_width) if video_width else None,
                "video_height": int(video_height) if video_height else None,
                "video_seed": int(video_seed) if video_seed is not None else -1,
            })

        def _on_backend_change(backend):
            """バックエンド切り替え時に各コンポーネントの表示を切り替える。"""
            is_comfy = backend == "ComfyUI"
            if is_comfy:
                _, latest_msg = comfyui_client.check_connection()
                conn_msg = f"ComfyUI: {latest_msg}"
            else:
                _, latest_msg = a1111_client.check_connection()
                conn_msg = f"WebUI Forge: {latest_msg}"
            return (
                gr.update(visible=is_comfy),        # comfyui_workflow_dropdown
                gr.update(visible=not is_comfy),    # steps_slider
                gr.update(visible=not is_comfy),    # cfg_slider
                gr.update(visible=not is_comfy),    # sampler_dropdown
                gr.update(visible=not is_comfy),    # width_input
                gr.update(visible=not is_comfy),    # height_input
                gr.update(visible=not is_comfy),    # seed_input
                gr.update(visible=not is_comfy),    # seed_random_btn
                gr.update(visible=not is_comfy),    # seed_from_image_btn
                gr.update(visible=is_comfy),        # comfyui_width_input
                gr.update(visible=is_comfy),        # comfyui_height_input
                gr.update(visible=is_comfy),        # comfyui_seed_input
                gr.update(visible=is_comfy),        # comfyui_seed_random_btn
                gr.update(visible=is_comfy),        # comfyui_seed_from_image_btn
                gr.update(value=conn_msg),          # connection_status
            )

        def _on_save_image_toggle(enabled):
            return gr.update(interactive=bool(enabled))

        def _on_save_video_toggle(enabled):
            return gr.update(interactive=bool(enabled))

        for _comp in _save_inputs:
            _comp.change(fn=_save_settings, inputs=_save_inputs, outputs=[])

        backend_radio.change(
            fn=_on_backend_change,
            inputs=[backend_radio],
            outputs=[
                comfyui_workflow_dropdown,
                steps_slider, cfg_slider, sampler_dropdown,
                width_input, height_input, seed_input,
                seed_random_btn, seed_from_image_btn,
                comfyui_width_input, comfyui_height_input, comfyui_seed_input,
                comfyui_seed_random_btn, comfyui_seed_from_image_btn,
                connection_status,
            ],
        )
        save_generated_image_checkbox.change(
            fn=_on_save_image_toggle,
            inputs=[save_generated_image_checkbox],
            outputs=[image_save_path_input],
        )
        save_generated_video_checkbox.change(
            fn=_on_save_video_toggle,
            inputs=[save_generated_video_checkbox],
            outputs=[video_save_path_input],
        )

        # ---- イベント接続 ----

        seed_random_btn.click(fn=on_set_seed_random, inputs=[], outputs=[seed_input])
        comfyui_seed_random_btn.click(fn=on_set_seed_random, inputs=[], outputs=[comfyui_seed_input])
        seed_from_image_btn.click(
            fn=on_set_seed_from_current_image,
            inputs=[state],
            outputs=[seed_input, image_status],
        )
        comfyui_seed_from_image_btn.click(
            fn=on_set_seed_from_current_image,
            inputs=[state],
            outputs=[comfyui_seed_input, image_status],
        )

        load_btn.click(
            fn=on_load_model,
            inputs=[model_dropdown],
            outputs=[model_status],
        )

        image_display.upload(
            fn=on_image_drop,
            inputs=[image_display, state, sampler_choices],
            outputs=[
                state,
                positive_prompt, negative_prompt,
                steps_slider, cfg_slider, sampler_dropdown,
                width_input, height_input, seed_input, comfyui_seed_input,
                image_status,
            ],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[state],
            outputs=[state, chatbot],
        )

        send_btn.click(
            fn=on_send,
            inputs=[user_input, state, chatbot, model_dropdown, positive_prompt, negative_prompt],
            outputs=[state, chatbot, positive_prompt, negative_prompt, user_input, model_status],
        )

        user_input.submit(
            fn=on_send,
            inputs=[user_input, state, chatbot, model_dropdown, positive_prompt, negative_prompt],
            outputs=[state, chatbot, positive_prompt, negative_prompt, user_input, model_status],
        )

        gen_event = generate_btn.click(
            fn=on_generate,
            inputs=[
                state,
                positive_prompt, negative_prompt,
                steps_slider, cfg_slider, sampler_dropdown,
                width_input, height_input, seed_input,
                backend_radio, comfyui_workflow_dropdown,
                comfyui_width_input, comfyui_height_input, comfyui_seed_input,
                count_input,
                save_generated_image_checkbox, image_save_path_input,
            ],
            outputs=[state, image_display, image_status],
        )

        stop_btn.click(fn=None, cancels=[gen_event])

        free_qwen_btn.click(fn=on_free_qwen,    inputs=[], outputs=[free_vram_status])
        free_forge_btn.click(fn=on_free_forge,  inputs=[], outputs=[free_vram_status])
        free_comfy_btn.click(fn=on_free_comfyui, inputs=[], outputs=[free_vram_status])

        # ---- 動画生成イベント ----

        video_res_640_btn.click(
            fn=lambda: (640, 480),
            inputs=[],
            outputs=[video_width_input, video_height_input],
        )
        video_res_1280_btn.click(
            fn=lambda: (1280, 720),
            inputs=[],
            outputs=[video_width_input, video_height_input],
        )

        gen_video_prompt_event = generate_video_prompt_btn.click(
            fn=on_generate_video_prompt,
            inputs=[state, positive_prompt, video_extra_instruction, video_section_checkboxes],
            outputs=[video_prompt],
        )

        gen_video_event = generate_video_btn.click(
            fn=on_generate_video,
            inputs=[state, video_prompt, video_workflow_dropdown, video_seed_input, video_width_input, video_height_input, save_generated_video_checkbox, video_save_path_input],
            outputs=[state, video_display, video_status],
        )

        video_stop_btn.click(
            fn=on_stop_video,
            inputs=[],
            outputs=[video_status],
            cancels=[gen_video_prompt_event, gen_video_event],
        )

        save_text_btn.click(
            fn=on_save_text,
            inputs=[state, save_folder_input, positive_prompt, video_extra_instruction, video_prompt],
            outputs=[save_text_status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue()
    app.launch(inbrowser=True)
