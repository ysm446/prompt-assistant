"""
app.py
SD × Qwen3-VL 画像生成アシスタント - Gradio アプリ本体
"""

import gradio as gr
from PIL import Image

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
            gr.update(), gr.update(), gr.update(),
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
            gr.update(), gr.update(), gr.update(),
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
        msg,
    )


def on_clear(state: dict) -> tuple:
    """会話履歴をリセットする。"""
    state["conversation_history"] = []
    return state, []


def on_send(
    user_input: str,
    state: dict,
    chatbot: list,
    model_label: str,
):
    """
    「送信」ボタン：Qwen3-VL にストリーミングで問い合わせる。
    モデル未ロードの場合は自動的にロードする。
    """
    if not user_input.strip():
        yield state, chatbot, gr.update(), gr.update(), "", gr.update()
        return

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
            positive_prompt=state.get("positive_prompt", ""),
            negative_prompt=state.get("negative_prompt", ""),
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
):
    """
    「画像生成」ボタン：選択中のバックエンドに生成リクエストを送る。
    count=1 で1枚、count>1 で指定枚数、count=0 で停止ボタンまで無限生成。
    """
    state["positive_prompt"] = positive
    state["negative_prompt"] = negative
    count = max(1, int(count) if count is not None else 1)
    _comfyui_seed_base = int(comfyui_seed) if comfyui_seed is not None else -1
    _forge_seed_base = int(seed) if seed is not None else -1
    for i in range(1, count + 1):
        _comfyui_seed_i = (_comfyui_seed_base + (i - 1)) if _comfyui_seed_base != -1 else -1
        _forge_seed_i = (_forge_seed_base + (i - 1)) if _forge_seed_base != -1 else -1
        try:
            if backend == "ComfyUI":
                workflow_path = comfyui_client.WORKFLOW_PRESETS.get(comfyui_workflow, comfyui_workflow)
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
            state["current_image"] = image
            suffix = f"（{i}/{count}枚）" if count > 1 else ""
            yield state, image, f"画像を生成しました。{suffix}"
        except Exception as e:
            yield state, None, f"画像生成エラー: {e}"
            break


# ---------------------------------------------------------------------------
# Gradio UI 定義
# ---------------------------------------------------------------------------

def build_ui():
    # アプリ起動時に A1111 接続確認とサンプラー一覧を取得
    a1111_ok, a1111_msg = a1111_client.check_connection()
    sampler_list = a1111_client.get_samplers() if a1111_ok else [DEFAULT_SAMPLER]

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

    # ComfyUI URL を comfyui_client に反映・接続確認
    comfyui_client.COMFYUI_URL = saved_comfyui_url
    comfyui_client.reload_workflows()
    workflow_list = list(comfyui_client.WORKFLOW_PRESETS.keys())
    comfyui_sampler_list = comfyui_client.get_samplers()
    _, comfyui_msg = comfyui_client.check_connection()

    # 保存されたサンプラーが一覧にない場合はフォールバック
    if saved_sampler not in sampler_list:
        saved_sampler = sampler_list[0] if sampler_list else DEFAULT_SAMPLER

    initial_state = {
        "conversation_history": [],
        "current_image": None,
        "positive_prompt": saved_positive,
        "negative_prompt": saved_negative,
    }

    with gr.Blocks(title="Prompt Assistant", css=".fullscreen-button { display: none !important; }") as demo:
        state = gr.State(value=initial_state)
        sampler_choices = gr.State(value=sampler_list)

        gr.Markdown("# Prompt Assistant")

        # ---- メインエリア（3カラム横並び）----
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
                    stop_btn = gr.Button("停止", variant="stop", scale=1)
                count_input = gr.Number(
                    value=saved_generate_count,
                    label="生成枚数",
                    precision=0,
                    minimum=1,
                )

                with gr.Accordion("生成パラメータ", open=False):
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
                    seed_input = gr.Number(
                        value=saved_seed, label="Seed（-1 でランダム）", precision=0,
                        visible=(saved_backend != "ComfyUI"),
                    )
                    with gr.Row():
                        comfyui_width_input = gr.Slider(
                            minimum=64, maximum=2048, value=saved_comfyui_width, step=8, label="Width",
                            visible=(saved_backend == "ComfyUI"),
                        )
                        comfyui_height_input = gr.Slider(
                            minimum=64, maximum=2048, value=saved_comfyui_height, step=8, label="Height",
                            visible=(saved_backend == "ComfyUI"),
                        )
                    comfyui_seed_input = gr.Number(
                        value=saved_comfyui_seed, label="Seed（-1 でランダム）", precision=0,
                        visible=(saved_backend == "ComfyUI"),
                    )

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
                    send_btn = gr.Button("送信", scale=1, variant="secondary")
                    clear_btn = gr.Button("クリア", scale=1, variant="stop")

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

        # ---- 設定自動保存 ----

        _save_inputs = [
            model_dropdown,
            positive_prompt, negative_prompt,
            steps_slider, cfg_slider, sampler_dropdown,
            width_input, height_input, seed_input,
            backend_radio, comfyui_workflow_dropdown,
            comfyui_width_input, comfyui_height_input, comfyui_seed_input,
            count_input,
        ]

        def _save_settings(model, positive, negative, steps, cfg, sampler, width, height, seed, backend, comfyui_workflow, comfyui_width, comfyui_height, comfyui_seed, generate_count):
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
            })

        def _on_backend_change(backend):
            """バックエンド切り替え時に各コンポーネントの表示を切り替える。"""
            is_comfy = backend == "ComfyUI"
            conn_msg = f"ComfyUI: {comfyui_msg}" if is_comfy else f"WebUI Forge: {a1111_msg}"
            return (
                gr.update(visible=is_comfy),        # comfyui_workflow_dropdown
                gr.update(visible=not is_comfy),    # steps_slider
                gr.update(visible=not is_comfy),    # cfg_slider
                gr.update(visible=not is_comfy),    # sampler_dropdown
                gr.update(visible=not is_comfy),    # width_input
                gr.update(visible=not is_comfy),    # height_input
                gr.update(visible=not is_comfy),    # seed_input
                gr.update(visible=is_comfy),        # comfyui_width_input
                gr.update(visible=is_comfy),        # comfyui_height_input
                gr.update(visible=is_comfy),        # comfyui_seed_input
                gr.update(value=conn_msg),          # connection_status
            )

        for _comp in _save_inputs:
            _comp.change(fn=_save_settings, inputs=_save_inputs, outputs=[])

        backend_radio.change(
            fn=_on_backend_change,
            inputs=[backend_radio],
            outputs=[
                comfyui_workflow_dropdown,
                steps_slider, cfg_slider, sampler_dropdown,
                width_input, height_input, seed_input,
                comfyui_width_input, comfyui_height_input, comfyui_seed_input,
                connection_status,
            ],
        )

        # ---- イベント接続 ----

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
                width_input, height_input, seed_input,
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
            inputs=[user_input, state, chatbot, model_dropdown],
            outputs=[state, chatbot, positive_prompt, negative_prompt, user_input, model_status],
        )

        user_input.submit(
            fn=on_send,
            inputs=[user_input, state, chatbot, model_dropdown],
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
            ],
            outputs=[state, image_display, image_status],
        )

        stop_btn.click(fn=None, cancels=[gen_event])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue()
    app.launch(inbrowser=True)
