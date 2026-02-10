"""
app.py
SD × Qwen3-VL 画像生成アシスタント - Gradio アプリ本体
"""

import gradio as gr
from PIL import Image

import a1111_client
import qwen_client
import settings_manager
from prompt_parser import parse_prompt_update, read_a1111_metadata

# 解像度の選択肢
SIZE_CHOICES = [512, 768, 832, 1024, 1216]
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
    meta = read_a1111_metadata(image)

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

    width_update = gr.update(value=meta["width"]) if "width" in meta and meta["width"] in SIZE_CHOICES else gr.update()
    height_update = gr.update(value=meta["height"]) if "height" in meta and meta["height"] in SIZE_CHOICES else gr.update()
    seed_update = gr.update(value=meta["seed"]) if "seed" in meta else gr.update()

    msg = "メタデータを読み込み、プロンプト・パラメータを反映しました。"
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


def on_send(
    user_input: str,
    state: dict,
    chatbot: list,
) -> tuple:
    """
    「送信」ボタン：Qwen3-VL に問い合わせて会話を進める。
    """
    if not user_input.strip():
        return state, chatbot, gr.update(), gr.update(), ""

    if not qwen_client.is_loaded():
        chatbot = chatbot + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": "モデルがロードされていません。先にモデルをロードしてください。"},
        ]
        return state, chatbot, gr.update(), gr.update(), ""

    try:
        response = qwen_client.query(
            conversation_history=state["conversation_history"],
            user_input=user_input,
            current_image=state.get("current_image"),
            positive_prompt=state.get("positive_prompt", ""),
            negative_prompt=state.get("negative_prompt", ""),
        )
    except Exception as e:
        chatbot = chatbot + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": f"エラーが発生しました: {e}"},
        ]
        return state, chatbot, gr.update(), gr.update(), ""

    positive_new, negative_new, display_text = parse_prompt_update(response)

    # 会話履歴を更新（messages 形式）
    user_content = []
    if state.get("current_image") is not None:
        user_content.append({"type": "image", "image": state["current_image"]})
    user_content.append({"type": "text", "text": user_input})
    state["conversation_history"].append({"role": "user", "content": user_content})
    state["conversation_history"].append({"role": "assistant", "content": response})

    # プロンプト更新があれば state に反映
    positive_update = gr.update()
    negative_update = gr.update()
    if positive_new is not None:
        state["positive_prompt"] = positive_new
        positive_update = gr.update(value=positive_new)
    if negative_new is not None:
        state["negative_prompt"] = negative_new
        negative_update = gr.update(value=negative_new)

    chatbot = chatbot + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": display_text},
    ]

    return state, chatbot, positive_update, negative_update, ""


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
) -> tuple:
    """
    「画像生成」ボタン：A1111 に生成リクエストを送る。
    """
    state["positive_prompt"] = positive
    state["negative_prompt"] = negative

    try:
        image = a1111_client.generate_image(
            positive=positive,
            negative=negative,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            width=width,
            height=height,
            seed=seed,
        )
        state["current_image"] = image
        return state, image, "画像を生成しました。"
    except Exception as e:
        return state, None, f"画像生成エラー: {e}"


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

    # 保存されたサンプラーが一覧にない場合はフォールバック
    if saved_sampler not in sampler_list:
        saved_sampler = sampler_list[0] if sampler_list else DEFAULT_SAMPLER

    initial_state = {
        "conversation_history": [],
        "current_image": None,
        "positive_prompt": saved_positive,
        "negative_prompt": saved_negative,
    }

    with gr.Blocks(title="Prompt Assistant") as demo:
        state = gr.State(value=initial_state)
        sampler_choices = gr.State(value=sampler_list)

        gr.Markdown("# Prompt Assistant")

        # ---- モデル設定バー ----
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
                scale=3,
            )

        # ---- 接続ステータス ----
        gr.Markdown(f"> **A1111 ステータス:** {a1111_msg}")

        # ---- メインエリア（3カラム横並び）----
        with gr.Row(equal_height=False):

            # 左: 画像表示エリア（ドロップ対応）
            with gr.Column(scale=1):
                image_display = gr.Image(
                    label="生成画像 / 画像をドロップしてロード",
                    type="pil",
                    interactive=True,
                    height=480,
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

                with gr.Accordion("生成パラメータ", open=True):
                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=1, maximum=150, value=saved_steps, step=1, label="Steps"
                        )
                        cfg_slider = gr.Slider(
                            minimum=1.0, maximum=30.0, value=saved_cfg, step=0.5, label="CFG Scale"
                        )
                    sampler_dropdown = gr.Dropdown(
                        choices=sampler_list,
                        value=saved_sampler,
                        label="Sampler",
                    )
                    with gr.Row():
                        width_dropdown = gr.Dropdown(
                            choices=SIZE_CHOICES,
                            value=saved_width if saved_width in SIZE_CHOICES else 512,
                            label="Width",
                        )
                        height_dropdown = gr.Dropdown(
                            choices=SIZE_CHOICES,
                            value=saved_height if saved_height in SIZE_CHOICES else 768,
                            label="Height",
                        )
                    seed_input = gr.Number(value=saved_seed, label="Seed（-1 でランダム）", precision=0)

                generate_btn = gr.Button("画像生成", variant="primary")

            # 右: Qwen3-VL 会話エリア
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="会話 (Qwen3-VL)",
                    height=480,
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Qwen3-VL へのメッセージを入力...",
                        label="",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("送信", scale=1, variant="secondary")

        # ---- 設定自動保存 ----

        _save_inputs = [
            model_dropdown,
            positive_prompt, negative_prompt,
            steps_slider, cfg_slider, sampler_dropdown,
            width_dropdown, height_dropdown, seed_input,
        ]

        def _save_settings(model, positive, negative, steps, cfg, sampler, width, height, seed):
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
            })

        for _comp in _save_inputs:
            _comp.change(fn=_save_settings, inputs=_save_inputs, outputs=[])

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
                width_dropdown, height_dropdown, seed_input,
                image_status,
            ],
        )

        send_btn.click(
            fn=on_send,
            inputs=[user_input, state, chatbot],
            outputs=[state, chatbot, positive_prompt, negative_prompt, user_input],
        )

        user_input.submit(
            fn=on_send,
            inputs=[user_input, state, chatbot],
            outputs=[state, chatbot, positive_prompt, negative_prompt, user_input],
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[
                state,
                positive_prompt, negative_prompt,
                steps_slider, cfg_slider, sampler_dropdown,
                width_dropdown, height_dropdown, seed_input,
            ],
            outputs=[state, image_display, image_status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(inbrowser=True)
