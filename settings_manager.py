"""
settings_manager.py
アプリ設定（生成パラメータ・プロンプト・モデル選択）の保存と読み込み。
settings.json にシリアライズする。
"""

import json
import os

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "model": "qwen3-vl-4b (推奨)",
    "positive_prompt": "",
    "negative_prompt": "",
    "steps": 28,
    "cfg": 7.0,
    "sampler": "Euler a",
    "width": 512,
    "height": 768,
    "seed": -1,
    "backend": "Forge 2",
    "comfyui_workflow": "",
    "comfyui_url": "http://127.0.0.1:8188",
}


def load() -> dict:
    """settings.json から設定を読み込む。ファイルがなければデフォルトを返す。"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            settings = DEFAULT_SETTINGS.copy()
            settings.update(saved)
            return settings
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save(settings: dict) -> None:
    """設定を settings.json に書き出す。失敗しても例外を外に出さない。"""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
