"""
discover_forge_api.py
Forge 2 の Gradio API エンドポイントと txt2img パラメータ一覧を調べるツール。
結果は forge_api_info.txt に書き出す（cp932 問題回避）。
使い方: python discover_forge_api.py
"""

import io
import sys
import contextlib

# stdout を utf-8 に強制
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from gradio_client import Client

FORGE_URL = "http://127.0.0.1:7861"
OUTPUT_FILE = "d:/GitHub/prompt-assistant/forge_api_info.txt"

print(f"Connecting to {FORGE_URL} ...")
client = Client(FORGE_URL, verbose=False)
print("接続成功。API 情報を取得中...")

# view_api の stdout 出力をキャプチャ（cp932 エラー回避）
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        api_info = client.view_api(return_format="dict")
    captured = buf.getvalue()
except Exception as e:
    print(f"view_api エラー: {e}")
    sys.exit(1)

# ---- /txt2img パラメータを整形して出力 ----
lines = []
txt2img = api_info.get("named_endpoints", {}).get("/txt2img")
if txt2img is None:
    lines.append("[!] /txt2img が named_endpoints に見つかりません。")
    lines.append("利用可能な named_endpoints:")
    for name in api_info.get("named_endpoints", {}):
        lines.append(f"  {name}")
    lines.append("\nunnamed_endpoints (fn_index):")
    for idx in api_info.get("unnamed_endpoints", {}):
        lines.append(f"  fn_index={idx}")
else:
    params = txt2img.get("parameters", [])
    lines.append(f"/txt2img パラメータ数: {len(params)}")
    lines.append("")
    for i, p in enumerate(params):
        label = p.get("label") or p.get("parameter_name") or f"param_{i}"
        ptype = p.get("python_type", {}).get("type", "?")
        default = p.get("default", "<required>")
        lines.append(f"  [{i:02d}] {label:45s} type={ptype:20s} default={default!r}")

result_text = "\n".join(lines)

# ファイルに書き出し
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(result_text + "\n")

print(result_text)
print(f"\n=> {OUTPUT_FILE} にも保存しました。")
