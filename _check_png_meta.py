from PIL import Image
import json
img = Image.open("d:/GitHub/prompt-assistant/sample_comfyui.png")
for k, v in img.info.items():
    print("KEY:", repr(k))
    val = str(v)
    print("VALUE (first 800 chars):", val[:800])
    print()
