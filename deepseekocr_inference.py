import time
import base64
import io
from openai import OpenAI
from PIL import Image

client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.10.50.50:8000/v1",
    timeout=3600
)

# ========== 配置区域 ==========
# 注意：base_size、image_size、crop_mode 等参数是服务器端配置参数，
# 需要在服务器启动时在 config.py 中配置，不能通过 API 请求传递。
# 
# 如需切换模式，请联系服务器管理员修改服务器端的 config.py 文件：
# 
# Tiny 模式:
#   BASE_SIZE = 512
#   IMAGE_SIZE = 512
#   CROP_MODE = False
#
# Small 模式:
#   BASE_SIZE = 640
#   IMAGE_SIZE = 640
#   CROP_MODE = False
#
# Base 模式:
#   BASE_SIZE = 1024
#   IMAGE_SIZE = 1024
#   CROP_MODE = False
#
# Large 模式:
#   BASE_SIZE = 1280
#   IMAGE_SIZE = 1280
#   CROP_MODE = False
#
# Gundam 模式（推荐用于复杂文档）:
#   BASE_SIZE = 1024
#   IMAGE_SIZE = 640
#   CROP_MODE = True

# 加载本地图片
image_path = "BDRC/val_images/I1KG811270370_1.png"  # 修改为您的本地图片路径
image = Image.open(image_path).convert("RGB")

# 将图片转换为base64编码
buffered = io.BytesIO()
image.save(buffered, format="PNG")
image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
image_data_url = f"data:image/png;base64,{image_base64}"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url  # 使用本地图片的base64编码
                }
            },
            {
                "type": "text",
                "text": "Free OCR."
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-OCR",
    messages=messages,
    max_tokens=4096,
    temperature=0.0,
    extra_body={
        "skip_special_tokens": False,
        # args used to control custom logits processor
        "vllm_xargs": {
            "ngram_size": 30,
            "window_size": 90,
            # whitelist: <td>, </td>
            "whitelist_token_ids": [128821, 128822],
        },
    },
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")