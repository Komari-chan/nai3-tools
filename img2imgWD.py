import requests
import base64
import random
import json
import os
from io import BytesIO
from zipfile import ZipFile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed  # 并行处理
from gradio_client import Client, handle_file  # 引入 wd-swinv2-tagger API 客户端
from tiled_tools import resize_image, crop_image_with_overlap, merge_images_with_alpha

# NovelAI API 的 URL
NOVELAI_API_URL = "https://image.novelai.net/ai/generate-image"
WD_TAGGER_API = "SmilingWolf/wd-tagger"  # wd tagger 模型

# 初始化 wd-swinv2-tagger 模型客户端
wd_client = Client(WD_TAGGER_API)

# 创建 output/temp 目录来存储缓存图片
os.makedirs("output/temp", exist_ok=True)

# 将图片转为 base64 格式
def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

# 解压缩 ZIP 文件并获取其中的图像
def extract_image_from_zip(zip_content):
    with ZipFile(BytesIO(zip_content)) as zip_file:
        for file_info in zip_file.infolist():
            with zip_file.open(file_info) as img_file:
                img = Image.open(BytesIO(img_file.read()))
                return img  # 假设 ZIP 文件中只有一个图像

# 使用 wd tagger 模型获取图像的提示词
def generate_prompts_for_image(image_path):
    result = wd_client.predict(
        image=handle_file(image_path),
        model_repo="SmilingWolf/wd-swinv2-tagger-v3",
        general_thresh=0.3,
        general_mcut_enabled=False,
        character_thresh=1,
        character_mcut_enabled=False,
        api_name="/predict"
    )
    prompts = result[0].split(", ")  # 解析并提取提示词部分
    return ", ".join(prompts)

# 生成 NovelAI 请求的 JSON 数据
def generate_novelai_payload(input_image_base64, positive_prompt, negative_prompt, width, height, scale, sampler, steps, strength, noise, seed=None):
    if seed is None:
        seed = random.randint(1000000000, 9999999999)

    payload = {
        "input": positive_prompt,
        "model": "nai-diffusion-3",
        "action": "img2img",
        "parameters": {
            "params_version": 1,
            "width": width,
            "height": height,
            "scale": scale,
            "sampler": sampler,
            "steps": steps,
            "n_samples": 1,
            "strength": strength,
            "noise": noise,
            "ucPreset": 0,
            "qualityToggle": True,
            "sm": False,
            "sm_dyn": False,
            "dynamic_thresholding": True,
            "controlnet_strength": 1,
            "legacy": False,
            "add_original_image": True,
            "cfg_rescale": 0,
            "noise_schedule": "karras",
            "legacy_v3_extend": False,
            "skip_cfg_above_sigma": 19,
            "seed": seed,
            "image": input_image_base64,
            "extra_noise_seed": seed,
            "negative_prompt": negative_prompt
        }
    }

    return payload

# 调用 NovelAI 处理每个裁剪的图像块
def novelai_img2img(image, positive_prompt, negative_prompt, width, height, scale, sampler, steps, strength, noise, api_key, idx):
    # 将输入图片转换为 base64
    input_image_base64 = img_to_base64(image)

    # 生成请求的 JSON 数据
    payload = generate_novelai_payload(
        input_image_base64, positive_prompt, negative_prompt,
        width, height, scale, sampler, steps, strength, noise
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 发送 POST 请求到 NovelAI API
    response = requests.post(NOVELAI_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        try:
            # 尝试从返回的 ZIP 文件中提取图像并保存
            reworked_img = extract_image_from_zip(response.content)
            if reworked_img:
                reworked_img.save(f"output/temp/novelai_crop_{idx + 1}.png")
            return reworked_img
        except Exception as e:
            print(f"无法从 ZIP 文件中提取图像: {e}")
            return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        return None

# 并行处理 NovelAI 和 WD tagger 的函数
def process_crop_with_wd_and_novelai(crop, idx, use_wd, positive_prompt, negative_prompt, scale, sampler, steps, strength, noise, api_key):
    temp_image_path = f"output/temp/crop_{idx + 1}.png"
    crop.save(temp_image_path)  # 保存临时图片到 temp 目录

    # 获取 WD tagger 提示词
    if use_wd:
        print(f"正在为第 {idx + 1} 块图像生成提示词...")
        auto_prompts = generate_prompts_for_image(temp_image_path)
        final_positive_prompt = f"{positive_prompt}, {auto_prompts}"
    else:
        final_positive_prompt = positive_prompt

    # 使用 NovelAI 重绘图像
    print(f"正在为第 {idx + 1} 块图像重绘...")
    reworked_img = novelai_img2img(crop, final_positive_prompt, negative_prompt, crop.width, crop.height, scale, sampler, steps, strength, noise, api_key, idx)

    # 返回重绘后的图像，如果失败则返回原图像
    return reworked_img if reworked_img else crop

# 将图片分块后发送给 NovelAI 重绘并拼接
def process_image_with_novelai(image_path, positive_prompt, negative_prompt, scale, sampler, steps, strength, noise, api_key, use_wd=False):
    # 打开并处理图像
    image = Image.open(image_path)
    original_size = (image.width, image.height)  # 保存原始尺寸

    # 第一步：先放大图片 2.5 倍
    print("正在放大图片...")
    image = resize_image(image)

    # 第二步：裁剪成9个块，每块的大小与原图一致
    print("正在裁剪图片...")
    crops = crop_image_with_overlap(image, original_size)

    # 第三步：并行处理每个块，使用 WD tagger 获取提示词并调用 NovelAI 重绘
    reworked_crops = [None] * 9  # 保持裁剪块的顺序
    with ThreadPoolExecutor(max_workers=4) as executor:  # 并行执行任务
        futures = {
            executor.submit(process_crop_with_wd_and_novelai, crop, idx, use_wd, positive_prompt, negative_prompt, scale, sampler, steps, strength, noise, api_key): idx
            for idx, crop in enumerate(crops)
        }

        for future in as_completed(futures):
            idx = futures[future]
            reworked_crops[idx] = future.result()  # 按照正确顺序保存结果

    # 第四步：拼接并应用渐变处理
    print("正在拼接图片...")
    final_image = merge_images_with_alpha(reworked_crops, image.size, original_size, overlap_percent=0.25)

    # 保存最终结果
    final_image.save("output/final_image_with_novelai_and_auto_prompts.png")
    print("图像处理完成，已保存 output/final_image_with_novelai_and_auto_prompts.png")

# 调用函数示例
if __name__ == "__main__":
    # 输入参数
    input_img_path = "input_image.png"  
    positive_prompt = "koshigaya komari, icecake, [olchas], fajyobore, agoto, dk.senie, reoen, qizhu, [sheya], torino_aqua, arsenixc, kase_daiki, blender_(medium), freng, alchemaniac, onineko, [1=2], [suerte], colorful, high contrast, light rays, multicolored background, flower, butterfly, dark, floral background, night"
    negative_prompt = "nsfw, lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    scale = 5
    sampler = "k_dpmpp_2m_sde"
    steps = 28
    strength = 0.3
    noise = 0
    api_key = "pst-Xl7JptkKqyg2Y6rPhYnBjdJpzDk2IPUmiG8VGNDpC2qWGCnZTJD0GJcRfZKkxQXl"  # 替换为你的 API 密钥
    use_wd = True  # 设置为 True 启用 wd tagger，设置为 False 使用默认提示词

    # 正确的函数调用
    process_image_with_novelai(
        input_img_path, positive_prompt, negative_prompt,
        scale, sampler, steps, strength, noise, api_key, use_wd
    )

