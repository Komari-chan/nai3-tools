import requests
import base64
import random
import json
import os
import time
from io import BytesIO
from zipfile import ZipFile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from gradio_client import Client, handle_file
from tiled_tools import resize_image, crop_image_with_overlap, merge_images_with_alpha

# NovelAI API 的 URL
NOVELAI_API_URL = "https://image.novelai.net/ai/generate-image"
WD_TAGGER_API = "SmilingWolf/wd-tagger"

# 初始化 wd-swinv2-tagger 模型客户端
wd_client = Client(WD_TAGGER_API)

# 创建输出目录
os.makedirs("output/gradients", exist_ok=True)
os.makedirs("output/reworked", exist_ok=True)

# 记录时间的装饰器
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 耗时: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

# 将图片转为 base64 格式
def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 解压缩 ZIP 文件并获取其中的图像
def extract_image_from_zip(zip_content):
    with ZipFile(BytesIO(zip_content)) as zip_file:
        for file_info in zip_file.infolist():
            with zip_file.open(file_info) as img_file:
                img = Image.open(BytesIO(img_file.read()))
                return img

# 使用 wd tagger 模型获取图像的提示词
@timer
def generate_prompts_for_image(crop_image, idx):
    temp_image_path = f"output/temp_crop_{idx + 1}.png"
    crop_image.save(temp_image_path)

    result = wd_client.predict(
        image=handle_file(temp_image_path),
        model_repo="SmilingWolf/wd-swinv2-tagger-v3",
        general_thresh=0.25,
        general_mcut_enabled=False,
        character_thresh=0.9,
        character_mcut_enabled=False,
        api_name="/predict"
    )
    prompts = result[0].split(", ")
    return ", ".join(prompts)

# 生成 NovelAI 请求的 JSON 数据
def generate_novelai_payload(input_image_base64, positive_prompt, negative_prompt, width, height, scale, sampler, steps, strength, noise, seed=None):
    if seed is None:
        seed = random.randint(1000000000, 9999999999)

    return {
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
            "negative_prompt": negative_prompt  # 保持默认负面提示词
        }
    }

# 调用 NovelAI 处理每个裁剪的图像块，增加重试机制
@timer
def novelai_img2img(image, positive_prompt, negative_prompt, width, height, scale, sampler, steps, strength, noise, api_key, idx, retries=3):
    input_image_base64 = img_to_base64(image)
    payload = generate_novelai_payload(input_image_base64, positive_prompt, negative_prompt, width, height, scale, sampler, steps, strength, noise)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    while retries > 0:
        try:
            response = requests.post(NOVELAI_API_URL, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                try:
                    reworked_img = extract_image_from_zip(response.content)
                    reworked_img.save(f"output/reworked/reworked_{idx + 1}.png")
                    return reworked_img
                except Exception as e:
                    print(f"无法提取图像: {e}")
                    return image
            else:
                print(f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
                retries -= 1
                print(f"重试剩余次数: {retries}")
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"ChunkedEncodingError: {e}")
            retries -= 1
            print(f"重试剩余次数: {retries}")
    return image

# 并行处理 WD tagger 和 NovelAI 重绘
def process_crop_for_wd_and_novelai(crop, positive_prompt, negative_prompt, idx, api_key, scale, sampler, steps, strength, noise):
    print(f"正在为第 {idx + 1} 块图像生成提示词...")
    auto_prompts = generate_prompts_for_image(crop, idx)
    final_positive_prompt = f"{positive_prompt}, {auto_prompts}"

    print(f"正在为第 {idx + 1} 块图像重绘...")
    reworked_img = novelai_img2img(crop, final_positive_prompt, negative_prompt, crop.width, crop.height, scale, sampler, steps, strength, noise, api_key, idx)

    return reworked_img

# 按顺序处理每个裁剪后的图像，并并行获取提示词和重绘
@timer
def process_image_with_novelai(image_path, positive_prompt, negative_prompt, scale, sampler, steps, strength, noise, api_key, use_wd=False):
    image = Image.open(image_path)
    original_size = (image.width, image.height)

    print("正在放大图片...")
    image = resize_image(image)

    print("正在裁剪图片...")
    crops = crop_image_with_overlap(image, original_size)

    reworked_crops = []
    with ThreadPoolExecutor(max_workers=2) as wd_executor, ThreadPoolExecutor(max_workers=2) as novelai_executor:
        wd_futures = {}
        for idx, crop in enumerate(crops):
            wd_future = wd_executor.submit(generate_prompts_for_image, crop, idx)
            wd_futures[wd_future] = idx

        for wd_future in as_completed(wd_futures):
            idx = wd_futures[wd_future]
            auto_prompts = wd_future.result()
            final_positive_prompt = f"{positive_prompt}, {auto_prompts}"
            print(f"正在为第 {idx + 1} 块图像重绘...")
            novelai_future = novelai_executor.submit(novelai_img2img, crops[idx], final_positive_prompt, negative_prompt, crops[idx].width, crops[idx].height, scale, sampler, steps, strength, noise, api_key, idx)
            reworked_crops.append(novelai_future.result())

    print("正在拼接图片...")
    final_image = merge_images_with_alpha(reworked_crops, image.size, original_size, overlap_percent=0.25)

    final_image.save("output/final_image_with_novelai_and_auto_prompts.png")
    print("图像处理完成，已保存 final_image_with_novelai_and_auto_prompts.png")

# 调用函数示例
if __name__ == "__main__":
    # 输入参数
    input_img_path = "input_image.png"  
    positive_prompt = "{{koshigaya komari}}, [[[[non non biyori]]]],[1=2],[sheya],onineko,blender_(medium),qizhu,freng, [olchas],torino_aqua,arsenixc,icecake,alchemaniac,fajyobore,[suerte],dk.senie,kase_daiki,agoto,reoen,colorful, high contrast, light rays, year 2023"
    negative_prompt = "nsfw, lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    scale = 5
    sampler = "k_dpmpp_2m_sde"
    steps = 28
    strength = 0.35
    noise = 0
    api_key = "pst-Xl7JptkKqyg2Y6rPhYnBjdJpzDk2IPUmiG8VGNDpC2qWGCnZTJD0GJcRfZKkxQXl"  # 替换为你的 API 密钥
    use_wd = True  # 设置为 True 启用 wd tagger，设置为 False 使用默认提示词

    # 正确的函数调用
    process_image_with_novelai(
        input_img_path, positive_prompt, negative_prompt,
        scale, sampler, steps, strength, noise, api_key, use_wd
    )

