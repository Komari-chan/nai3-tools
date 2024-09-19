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
def generate_prompts_for_image(crop_image, idx, general_thresh, character_thresh):
    temp_image_path = f"output/temp_crop_{idx + 1}.png"
    crop_image.save(temp_image_path)

    result = wd_client.predict(
        image=handle_file(temp_image_path),
        model_repo="SmilingWolf/wd-swinv2-tagger-v3",
        general_thresh=general_thresh,
        general_mcut_enabled=False,
        character_thresh=character_thresh,
        character_mcut_enabled=False,
        api_name="/predict"
    )
    prompts = result[0].split(", ")
    return ", ".join(prompts)

# 生成 NovelAI 请求的 JSON 数据
def generate_novelai_payload(input_image_base64, positive_prompt, negative_prompt, width, height, scale, sampler, steps, strength, noise, noise_schedule, seed=None):
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
            "noise_schedule": noise_schedule,
            "legacy_v3_extend": False,
            "skip_cfg_above_sigma": 19,
            "seed": seed,
            "image": input_image_base64,
            "extra_noise_seed": seed,
            "negative_prompt": negative_prompt
        }
    }

# 调用 NovelAI 处理每个裁剪的图像块
@timer
def novelai_img2img(image, positive_prompt, negative_prompt, width, height, scale, sampler, steps, strength, noise, noise_schedule, api_key, idx, retries=10):
    input_image_base64 = img_to_base64(image)
    payload = generate_novelai_payload(input_image_base64, positive_prompt, negative_prompt, width, height, scale, sampler, steps, strength, noise, noise_schedule)

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

# 顺序处理每个裁剪后的图像，确保提示词获取和重绘按正确顺序执行
@timer
def process_image_with_novelai(image_path, positive_prompt, negative_prompt, scale, sampler, steps, strength, noise, noise_schedule, api_key, general_thresh, character_thresh, use_wd=False):
    image = Image.open(image_path)
    original_size = (image.width, image.height)

    print("正在放大图片...")
    image = resize_image(image)

    print("正在裁剪图片...")
    crops = crop_image_with_overlap(image, original_size)

    reworked_crops = []
    wd_prompts = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        # 首先为第一个分块生成提示词
        print(f"正在为第 1 块图像生成提示词...")
        auto_prompt = generate_prompts_for_image(crops[0], 0, general_thresh, character_thresh)
        wd_prompts.append(auto_prompt)

        # 为第一个分块执行 NovelAI 重绘，同时为第二个分块生成提示词
        print(f"正在为第 1 块图像重绘...")
        novelai_future = executor.submit(novelai_img2img, crops[0], f"{positive_prompt}, {auto_prompt}", negative_prompt, crops[0].width, crops[0].height, scale, sampler, steps, strength, noise, noise_schedule, api_key, 0)
        
        print(f"正在为第 2 块图像生成提示词...")
        wd_future = executor.submit(generate_prompts_for_image, crops[1], 1, general_thresh, character_thresh)

        reworked_crops.append(novelai_future.result())
        wd_prompts.append(wd_future.result())

        # 之后的每个分块都按顺序处理
        for idx in range(1, len(crops) - 1):
            # 开始重绘当前块
            print(f"正在为第 {idx + 1} 块图像重绘...")
            novelai_future = executor.submit(novelai_img2img, crops[idx], f"{positive_prompt}, {wd_prompts[idx]}", negative_prompt, crops[idx].width, crops[idx].height, scale, sampler, steps, strength, noise, noise_schedule, api_key, idx)
            
            # 同时为下一块生成提示词
            print(f"正在为第 {idx + 2} 块图像生成提示词...")
            wd_future = executor.submit(generate_prompts_for_image, crops[idx + 1], idx + 1, general_thresh, character_thresh)

            reworked_crops.append(novelai_future.result())
            wd_prompts.append(wd_future.result())

        # 为最后一块重绘
        print(f"正在为第 {len(crops)} 块图像重绘...")
        reworked_crops.append(novelai_img2img(crops[-1], f"{positive_prompt}, {wd_prompts[-1]}", negative_prompt, crops[-1].width, crops[-1].height, scale, sampler, steps, strength, noise, noise_schedule, api_key, len(crops) - 1))

    print("正在拼接图片...")
    final_image = merge_images_with_alpha(reworked_crops, image.size, original_size, overlap_percent=0.25)

    final_image.save("output/final_image_with_novelai_and_auto_prompts.png")
    print("图像处理完成，已保存 final_image_with_novelai_and_auto_prompts.png")
# 调用函数示例

if __name__ == "__main__":
    input_img_path = "input_image.png"
    positive_prompt = "koshigaya komari, [[[[non non biyori]]]], tab head, torino aqua, ogipote, 58 (opal 00 58), sheya, wlop, ciloranko, sho (sho lwlw), reoen, ask (askzy), as109, ye jji, noyu (noyu23386566), konya karasue, kedama milk, u u zan,colorful, high contrast, light rays, year 2023"
    negative_prompt = "nsfw, lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    scale = 5
    sampler = "k_dpmpp_2m_sde"  # 用户可以选择的采样器
    steps = 28
    strength = 0.35
    noise = 0
    noise_schedule = "karras"  # 用户可以选择的 noise_schedule
    api_key = "pst-Xl7JptkKqyg2Y6rPhYnBjdJpzDk2IPUmiG8VGNDpC2qWGCnZTJD0GJcRfZKkxQXl"
    general_thresh = 0.3  # 用户输入的 wd general_thresh
    character_thresh = 0.9  # 用户输入的 wd character_thresh
    use_wd = True

    process_image_with_novelai(
        input_img_path, positive_prompt, negative_prompt,
        scale, sampler, steps, strength, noise, noise_schedule, api_key,
        general_thresh, character_thresh, use_wd
    )