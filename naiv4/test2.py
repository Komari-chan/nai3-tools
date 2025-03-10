import requests
import random
import os
import zipfile
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置信息
API_URL = "https://staging-image.novelai.net/ai/generate-image"
API_TOKEN = "pst-3Z3b89Pu32WYKBHwmzsh1JVrhxFPIZd6sjGqgrBclwbtp09x0DU8QsPr7qNrTJ7n"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}
OUTPUT_DIR = "./output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 基础请求体模板
BASE_PAYLOAD = {
    "input": "1girl, loli, child, aged down, mon-musu quest!, mon-musu quest: paradox, blonde hair, angel wings, barefoot, lens flare, light particles, depth of field, wide shot, sidelighting, light rays, shiny clothes, shiny skin, dark, tsurime, eyeliner, eyeshadow, eyelashes, upturned eyes, year 2023, year 2024, ilias, {{absurdres, amazing quality, very aesthetic}}, lucifina (mon-musu quest!), [tianliang duohe fangdongye], [ogipote], [torino aqua], bukurote, yoneyama mai, fh moya, rolua, {otokoe}, ",
    "model": "custom",
    "action": "generate",
    "parameters": {
        "params_version": 3,
        "width": 1024,
        "height": 1536,
        "scale": 5,
        "sampler": "k_dpmpp_2s_ancestral",
        "steps": 50,
        "n_samples": 1,
        "ucPreset": 2,
        "qualityToggle": False,
        "dynamic_thresholding": True,
        "controlnet_strength": 1,
        "legacy": False,
        "add_original_image": True,
        "cfg_rescale": 0,
        "noise_schedule": "karras",
        "legacy_v3_extend": False,
        "skip_cfg_above_sigma": 23.69030960605558,
        "use_coords": True,
        "v4_prompt": {
            "caption": {
                "base_caption": "1girl, loli, child, aged down, mon-musu quest!, mon-musu quest: paradox, blonde hair, angel wings, barefoot, lens flare, light particles, depth of field, wide shot, sidelighting, light rays, shiny clothes, shiny skin, dark, tsurime, eyeliner, eyeshadow, eyelashes, upturned eyes, year 2023, year 2024, ilias, {{absurdres, amazing quality, very aesthetic}}, lucifina (mon-musu quest!), [tianliang duohe fangdongye], [ogipote], [torino aqua], bukurote, yoneyama mai, fh moya, rolua, {otokoe}, ",
                "char_captions": []
            },
            "use_coords": True,
            "use_order": True
        },
        "v4_negative_prompt": {
            "caption": {
                "base_caption": "{{very displeasing, worst quality, bad quality}}, {displeasing, blurry, lowres, error}, film grain, scan artifacts, jpeg artifacts, chromatic aberration, multiple views, gigantic breasts, normal quality, multiple boys, male focus, everyone, 6+girls, grayscale, signature, logo, text, deformed, @_@, pov, ",
                "char_captions": []
            }
        },
        "characterPrompts": [],
        "negative_prompt": "{{very displeasing, worst quality, bad quality}}, {displeasing, blurry, lowres, error}, film grain, scan artifacts, jpeg artifacts, chromatic aberration, multiple views, gigantic breasts, normal quality, multiple boys, male focus, everyone, 6+girls, grayscale, signature, logo, text, deformed, @_@, pov, ",
        "reference_image_multiple": [],
        "reference_information_extracted_multiple": [],
        "reference_strength_multiple": []
    },
    "url": "http://custom-exposed-full-anime-v4-staging.ai.svc.cluster.local/"
}

def process_image(index):
    try:
        # 生成唯一种子
        payload = BASE_PAYLOAD.copy()
        payload["parameters"] = {**payload["parameters"], "seed": random.randint(1, 9999999999)}

        # 发送请求
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()

        # 处理ZIP文件
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # 获取第一个PNG文件
            png_files = [f for f in zip_ref.namelist() if f.lower().endswith('.png')]
            if not png_files:
                raise ValueError("ZIP文件中未找到PNG文件")
            
            # 提取并保存PNG
            png_data = zip_ref.read(png_files[0])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
            filename = f"image_{timestamp}_n{index}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            with open(save_path, "wb") as f:
                f.write(png_data)

        return f"成功生成 {filename}"
    
    except Exception as e:
        return f"请求 {index} 失败: {str(e)}"

def main():
    try:
        num_requests = int(input("请输入要生成的图片数量: "))
    except ValueError:
        print("请输入有效的整数！")
        return

    # 使用线程池并行处理（每轮10个）
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_image, i+1) for i in range(num_requests)]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"处理异常: {str(e)}")

if __name__ == "__main__":
    main()