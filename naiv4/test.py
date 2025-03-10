import requests
import random
import os
from datetime import datetime

# 配置信息
API_URL = "https://staging-image.novelai.net/ai/generate-image"
API_TOKEN = "pst-3Z3b89Pu32WYKBHwmzsh1JVrhxFPIZd6sjGqgrBclwbtp09x0DU8QsPr7qNrTJ7n"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}
OUTPUT_DIR = "./output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 最新请求体模板
BASE_PAYLOAD = {
    "input": "multiple girls, 2girls, loli, child, full body, indoors, bedroom, mother and daughter, shibari over clothes, blindfold, arknights, from side, bondage, bdsm, looking at another, blush, sitting, on back, pantyhose, mesugaki, yuri, lolidom, steam, hot, steaming body, femdom, nipple tweak through clothes, foot on head, smelling, smelling feet, foot worship, stepped on, foot on another's face, foot on breast, cheek press, cheek squash, light particles, depth of field, wide shot, sidelighting, light rays, shiny clothes, shiny skin, dark, night, tsurime, eyeliner, eyeshadow, eyelashes, upturned eyes, year 2023, year 2024, {{absurdres, amazing quality, very aesthetic}}, [[[tsubasa tsubasa, onineko, sho (sho lwlw), villainchin]]], [[chen bin, qizhu, ogipote, torino aqua]], [tianliang duohe fangdongye, bukurote, luode huayuan], yoneyama mai, fh moya, rolua, {otokoe}, ",
    "model": "custom",
    "action": "generate",
    "parameters": {
        "params_version": 3,
        "width": 1536,
        "height": 1024,
        "scale": 5,
        "sampler": "k_dpmpp_2s_ancestral",
        "steps": 50,
        "n_samples": 1,
        "ucPreset": 2,
        "qualityToggle": False,
        "autoSmea": False,
        "sm": True,
        "dynamic_thresholding": False,
        "controlnet_strength": 1,
        "legacy": False,
        "add_original_image": True,
        "cfg_rescale": 0,
        "noise_schedule": "karras",
        "legacy_v3_extend": False,
        "skip_cfg_above_sigma": 23.69030960605558,
        "use_coords": False,  # 关闭坐标使用
        "v4_prompt": {
            "caption": {
                "base_caption": "multiple girls, 2girls, loli, child, full body, indoors, bedroom, mother and daughter, shibari over clothes, blindfold, arknights, from side, bondage, bdsm, looking at another, blush, sitting, on back, pantyhose, mesugaki, yuri, lolidom, steam, hot, steaming body, femdom, nipple tweak through clothes, foot on head, smelling, smelling feet, foot worship, stepped on, foot on another's face, foot on breast, cheek press, cheek squash, light particles, depth of field, wide shot, sidelighting, light rays, shiny clothes, shiny skin, dark, night, tsurime, eyeliner, eyeshadow, eyelashes, upturned eyes, year 2023, year 2024, {{absurdres, amazing quality, very aesthetic}}, [[[tsubasa tsubasa, onineko, sho (sho lwlw), villainchin]]], [[chen bin, qizhu, ogipote, torino aqua]], [tianliang duohe fangdongye, bukurote, luode huayuan], yoneyama mai, fh moya, rolua, {otokoe}, ",
                "char_captions": [
                    {
                        "char_caption": "vulpisfoglia (arknights)",
                        "centers": [{"x": 0.5, "y": 0.5}]
                    },
                    {
                        "char_caption": "suzuran (arknights)",
                        "centers": [{"x": 0.5, "y": 0.5}]
                    }
                ]
            },
            "use_coords": False,
            "use_order": True
        },
        "v4_negative_prompt": {
            "caption": {
                "base_caption": "{{very displeasing, worst quality, bad quality}}, {displeasing, blurry, lowres, error}, film grain, scan artifacts, jpeg artifacts, chromatic aberration, multiple views, gigantic breasts, normal quality, everyone, 6+girls, grayscale, signature, logo, text, deformed, @_@, pov, ",
                "char_captions": [
                    {
                        "char_caption": "",
                        "centers": [{"x": 0.5, "y": 0.5}]
                    },
                    {
                        "char_caption": "",
                        "centers": [{"x": 0.5, "y": 0.5}]
                    }
                ]
            }
        },
        "seed": 1601595792,
        "characterPrompts": [
            {
                "prompt": "vulpisfoglia (arknights)",
                "uc": "",
                "center": {"x": 0.5, "y": 0.5}
            },
            {
                "prompt": "suzuran (arknights)",
                "uc": "",
                "center": {"x": 0.5, "y": 0.5}
            }
        ],
        "negative_prompt": "{{very displeasing, worst quality, bad quality}}, {displeasing, blurry, lowres, error}, film grain, scan artifacts, jpeg artifacts, chromatic aberration, multiple views, gigantic breasts, normal quality, everyone, 6+girls, grayscale, signature, logo, text, deformed, @_@, pov, ",
        "reference_image_multiple": [],
        "reference_information_extracted_multiple": [],
        "reference_strength_multiple": []
    },
    "url": "http://custom-exposed-full-test-anime-v4-staging.ai.svc.cluster.local/"
}

def main():
    try:
        num_requests = int(input("请输入要生成的图片数量: "))
    except ValueError:
        print("请输入有效的整数！")
        return

    for i in range(1, num_requests+1):
        try:
            # 深拷贝请求体
            payload = BASE_PAYLOAD.copy()
            payload["parameters"] = payload["parameters"].copy()
            payload["parameters"]["seed"] = random.randint(1, 9999999999)

            # 发送请求
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()

            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(OUTPUT_DIR, f"image_{timestamp}_{i}.zip")
            
            # 保存文件
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"成功生成第 {i}/{num_requests} 张图片，保存至：{filename}")

        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误（第 {i} 次）: {e.response.status_code}")
            if e.response.status_code == 429:
                print("触发速率限制，建议稍后重试")
        except requests.exceptions.RequestException as e:
            print(f"请求失败（第 {i} 次）: {str(e)}")
        except Exception as e:
            print(f"系统异常（第 {i} 次）: {str(e)}")

if __name__ == "__main__":
    main()      