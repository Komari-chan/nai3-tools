import os
import time
from PIL import Image
from img2imgWD import process_image_with_novelai

# 设置输入和输出目录
INPUT_DIR = "input"
OUTPUT_DIR = "output/final"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def batch_process_images(input_dir, output_dir, positive_prompt, negative_prompt, scale, sampler, steps, strength, noise, noise_schedule, api_key, general_thresh, character_thresh, use_wd):
    # 扫描输入目录下的所有图片
    images = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not images:
        print("输入目录中没有找到任何图片。")
        return

    for idx, image_file in enumerate(images):
        input_image_path = os.path.join(input_dir, image_file)
        output_image_path = os.path.join(output_dir, f"final_{idx + 1}_{os.path.splitext(image_file)[0]}.png")

        if os.path.exists(output_image_path):
            print(f"图片 {output_image_path} 已存在，跳过处理。")
            continue

        print(f"正在处理第 {idx + 1} 张图片: {image_file}")

        try:
            process_image_with_novelai(
                input_image_path, positive_prompt, negative_prompt,
                scale, sampler, steps, strength, noise, noise_schedule,
                api_key, general_thresh, character_thresh, use_wd
            )
            # 将处理后的图片移到 final 目录
            processed_image_path = "output/final_image_with_novelai_and_auto_prompts.png"
            if os.path.exists(processed_image_path):
                os.rename(processed_image_path, output_image_path)
                print(f"图片已处理并保存至: {output_image_path}")
            else:
                print(f"处理完成但未找到预期输出文件 {processed_image_path}。")
        except Exception as e:
            print(f"处理图片 {image_file} 时发生错误: {e}")

        time.sleep(1)  # 避免过快请求服务器

    print("所有图片处理完成。")

if __name__ == "__main__":
    positive_prompt = "{{momoi_(blue_archive)}}, maccha_(mochancc), {au_(d_elete)}, {mana_(remana)}, {fuzichoco}, {momoko_(momopoco)}, {nababa}, {mafuyu_(chibi21)}, {rin_yuu}, {sheya}, [dk.senie], [kedama_milk], [[agoto]], fkey, year 2023"
    negative_prompt = "nsfw, lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    scale = 5
    sampler = "k_dpmpp_2m_sde"  
    steps = 28
    strength = 0.35
    noise = 0
    noise_schedule = "karras"  
    api_key = "pst-UeGZGd8GDV7utKhemgm8tvcHxtapFaPFcrtaQq4fIZGAniu7KdsH3JqMf3c38H0R"
    general_thresh = 0.3  
    character_thresh = 0.9  
    use_wd = True

    # 运行批量处理
    batch_process_images(
        INPUT_DIR, OUTPUT_DIR, positive_prompt, negative_prompt,
        scale, sampler, steps, strength, noise, noise_schedule,
        api_key, general_thresh, character_thresh, use_wd
    )
