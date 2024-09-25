import os
from PIL import Image, ImageChops
import numpy as np

# 设置输入图片路径和输出目录
INPUT_IMAGE_PATH = "input_image.png"
OUTPUT_DIR = "output"
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
GRADIENTS_DIR = os.path.join(OUTPUT_DIR, "gradients")
FINAL_IMAGE_PATH = os.path.join(OUTPUT_DIR, "final_image.png")

# 创建输出目录
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(GRADIENTS_DIR, exist_ok=True)

# 图像处理函数
def crop_image(image, crop_width, crop_height, overlap_x, overlap_y):
    """裁剪图像为指定大小的块，并保存重叠部分"""
    crops = []
    width, height = image.size

    # 水平方向步长
    step_x = crop_width - overlap_x
    # 垂直方向步长
    step_y = crop_height - overlap_y

    # 遍历裁剪矩阵
    for row in range(8):  # 垂直方向分成8块
        for col in range(4):  # 水平方向分成4块
            left = col * step_x
            upper = row * step_y
            right = left + crop_width
            lower = upper + crop_height

            # 确保裁剪区域不会超出图像范围
            if right > width:
                right = width
                left = width - crop_width
            if lower > height:
                lower = height
                upper = height - crop_height

            # 裁剪图像
            crop = image.crop((left, upper, right, lower))
            crop_filename = os.path.join(CROPS_DIR, f"crop_{col}_{row}.png")
            crop.save(crop_filename)
            crops.append(crop)

    return crops

def create_gradient_mask(crop_size, overlap_x, overlap_y, fade_left=False, fade_top=False):
    """创建渐变遮罩，用于图像合并时的平滑处理"""
    width, height = crop_size
    mask = Image.new("L", (width, height), 255)  # 默认全不透明
    mask_data = np.array(mask)

    # 左侧渐变
    if fade_left:
        fade_x = np.linspace(0, 255, overlap_x)
        for x in range(overlap_x):
            mask_data[:, x] = fade_x[x]

    # 上侧渐变
    if fade_top:
        fade_y = np.linspace(0, 255, overlap_y)
        for y in range(overlap_y):
            mask_data[y, :] = np.minimum(mask_data[y, :], fade_y[y])

    return Image.fromarray(mask_data)

def merge_images_with_gradient(crops, crop_width, crop_height, overlap_x, overlap_y, grid_size):
    """将图像块进行合并，使用渐变遮罩平滑处理重叠区域"""
    merged_width = crop_width * grid_size[0] - overlap_x * (grid_size[0] - 1)
    merged_height = crop_height * grid_size[1] - overlap_y * (grid_size[1] - 1)
    merged_image = Image.new("RGB", (merged_width, merged_height))

    for idx, crop in enumerate(crops):
        col = idx % grid_size[0]  # 水平方向的列
        row = idx // grid_size[0]  # 垂直方向的行

        # 创建对应块的渐变遮罩
        if col == 0 and row == 0:
            # 左上角第一块，不加渐变
            gradient_mask = None
        elif col == 0:
            # 第一列剩下的块，上侧渐变
            gradient_mask = create_gradient_mask((crop_width, crop_height), overlap_x, overlap_y, fade_top=True)
        elif row == 0:
            # 第一排剩下的块，左侧渐变
            gradient_mask = create_gradient_mask((crop_width, crop_height), overlap_x, overlap_y, fade_left=True)
        else:
            # 其余块，左侧和上侧都渐变
            gradient_mask = create_gradient_mask((crop_width, crop_height), overlap_x, overlap_y, fade_left=True, fade_top=True)

        # 保存渐变处理后的图像
        gradient_crop = crop.convert("RGBA")
        if gradient_mask:
            gradient_crop.putalpha(gradient_mask)
            gradient_filename = os.path.join(GRADIENTS_DIR, f"gradient_{idx}.png")
            gradient_crop.save(gradient_filename)

        x_offset = col * (crop_width - overlap_x)
        y_offset = row * (crop_height - overlap_y)
        merged_image.paste(gradient_crop.convert("RGB"), (x_offset, y_offset), gradient_mask)

    return merged_image

def process_image(image_path):
    """主函数，执行图像的裁剪、渐变处理和拼接"""
    # 打开图片，图片尺寸为 2880x8960
    image = Image.open(image_path)

    # 定义裁剪参数
    crop_width = 832
    crop_height = 1216
    overlap_x = 150  # 水平方向重叠部分（20% 重叠比例为832*0.2 = 166）
    overlap_y = 110  # 垂直方向重叠部分（20% 重叠比例为1216*0.2 = 243）

    # 裁剪图像
    print("正在裁剪图像...")
    crops = crop_image(image, crop_width, crop_height, overlap_x, overlap_y)

    # 合并图像
    print("正在合并图像...")
    final_image = merge_images_with_gradient(crops, crop_width, crop_height, overlap_x, overlap_y, grid_size=(4, 8))

    # 保存最终合并的图像
    final_image.save(FINAL_IMAGE_PATH)
    print(f"图像处理完成，已保存 {FINAL_IMAGE_PATH}")

# 运行主函数
if __name__ == "__main__":
    process_image(INPUT_IMAGE_PATH)
