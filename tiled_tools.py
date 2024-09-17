from PIL import Image, ImageFilter, ImageChops
import numpy as np
import os

# 确保保存图像的输出目录存在
os.makedirs("output/crops", exist_ok=True)
os.makedirs("output/gradients", exist_ok=True)

def resize_image(image, scale_factor=2.5):
    # 先将图片放大 2.5 倍
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    return image.resize(new_size, Image.LANCZOS)

def crop_image_with_overlap(image, original_size, overlap=0.25):
    # 根据给定的原始大小和重叠比例进行裁剪
    crops = []
    step_x = int(original_size[0] * (1 - overlap))  # 水平方向步长
    step_y = int(original_size[1] * (1 - overlap))  # 垂直方向步长

    for i in range(3):
        for j in range(3):
            left = j * step_x
            upper = i * step_y
            right = left + original_size[0]
            lower = upper + original_size[1]

            # 裁剪并保存区域
            crop = image.crop((left, upper, right, lower))
            crop.save(f"output/crops/crop_{i}_{j}.png")  # 导出每个裁切的图像
            crops.append(crop)

    return crops

def create_alpha_gradient_mask(image_size, overlap_percent, fade_left=False, fade_top=False):
    # 创建一个边缘渐变的alpha通道，覆盖重叠区域
    width, height = image_size
    mask = Image.new("L", (width, height), 255)
    mask_data = np.array(mask)

    # 渐变处理 (左侧)
    if fade_left:
        gradient = np.linspace(0, 255, int(width * overlap_percent))
        for i in range(len(gradient)):
            mask_data[:, i] = np.minimum(mask_data[:, i], int(gradient[i]))

    # 渐变处理 (上侧)
    if fade_top:
        gradient = np.linspace(0, 255, int(height * overlap_percent))
        for i in range(len(gradient)):
            mask_data[i, :] = np.minimum(mask_data[i, :], int(gradient[i]))

    return Image.fromarray(mask_data)

def merge_images_with_alpha(images, image_size, original_size, overlap_percent):
    # 创建一个空白的大图，尺寸与放大后的图片相同
    base_image = Image.new("RGBA", image_size)

    step_x = int(original_size[0] * (1 - overlap_percent))
    step_y = int(original_size[1] * (1 - overlap_percent))

    positions = [
        (0, 0), (step_x, 0), (2 * step_x, 0),  # 第一行
        (0, step_y), (step_x, step_y), (2 * step_x, step_y),  # 第二行
        (0, 2 * step_y), (step_x, 2 * step_y), (2 * step_x, 2 * step_y)  # 第三行
    ]

    # 对每个裁剪的图像进行拼接，并根据位置进行渐变处理
    for index, (x_offset, y_offset) in enumerate(positions):
        img = images[index].convert("RGBA")

        # 创建渐变掩码
        if index == 0:
            # 左上角，不做任何渐变
            alpha_mask = None
        elif index in [1, 2]:  # 中上和右上块
            alpha_mask = create_alpha_gradient_mask(img.size, overlap_percent, fade_left=True)
        elif index in [3, 6]:  # 左中和左下块
            alpha_mask = create_alpha_gradient_mask(img.size, overlap_percent, fade_top=True)
        elif index == 4:  # 中中块
            alpha_mask = create_alpha_gradient_mask(img.size, overlap_percent, fade_left=True, fade_top=True)
        else:  # 其他块
            alpha_mask = create_alpha_gradient_mask(img.size, overlap_percent, fade_left=True, fade_top=True)

        if alpha_mask:
            img.putalpha(alpha_mask)

        # 保存渐变处理后的图像
        img.save(f"output/gradients/gradient_{index}.png")

        # 将图块合成到基础图像
        base_image.alpha_composite(img, (x_offset, y_offset))

    return base_image

# def process_image(image_path):
#     # 打开并处理图像
#     image = Image.open(image_path)
#     original_size = (image.width, image.height)  # 保存原始尺寸
    
#     # 第一步：先放大图片 2.5 倍
#     image = resize_image(image)

#     # 第二步：裁剪成9个块，每块的大小与原图一致
#     crops = crop_image_with_overlap(image, original_size)

#     # 第三步：对每个块进行图生图处理 (此处省略图生图的部分)

#     # 第四步：拼接并应用渐变处理，重叠区域为原图大小的 0.25
#     final_image = merge_images_with_alpha(crops, image.size, original_size, overlap_percent=0.25)

#     # 保存最终结果
#     final_image.save("final_image_with_alpha.png")

#     print("图像处理完成，已保存 final_image_with_alpha.png")

# # 示例使用
# process_image("input_image.png")
