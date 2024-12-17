import onnxruntime as rt
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import os
import time
import base64
from concurrent.futures import ThreadPoolExecutor
from tiled_tools import resize_image, crop_image_with_overlap, merge_images_with_alpha

# 模型路径
MODEL_DIR = "wd-swinv2-tagger-v3"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
LABEL_PATH = os.path.join(MODEL_DIR, "selected_tags.csv")

# 初始化模型
class LocalPredictor:
    def __init__(self, model_path, label_path):
        self.model = rt.InferenceSession(model_path)
        self.labels = self.load_labels(label_path)
        _, self.model_target_size, _, _ = self.model.get_inputs()[0].shape

    @staticmethod
    def load_labels(label_path):
        tags_df = pd.read_csv(label_path)
        return {
            "names": tags_df["name"].tolist(),
            "rating_indexes": list(np.where(tags_df["category"] == 9)[0]),
            "general_indexes": list(np.where(tags_df["category"] == 0)[0]),
            "character_indexes": list(np.where(tags_df["category"] == 4)[0]),
        }

    def prepare_image(self, image):
        # 调整图像为正方形并适配模型输入
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != self.model_target_size:
            padded_image = padded_image.resize(
                (self.model_target_size, self.model_target_size), Image.BICUBIC
            )

        image_array = np.asarray(padded_image, dtype=np.float32)[:, :, ::-1]
        return np.expand_dims(image_array, axis=0)

    def predict(self, image, general_thresh=0.35, character_thresh=0.85):
        input_data = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        preds = self.model.run([output_name], {input_name: input_data})[0]

        labels = list(zip(self.labels["names"], preds[0].astype(float)))

        general_res = {
            label: score
            for label, score in [labels[i] for i in self.labels["general_indexes"]]
            if score > general_thresh
        }

        character_res = {
            label: score
            for label, score in [labels[i] for i in self.labels["character_indexes"]]
            if score > character_thresh
        }

        sorted_general = sorted(general_res.keys(), key=lambda x: general_res[x], reverse=True)
        return ", ".join(sorted_general), general_res, character_res


# 初始化本地模型预测器
predictor = LocalPredictor(MODEL_PATH, LABEL_PATH)

# 修改 generate_prompts_for_image 函数
def generate_prompts_for_image(crop_image, idx, general_thresh, character_thresh):
    print(f"正在为第 {idx + 1} 块图像生成提示词...")
    general_tags, general_res, character_res = predictor.predict(
        crop_image, general_thresh=general_thresh, character_thresh=character_thresh
    )
    return general_tags

# 示例调用
if __name__ == "__main__":
    image_path = "input_image.png"
    general_thresh = 0.3
    character_thresh = 0.9

    # 加载输入图像
    input_image = Image.open(image_path).convert("RGBA")

    # 使用预测器生成提示词
    prompts = generate_prompts_for_image(input_image, 0, general_thresh, character_thresh)
    print("生成的提示词:", prompts)
