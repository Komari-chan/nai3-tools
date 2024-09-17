from gradio_client import Client, handle_file

# 初始化客户端，指向目标模型
client = Client("SmilingWolf/wd-tagger")

# 使用模型生成提示词
result = client.predict(
    image=handle_file("input_image.png"),  # 替换为你要识别的图像路径
    model_repo="SmilingWolf/wd-swinv2-tagger-v3",  # 使用的模型
    general_thresh=0.3,  # 一般标签的阈值
    general_mcut_enabled=False,  # 是否启用 MCut 阈值
    character_thresh=1,  # 角色标签的阈值
    character_mcut_enabled=False,  # 是否启用 MCut 阈值
    api_name="/predict"  # API 名称
)

# 打印生成的提示词
print(result)
