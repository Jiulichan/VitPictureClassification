#加载模型进行推理演示
import torch
from PIL import Image
import typing_extensions
import matplotlib.pyplot as plt
from model import setup_model  # 从model.py中导入创建模型的函数
from data_loader import get_dataloaders  # 为了获取class_names
from transformers import ViTImageProcessor
import config

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device) # 检查当前使用的设备

def predict_single_image(image_path, device):
    """
    对单张图像进行预测并显示结果
    """
    # 加载类别标签（假设data_loader可以返回class_names）
    _, _, class_names = get_dataloaders()

    # 加载模型结构
    model = setup_model(num_classes=len(class_names))

    # 加载训练好的权重
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model = model.to(device)

    # 加载处理器（如果需要从本地加载，确保本地有preprocessor_config.json）
    processor = ViTImageProcessor.from_pretrained('./local_vit_model')  # 使用之前下载的本地模型路径

    # 1. 打开图像
    image = Image.open(image_path).convert('RGB')

    # 2. 使用处理器预处理图像（与训练时一致）
    inputs = processor(images=image, return_tensors="pt")  # 返回PyTorch张量
    inputs = inputs.to(device)  # 将输入数据移动到GPU或CPU

    # 3. 将模型设置为评估模式并进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)  # 计算概率
        predicted_class_idx = logits.argmax(-1).item()  # 获取最高概率的类别索引
        confidence = probabilities[0][predicted_class_idx].item()  # 获取置信度

    # 4. 获取预测的类别名称和置信度
    predicted_class = class_names[predicted_class_idx]

    # 5. 显示结果
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    plt.show()

    return predicted_class, confidence

if __name__ == '__main__':


    # 对新图片进行预测
    image_path = "./PredictNew/img_05.jpg"  # 替换成待预测的图片路径
    pred_class, conf = predict_single_image(image_path, model, processor, class_names, device)

    print(f"预测结果: {pred_class}")
    print(f"置信度: {conf:.2%}")