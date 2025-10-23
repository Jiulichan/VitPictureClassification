# 加载LoRA模型进行推理
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import setup_model
from data_loader import get_dataloaders
from transformers import ViTImageProcessor
from peft import PeftModel
import config


def setup_lora_for_inference(base_model, lora_path):
    """为推理设置LoRA模型"""
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    return lora_model


def predict_single_image_lora(image_path, device):
    """使用LoRA模型对单张图像进行预测"""
    # 加载类别标签
    _, _, class_names = get_dataloaders()

    # 加载基础模型
    base_model = setup_model(num_classes=len(class_names))

    # 加载LoRA适配器
    model = setup_lora_for_inference(base_model, config.LORA_SAVE_PATH)
    model = model.to(device)
    model.eval()

    # 加载处理器
    processor = ViTImageProcessor.from_pretrained('./local_vit_model')

    # 打开并预处理图像
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class_idx].item()

    predicted_class = class_names[predicted_class_idx]

    # 显示结果
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}\n(LoRA Model)")
    plt.axis('off')
    plt.show()

    return predicted_class, confidence


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 对新图片进行预测
    image_path = "./PredictNew/img_05.jpg"  # 替换成待预测的图片路径
    pred_class, conf = predict_single_image_lora(image_path, device)

    print(f"LoRA模型预测结果: {pred_class}")
    print(f"置信度: {conf:.2%}")