# 主运行脚本 - LoRA版本
from model import setup_model
from data_loader import get_dataloaders
from train_with_LoRA import train_model
import config


def main():
    print("=== 使用添加了LoRA方法的分层冻结策略进行ViT微调 ===")

    # 获取数据
    dataloaders, dataset_sizes, class_names = get_dataloaders()
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")

    # 创建模型
    model = setup_model(num_classes=len(class_names))

    # 使用LoRA训练
    trained_model, history = train_model(model, dataloaders, dataset_sizes, class_names)

    print("添加了LoRA方法的分层冻结策略模型训练完成！")


if __name__ == '__main__':
    main()