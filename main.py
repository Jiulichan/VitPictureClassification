#主程序，组织整个流程

from data_loader import get_dataloaders
from model import setup_model, print_model_parameters
from train import train_model
import config

def main():
    print("1. Loading data...")
    dataloaders, dataset_sizes, class_names = get_dataloaders()

    print("2. Setting up model...")
    model = setup_model(num_classes=len(class_names))
    #print(f"Model has been setup for {len(class_names)} classes.")
    #四周目版本：
    print_model_parameters(model) # 打印初始参数状态

    print("3. Starting training...")
    # 二周目修改：接收返回的模型和历史记录
    model, history = train_model(model, dataloaders, dataset_sizes, class_names)

    print("4. All done!")


# 这个if语句确保只有当这个文件被直接运行时，main()才会被调用。
# 如果它被其他文件import，则不会自动运行。
if __name__ == '__main__':
    main()