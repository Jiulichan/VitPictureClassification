#训练和验证循环
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
from tqdm import tqdm  # 用于显示进度条
import matplotlib.pyplot as plt #二周目新增
import json # 二周目新增：保存历史数据到JSON文件
import numpy as np # 同上
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 三周目：导入调度器
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
#四周目导入新函数，需要创建两个训练阶段
from model import freeze_backbone, unfreeze_for_finetuning, print_model_parameters  # 导入新函数
import os
import config

def train_model(model, dataloaders, dataset_sizes, class_names):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 注意：这里为了简单改为训练所有参数。也可以选择只训练分类头：
    #optimizer = optim.Adam(model.classifier.parameters(), lr=config.LEARNING_RATE)

    # 三周目新增：初始化学习率调度器
    if config.USE_SCHEDULER and config.SCHEDULER_TYPE == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.SCHEDULER_MODE,
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            cooldown=config.SCHEDULER_COOLDOWN,
            min_lr=config.SCHEDULER_MIN_LR,
            # verbose=True # 打印学习率变化信息，报错了，应该是我的pytorch版本没有这个东西
        )
    # 余弦退火
    elif config.SCHEDULER_TYPE == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.COSINE_T_MAX,
            eta_min=config.COSINE_ETA_MIN
        )
    #步进衰减
    elif config.SCHEDULER_TYPE == 'StepLR':
        scheduler = StepLR(
            optimizer,
            step_size=config.STEP_SIZE,
            gamma=config.GAMMA
        )
    else:
        scheduler = None

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    since = time()
    best_acc = 0.0

    #三周目：初始化一个变量来记录上一次的学习率
    #last_lr = optimizer.param_groups[0]['lr']

    # 二周目新增：初始化历史数据列表来记录历史数据
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': [], # 三周目新增：记录每个epoch的学习率
        'phase': [] #四周目新增：phase记录训练阶段
    }

    # ==================== 第一阶段：冻结主干，只训练分类头 ====================
    if config.FREEZE_BACKBONE:
        print("\n" + "=" * 60)
        print("第一阶段：冻结主干网络，只训练分类头")
        print("=" * 60)

        # 冻结主干网络
        freeze_backbone(model, config.UNFREEZE_LAYERS)

        # 第一阶段优化器：只训练需要梯度的参数（即分类头）
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE  # 使用较大的学习率
        )

        # 训练冻结阶段
        model, history, best_acc = train_phase(
            model, optimizer, None, criterion, dataloaders,
            dataset_sizes, history, config.NUM_EPOCHS_FROZEN,
            "frozen", best_acc, device
        )
        # ==================== 第二阶段：解冻主干，全模型微调 ====================
        print("\n" + "=" * 60)
        print("第二阶段：解冻主干网络，全模型微调")
        print("=" * 60)

        # 解冻主干网络
        unfreeze_for_finetuning(model, config.UNFREEZE_LAYERS)

        # 第二阶段优化器：训练所有参数，但使用更小的学习率
        optimizer_ft = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.FINETUNE_LR  # 使用较小的学习率微调
        )

        # 计算第二阶段需要的epoch数
        remaining_epochs = config.NUM_EPOCHS - (config.NUM_EPOCHS_FROZEN if config.FREEZE_BACKBONE else 0)

        # 训练微调阶段
        model, history, best_acc = train_phase(
            model, optimizer_ft, scheduler, criterion, dataloaders,
            dataset_sizes, history, remaining_epochs,
            "fine_tune", best_acc, device
        )

    time_elapsed = time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 二周目新增：绘制损失和准确率曲线
    plot_training_history(history)
    # 二周目新增：保存训练历史数据到JSON文件
    save_training_history(history)

    return model, history

#四周目重塑：
def train_phase(model, optimizer, scheduler, criterion, dataloaders, dataset_sizes, history, num_epochs, phase_name, best_acc, device):
    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 10)

        #记录当前学习率
        current_lr = float(optimizer.param_groups[0]['lr'])
        print(f'Current learning rate: {current_lr:.2e}')
        # 记录当前学习率到历史数据中
        history['learning_rates'].append(current_lr)
        history['phase'].append(phase_name)
        print(f'Current learning rate: {current_lr:.2e}')
        print(f'Phase: {phase_name}')

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 使用tqdm包装dataloader以显示进度条
            loop = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch}')
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)
                    _, preds = torch.max(outputs.logits, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # 更新进度条描述
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 二周目新增：记录当前epoch的损失和准确率
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            #余弦退火
            if phase == 'train' and scheduler is not None and config.SCHEDULER_TYPE == 'CosineAnnealingLR':
                scheduler.step() # Cosine通常在每个训练阶段后更新
            #步进衰减，也是在每个训练阶段后step
            if phase == 'train' and scheduler is not None and config.SCHEDULER_TYPE == 'StepLR':
                scheduler.step()

            # 三周目新增：在每个验证阶段结束后，根据验证准确率更新学习率
            if phase == 'val' and scheduler is not None and config.SCHEDULER_TYPE == 'ReduceLROnPlateau':
                if config.SCHEDULER_MODE == 'max':
                    scheduler.step(epoch_acc) # 对于准确率。我们希望它最大化
                else:
                    scheduler.step(epoch_loss) # 对于损失，我们希望它最小化

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 保存最佳模型
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f'==> New best model saved with Acc: {best_acc:4f}')

        print()

    return model, history, best_acc


# 二周目新增：绘制训练历史的函数
def plot_training_history(history):
    """绘制训练和验证的损失曲线和准确率曲线"""
    # 创建两个子图
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 三周目新更改：绘制三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,4))

    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 绘制准确率曲线
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # 三周目新增：绘制学习率变化曲线
    ax3.plot(history['learning_rates'], label='Learning Rate', marker='o')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()

    # 保存图像
    plt.savefig('./outputs/training_history.png')
    plt.tight_layout()
    plt.show()

# 转换numpy类型为python原生类型以便JSON序列化
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 二周目新增：保存训练历史数据到JSON文件
def save_training_history(history):
    '''将训练历史保存为JSON文件'''
    #确保输出目录存在
    os.makedirs('./outputs', exist_ok=True)

    #创建可序列化的历史记录副本
    serializable_history = {}
    for key, values in history.items():
        serializable_history[key] = [convert_to_serializable(v) for v in values]

    #保存到文件
    with open('./outputs/training_history.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, ensure_ascii=False, indent=4)

    print("训练数据已保存到：'./outputs/training_history.json'")
