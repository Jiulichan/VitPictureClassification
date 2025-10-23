#模型定义和加载。创建和修改模型结构
from transformers import ViTForImageClassification
import torch.nn as nn
from peft import LoraConfig, get_peft_model  # 新增导入
import config

def setup_model(num_classes):
    """加载预训练模型并修改分类头"""
    model = ViTForImageClassification.from_pretrained(config.local_model_path) # 从本地加载
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    return model


def setup_lora_model(model):
    """配置LoRA模型"""
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model

def freeze_backbone(model, unfreeze_layers='all'):
    # 冻结ViT模型的主干网络
    if config.USE_LORA:
        # 使用LoRA时，只需要确保分类头可训练，LoRA参数会自动可训练
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("使用LoRA微调，自动管理可训练参数")
    else:
        # 首先冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 然后解冻分类头
        for param in model.classifier.parameters():
            param.requires_grad = True
        # 根据配置解冻主干的部分层
        if unfreeze_layers != 'all':
            # ViT的encoder由多个层（layer）组成，我们解冻最后几层
            total_layers = len(model.vit.encoder.layer)
            if unfreeze_layers == 'last2':
                layers_to_unfreeze = model.vit.encoder.layer[-2:]
            elif unfreeze_layers == 'last4':
                layers_to_unfreeze = model.vit.encoder.layer[-4:]
            else:
                layers_to_unfreeze = []  # 默认只训练分类头

            # 解冻选中的层
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

    print_model_parameters(model) # 打印参数状态

def unfreeze_for_finetuning(model, unfreeze_layers='all'):
    # 为微调解冻模型参数
    if config.USE_LORA:
        # 使用LoRA时，第二阶段可以继续使用LoRA或完全解冻
        if unfreeze_layers == 'all':
            # 完全解冻，移除LoRA适配器
            model = model.merge_and_unload()
            for param in model.parameters():
                param.requires_grad = True
            print("第二阶段：完全解冻所有参数进行微调")
        else:
            # 继续使用LoRA，但解冻更多层
            print("第二阶段：继续使用LoRA进行微调")
    if unfreeze_layers == 'all':
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True
    else:
        # 只解冻指定的层
        total_layers = len(model.vit.encoder.layer)

        if unfreeze_layers == 'last2':
            layers_to_unfreeze = model.vit.encoder.layer[-2:]
        elif unfreeze_layers == 'last4':
            layers_to_unfreeze = model.vit.encoder.layer[-4:]
        else:
            layers_to_unfreeze = []

        # 确保分类头是解冻的
        for param in model.classifier.parameters():
            param.requires_grad = True
        #解冻选中的层
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    print_model_parameters(model) # 打印参数状态

def print_model_parameters(model):
    # 打印模型可训练参数的数量和状态
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数量：{total_params:,}')
    print(f'可训练参数量：{trainable_params:,}')
    print(f'冻结参数量：{total_params - trainable_params:,}')
    print(f'可训练参数占比：{100 * trainable_params / total_params:.2f}%')