#超参数和配置集中管理
DATA_DIR = './data'
MODEL_SAVE_PATH = './outputs/best_model.pth'

#模型相关
PRETRAINED_MODEL_NAME = 'google/vit-base-patch16-224'

#训练超参数
#以后可能需要更改batch_size，以适配我的电脑配置
BATCH_SIZE = 16
NUM_EPOCHS = 10 #在学习调度器方法中尝试过20、50
LEARNING_RATE = 1e-4 # 首次训练使用的学习率是1e-3，第二次是5e-5，ACC得到了巨大提升

#三周目：学习率调度器参数
USE_SCHEDULER = False # 是否使用学习率调度器 在差分学习率/分阶段微调方法中，我将它设置为False
SCHEDULER_TYPE = 'ReduceLROnPlateau' # 学习率调度器：①当验证损失不再下降时自动降低学习率；②余弦退火
SCHEDULER_MODE = 'max'  # 监控指标的模式：'min'用于损失，'max'用于准确率
SCHEDULER_PATIENCE = 1  # 容忍度，多少个epoch没有提升就降低学习率
SCHEDULER_FACTOR = 0.5  # 学习率降低的因子
SCHEDULER_COOLDOWN = 1  # 在调整学习率后的冷却期（不监控改进）
SCHEDULER_MIN_LR = 1e-7  # 学习率下限
# 余弦退火
SCHEDULER_TYPE = 'CosineAnnealingLR' #余弦退火
COSINE_T_MAX = 5  # 半个余弦周期的epoch数
COSINE_ETA_MIN = 1e-6  # 最低学习率
# 步进衰减
SCHEDULER_TYPE = 'StepLR'
STEP_SIZE = 3  # 每多少个epoch衰减一次
GAMMA = 0.1     # 衰减因子

# --- 分阶段训练配置 ---
FREEZE_BACKBONE = True          # 是否开启冻结主干网络训练
NUM_EPOCHS_FROZEN = 5           # 冻结主干训练的轮数
UNFREEZE_LAYERS = "all"         # 解冻范围: "all" 全部层, "last2" 最后2层, "last4" 最后4层
FINETUNE_LR = 1e-5              # 解冻后微调的学习率 (通常比初始LR小10倍或更多)

# LoRA相关的一系列配置
USE_LORA = True
LORA_R = 8  # LoRA的秩
LORA_ALPHA = 32  # LoRA的alpha参数
LORA_DROPOUT = 0.1  # LoRA的dropout率
LORA_TARGET_MODULES = ['query', 'value', 'key',
                       'dense' #前馈网络第一层
                       ]  # 目标模块，ViT中通常是attention的query和value
LORA_LEARNING_RATE = 1e-4

# LoRA模型保存路径
LORA_SAVE_PATH = './outputs/lora_model'


#其他
NUM_WORKERS = 2 # 用于DataLoader加载数据的进程数，一开始的训练进程数量是2
DEVICE = 'cuda' # 默认使用GPU

local_model_path = "./local_vit_model"  # 把模型下载到本地了，存放在该位置
