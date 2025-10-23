from huggingface_hub import snapshot_download

# 指定模型名称
model_id = "google/vit-base-patch16-224"
# 指定本地保存路径
local_dir = "./local_vit_model"

# 下载所有模型文件
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)