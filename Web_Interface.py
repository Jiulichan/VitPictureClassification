import gradio as gr
import torch
from PIL import Image
from predict import predict_single_image  # 从项目模块中导入预测函数
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个处理函数，接收上传的图片，返回预测结果和置信度
def classify_flower(image):
    # 将Gradio的图片输入转换为模型需要的格式
    # 调用你的预测函数
    predicted_class, confidence = predict_single_image(image,device)
    # 返回结果，Gradio会自动格式化显示
    return f"预测类别: {predicted_class}", f"置信度: {confidence:.2%}"

# 创建Gradio界面
demo = gr.Interface(
    fn=classify_flower,  # 处理函数
    inputs=gr.Image(type="filepath", label="上传花卉图片"),  # 输入组件
    outputs=[gr.Textbox(label="预测结果"), gr.Textbox(label="置信度")],  # 输出组件
    title="花卉分类器",
    description="上传一张花卉图片，模型会预测它属于102种花卉中的哪一种。",
    examples=[os.path.join("img", img) for img in ["rose.jpg", "sunflower.jpg", "tulip.jpg"]]  # 提供示例图片
)

# 启动应用 - 仅在本地测试时使用 share=False
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")  # server_name="0.0.0.0" 允许局域网内其他设备访问