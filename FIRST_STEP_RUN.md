# 第一阶段运行说明：AeroRescue-AI 灾害目标检测 Demo

本阶段只跑通 AeroRescue-AI（面向低空应急救援的无人机多模态灾情识别与辅助决策系统）的 Gradio + YOLOv11 灾害目标检测 Demo，不接入 ARGUS、Detection-Models、RescueNet，也不重新训练模型。

## 环境建议

- Python：建议 3.10-3.12；当前本机验证使用 Python 3.12。Gradio 5.x 不适合继续使用 Python 3.9。
- GPU：不是必需。CPU 可以运行图片检测；视频检测在 CPU 上会慢一些。
- 模型文件：仓库内需要存在以下权重文件：
  - `models/yolov11n/best.pt`
  - `models/yolov11s/best.pt`
  - `models/yolov11m/best.pt`
  - `models/yolov11l/best.pt`

## 启动命令

```bash
cd app
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

启动后打开终端输出中的本地 Gradio 地址，通常是：

```text
http://127.0.0.1:7860
```

## 当前 Demo 能力

- 上传灾害图片。
- 选择 YOLOv11n / YOLOv11s / YOLOv11m / YOLOv11l。
- 输出带检测框的结果图。
- 输出每个目标的类别、置信度和检测框坐标。
- 支持类别：`civilian`、`rescuer`、`dog`、`cat`、`horse`、`cow`。

## 已做的最小兼容性修复

- 模型路径改为优先读取仓库本地 `models/<variant>/best.pt`。
- 如果模型缺失，会明确提示应该放到哪个目录。
- 移除运行时对 Hugging Face 模型下载的依赖。
- 视频示例改为使用仓库内 `static/video/rescuer.mp4`。
- Gradio `allowed_paths` 改为基于当前项目目录，避免写死 Linux 用户路径。
- 精简 `app/requirements.txt`，去掉 Linux CUDA 专用依赖，便于 Mac / Windows / Linux 安装。
