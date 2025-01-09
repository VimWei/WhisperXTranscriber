# WhisperX Transcriber

基于 WhisperX 的音频转写工具，提供了更灵活的配置选项和批量处理能力。

## 功能特点

### WhisperX 核心功能
- 自动 VAD (Voice Activity Detection) 语音检测和分段
- Whisper 大规模语音识别
- Forced Alignment 音素级别对齐
- 说话人分离 (Speaker Diarization)

### 增强功能
- 完整的配置系统
  - 模型参数配置
  - 输入输出配置
  - 功能模块开关
- 批量文件处理
  - 支持多种音频格式
  - 支持指定文件处理
  - 支持目录批处理
- 多格式输出
  - JSON 格式（完整信息）
  - SRT 字幕格式
  - TXT 纯文本格式
- 内存管理优化

## 快速开始

1. 安装 Python 环境

    conda create --name whisperx python=3.10
    conda activate whisperx

2. Install PyTorch, e.g. for Linux and Windows

* CUDA11.8:
    conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

* CPU Only
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch

3. 安装 whisperx 及相关库（适配CPU Only）

    pip install -r requirements.txt

4. 配置 config.yaml 及 .env 文件

请在 .env 文件中设置 diarization 所需的 token 及 proxy。

```bash
HF_AUTH_TOKEN=*your_token*
HTTP_PROXY=*your_proxy*
```

而且要在 HuggingFace 网站上手动接受相关模型的使用授权条款：

* 访问 https://huggingface.co/pyannote/segmentation-3.0
* 访问 https://huggingface.co/pyannote/speaker-diarization-3.0
* 在每个页面上：
    - 登录你的 HuggingFace 账号（使用与 token 相关联的账号）
    - 点击 "Accept" 或 "Access repository" 按钮接受使用条款
    - 同意相关的许可条款

5. 运行程序

可以使用命令：
```bash
    python WhisperXTranscriber.py
```
也可以使用快捷键 “WhisperXConda.lnk” （需要根据项目位置先修订其属性值）

## 核心类说明

### WhisperXTranscriber

主要组件：

1. 配置系统
   - `__init__`: 初始化配置
   - `load_config`: 加载配置文件

2. 文件处理系统
   - `get_media_files`: 获取待处理文件列表
   - `process_file`: 单文件处理
   - `process_all`: 批量处理入口

3. 模型管理
   - `load_whisper_model`: 加载 Whisper 模型
   - `load_alignment_model`: 加载对齐模型
   - `load_diarization_model`: 加载说话人分离模型

4. 结果保存系统
   - `save_results`: 多格式结果保存
