# WhisperX Transcriber

基于 WhisperX 的音频转写工具，提供了更灵活的配置选项和批量处理能力。

## 功能特点

1. WhisperX 核心功能
    - 自动 VAD (Voice Activity Detection) 语音检测和分段
    - Whisper 大规模语音识别
    - Forced Alignment 音素级别对齐
    - 说话人分离 (Speaker Diarization)
2. 增强功能
    * 完整的配置系统
        - 模型参数配置
        - 输入输出配置
        - 功能模块开关
    * 批量文件处理
        - 支持多种音频格式
        - 支持指定文件处理
        - 支持目录批处理
    * 多格式输出：JSON，SRT, TXT
    * 内存管理优化

## 安装使用

1. 安装 Python 环境
```bash
    conda create --name whisperx python=3.10
    conda activate whisperx
```

2. Install PyTorch, e.g. for Linux and Windows

* CUDA11.8:
```bash
    conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
* CPU Only
```bash
    # 安装符合whisperx 3.3.4要求的PyTorch版本（必须CPU版）
    conda install pytorch==2.5.1 torchaudio==2.5.1 cpuonly -c pytorch
```

3. 安装 whisperx 及相关库（适配CPU Only）
```bash
    pip install "whisperx==3.3.4" --upgrade-strategy only-if-needed --no-cache-dir --force-reinstall --no-deps
    pip install "nltk>=3.9.1" "pandas>=2.2.3" "pyannote-audio>=3.3.2" "transformers>=4.48.0" "ctranslate2<4.5.0" "faster-whisper>=1.1.1" "numpy>=2.0.2"
```

4. 配置 config.yaml 及 secrets.yaml 文件

请在 secrets.yaml 文件中设置 diarization 所需的 token 及 proxy。
```bash
    auth_token: "your_token"
    proxy: "your_proxy"
```

而且要在 HuggingFace 网站上手动接受相关模型的使用授权条款：

* 访问 https://huggingface.co/pyannote/segmentation-3.0
* 访问 https://huggingface.co/pyannote/speaker-diarization-3.0
* 在每个页面上：
    - 登录你的 HuggingFace 账号（使用与 token 相关联的账号）
    - 点击 "Accept" 或 "Access repository" 按钮接受使用条款
    - 同意相关的许可条款

5. 运行程序

可以使用 python 命令：
```bash
    python WhisperXTranscriber.py
```
也可以使用快捷键 “WhisperXConda.lnk” （需要根据项目位置先修订其属性值）

## ref

https://github.com/VimWei/WhisperTranscriber
