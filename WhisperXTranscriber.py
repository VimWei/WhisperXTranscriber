import os
import gc 
import yaml
import json
from pathlib import Path
import whisperx
from typing import List

# 使用新的 inference 接口
from speechbrain.inference import SpeakerRecognition

class WhisperXTranscriber:
    # 1. 配置系统
    def __init__(self, config_path: str = "config.yaml"):
        # 加载主配置
        self.config = self.load_config(config_path)
        
        # 加载敏感配置
        try:
            with open("secrets.yaml", 'r', encoding='utf-8') as f:
                secrets = yaml.safe_load(f)
                # 更新配置
                if 'auth_token' in secrets:
                    self.config['model']['diarization']['auth_token'] = secrets['auth_token']
                if 'proxy' in secrets:
                    self.config['global']['proxy'] = secrets['proxy']
                
                # 如果有代理设置，立即应用
                if proxy := self.config['global'].get('proxy'):
                    for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                        os.environ[var] = proxy
                    os.environ['REQUESTS_CA_BUNDLE'] = ''
                    print(f"Proxy set to: {proxy}")
                    
        except FileNotFoundError:
            print("Warning: secrets.yaml not found")

    @staticmethod
    def load_config(config_path: str) -> dict:
        """配置文件加载"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # 2. 文件处理系统
    def get_media_files(self) -> List[str]:
        """批量文件处理支持"""
        input_dir = self.config['input']['directory']
        formats = self.config['input']['formats']
        specific_files = self.config['input']['specific_files']

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            print(f"Created input directory: {input_dir}")
            return []

        if specific_files:
            return [os.path.join(input_dir, f) for f in specific_files 
                    if os.path.exists(os.path.join(input_dir, f))]

        media_files = []
        for format in formats:
            media_files.extend(str(p) for p in Path(input_dir).glob(f"*{format}"))
        
        return sorted(media_files)

    def load_whisper_model(self):
        """Load the Whisper model"""
        config = self.config['model']['whisper']
        return whisperx.load_model(
            config['name'],
            device=config['device'],
            compute_type=config['compute_type'],
            language=config['language']
        )

    def load_alignment_model(self, language: str):
        """Load the alignment model"""
        config = self.config['model']['aligner']
        if not config['enabled']:
            return None, None
        
        return whisperx.load_align_model(
            language_code=language,
            device=self.config['model']['whisper']['device']
        )

    def load_diarization_model(self):
        """Load the diarization model"""
        config = self.config['model']['diarization']
        if not config['enabled']:
            return None

        return whisperx.DiarizationPipeline(
            use_auth_token=config['auth_token'],
            device=self.config['model']['whisper']['device']
        )

    def process_file(self, audio_path: str):
        """Process a single audio file"""
        print(f"Processing: {audio_path}")
        
        # Load audio
        print("Loading audio file...")
        audio = whisperx.load_audio(audio_path)
        
        # 1. Transcribe with Whisper
        print("Loading Whisper model...")
        model = self.load_whisper_model()
        print("Transcribing audio...")
        result = model.transcribe(
            audio,
            batch_size=self.config['model']['whisper']['batch_size'],
            language=self.config['model']['whisper']['language']
        )
        
        print("Cleaning up Whisper model...")
        del model
        gc.collect()

        # 2. Align if enabled
        if self.config['model']['aligner']['enabled']:
            print("Loading alignment model...")
            model_a, metadata = self.load_alignment_model(
                language=self.config['model']['aligner']['language']
            )
            if model_a is not None:
                print("Aligning transcription...")
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.config['model']['whisper']['device'],
                    return_char_alignments=self.config['transcription']['return_char_alignments']
                )
                
                print("Cleaning up alignment model...")
                del model_a
                gc.collect()

        # 3. Diarize if enabled
        if self.config['model']['diarization']['enabled']:
            print("Loading diarization model...")
            diarize_model = self.load_diarization_model()
            if diarize_model is not None:
                try:
                    print("Performing speaker diarization...")
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=self.config['model']['diarization']['min_speakers'],
                        max_speakers=self.config['model']['diarization']['max_speakers']
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    print("Cleaning up diarization model...")
                    del diarize_model
                    gc.collect()
                except Exception as e:
                    print(f"Diarization failed: {str(e)}")

        # Save results
        print("Saving results...")
        self.save_results(result, audio_path)
        print(f"Completed processing: {os.path.basename(audio_path)}\n")

    # 3. 结果保存系统
    def save_results(self, result: dict, audio_path: str):
        """使用 WhisperX 内置的输出功能"""
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = self.config['output']['directory']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 使用 WhisperX 的内置 writer
        for fmt in self.config['output']['formats']:
            output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
            
            if fmt == "txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    # 简单的文本输出
                    for segment in result["segments"]:
                        f.write(f"{segment['text'].strip()}\n")
            elif fmt == "srt":
                with open(output_path, "w", encoding="utf-8") as f:
                    # 直接使用 segments 数组
                    segments = []
                    for segment in result["segments"]:
                        segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"]
                        })
                    # 写入 SRT 格式
                    for i, segment in enumerate(segments, start=1):
                        f.write(f"{i}\n")
                        f.write(f"{self.format_timestamp(segment['start'])} --> {self.format_timestamp(segment['end'])}\n")
                        f.write(f"{segment['text'].strip()}\n\n")
            elif fmt == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

        if self.config['output']['verbose']:
            print(f"Results saved to {output_dir}")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

    # 4. 批处理入口
    def process_all(self):
        """批量处理入口"""
        media_files = self.get_media_files()
        
        if not media_files:
            print(f"No media files found in {self.config['input']['directory']}")
            print(f"Supported formats: {', '.join(self.config['input']['formats'])}")
            return

        print(f"Found {len(media_files)} files to process:")
        for file in media_files:
            print(f"  - {os.path.basename(file)}")
        print()

        for file in media_files:
            self.process_file(file)

if __name__ == "__main__":
    transcriber = WhisperXTranscriber()
    transcriber.process_all()