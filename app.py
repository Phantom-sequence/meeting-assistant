import pyaudio
import wave
import tempfile
import os
import threading
import queue
import time
import numpy as np
import pickle
from datetime import datetime

# 语音识别
from onnx_asr import load_model

# 声纹识别
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAS_SPEAKER_REC = True
except ImportError:
    HAS_SPEAKER_REC = False
    print("⚠️ 未安装 resemblyzer，将使用默认说话人")


class SpeakerManager:
    """声纹识别管理器"""
    
    def __init__(self, db_path="speaker_db.pkl"):
        self.db_path = db_path
        self.database = {}
        self.encoder = None
        self.speaker_counter = 1  
        self._load_encoder()
        self._load_database()
    
    def _load_encoder(self):
        """加载声纹编码器"""
        if HAS_SPEAKER_REC:
            try:
                self.encoder = VoiceEncoder()
                print("✅ 声纹识别模型加载成功")
            except Exception as e:
                print(f"❌ 声纹模型加载失败: {e}")
    
    def _load_database(self):
        """加载声纹数据库"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.database = pickle.load(f)
                print(f"✅ 加载了声纹库，共 {len(self.database)} 位说话人")
                
                # 更新计数器
                for name in self.database.keys():
                    if name.startswith("说话人"):
                        try:
                            num = int(name[3:])
                            if num >= self.speaker_counter:
                                self.speaker_counter = num + 1
                        except:
                            pass
            except Exception as e:
                print(f"加载声纹库失败: {e}")
    
    def _save_database(self):
        """保存声纹数据库"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.database, f)
            print(f"✅ 声纹库已保存，共 {len(self.database)} 位说话人")
        except Exception as e:
            print(f"保存声纹库失败: {e}")
    
    def extract_embedding(self, audio_file):
        """提取声纹特征"""
        if self.encoder is None:
            return None
        try:
            wav = preprocess_wav(audio_file)
            if len(wav) < 16000:  # 至少1秒
                return None
            return self.encoder.embed_utterance(wav)
        except Exception as e:
            return None
    
    def register_speaker(self, audio_file, name=None):
        """注册新说话人"""
        if self.encoder is None:
            return None, "声纹识别未启用"
        
        embedding = self.extract_embedding(audio_file)
        if embedding is None:
            return None, "录音太短或无效"
        
        # 自动分配名称
        if name is None:
            name = f"说话人{self.speaker_counter}"
            self.speaker_counter += 1
        
        # 存储
        if name not in self.database:
            self.database[name] = {
                'embeddings': [],
                'register_time': datetime.now().isoformat(),
                'sample_count': 0
            }
        
        self.database[name]['embeddings'].append(embedding)
        self.database[name]['sample_count'] += 1
        self._save_database()
        
        return name, f"✅ 注册成功: {name}"
    
    def identify_speaker(self, audio_file, threshold=0.5):
        """识别说话人"""
        if not self.database or self.encoder is None:
            return "未知", 0.0
        
        embedding = self.extract_embedding(audio_file)
        if embedding is None:
            return "无效音频", 0.0
        
        best_name = "未知"
        best_score = 0
        
        for name, data in self.database.items():
            for reg_emb in data['embeddings']:
                # 计算余弦相似度
                similarity = np.dot(embedding, reg_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(reg_emb)
                )
                if similarity > best_score:
                    best_score = similarity
                    best_name = name
        
        if best_score >= threshold:
            return best_name, best_score
        return "未知", best_score
    
    def get_speaker_list(self):
        """获取所有说话人列表"""
        return list(self.database.keys())
    
    def clear_database(self):
        """清空声纹库"""
        self.database = {}
        self.speaker_counter = 1
        self._save_database()
        print("✅ 声纹库已清空")


class RealtimeASR:
    """多线程实时语音识别器（带声纹识别）"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = 1
        
        self.is_recording = False
        self.is_running = False
        
        self.audio_queue = queue.Queue(maxsize=50)
        
        self.record_thread = None
        self.asr_thread = None
        self.display_thread = None
        self.speaker_thread = None  # 新增：声纹识别线程
        
        self.asr_model = None
        self.speaker_manager = SpeakerManager()  # 声纹管理器
        
        self.p = None
        self.stream = None
        
        self.audio_buffer = []
        self.speaker_buffer = []  # 声纹识别专用缓冲区
        self.buffer_duration = 2.0
        self.samples_per_buffer = int(sample_rate * self.buffer_duration)
        
        # 声纹识别间隔（每3秒识别一次说话人）
        self.speaker_interval = 3.0
        self.speaker_samples = int(sample_rate * self.speaker_interval)
        
        # 识别结果队列
        self.result_queue = queue.Queue()
        
        # 会议记录存储
        self.meeting_transcript = []
        
        # 当前说话人（用于显示）
        self.current_speaker = "识别中"
        self.current_confidence = 0.0
    
    def load_asr_model(self, model_name="whisper-base"):
        """加载语音识别模型"""
        try:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            self.asr_model = load_model(model_name)
            print(f"✅ 语音识别模型加载成功: {model_name}")
            return True
        except Exception as e:
            print(f"❌ 语音识别模型加载失败: {e}")
            return False
    
    def register_current_speaker(self):
        """注册当前说话人（用于快速注册）"""
        if len(self.speaker_buffer) < self.sample_rate:
            print("⚠️ 录音太短，请多说一会儿")
            return None
        
        # 保存缓冲区为临时文件
        temp_file = self._save_buffer_to_file(self.speaker_buffer)
        if temp_file:
            name, msg = self.speaker_manager.register_speaker(temp_file)
            os.unlink(temp_file)
            print(msg)
            return name
        return None
    
    def _save_buffer_to_file(self, buffer_data):
        """保存缓冲区数据到临时文件"""
        if not buffer_data:
            return None
        
        try:
            audio_bytes = np.array(buffer_data, dtype=np.int16).tobytes()
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_bytes)
            
            return temp_file
        except Exception as e:
            print(f"保存文件失败: {e}")
            return None
    
    def start(self):
        """启动识别"""
        if self.asr_model is None:
            print("请先加载语音识别模型")
            return False
        
        self.is_recording = True
        self.is_running = True
        self.audio_buffer = []
        self.speaker_buffer = []
        self.meeting_transcript = []
        self.current_speaker = "识别中"
        
        # 启动线程
        self.record_thread = threading.Thread(target=self._record_loop)
        self.asr_thread = threading.Thread(target=self._asr_loop)
        self.speaker_thread = threading.Thread(target=self._speaker_loop)
        self.display_thread = threading.Thread(target=self._display_loop)
        
        self.record_thread.daemon = True
        self.asr_thread.daemon = True
        self.speaker_thread.daemon = True
        self.display_thread.daemon = True
        
        self.record_thread.start()
        self.asr_thread.start()
        self.speaker_thread.start()
        self.display_thread.start()
        
        return True
    
    def stop(self):
        """停止识别"""
        self.is_recording = False
        self.is_running = False
        
        time.sleep(1.5)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        
        # 处理最后的音频
        if len(self.audio_buffer) > self.sample_rate * 0.5:
            self._recognize_buffer()
    
    def _record_loop(self):
        """录音线程"""
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                try:
                    self.audio_queue.put_nowait(audio_data)
                except queue.Full:
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(audio_data)
                    except queue.Empty:
                        pass
            except Exception as e:
                print(f"录音错误: {e}")
    
    def _asr_loop(self):
        """语音识别线程"""
        while self.is_running:
            try:
                # 收集音频数据
                while len(self.audio_buffer) < self.samples_per_buffer:
                    try:
                        chunk = self.audio_queue.get(timeout=0.5)
                        self.audio_buffer.extend(chunk)
                        # 同时给声纹缓冲区
                        self.speaker_buffer.extend(chunk)
                    except queue.Empty:
                        if not self.is_recording:
                            break
                        continue
                
                if len(self.audio_buffer) >= self.samples_per_buffer:
                    self._recognize_buffer()
                    
                    overlap = int(self.samples_per_buffer * 0.5)
                    self.audio_buffer = self.audio_buffer[-overlap:] if overlap < len(self.audio_buffer) else []
                    
            except Exception as e:
                pass
    
    def _speaker_loop(self):
        """声纹识别线程"""
        while self.is_running:
            try:
                # 收集足够的音频用于声纹识别
                if len(self.speaker_buffer) >= self.speaker_samples:
                    self._identify_speaker()
                    
                    # 清空声纹缓冲区（或保留重叠）
                    overlap = int(self.speaker_samples * 0.5)
                    self.speaker_buffer = self.speaker_buffer[-overlap:] if overlap < len(self.speaker_buffer) else []
                
                time.sleep(0.5)
                
            except Exception as e:
                pass
    
    def _recognize_buffer(self):
        """语音识别当前缓冲区"""
        if len(self.audio_buffer) < self.sample_rate * 0.5:
            return
        
        try:
            temp_file = self._save_buffer_to_file(self.audio_buffer)
            if not temp_file:
                return
            
            # 识别
            result = self.asr_model.recognize(temp_file, language="zh")
            
            if hasattr(result, 'text'):
                text = result.text
            else:
                text = str(result)
            
            os.unlink(temp_file)
            
            # 输出结果
            if text and text.strip():
                if "[未识别" not in text and "[识别错误" not in text:
                    entry = {
                        'text': text.strip(),
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'datetime': datetime.now(),
                        'speaker': self.current_speaker,
                        'confidence': f"{self.current_confidence:.1%}" if self.current_confidence > 0 else "N/A"
                    }
                    self.result_queue.put(entry)
                    self.meeting_transcript.append(entry)
                
        except Exception as e:
            pass
    
    def _identify_speaker(self):
        """识别当前说话人"""
        if len(self.speaker_buffer) < self.sample_rate:
            return
        
        temp_file = self._save_buffer_to_file(self.speaker_buffer)
        if temp_file:
            speaker, confidence = self.speaker_manager.identify_speaker(temp_file)
            os.unlink(temp_file)
            
            if speaker != self.current_speaker:
                self.current_speaker = speaker
                self.current_confidence = confidence
                # 显示说话人切换
                print(f"\n👤 [说话人切换: {speaker} 置信度:{confidence:.1%}]")
    
    def _display_loop(self):
        """显示线程 - 实时打印识别结果"""
        last_display_time = 0
        
        while self.is_running:
            try:
                # 获取新结果
                while not self.result_queue.empty():
                    try:
                        result = self.result_queue.get_nowait()
                        # 实时显示（带说话人）
                        print(f"\n[{result['time']}] {result['speaker']} [{result['confidence']}]:")
                        print(f"  {result['text']}")
                    except queue.Empty:
                        break
                
                time.sleep(0.1)
                
            except Exception as e:
                pass
    
    def save_meeting_minutes(self, format='txt'):
        """保存会议纪要"""
        if not self.meeting_transcript:
            print("\n⚠️ 没有会议记录可保存")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'txt':
            filename = f"meeting_minutes_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("           AI会议纪要（实时识别+声纹识别版）\n")
                f.write("="*60 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总发言次数: {len(self.meeting_transcript)}\n")
                f.write("="*60 + "\n\n")
                
                # 说话人统计
                speaker_stats = {}
                for entry in self.meeting_transcript:
                    speaker = entry['speaker']
                    speaker_stats[speaker] = speaker_stats.get(speaker, 0) + 1
                
                f.write("【说话人统计】\n")
                for speaker, count in speaker_stats.items():
                    f.write(f"  {speaker}: {count} 次\n")
                f.write("\n" + "="*60 + "\n\n")
                
                # 详细记录
                f.write("【详细记录】\n\n")
                for i, entry in enumerate(self.meeting_transcript, 1):
                    f.write(f"[{i}] {entry['time']} | {entry['speaker']} [{entry['confidence']}]\n")
                    f.write(f"    {entry['text']}\n")
                    f.write("-"*40 + "\n\n")
            
            print(f"\n✅ 会议纪要已保存: {filename}")
            return filename
            
        elif format == 'json':
            import json
            filename = f"meeting_minutes_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'meeting_info': {
                        'title': 'AI会议纪要',
                        'generate_time': datetime.now().isoformat(),
                        'total_speeches': len(self.meeting_transcript)
                    },
                    'speaker_statistics': speaker_stats if 'speaker_stats' in dir() else {},
                    'transcript': self.meeting_transcript
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 会议纪要已保存: {filename}")
            return filename
    
    def get_summary(self):
        """获取会议摘要"""
        if not self.meeting_transcript:
            return "暂无会议记录"
        
        # 统计说话人
        speaker_stats = {}
        for entry in self.meeting_transcript:
            speaker = entry['speaker']
            speaker_stats[speaker] = speaker_stats.get(speaker, 0) + 1
        
        total_chars = sum(len(entry['text']) for entry in self.meeting_transcript)
        
        summary = f"""
============================================================
                    会议摘要
============================================================
总发言次数: {len(self.meeting_transcript)}
总字数: {total_chars}
说话人数: {len(speaker_stats)}
开始时间: {self.meeting_transcript[0]['time'] if self.meeting_transcript else 'N/A'}
结束时间: {self.meeting_transcript[-1]['time'] if self.meeting_transcript else 'N/A'}

【说话人统计】
"""
        for speaker, count in speaker_stats.items():
            summary += f"  {speaker}: {count} 次\n"
        
        summary += "============================================================"
        return summary


def print_menu():
    """打印菜单"""
    print("\n" + "="*50)
    print("实时语音识别系统")
    print("="*50)
    print("  [1] 开始新会议")
    print("  [2] 注册新说话人")
    print("  [3] 查看已注册说话人")
    print("  [4] 保存当前会议纪要")
    print("  [5] 查看会议摘要")
    print("  [6] 清空声纹库")
    print("  [7] 退出")
    print("="*50)


def register_speaker_interactive(speaker_manager, recognizer):
    """交互式注册说话人"""
    print("\n" + "="*50)
    print("          注册新说话人")
    print("="*50)
    print("请对着麦克风清晰说话 3-5 秒...")
    
    # 临时录音注册
    import pyaudio
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    DURATION = 3
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    
    print("🔴 录音中...", end="", flush=True)
    frames = []
    for _ in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)
        print(".", end="", flush=True)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("\n✅ 录音完成")
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    # 注册
    name = input("请输入姓名（直接回车自动分配编号）: ").strip()
    if not name:
        name = None
    
    speaker_name, msg = speaker_manager.register_speaker(temp_file, name)
    print(msg)
    
    # 可选：试听录音
    play = input("是否试听录音？(y/n): ").strip().lower()
    if play == 'y':
        import platform
        if platform.system() == "Windows":
            os.system(f"start {temp_file}")
        elif platform.system() == "Darwin":
            os.system(f"open {temp_file}")
        else:
            os.system(f"xdg-open {temp_file}")
    
    os.unlink(temp_file)


def main():
    """主函数"""
    print("="*60)
    print("    实时语音识别系统 - 声纹识别版")
    print("="*60)
    
    # 创建识别器
    recognizer = RealtimeASR()
    
    # 加载语音识别模型
    print("\n🎤 正在加载语音识别模型...")
    if not recognizer.load_asr_model("whisper-base"):
        print("❌ 模型加载失败，请检查网络后重试")
        return
    
    print("\n✅ 系统初始化完成！")
    print(f"📊 当前声纹库中有 {len(recognizer.speaker_manager.get_speaker_list())} 位已注册说话人")
    
    is_meeting_active = False
    
    while True:
        print_menu()
        choice = input("\n请选择操作: ").strip()
        
        if choice == '1':
            # 开始新会议
            if is_meeting_active:
                print("\n⚠️ 已有会议在进行中，请先结束当前会议")
                continue
            
            print("\n" + "="*60)
            print("🎙️ 会议已开始，请说话...")
            print("📝 识别结果将实时显示")
            print("👤 声纹识别会自动标注说话人")
            print("⏹️ 按 Enter 键结束会议")
            print("="*60 + "\n")
            
            # 启动识别
            recognizer.start()
            is_meeting_active = True
            
            # 等待用户结束
            input()
            
            # 停止识别
            recognizer.stop()
            is_meeting_active = False
            
            print("\n" + "="*60)
            print("✅ 会议已结束")
            print("="*60)
            
            # 询问是否保存
            if recognizer.meeting_transcript:
                save_choice = input("\n是否保存会议纪要？(y/n): ").strip().lower()
                if save_choice == 'y':
                    recognizer.save_meeting_minutes('txt')
        
        elif choice == '2':
            # 注册新说话人
            if is_meeting_active:
                print("\n⚠️ 会议进行中，请先结束会议")
                continue
            
            register_speaker_interactive(recognizer.speaker_manager, recognizer)
        
        elif choice == '3':
            # 查看已注册说话人
            speakers = recognizer.speaker_manager.get_speaker_list()
            if speakers:
                print("\n📋 已注册说话人:")
                for i, speaker in enumerate(speakers, 1):
                    data = recognizer.speaker_manager.database[speaker]
                    print(f"  {i}. {speaker} (样本数: {data['sample_count']})")
            else:
                print("\n📋 暂无注册说话人")
        
        elif choice == '4':
            # 保存会议纪要
            if not recognizer.meeting_transcript:
                print("\n⚠️ 没有会议记录可保存")
                continue
            
            print("\n请选择导出格式:")
            print("  [1] TXT 格式")
            print("  [2] JSON 格式")
            format_choice = input("请选择 (1/2): ").strip()
            
            if format_choice == '1':
                recognizer.save_meeting_minutes('txt')
            elif format_choice == '2':
                recognizer.save_meeting_minutes('json')
            else:
                print("无效选择")
        
        elif choice == '5':
            # 查看会议摘要
            print(recognizer.get_summary())
        
        elif choice == '6':
            # 清空声纹库
            confirm = input("\n⚠️ 确定要清空所有声纹数据吗？(y/n): ").strip().lower()
            if confirm == 'y':
                recognizer.speaker_manager.clear_database()
        
        elif choice == '7':
            # 退出
            if is_meeting_active:
                print("\n⚠️ 会议正在进行中，请先结束会议")
                continue
            
            print("\n👋 再见！")
            break
        
        else:
            print("\n❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()