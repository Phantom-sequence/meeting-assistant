from __future__ import annotations
import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
try:
    import pyaudio as pyaudio_mod
except Exception:  
    pyaudio_mod = None

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 样本 = 一帧
CHANNELS = 1

VAD_MODE = 2              # 0~3，越大越严格（越不容易把噪声当人声）
PRE_ROLL_FRAMES = 8       # 240ms 预卷，避免"字头"被吃掉
HANGOVER_FRAMES = 16      # 480ms 静音判定一句结束
TRAIL_KEEP_FRAMES = 8     # 句末保留 240ms，避免尾音截断
MIN_SPEECH_MS = 350       # 有效语音短于此值直接丢弃
MAX_UTTERANCE_S = 20      # 超长强制切句

SPK_ACCEPT = 0.68         # 声纹接受阈值（Resemblyzer 余弦相似度）
SPK_MARGIN = 0.06         # best - second 至少差这么多，否则视为"分不清"
SPK_UPDATE_ALPHA = 0.25   # 质心 EMA 更新系数
SPK_SHORT_S = 0.9         # 短于 0.9s 不单独认人，沿用上一句
SPK_MERGE = 0.80          # 会后自动合并同一人的阈值
LIVE_PROBE_INTERVAL = 0.8  # 实时说话人探测间隔
LIVE_PROBE_WINDOW_S = 1.8  # 探测使用的最近语音长度
DEFAULT_MODEL = "whisper-base"
DEFAULT_LANGUAGE = "zh"
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "recordings"
DB_PATH = BASE_DIR / "speakers_v3.npz"
LEGACY_DB = BASE_DIR / "speaker_db.pkl"
CONFIG_FILE = BASE_DIR / "config.json"

PRINT_LOCK = threading.Lock()

def _enable_utf8_output() -> None:
    """Windows 默认 GBK 控制台打不出 emoji/生僻字，统一切到 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_enable_utf8_output()

def log(msg: str = "") -> None:
    with PRINT_LOCK:
        print(msg, flush=True)

def ask(prompt: str) -> Optional[str]:
    """input() 的容错版：管道输入结束/关闭时返回 None，而不是抛异常。"""
    try:
        return input(prompt)
    except EOFError:
        log("")
        return None

def resample_frame(pcm: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == SAMPLE_RATE or pcm.size == 0:
        return pcm
    ratio = src_rate / SAMPLE_RATE
    if float(ratio).is_integer():
        k = int(ratio)
        usable = (pcm.size // k) * k
        block = pcm[:usable].astype(np.float32).reshape(-1, k).mean(axis=1)
        if block.size >= FRAME_SAMPLES:
            return block[:FRAME_SAMPLES].astype(np.int16)
        padded = np.zeros(FRAME_SAMPLES, np.float32)
        padded[: block.size] = block
        return padded.astype(np.int16)
    idx = np.clip(np.arange(FRAME_SAMPLES) * ratio, 0, pcm.size - 1)
    return np.interp(idx, np.arange(pcm.size), pcm.astype(np.float32)).astype(np.int16)

class FrameGate:
    
    def __init__(self, mode: int = VAD_MODE):
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(mode)
            self.backend = "webrtcvad"
        except Exception:
            self._vad = None
            self.backend = "energy"
            self._floor = 200.0

    def is_speech(self, frame: np.ndarray) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame.astype(np.int16).tobytes(), SAMPLE_RATE)
            except Exception:
                pass
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-6)
        # 噪声地板：快速下降、缓慢上升
        if rms < self._floor:
            self._floor = 0.7 * self._floor + 0.3 * rms
        else:
            self._floor = 0.995 * self._floor + 0.005 * rms
        return rms > max(450.0, self._floor * 3.0)

@dataclass
class Utterance:
    index: int
    start: float          # 相对会议开始的秒数
    end: float
    pcm: np.ndarray       # int16 mono 16k
    speech_ms: int


class Segmenter:
    """把连续音频流切成一句一句的 utterance。"""

    def __init__(self, gate: FrameGate):
        self.gate = gate
        self.pre: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES)
        self.cur: list[np.ndarray] = []
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.n_samples = 0
        self.utt_index = 0

    def push(self, frame: np.ndarray) -> list[Utterance]:
        """喂一帧 480 样本，返回本帧结束时"已完成"的句子列表。"""
        self.n_samples += FRAME_SAMPLES
        out: list[Utterance] = []
        voiced = self.gate.is_speech(frame)

        if not self.in_speech:
            if voiced:
                self.in_speech = True
                self.cur = list(self.pre) + [frame]
                self.speech_frames = 1
                self.silence_frames = 0
            else:
                self.pre.append(frame)
            return out

        self.cur.append(frame)
        if voiced:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        if self.silence_frames >= HANGOVER_FRAMES:
            utt = self._finish()
            if utt is not None:
                out.append(utt)
        elif len(self.cur) * FRAME_MS / 1000.0 >= MAX_UTTERANCE_S:
            utt = self._finish(force=True)
            if utt is not None:
                out.append(utt)
        return out

    def _finish(self, force: bool = False) -> Optional[Utterance]:
        frames = self.cur
        # 去掉句末多余静音（触发器需要 480ms 静音，但句子不需要这么多）
        drop = max(0, self.silence_frames - TRAIL_KEEP_FRAMES)
        if drop:
            frames = frames[:-drop] if drop < len(frames) else frames

        pcm = np.concatenate(frames) if frames else np.zeros(0, np.int16)
        speech_ms = self.speech_frames * FRAME_MS
        start = (self.n_samples - len(pcm)) / SAMPLE_RATE

        self.in_speech = False
        self.cur = []
        self.pre.clear()
        self.speech_frames = 0
        self.silence_frames = 0

        if len(pcm) < SAMPLE_RATE * 0.3 or speech_ms < MIN_SPEECH_MS:
            return None

        utt = Utterance(
            index=self.utt_index,
            start=start,
            end=start + len(pcm) / SAMPLE_RATE,
            pcm=pcm,
            speech_ms=speech_ms,
        )
        self.utt_index += 1
        return utt

    def flush(self) -> Optional[Utterance]:
        """停止录音时把最后一句收尾。"""
        if self.in_speech:
            return self._finish(force=True)
        return None

    def current_duration(self) -> float:
        return len(self.cur) * FRAME_MS / 1000.0 if self.in_speech else 0.0

    def current_audio(self, window_s: float = LIVE_PROBE_WINDOW_S) -> Optional[np.ndarray]:
        if not self.in_speech or not self.cur:
            return None
        keep = max(1, int(window_s * 1000 / FRAME_MS))
        return np.concatenate(self.cur[-keep:])

class SpeakerDB:
    """说话人质心库。用 npz 而不是 pickle：无代码执行风险，维度可校验。"""

    def __init__(self, path: Path = DB_PATH, dim: int = 256):
        self.path = Path(path)
        self.dim = dim
        self.names: list[str] = []
        self.centroids = np.zeros((0, dim), dtype=np.float32)
        self.counts: list[int] = []
        self._lock = threading.Lock()

    def load(self) -> bool:
        if self.path.exists():
            try:
                data = np.load(self.path, allow_pickle=False)
                c = data["centroids"].astype(np.float32)
                if c.ndim != 2 or c.shape[1] != self.dim:
                    log(f"⚠️ 声纹库维度 {getattr(c, 'shape', '?')} 与当前模型 {self.dim} 不一致，忽略旧库")
                    return False
                self.centroids = c
                self.names = [str(x) for x in data["names"]]
                self.counts = [int(x) for x in data["counts"]]
                return True
            except Exception as e:
                log(f"⚠️ 声纹库读取失败：{e}")
                return False
        return self._migrate_legacy()

    def _migrate_legacy(self) -> bool:
        """把 v2 的 speaker_db.pkl 迁移成质心库（只读，不改原文件）。"""
        if not LEGACY_DB.exists():
            return False
        try:
            import pickle

            with open(LEGACY_DB, "rb") as f:
                old = pickle.load(f)
            for name, data in old.items():
                embs = [np.asarray(e, dtype=np.float32).reshape(-1) for e in data.get("embeddings", [])]
                embs = [e for e in embs if e.shape[0] == self.dim]
                if not embs:
                    continue
                c = np.mean(embs, axis=0)
                n = np.linalg.norm(c)
                if n == 0:
                    continue
                self.names.append(str(name))
                self.counts.append(len(embs))
                self.centroids = np.vstack([self.centroids, (c / n).astype(np.float32)])
            if self.names:
                log(f"✅ 已从 speaker_db.pkl 迁移 {len(self.names)} 位说话人（原文件保留）")
                self.save()
            return bool(self.names)
        except Exception as e:
            log(f"⚠️ 旧声纹库迁移失败：{e}")
            return False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path,
            centroids=self.centroids.astype(np.float32),
            names=np.array(self.names, dtype=str),
            counts=np.array(self.counts, dtype=np.int64),
        )

    def match(self, emb: np.ndarray) -> tuple[Optional[int], float, float]:
        if not self.names or self.centroids.shape[0] == 0:
            return None, 0.0, 0.0
        sims = self.centroids @ emb
        order = np.argsort(sims)[::-1]
        best = int(order[0])
        second = float(sims[order[1]]) if len(order) > 1 else -1.0
        return best, float(sims[best]), second

    def add(self, name: str, emb: np.ndarray) -> int:
        with self._lock:
            idx = len(self.names)
            self.names.append(name)
            self.counts.append(1)
            self.centroids = np.vstack([self.centroids, emb[None, :].astype(np.float32)])
            self.save()
            return idx

    def add_unknown(self, emb: np.ndarray) -> int:
        return self.add(self._next_unknown_name(), emb)

    def _next_unknown_name(self) -> str:
        used = set()
        for n in self.names:
            if n.startswith("说话人"):
                try:
                    used.add(int(n[3:]))
                except ValueError:
                    pass
        k = 1
        while k in used:
            k += 1
        return f"说话人{k}"

    def update(self, idx: int, emb: np.ndarray, alpha: float = SPK_UPDATE_ALPHA) -> None:
        with self._lock:
            if not (0 <= idx < len(self.names)):
                return
            c = (1 - alpha) * self.centroids[idx] + alpha * emb
            n = np.linalg.norm(c)
            if n > 0:
                self.centroids[idx] = (c / n).astype(np.float32)
            self.counts[idx] = self.counts[idx] + 1

    def rename(self, idx: int, name: str) -> bool:
        with self._lock:
            if not (0 <= idx < len(self.names)):
                return False
            self.names[idx] = name
            self.save()
            return True

    def merge(self, src: int, dst: int) -> Optional[int]:
        if src == dst or not (0 <= src < len(self.names)) or not (0 <= dst < len(self.names)):
            return None
        with self._lock:
            keep_name = self.names[dst]
            lo, hi = (src, dst) if src < dst else (dst, src)
            w_lo, w_hi = max(self.counts[lo], 1), max(self.counts[hi], 1)
            c = self.centroids[lo] * w_lo + self.centroids[hi] * w_hi
            n = np.linalg.norm(c)
            if n > 0:
                self.centroids[lo] = (c / n).astype(np.float32)
            self.counts[lo] = w_lo + w_hi
            self.names[lo] = keep_name
            del self.names[hi]
            del self.counts[hi]
            self.centroids = np.delete(self.centroids, hi, axis=0)
            self.save()
            return lo

    def auto_merge(self, threshold: float = SPK_MERGE, on_merge=None) -> int:
        """会后把相似度极高（>0.80）的簇合并，消除"同一个人被拆成两簇"。

        on_merge(lo, hi, similarity) 会在每次合并后回调，供上层修正历史记录。
        """
        merged = 0
        while len(self.names) > 1:
            sims = self.centroids @ self.centroids.T
            np.fill_diagonal(sims, -1.0)
            i, j = np.unravel_index(np.argmax(sims), sims.shape)
            if sims[i, j] < threshold:
                break
            keep = str(self.names[int(i)])
            hi = int(max(i, j))
            drop = str(self.names[hi])
            lo = self.merge(hi, int(i))
            if lo is None:
                break
            log(f"  自动合并：{drop} 并入 {keep}（相似度 {sims[i, j]:.2f}）")
            if on_merge is not None:
                on_merge(lo, hi)
            merged += 1
        return merged


class SpeakerIdentifier:
    """Resemblyzer 声纹：把一段 PCM 变成归一化 embedding，并决定它属于谁。"""

    def __init__(self, db: SpeakerDB):
        self.db = db
        self.encoder = None
        self._load()

    def _load(self) -> None:
        try:
            from resemblyzer import VoiceEncoder

            self.encoder = VoiceEncoder(verbose=False)
            # 预热：第一次 embed 要初始化 ONNX/线程池，约 1~2s，不能留给第一句话
            self.encoder.embed_utterance((np.random.randn(SAMPLE_RATE * 2) * 0.01).astype(np.float32))
            log("✅ 声纹模型已加载（Resemblyzer）")
        except Exception as e:
            log(f"⚠️ 声纹模型不可用：{e}\n   本次会议将不区分说话人。")

    def embed(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        if self.encoder is None or pcm is None or len(pcm) < SAMPLE_RATE * 0.3:
            return None
        try:
            from resemblyzer import preprocess_wav

            wav = pcm.astype(np.float32) / 32768.0
            try:
                wav = preprocess_wav(wav, source_sr=SAMPLE_RATE)
            except Exception:
                pass
            if len(wav) < SAMPLE_RATE * 0.3:
                # preprocess_wav 的 trim_long_silences 偶尔剪过头，退回原始波形
                wav = pcm.astype(np.float32) / 32768.0
            if len(wav) < SAMPLE_RATE * 0.3:
                return None
            emb = self.encoder.embed_utterance(wav)
            n = float(np.linalg.norm(emb))
            return (emb / n).astype(np.float32) if n else None
        except Exception:
            return None

    def identify(self, utt: Utterance, prev_idx: Optional[int]) -> tuple[Optional[int], float, str]:
        """返回 (说话人编号, 置信度, 判定依据)。"""
        if self.encoder is None:
            return None, 0.0, "no_model"

        emb = self.embed(utt.pcm)
        if emb is None:
            return prev_idx, 0.0, "no_embed"

        dur = len(utt.pcm) / SAMPLE_RATE
        idx, best, second = self.db.match(emb)

        # 短句（"嗯""对"）声纹不可靠，沿用上一句
        if dur < SPK_SHORT_S and prev_idx is not None:
            self.db.update(prev_idx, emb, alpha=0.1)
            return prev_idx, best, "short"

        if idx is not None and best >= SPK_ACCEPT:
            if best - second >= SPK_MARGIN:
                self.db.update(idx, emb)
                return idx, best, "match"
            # 两个人都像：优先沿用上一句，避免来回跳
            if prev_idx is not None and float(self.db.centroids[prev_idx] @ emb) >= best - 0.02:
                self.db.update(prev_idx, emb, alpha=0.15)
                return prev_idx, best, "ambiguous"
            self.db.update(idx, emb, alpha=0.15)
            return idx, best, "best"

        # 陌生声音：开一个新簇
        if dur >= SPK_SHORT_S + 0.3:
            return self.db.add_unknown(emb), best, "new"

        if prev_idx is not None:
            self.db.update(prev_idx, emb, alpha=0.1)
            return prev_idx, best, "short"
        return self.db.add_unknown(emb), best, "new"


# --------------------------------------------------------------------------- #
# ASR 引擎
# --------------------------------------------------------------------------- #
_HALLUCINATION = ("[未识别", "[识别错误", "[BLANK", "（音乐）", "(音乐)", "♪", "字幕由")
_PAREN_ONLY = re.compile(r"^[（(][^）)]{0,12}[）)]$")

# --------------------------------------------------------------------------- #
# 繁体 -> 简体
# whisper 对中文会不定时输出繁体（尤其中英混说、口音重时），这里做后处理。
# 表只收"单字唯一对应"的常用字，避免 著/乾/於 这类多义字被误改。
# --------------------------------------------------------------------------- #
_T2S_PAIRS = """
燒烧 條条 圍围 瀰弥 們们 這这 說说 話话 來来 時时 個个 會会 對对 為为 開开 關关 後后 麼么 沒没 長长 間间 點点從从 現现 樣样 見见 覺觉 讓让 題题 學学 實实 際际 動动 員员 經经 驗验 過过 還还 進进 認认
識识 裡里 裏里 邊边 車车 機机 電电 腦脑 網网 線线 號号 碼码 數数 據据 庫库 檔档 頁页 書书
報报 紙纸 筆笔 記记 錄录 資资 訊讯 傳传 輸输 壞坏 難难 簡简 單单 複复 雜杂 齊齐 總总 監监
組组 師师 發发 結结 續续 斷断 暫暂 聲声 語语 別别 轉转 換换 準准 確确 誤误 幾几 種种 類类
應应 該该 須须 決决 議议 討讨 論论 計计 劃划 執执 標标 務务 責责 負负 擔担 節节 導导 領领
辦办 夠够 嗎吗 囉啰 誰谁 處处 檔档 團团 圓圆 壓压 歷历 廳厅 廣广 廠厂 張张 強强 歸归 當当
徵征 憂忧 懷怀 態态 戰战 擬拟 攝摄 敗败 敵敌 無无 舊旧 術术 東东 貨货 質质 費费 賣卖 買买
貼贴 貸贷 貿贸 賞赏 賜赐 賠赔 賦赋 賬账 賭赌 賽赛 贊赞 趕赶 趨趋 跡迹 蹤踪 軌轨 軍军 軟软
較较 載载 轄辖 農农 適适 遲迟 選选 遺遗 遙遥 遞递 遠远 邏逻 郵邮 醫医 釋释 鋼钢 錢钱 錯错
鍵键 鐘钟 鐵铁 銀银 鎮镇 閃闪 閱阅 陣阵 陽阳 陰阴 陳陈 陸陆 駕驾 髮发 鬆松 魚鱼 鳥鸟 鳴鸣
麗丽 麥麦 黃黄 黨党 齒齿 齡龄 龍龙 龜龟 給给 統统 絲丝 綜综 綠绿 緊紧 練练 縮缩 績绩 織织
繼继 罰罚 罷罢 羅罗 義义 習习 聖圣 聞闻 聯联 聰聪 職职 聽听 脅胁 臉脸 膽胆 臨临 舉举 艱艰
藝艺 葉叶 蓋盖 藍蓝 藥药 蘇苏 蘋苹 蟲虫 補补 裝装 褲裤 觀观 規规 視视 親亲 訂订 訓训 講讲
許许 設设 訪访 證证 評评 譯译 護护 讀读 變变 讚赞 貝贝 財财 貢贡 貧贫 買买 賀贺 貴贵 賓宾
賢贤 賴赖 購购 贈赠 贏赢 贖赎 趙赵 踐践 躍跃 輔辅 輕轻 輛辆 輝辉 輪轮 轟轰 辭辞 運运 達达
違违 遜逊 遼辽 邁迈 鄉乡 針针 釘钉 釣钓 鈴铃 銀银 銅铜 銘铭 錫锡 鍛锻 鏡镜 鑑鉴 門门 閉闭
闡阐 隊队 階阶 隨随 險险 隱隐 雖虽 雙双 雞鸡 離离 雲云 霧雾 靈灵 靜静 韓韩 頂顶 項项 順顺
預预 頓顿 領领 頭头 頸颈 頻频 顆颗 額额 顏颜 願愿 顧顾 顯显 風风 飛飞 飢饥 飯饭 飲饮 飾饰
飽饱 館馆 馬马 騎骑 騙骗 驅驱 驚惊 體体 鬧闹 與与 於于 係系 嘗尝 嗎吗 噸吨 壯壮 夢梦 奪夺
奮奋 嬰婴 寧宁 寶宝 寵宠 審审 寫写 寬宽 將将 專专 尋寻 屆届 屍尸 屬属 岡冈 島岛 嶺岭 峽峡
巖岩 幫帮 幣币 廢废 廬庐 張张 彌弥 彎弯 彙汇 徑径 復复 恆恒 悅悦 悶闷 惡恶 惱恼 愛爱 愜惬
慚惭 慘惨 慣惯 慮虑 慶庆 憑凭 懇恳 懲惩 懸悬 懶懒 懼惧 戀恋 搖摇 撈捞 撐撑 撥拨 撫抚 擁拥
擇择 擋挡 擠挤 擾扰 攤摊 攪搅 權权 攬揽 曬晒 爭争 牆墙 獨独 獎奖 獲获 現现 現现 璽玺 產产
畢毕 異异 疊叠 瘋疯 療疗 瘡疮 癒愈 發发 皚皑 盜盗 睜睁 碼码 礎础 礙碍 祕秘 祿禄 禍祸 禱祷
稅税 稱称 積积 穩稳 窮穷 竊窃 競竞 簾帘 籃篮 籤签 粵粤 糧粮 紀纪 約约 級级 紅红 純纯 細细
終终 組组 結结 絕绝 給给 統统 絲丝 經经 綠绿 維维 綱纲 網网 罰罚 罷罢 翱翱 恥耻 聳耸 肅肃
肝肝 脅胁 膽胆 臟脏 臨临 舉举 舍舍 藝艺 節节 範范 築筑 簡简 籃篮 臨临 艱艰 蘭兰 處处 號号
虛虚 虧亏 蠻蛮 裝装 製制 複复 襄襄 見见 親亲 覽览 觀观 規规 覺觉 觸触 計计 訂订 討讨 訓训
記记 訊讯 託托 記记 訪访 設设 許许 訴诉 診诊 註注 詞词 試试 詩诗 話话 詳详 認认 誇夸 誌志
語语 誠诚 說说 誰谁 課课 調调 談谈 請请 諸诸 諾诺 謀谋 謂谓 講讲 謝谢 證证 識识 譜谱 警警
譯译 議议 護护 讀读 變变 讚赞 豬猪 貓猫 貝贝 貞贞 負负 財财 貢贡 貧贫 貨货 販贩 貪贪 責责
貴贵 買买 貸贷 費费 賀贺 貼贴 貿贸 資资 賈贾 賓宾 賃赁 賜赐 賞赏 賠赔 賣卖 賴赖 購购 賽赛
贈赠 贊赞 趕赶 起起 跟跟 跡迹 踐践 車车 軌轨 軍军 軟软 較较 載载 輔辅 輕轻 輛辆 輝辉 輪轮
輸输 轉转 轟轰 辭辞 農农 迴回 這这 連连 週周 進进 運运 過过 達达 違违 遙遥 遞递 遠远 遜逊
適适 遲迟 選选 遺遗 遼辽 邁迈 還还 邊边 邏逻 郵邮 鄉乡 醫医 釋释 金金 針针 釘钉 釣钓 鈴铃
鉛铅 銀银 銅铜 銘铭 鋼钢 錄录 錢钱 錫锡 鍵键 鍊链 鎖锁 鍋锅 鏈链 鐘钟 鐵铁 鑄铸 鏡镜 鑑鉴
長长 門门 閉闭 開开 閏闰 閑闲 間间 閘闸 關关 闡阐 陣阵 陰阴 陳陈 陸陆 陽阳 隊队 階阶 隨随
險险 隱隐 雖虽 雙双 雜杂 雞鸡 離离 難难 雲云 電电 霧雾 靈灵 靜静 韓韩 頁页 頂顶 項项 順顺
須须 預预 頓顿 領领 頭头 頸颈 頻频 顆颗 題题 額额 顏颜 願愿 類类 顧顾 顯显 風风 飛飞 飢饥
飯饭 飲饮 飾饰 飽饱 館馆 馬马 駕驾 騎骑 騙骗 驅驱 驗验 驚惊 體体 髮发 鬆松 鬥斗 鬧闹 魚鱼
鳥鸟 鳴鸣 鴻鸿 鵝鹅 麗丽 麥麦 黃黄 點点 黨党 齊齐 齒齿 齡龄 龍龙 龜龟
問问 檢检 備备 內内 並并 測测 紹绍 兩两 業业 產产 樂乐 勢势 區区 華华 萬万 億亿 衛卫 衝冲
競竞 層层 屬属 嚴严 嚇吓 媽妈 孫孙 婦妇 啟启 嘆叹 喬乔 墊垫 壘垒 巒峦 寢寝 囑嘱 囂嚣 壩坝
"""

_T2S: dict[str, str] = {}
for _tok in _T2S_PAIRS.split():
    if len(_tok) == 2:
        _T2S[_tok[0]] = _tok[1]
del _tok


def to_simplified(text: str) -> str:
    """把 whisper 偶尔吐出的繁体字转成简体。表里没有的字原样保留。"""
    if not text:
        return text
    return "".join(_T2S.get(ch, ch) for ch in text)


# --------------------------------------------------------------------------- #
# 热词表：修正 ASR 固定错法（如 连调 -> 联调）
# --------------------------------------------------------------------------- #
HOTWORDS_FILE = BASE_DIR / "hotwords.txt"
_HOTWORDS: list[tuple[str, str]] = []


def load_hotwords(path: Path = HOTWORDS_FILE, quiet: bool = False) -> list[tuple[str, str]]:
    """读取 错词=>正确写法 的映射；文件不存在就提示一次，不影响主流程。"""
    global _HOTWORDS
    items: list[tuple[str, str]] = []
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                sep = "=>" if "=>" in line else ("->" if "->" in line else None)
                if sep is None:
                    continue
                wrong, right = (x.strip() for x in line.split(sep, 1))
                if wrong and right:
                    items.append((wrong, right))
    except Exception as e:
        if not quiet:
            log(f"⚠️ 热词表读取失败（忽略）：{e}")
    # 长词优先，避免 "联调" 被 "调" 抢先替换
    items.sort(key=lambda kv: -len(kv[0]))
    _HOTWORDS = items
    if items and not quiet:
        log(f"📚 已加载热词表：{len(items)} 条（{path.name}）")
    return items


def apply_hotwords(text: str) -> str:
    for wrong, right in _HOTWORDS:
        text = text.replace(wrong, right)
    return text


def clean_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    for bad in _HALLUCINATION:
        if bad in t:
            return ""
    # "(水)"、"（笑声）" 这类纯音效/噪声标注，对纪要是噪声
    if _PAREN_ONLY.match(t):
        return ""
    if len(t) <= 2 and t.strip("。，、.!? ") == "":
        return ""
    t = to_simplified(t)
    if _HOTWORDS:
        t = apply_hotwords(t)
    return t


class WhisperEngine:
    """onnx-asr 的 whisper 封装。输入 PCM int16，直接给 numpy，不走临时文件。"""

    def __init__(self, model: str = DEFAULT_MODEL, language: str = DEFAULT_LANGUAGE):
        self.model_name = model
        self.language = language
        self.model = None

    def load(self) -> bool:
        try:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from onnx_asr import load_model

            self.model = load_model(self.model_name)
            # 预热：第一次 recognize 要建 session / 预处理资源
            try:
                self.model.recognize(
                    (np.random.randn(SAMPLE_RATE) * 0.003).astype(np.float32),
                    sample_rate=SAMPLE_RATE,
                    language=self.language,
                )
            except Exception:
                pass
            log(f"✅ 语音识别模型已加载（{self.model_name}）")
            return True
        except Exception as e:
            log(f"❌ 语音识别模型加载失败：{e}")
            log(f"   提示：{self.model_name} 首次使用需要联网下载；离线时请改用已缓存的 whisper-base。")
            return False

    def transcribe(self, pcm: np.ndarray) -> str:
        if self.model is None or pcm is None or len(pcm) < SAMPLE_RATE * 0.2:
            return ""
        try:
            wav = pcm.astype(np.float32) / 32768.0
            res = self.model.recognize(wav, sample_rate=SAMPLE_RATE, language=self.language)
            text = getattr(res, "text", None) or str(res)
            return clean_text(text)
        except Exception as e:
            return ""


# --------------------------------------------------------------------------- #
# 会议会话
# --------------------------------------------------------------------------- #


@dataclass


class Turn:
    index: int
    start: float
    end: float
    speaker_idx: Optional[int]
    speaker_conf: float
    text: str
    speech_ms: int = 0
    asr_ms: float = 0.0
    reason: str = ""

    def to_json(self, db: SpeakerDB) -> dict:
        if self.speaker_idx is None:
            name = "未知"
        elif self.speaker_idx < len(db.names):
            name = db.names[self.speaker_idx]
        else:
            name = f"说话人{self.speaker_idx + 1}"
        return {
            "index": self.index,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker": name,
            "speaker_conf": round(self.speaker_conf, 3),
            "text": self.text,
        }


class MeetingSession:
    def __init__(
        self,
        engine: WhisperEngine,
        identifier: SpeakerIdentifier,
        db: SpeakerDB,
        device_index: Optional[int] = None,
        out_dir: Path = OUT_DIR,
    ):
        self.engine = engine
        self.identifier = identifier
        self.db = db
        self.device_index = device_index
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wav_path = self.out_dir / f"meeting_{self.session_id}.wav"
        self.jsonl_path = self.out_dir / f"meeting_{self.session_id}.jsonl"

        self.audio_q: queue.Queue = queue.Queue(maxsize=200)
        self.asr_q: queue.Queue = queue.Queue()

        self.segmenter = Segmenter(FrameGate())
        self.turns: list[Turn] = []
        self.running = False
        self._t0 = 0.0
        self._prev_speaker: Optional[int] = None
        self._last_probe = 0.0
        self._probe_warned = False
        self._asr_errors = 0
        self._level = 0.0
        self._turns_lock = threading.Lock()

        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._asr_thread: Optional[threading.Thread] = None
        self._wav: Optional[wave.Wave_write] = None
        self._stream = None
        self._pyaudio = None
        self.capture_rate = SAMPLE_RATE
        self.capture_chunks = FRAME_SAMPLES
        self.audio_seconds = 0.0

    # ---- 生命周期 --------------------------------------------------------- #
    def _open_input(self):
        """优先 16k 单声道；不支持就退到设备原生采样率，再重采样到 16k。"""
        import pyaudio

        candidates = [(SAMPLE_RATE, CHANNELS)]
        try:
            info = self._pyaudio.get_device_info_by_index(self.device_index) if self.device_index is not None else None
            native = int(info["defaultSampleRate"]) if info else 48000
        except Exception:
            native = 48000
        if native != SAMPLE_RATE:
            candidates.append((native, 1))
        candidates.append((48000, 1))

        last_err: Optional[Exception] = None
        for rate, ch in candidates:
            try:
                stream = self._pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=ch,
                    rate=rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=int(FRAME_SAMPLES * rate / SAMPLE_RATE),
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            self.capture_rate = rate
            self.capture_chunks = int(FRAME_SAMPLES * rate / SAMPLE_RATE)
            if rate != SAMPLE_RATE:
                log(f"ℹ️ 设备不支持 16kHz，改用 {rate}Hz 采集后重采样（音质无影响）")
            return stream
        log(f"❌ 无法打开麦克风：{last_err}")
        return None

    def start(self, stream=None) -> bool:
        """stream 传入一个鸭子类型的音频源即可离线回放/自测，不碰真实麦克风。"""
        if stream is None:
            self._pyaudio = pyaudio_mod.PyAudio() if pyaudio_mod else None
            if self._pyaudio is None:
                log("❌ 未安装 pyaudio：pip install pyaudio")
                return False
            stream = self._open_input()
            if stream is None:
                self._pyaudio.terminate()
                self._pyaudio = None
                return False
        self._stream = stream

        self._wav = wave.open(str(self.wav_path), "wb")
        self._wav.setnchannels(CHANNELS)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)

        self.running = True
        self._t0 = time.time()
        self._prev_speaker = None
        self._last_probe = 0.0

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()
        self._asr_thread.start()
        return True

    def stop(self) -> None:
        self.running = False
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        for t in (self._capture_thread, self._process_thread):
            if t is not None:
                t.join(timeout=5)
        if self._asr_thread is not None:
            self._asr_thread.join(timeout=120)  # 等最后几句识别完

        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                pass
            self._wav = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None

    # ---- 采集 ------------------------------------------------------------- #
    def _capture_loop(self) -> None:
        try:
            while self.running:
                try:
                    data = self._stream.read(self.capture_chunks, exception_on_overflow=False)
                except Exception:
                    if not self.running:
                        break
                    time.sleep(0.01)
                    continue
                pcm = np.frombuffer(data, dtype=np.int16)
                if self.capture_rate != SAMPLE_RATE:
                    pcm = resample_frame(pcm, self.capture_rate)
                else:
                    pcm = pcm.copy()
                self.audio_seconds += len(pcm) / SAMPLE_RATE
                self._level = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) / 32768.0
                try:
                    self.audio_q.put_nowait(pcm)
                except queue.Full:
                    # 处理端落后：丢最旧的，保证延迟不无限增长
                    try:
                        self.audio_q.get_nowait()
                        self.audio_q.put_nowait(pcm)
                    except queue.Empty:
                        pass
                if self._wav is not None:
                    try:
                        self._wav.writeframes(pcm.tobytes())
                    except Exception:
                        pass
        finally:
            try:
                self.audio_q.put(None, timeout=1)
            except Exception:
                pass

    # ---- 断句 + 实时说话人 ------------------------------------------------- #
    def _process_loop(self) -> None:
        while True:
            try:
                item = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                if not self.running:
                    break
                continue
            if item is None:
                break
            for utt in self.segmenter.push(item):
                self.asr_q.put(utt)
            try:
                self._maybe_live_probe()
            except Exception as e:  # 实时预览失败不能拖垮整条管线
                if not self._probe_warned:
                    self._probe_warned = True
                    log(f"⚠️ 实时说话人预览异常（不影响最终结果）：{e}")

        last = self.segmenter.flush()
        if last is not None:
            self.asr_q.put(last)
        self.asr_q.put(None)  # 通知识别线程收尾

    def _maybe_live_probe(self) -> None:
        if getattr(self.identifier, "encoder", None) is None or not self.segmenter.in_speech:
            return
        now = time.time()
        if now - self._last_probe < LIVE_PROBE_INTERVAL:
            return
        if self.segmenter.current_duration() < 0.8:
            return
        self._last_probe = now

        audio = self.segmenter.current_audio()
        emb = self.identifier.embed(audio) if audio is not None else None
        if emb is None:
            return
        idx, best, _ = self.db.match(emb)
        if idx is not None and best >= SPK_ACCEPT:
            name = self.db.names[idx]
        else:
            name = "识别中…"
        dur = self.segmenter.current_duration()
        bar = "█" * int(min(1.0, self._level * 6) * 12)
        with PRINT_LOCK:
            sys.stdout.write(f"\r🎙 {dur:4.1f}s  👤 {name:<8}  {bar:<12}")
            sys.stdout.flush()

    # ---- 识别 ------------------------------------------------------------- #
    def _asr_loop(self) -> None:
        while True:
            utt = self.asr_q.get()
            if utt is None:
                break
            try:
                self._handle_utterance(utt)
            except Exception as e:  # 单句失败不能拖垮整场会议
                self._asr_errors += 1
                if self._asr_errors <= 3:
                    log(f"⚠️ 这一段识别失败（已跳过）：{e}")

    def _handle_utterance(self, utt: Utterance) -> None:
        t1 = time.time()
        text = self.engine.transcribe(utt.pcm)
        asr_ms = (time.time() - t1) * 1000

        if not text:
            return

        idx, conf, reason = self.identifier.identify(utt, self._prev_speaker)
        self._prev_speaker = idx

        with self._turns_lock:
            turn = Turn(
                index=len(self.turns) + 1,
                start=utt.start,
                end=utt.end,
                speaker_idx=idx,
                speaker_conf=conf,
                text=text,
                speech_ms=utt.speech_ms,
                asr_ms=asr_ms,
                reason=reason,
            )
            self.turns.append(turn)
            self._append_jsonl(turn)

        name = self.db.names[idx] if idx is not None and idx < len(self.db.names) else "未知"
        stamp = f"{int(utt.start // 60):02d}:{int(utt.start % 60):02d}"
        with PRINT_LOCK:
            sys.stdout.write("\r" + " " * 70 + "\r")
            print(f"[{stamp}] {name}: {text}", flush=True)

    def _append_jsonl(self, turn: Turn) -> None:
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn.to_json(self.db), ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---- 统计 ------------------------------------------------------------- #
    def duration(self) -> float:
        return time.time() - self._t0 if self._t0 else 0.0

    # ---- 会后整理说话人（必须同时修正历史 turn 的编号） ------------------- #
    def _remap_turns(self, lo: int, hi: int) -> None:
        """db 删掉编号 hi、保留 lo 之后，修正历史 turn 的编号：

        - 原本属于 hi 的 → 并到 lo
        - 原本大于 hi 的 → 整体减一
        - lo < idx < hi 的不动（被删的是它们上面那个）
        """
        with self._turns_lock:
            for t in self.turns:
                if t.speaker_idx is None:
                    continue
                if t.speaker_idx == hi:
                    t.speaker_idx = lo
                elif t.speaker_idx > hi:
                    t.speaker_idx -= 1

    def merge_speakers(self, src: int, dst: int) -> Optional[int]:
        keep_name = self.db.names[dst] if 0 <= dst < len(self.db.names) else "?"
        lo = self.db.merge(src, dst)
        if lo is None:
            return None
        self._remap_turns(lo, max(src, dst))
        log(f"✅ 已合并：编号 {max(src, dst)} 并入 {lo}（{keep_name}）")
        return lo

    def auto_merge_speakers(self, threshold: float = SPK_MERGE) -> int:
        return self.db.auto_merge(threshold, on_merge=lambda lo, hi: self._remap_turns(lo, hi))

    def speaker_stats(self) -> dict[int, dict]:
        stats: dict[int, dict] = {}
        with self._turns_lock:
            turns = list(self.turns)
        for t in turns:
            if t.speaker_idx is None:
                continue
            s = stats.setdefault(t.speaker_idx, {"turns": 0, "seconds": 0.0, "chars": 0})
            s["turns"] += 1
            s["seconds"] += max(0.0, t.end - t.start)
            s["chars"] += len(t.text)
        return stats


# --------------------------------------------------------------------------- #
# 纪要生成
# --------------------------------------------------------------------------- #


def coalesce(turns: list[Turn], max_gap: float = 1.0, max_span: float = 30.0) -> list[Turn]:
    """把同一人、间隔很短的碎句并成一段，读起来才像会议记录。"""
    out: list[Turn] = []
    for t in turns:
        if out:
            prev = out[-1]
            same = t.speaker_idx == prev.speaker_idx
            gap = t.start - prev.end
            span = t.end - prev.start
            if same and gap <= max_gap and span <= max_span:
                if t.text == prev.text:
                    continue
                prev.text = f"{prev.text} {t.text}".strip()
                prev.end = t.end
                prev.speech_ms += t.speech_ms
                continue
        out.append(
            Turn(
                index=0,
                start=t.start,
                end=t.end,
                speaker_idx=t.speaker_idx,
                speaker_conf=t.speaker_conf,
                text=t.text,
                speech_ms=t.speech_ms,
                asr_ms=t.asr_ms,
                reason=t.reason,
            )
        )
    for i, t in enumerate(out, 1):
        t.index = i
    return out


def build_minutes(session: MeetingSession, db: SpeakerDB, title: str = "会议纪要") -> dict:
    turns = coalesce(list(session.turns))

    stats: dict[int, dict] = {}
    for t in turns:
        if t.speaker_idx is None:
            continue
        s = stats.setdefault(t.speaker_idx, {"turns": 0, "seconds": 0.0, "chars": 0})
        s["turns"] += 1
        s["seconds"] += max(0.0, t.end - t.start)
        s["chars"] += len(t.text)
    total_speech = sum(s["seconds"] for s in stats.values()) or 1.0

    speakers = []
    for idx, s in sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]):
        name = db.names[idx] if idx < len(db.names) else f"说话人{idx + 1}"
        speakers.append(
            {
                "name": name,
                "turns": s["turns"],
                "speech_seconds": round(s["seconds"], 1),
                "chars": s["chars"],
                "ratio": round(s["seconds"] / total_speech, 3),
            }
        )

    turn_dicts = []
    for t in turns:
        d = t.to_json(db)
        # 兜底：即使某句在识别环节漏做，导出前再转一次（两个函数都是幂等的）
        d["text"] = to_simplified(apply_hotwords(d["text"]))
        turn_dicts.append(d)

    return {
        "title": title,
        "session_id": session.session_id,
        "started_at": datetime.fromtimestamp(session._t0).isoformat(timespec="seconds") if session._t0 else "",
        "recorded_seconds": round(session.audio_seconds, 1),
        "audio_file": str(session.wav_path),
        "turn_count": len(turns),
        "speakers": speakers,
        "turns": turn_dicts,
    }


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def minutes_to_markdown(m: dict, summary: Optional[dict] = None) -> str:
    lines = [f"# {m['title']}", ""]
    lines.append(f"- 会议编号：{m['session_id']}")
    if m.get("started_at"):
        lines.append(f"- 开始时间：{m['started_at'].replace('T', ' ')}")
    lines.append(f"- 录音时长：{m['recorded_seconds']} 秒")
    lines.append(f"- 录音文件：`{Path(m['audio_file']).name}`")
    lines.append(f"- 发言段数：{m['turn_count']}，识别到 {len(m['speakers'])} 位说话人")
    lines.append("")

    lines.append("## 参会人（按发言时长）")
    lines.append("")
    lines.append("| 说话人 | 发言段数 | 发言时长 | 占比 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for s in m["speakers"]:
        minutes = s["speech_seconds"] / 60
        lines.append(f"| {s['name']} | {s['turns']} | {minutes:.1f} 分钟 | {s['ratio'] * 100:.0f}% |")
    lines.append("")

    if summary:
        lines.append("## 一、会议摘要")
        lines.append("")
        for p in summary.get("summary", []):
            lines.append(f"- {p}")
        lines.append("")
        if summary.get("topics"):
            lines.append("## 二、讨论议题")
            lines.append("")
            for t in summary["topics"]:
                lines.append(f"### {t.get('title', '议题')}")
                for p in t.get("points", []):
                    lines.append(f"- {p}")
                lines.append("")
        if summary.get("decisions"):
            lines.append("## 三、决议")
            lines.append("")
            for d in summary["decisions"]:
                lines.append(f"- {d}")
            lines.append("")
        if summary.get("actions"):
            lines.append("## 四、待办事项")
            lines.append("")
            lines.append("| 事项 | 负责人 | 截止时间 |")
            lines.append("| --- | --- | --- |")
            for a in summary["actions"]:
                lines.append(f"| {a.get('task', '')} | {a.get('owner', '待定')} | {a.get('due', '待定')} |")
            lines.append("")
        if summary.get("risks"):
            lines.append("## 五、风险与未决问题")
            lines.append("")
            for r in summary["risks"]:
                lines.append(f"- {r}")
            lines.append("")

    lines.append("## 会议原文")
    lines.append("")
    for t in m["turns"]:
        lines.append(f"**[{_fmt_ts(t['start'])}] {t['speaker']}**")
        lines.append("")
        lines.append(t["text"])
        lines.append("")
    return "\n".join(lines)


def save_minutes(m: dict, summary: Optional[dict], out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    md_path = out_dir / f"minutes_{m['session_id']}.md"
    json_path = out_dir / f"minutes_{m['session_id']}.json"
    md_path.write_text(minutes_to_markdown(m, summary), encoding="utf-8")

    payload = dict(m)
    payload["summary"] = summary or {}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path


def transcript_to_txt(m: dict, summary: Optional[dict] = None) -> str:
    """纯文本纪要：说话人 + 每句原话 + 时间戳，用记事本就能直接看。"""
    lines: list[str] = []
    lines.append(f"{m['title']}")
    lines.append("=" * 46)
    if m.get("started_at"):
        lines.append(f"会议时间：{m['started_at'].replace('T', ' ')}")
    secs = m.get("recorded_seconds", 0)
    lines.append(f"录音时长：{int(secs) // 60} 分 {int(secs) % 60} 秒")
    lines.append(f"发言段数：{m['turn_count']}")
    if m["speakers"]:
        names = "、".join(f"{s['name']}（{s['ratio'] * 100:.0f}%）" for s in m["speakers"])
        lines.append(f"参会人：{names}")
    lines.append("")

    if summary:
        lines.append("-" * 46)
        lines.append("【会议摘要】")
        for p in summary.get("summary", []):
            lines.append(f"  · {p}")
        if summary.get("topics"):
            lines.append("")
            lines.append("【讨论议题】")
            for t in summary["topics"]:
                lines.append(f"  ◆ {t.get('title', '议题')}")
                for p in t.get("points", []):
                    lines.append(f"    - {p}")
        if summary.get("decisions"):
            lines.append("")
            lines.append("【决议】")
            for d in summary["decisions"]:
                lines.append(f"  · {d}")
        if summary.get("actions"):
            lines.append("")
            lines.append("【待办事项】")
            for a in summary["actions"]:
                lines.append(f"  □ {a.get('task', '')}｜负责人：{a.get('owner', '待定')}｜截止：{a.get('due', '待定')}")
        if summary.get("risks"):
            lines.append("")
            lines.append("【风险与未决问题】")
            for r in summary["risks"]:
                lines.append(f"  ! {r}")
        lines.append("")

    lines.append("-" * 46)
    lines.append("【会议内容】")
    lines.append("")
    prev = None
    for t in m["turns"]:
        if prev is not None and t["speaker"] != prev:
            lines.append("")
        lines.append(f"[{_fmt_ts(t['start'])}] {t['speaker']}：{t['text']}")
        prev = t["speaker"]
    return "\n".join(lines).rstrip() + "\n"


def export_transcript_txt(
    m: dict, summary: Optional[dict] = None, out_dir: Optional[Path] = None, path: Optional[Path] = None
) -> Path:
    """把纪要写成 txt，默认放在本 py 文件同一目录。"""
    if path is None:
        target_dir = Path(out_dir) if out_dir is not None else BASE_DIR
        path = target_dir / f"会议纪要_{m['session_id']}.txt"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript_to_txt(m, summary), encoding="utf-8")
    return path


def llm_summary(m: dict, model: str = "deepseek-chat", timeout: int = 120) -> Optional[dict]:
    """可选：调用 DeepSeek 生成结构化摘要。没配 key 就返回 None，不影响主流程。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None

    text = "\n".join(f"[{_fmt_ts(t['start'])}] {t['speaker']}: {t['text']}" for t in m["turns"])
    if len(text) > 24000:  # 粗略截断，避免超长
        head, tail = text[:16000], text[-8000:]
        text = head + "\n……（中间省略）……\n" + tail

    prompt = (
        "你是会议记录员。下面是一场会议的逐字转写。请只输出 JSON，不要输出任何解释。\n"
        "JSON 结构："
        '{"summary":["3-5条概括"],'
        '"topics":[{"title":"议题名","points":["要点"]}],'
        '"decisions":["已明确的结论"],'
        '"actions":[{"task":"待办","owner":"负责人或待定","due":"截止时间或待定"}],'
        '"risks":["风险或未决问题"]}\n'
        "要求：只写转写里真实出现的内容，不要编造；负责人和时间没提到就写“待定”。\n\n"
        f"会议转写：\n{text}"
    )

    try:
        import requests

        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        log(f"⚠️ LLM 摘要失败（已跳过）：{e}")
        return None


# --------------------------------------------------------------------------- #
# 设备 / 注册
# --------------------------------------------------------------------------- #


def list_devices() -> list[dict]:
    import pyaudio

    p = pyaudio.PyAudio()
    out = []
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                out.append(
                    {
                        "index": i,
                        "name": info.get("name", ""),
                        "channels": int(info.get("maxInputChannels", 0)),
                        "rate": int(info.get("defaultSampleRate", 0)),
                    }
                )
    finally:
        p.terminate()
    return out


def print_devices() -> None:
    devices = list_devices()
    if not devices:
        log("没有找到输入设备")
        return
    log("可用输入设备：")
    for d in devices:
        log(f"  [{d['index']}] {d['name']}  通道:{d['channels']}  默认采样率:{d['rate']}")


def record_clip(seconds: float, device_index: Optional[int] = None) -> Optional[np.ndarray]:
    """录音若干秒，返回 16k int16 PCM（设备不支持 16k 时自动按 48k 录再重采样）。"""
    import pyaudio

    p = pyaudio.PyAudio()
    stream = None
    rate = SAMPLE_RATE
    chunks = FRAME_SAMPLES
    try:
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=FRAME_SAMPLES,
            )
        except Exception:
            rate, chunks = 48000, FRAME_SAMPLES * 3
            stream = p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunks,
            )
            log(f"ℹ️ 设备不支持 16kHz，改用 {rate}Hz 采集后重采样")
        frames = []
        n = int(seconds * SAMPLE_RATE / FRAME_SAMPLES)
        for i in range(n):
            pcm = np.frombuffer(stream.read(chunks, exception_on_overflow=False), np.int16)
            if rate != SAMPLE_RATE:
                pcm = resample_frame(pcm, rate)
            frames.append(pcm)
            if i % 10 == 0:
                sys.stdout.write(f"\r🔴 录音中… {i * FRAME_MS / 1000:4.1f}s / {seconds:.1f}s")
                sys.stdout.flush()
        sys.stdout.write("\r" + " " * 40 + "\r")
        return np.concatenate(frames) if frames else None
    except Exception as e:
        log(f"❌ 录音失败：{e}")
        return None
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()


def register_speaker(identifier: SpeakerIdentifier, db: SpeakerDB, device_index: Optional[int]) -> None:
    log("\n===== 注册说话人 =====")
    log("请对着麦克风连续说 5 秒左右（用开会时的音量，别太近）")
    name = (ask("姓名：") or "").strip()
    if not name:
        log("已取消")
        return

    pcm = record_clip(5.0, device_index)
    if pcm is None:
        return
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms < 200:
        log("⚠️ 音量太小，几乎没录到声音，请检查麦克风后重试")
        return

    emb = identifier.embed(pcm)
    if emb is None:
        log("⚠️ 声纹提取失败（录音太短或静音太多）")
        return

    if name in db.names:
        idx = db.names.index(name)
        db.update(idx, emb, alpha=0.5)
        log(f"✅ 已更新 {name} 的声纹（样本数 {db.counts[idx]}）")
    else:
        db.add(name, emb)
        log(f"✅ 已注册 {name}")


# --------------------------------------------------------------------------- #
# 菜单
# --------------------------------------------------------------------------- #


def show_speakers(db: SpeakerDB) -> None:
    if not db.names:
        log("还没有任何说话人")
        return
    log("\n说话人列表：")
    for i, (name, cnt) in enumerate(zip(db.names, db.counts)):
        log(f"  [{i}] {name}  样本数:{cnt}")


def manage_speakers(db: SpeakerDB, session: Optional[MeetingSession] = None) -> None:
    """会后整理：改名 / 合并 / 自动合并。

    传入 session 时会同步修正历史 turn 的编号，否则合并后纪要里的人名会错位。
    """
    if not db.names:
        log("还没有任何说话人")
        return
    if session is not None:
        session.auto_merge_speakers()
    else:
        db.auto_merge()
    show_speakers(db)
    log("\n命令：")
    log("  r <编号> <新名字>   重命名（比如把 说话人1 改成 张三）")
    log("  m <编号> <编号>     合并两个说话人（把后一个并进前一个）")
    log("  a                    再次自动合并")
    log("  回车                 返回")
    while True:
        cmd = (ask("> ") or "").strip()
        if not cmd:
            return
        parts = cmd.split(maxsplit=1)
        try:
            if parts[0] == "r" and len(parts) == 2 and " " in parts[1]:
                idx_s, new_name = parts[1].split(maxsplit=1)
                if db.rename(int(idx_s), new_name.strip()):
                    log(f"✅ 已改名：{new_name.strip()}")
            elif parts[0] == "r":
                log("用法：r <编号> <新名字>")
            elif parts[0] == "m":
                a, b = (int(x) for x in parts[1].split())
                if session is not None:
                    session.merge_speakers(b, a)
                elif db.merge(b, a) is not None:
                    log(f"✅ 已合并 {b} → {a}")
            elif parts[0] == "a":
                n = session.auto_merge_speakers() if session is not None else db.auto_merge()
                log(f"✅ 自动合并了 {n} 组")
            else:
                log("未知命令")
        except Exception as e:
            log(f"⚠️ {e}")
        show_speakers(db)


def run_meeting(engine: WhisperEngine, identifier: SpeakerIdentifier, db: SpeakerDB, device: Optional[int]) -> None:
    session = MeetingSession(engine, identifier, db, device_index=device)

    log("=" * 66)
    log("🎙️  会议开始。正常说话即可，我会自动断句、认人。")
    log("    按 Enter 结束会议（结束后可以改名/合并说话人、生成纪要）")
    log("=" * 66)

    if not session.start():
        return
    try:
        ask("") or None
    except KeyboardInterrupt:
        pass

    log("\n⏹  正在收尾（等最后几句识别完）…")
    session.stop()

    turns = session.turns
    if not turns:
        log("没有识别到任何内容。请检查：麦克风是否被其它程序占用、VAD_MODE 是否过严。")
        return

    log(f"\n✅ 会议结束：{len(turns)} 段发言，录音已保存 {session.wav_path.name}")

    summary: Optional[dict] = None
    exported: Optional[Path] = None
    while True:
        log("")
        log("-" * 50)
        log("  [1] 查看会议内容（只是看看，不存文件）")
        log("  [2] 导出 txt 纪要（存到本程序所在文件夹）")
        log("  [3] 整理说话人（改名 / 合并）")
        log("  [4] 查看参会人统计")
        log("  [5] AI 摘要（DeepSeek，可选）")
        log("  [0] 退出")
        log("-" * 50)
        raw = ask("请选择（可反复操作，0 退出）: ")
        if raw is None:
            return
        choice = raw.strip() or "1"

        if choice == "0":
            if exported is not None:
                log(f"\n👋 纪要文件：{exported}")
            else:
                log("\n👋 没有导出 txt 纪要。可以下次重新运行程序再导出，录音还在 recordings/ 里。")
            return
        if choice == "1":
            m = build_minutes(session, db)
            log("")
            log(transcript_to_txt(m, summary))
        elif choice == "2":
            m = build_minutes(session, db)
            try:
                exported = export_transcript_txt(m, summary)
                log(f"✅ txt 纪要已保存：{exported}")
            except Exception as e:  # noqa: BLE001
                log(f"❌ 写入失败：{e}（可能是文件正被记事本占用，关掉重试）")
        elif choice == "3":
            manage_speakers(db, session)
        elif choice == "4":
            m = build_minutes(session, db)
            log(f"\n时长 {m['recorded_seconds']:.0f}s，{m['turn_count']} 段发言，{len(m['speakers'])} 位说话人")
            for s in m["speakers"]:
                log(f"  {s['name']}: {s['turns']} 段，{s['speech_seconds'] / 60:.1f} 分钟（{s['ratio'] * 100:.0f}%）")
        elif choice == "5":
            m = build_minutes(session, db)
            if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
                log("未配置 DEEPSEEK_API_KEY，跳过。设置方法：$env:DEEPSEEK_API_KEY=\"sk-...\"")
                continue
            log("⏳ 正在生成摘要…")
            result = llm_summary(m)
            if result is None:
                log("❌ 摘要生成失败（网络或 key 问题），其余功能不受影响")
            else:
                summary = result
                log("✅ 摘要已生成，之后导出的 txt 会带上摘要")
                md, js = save_minutes(m, summary)
                log(f"   详细版：{md}")
                log(f"   数据版：{js}")
        else:
            log("请输入 0~5")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI 会议纪要助手 v3")
    p.add_argument("--device", type=int, default=None, help="麦克风设备编号")
    p.add_argument("--model", default=None, help="onnx-asr 模型名，如 whisper-base；不写就用上次在菜单里选的")
    p.add_argument("--language", default=DEFAULT_LANGUAGE, help="识别语言，默认 zh")
    p.add_argument("--list-devices", action="store_true", help="列出输入设备")
    p.add_argument("--selftest", action="store_true", help="运行不依赖麦克风的自检")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        print_devices()
        return 0
    if args.selftest:
        return selftest()

    print_devices()
    log("")

    load_hotwords()

    model_name = args.model or load_config().get("model") or DEFAULT_MODEL
    engine = WhisperEngine(model_name, args.language)
    if not engine.load():
        return 1

    db = SpeakerDB()
    db.load()
    identifier = SpeakerIdentifier(db)
    log(f"声纹库：{len(db.names)} 位说话人" + (f"（{', '.join(db.names[:6])}…）" if len(db.names) > 6 else f"（{', '.join(db.names)}）" if db.names else ""))

    while True:
        log("\n" + "=" * 50)
        log("  [1] 开始新会议")
        log("  [2] 注册新说话人")
        log("  [3] 查看说话人")
        log("  [4] 整理说话人（改名/合并）")
        log("  [5] 列出音频设备")
        log(f"  [6] 切换识别模型（当前：{engine.model_name}）")
        log("  [7] 退出")
        log("=" * 50)
        choice = ask("请选择: ")
        if choice is None:
            return 0
        choice = choice.strip()

        if choice == "1":
            run_meeting(engine, identifier, db, args.device)
        elif choice == "2":
            register_speaker(identifier, db, args.device)
        elif choice == "3":
            show_speakers(db)
        elif choice == "4":
            manage_speakers(db)
        elif choice == "5":
            print_devices()
        elif choice == "6":
            new_engine = switch_model(engine)
            if new_engine is not None:
                engine = new_engine
        elif choice in ("7", "0", "q"):
            log("👋 再见")
            return 0
        else:
            log("无效选择")


MODEL_CHOICES = (
    ("whisper-base", "最快，约 0.4 倍速，准确率一般（当前默认）"),
    ("whisper-small", "慢一倍左右，中文明显更准（需要联网下载约 500MB）"),
    ("whisper-large-v3-turbo", "最准，但 CPU 上会明显跟不上实时（需要联网下载约 1.6GB）"),
)


def switch_model(current: "WhisperEngine") -> Optional["WhisperEngine"]:
    """换个模型不用改代码、不用重启，换坏了自动退回原来的。"""
    log("\n可用模型：")
    for i, (name, desc) in enumerate(MODEL_CHOICES, 1):
        mark = " ←当前" if name == current.model_name else ""
        log(f"  [{i}] {name} —— {desc}{mark}")
    log("  [0] 不换了")
    sel = (ask("请选择: ") or "").strip()
    if sel in ("", "0"):
        return None
    try:
        name = MODEL_CHOICES[int(sel) - 1][0]
    except (ValueError, IndexError):
        log("无效选择")
        return None
    if name == current.model_name:
        return None

    log(f"⏳ 正在加载 {name}（第一次用要下载，请耐心等）…")
    candidate = WhisperEngine(name, current.language)
    if candidate.load():
        save_config({"model": name})
        log(f"✅ 已切换到 {name}，下次启动直接用它")
        return candidate
    log(f"❌ {name} 加载失败，继续用 {current.model_name}")
    return None


def load_config() -> dict:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log(f"⚠️ 配置保存失败（不影响本次使用）：{e}")

class FakeGate:
    """按预设模式返回语音/非语音，用来测断句状态机。"""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.i = 0

    def is_speech(self, frame: np.ndarray) -> bool:  # noqa: ARG002
        c = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return c == "1"

def selftest() -> int:
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        print(f"{'✅' if cond else '❌'} {name} {extra}")
        ok = ok and cond

    # 1) 断句：1s 语音 + 0.6s 静音 + 1s 语音 + 长静音 → 2 句
    seg = Segmenter(FakeGate("0" * 5 + "1" * 33 + "0" * 20 + "1" * 33 + "0" * 400))
    utts = []
    for _ in range(200):
        utts += seg.push(np.zeros(FRAME_SAMPLES, np.int16))
    check("断句数量", len(utts) == 2, f"实际 {len(utts)}")

    # 2) 最短语音过滤：只有 3 帧语音不应成句
    seg2 = Segmenter(FakeGate("0" * 5 + "1" * 3 + "0" * 30))
    got = []
    for _ in range(60):
        got += seg2.push(np.zeros(FRAME_SAMPLES, np.int16))
    check("短语音被过滤", len(got) == 0, f"实际 {len(got)}")

    # 3) 强制切分：连续语音超过 MAX_UTTERANCE_S
    seg3 = Segmenter(FakeGate("1" * 4000))
    got3 = []
    for _ in range(1000):
        got3 += seg3.push(np.zeros(FRAME_SAMPLES, np.int16))
    check("超长语音强制切句", len(got3) >= 1)

    # 4) 声纹库：同人归一、陌生人新建、合并
    rng = np.random.default_rng(0)
    db = SpeakerDB(path=BASE_DIR / "._selftest_speakers.npz")
    a = rng.normal(size=256).astype(np.float32)
    a /= np.linalg.norm(a)
    b = rng.normal(size=256).astype(np.float32)
    b /= np.linalg.norm(b)
    ia = db.add("张三", a)
    ib = db.add("李四", b)
    noisy_a = a + rng.normal(scale=0.02, size=256).astype(np.float32)
    noisy_a /= np.linalg.norm(noisy_a)
    idx, best, second = db.match(noisy_a)
    check("同人识别", idx == ia and best > 0.9 and (best - second) > 0.1, f"best={best:.2f}")

    stranger = rng.normal(size=256).astype(np.float32)
    stranger /= np.linalg.norm(stranger)
    idx2, best2, _ = db.match(stranger)
    check("陌生人不应误认", best2 < SPK_ACCEPT, f"best={best2:.2f}")

    lo = db.merge(ib, ia)
    check("合并后人数", len(db.names) == 1 and lo == 0, f"len={len(db.names)} lo={lo}")

    # 5) 合并说话人时必须同步修正历史 turn 的编号
    db2 = SpeakerDB(path=BASE_DIR / "._selftest_speakers2.npz")
    for nm in ("A", "B", "C"):
        e = rng.normal(size=256).astype(np.float32)
        db2.add(nm, e / np.linalg.norm(e))
    sess = MeetingSession(engine=None, identifier=None, db=db2, out_dir=BASE_DIR / "._selftest_out")  # type: ignore[arg-type]
    sess.turns = [
        Turn(1, 0.0, 1.0, 0, 0.9, "a"),
        Turn(2, 1.0, 2.0, 1, 0.9, "b"),
        Turn(3, 2.0, 3.0, 2, 0.9, "c"),
    ]
    new_lo = sess.merge_speakers(2, 0)  # 把 C 并进 A
    idxs = [t.speaker_idx for t in sess.turns]
    check("合并后 turn 编号", new_lo == 0 and idxs == [0, 1, 0], f"lo={new_lo} idxs={idxs}")
    check("合并保留目标名字", len(db2.names) == 2 and db2.names[0] == "A", f"{db2.names}")

    # 6) 设备只支持 48k/44.1k 时的重采样
    t = np.arange(1440) / 48000.0
    sig48 = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    out48 = resample_frame(sig48, 48000)
    check("48k->16k 帧长", out48.size == FRAME_SAMPLES and out48.dtype == np.int16, f"{out48.size}")
    t2 = np.arange(1323) / 44100.0
    sig44 = (np.sin(2 * np.pi * 440 * t2) * 20000).astype(np.int16)
    out44 = resample_frame(sig44, 44100)
    check("44.1k->16k 帧长", out44.size == FRAME_SAMPLES, f"{out44.size}")
    # 重采样后能量不应消失（否则 VAD 会把人声判成静音）
    check("重采样保住能量", float(np.abs(out48).max()) > 15000, f"max={np.abs(out48).max()}")

    # 7) 纪要 JSON 可序列化 + Markdown 可生成
    class FakeSession:
        session_id = "selftest"
        _t0 = time.time()
        audio_seconds = 12.0
        wav_path = Path("x.wav")
        turns = [
            Turn(1, 0.0, 3.2, 0, 0.8, "今天讨论一下进度。"),
            Turn(2, 3.4, 6.0, 0, 0.7, "这周先把接口联调完。"),
            Turn(3, 6.2, 9.0, None, 0.0, "没问题。"),
        ]

    m = build_minutes(FakeSession(), db)
    try:
        json.dumps(m, ensure_ascii=False)
        json_ok = True
    except Exception as e:  # noqa: BLE001
        json_ok = False
        print("   json error:", e)
    check("纪要 JSON 可序列化", json_ok)
    md = minutes_to_markdown(m)
    check("纪要 Markdown 非空", len(md) > 200 and "张三" in md)

    # 8) 繁体转简体
    check(
        "繁体转简体",
        to_simplified("我們現在討論這個問題") == "我们现在讨论这个问题",
        to_simplified("我們現在討論這個問題"),
    )
    check("简体不受影响", to_simplified("今天先对一下进度") == "今天先对一下进度")
    check("多义字不动", to_simplified("著名") == "著名" and to_simplified("乾坤") == "乾坤")

    # 9) 热词替换（长词优先）
    hw = BASE_DIR / "._selftest_hotwords.txt"
    hw.write_text("连调 => 联调\n接口连调 => 接口联调\n", encoding="utf-8")
    load_hotwords(hw, quiet=True)
    check("热词替换", clean_text("这周把接口连调完") == "这周把接口联调完", clean_text("这周把接口连调完"))

    # 10) txt 纪要导出（落在 py 同目录）
    txt_path = export_transcript_txt(m, None, path=BASE_DIR / "._selftest_minutes.txt")
    body = txt_path.read_text(encoding="utf-8")
    check(
        "txt 纪要含说话人和语句",
        "张三：" in body and "今天讨论一下进度。" in body and "【会议内容】" in body and "[00:00]" in body,
    )
    txt_path.unlink(missing_ok=True)
    hw.unlink(missing_ok=True)

    for leftover in (
        BASE_DIR / "._selftest_speakers.npz",
        BASE_DIR / "._selftest_speakers2.npz",
    ):
        if leftover.exists():
            leftover.unlink()
    out_dir = BASE_DIR / "._selftest_out"
    if out_dir.exists():
        for f in out_dir.glob("*"):
            f.unlink()
        out_dir.rmdir()

    print("\n自检" + ("通过 ✅" if ok else "失败 ❌"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())