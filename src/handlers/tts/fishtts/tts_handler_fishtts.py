import io
import os
import re
import time
from typing import Dict, Optional, cast
import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.directory_info import DirectoryInfo
import requests
from typing import Iterator

class TTSConfig(HandlerBaseConfigModel, BaseModel):
    ref_audio_path: str = Field(default=None)
    ref_audio_text: str = Field(default=None)
    voice: str = Field(default=None)
    sample_rate: int = Field(default=24000)
    api_url: str = Field(default=None)


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None

class HandlerTTS(HandlerBase, ABC):


    def __init__(self):
        super().__init__()
        self.ref_audio_path = None
        self.ref_audio_text = None
        self.voice = None
        self.ref_audio_buffer = None
        self.sample_rate = None
        self.api_url = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        config = cast(TTSConfig, handler_config)
        self.voice = config.voice
        self.sample_rate = config.sample_rate
        self.ref_audio_path = config.ref_audio_path
        self.ref_audio_text = config.ref_audio_text
        self.api_url = config.api_url

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.input_text = ''
        if context.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp',
                                          f"dump_avatar_audio_{context.session_id}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pcm")
            context.audio_dump_file = open(dump_file_path, "wb")
        return context

    def start_context(self, session_context: SessionContext, context: HandlerContext):
        context = cast(TTSContext, context)

    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        
        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            context.input_text += self.filter_text(text)

        text_end = inputs.data.get_meta("avatar_text_end", False)
        
        if not text_end:
            # 按句子分隔符拆分文本
            sentences = re.split(r'(?<=[,.~!?，。！？])', context.input_text)
            if len(sentences) > 1:  # 至少有一个完整句子
                complete_sentences = sentences[:-1]  # 完整句子
                context.input_text = sentences[-1]  # 剩余的未完成部分

                # 对完整句子进行处理
                for sentence in complete_sentences:
                    if len(sentence.strip()) < 1:
                        continue
                    logger.info(f'current sentence: {sentence}')
                    
                    # 调用 Fish TTS API
                    audio_chunks = self.fish_speech(
                        text=sentence,
                        reffile=self.ref_audio_path,
                        reftext=self.ref_audio_text,
                        language="zh",
                        server_url=self.api_url
                    )
                    
                    # 处理音频流
                    self.process_audio_stream(audio_chunks, context, output_definition)
        else:
            # 处理最后一句文本
            logger.info(f'last sentence: {context.input_text}')
            if context.input_text is not None and len(context.input_text.strip()) > 0:
                audio_chunks = self.fish_speech(
                    text=context.input_text,
                    reffile=self.ref_audio_path,
                    reftext=self.ref_audio_text,
                    language="zh",
                    server_url=self.api_url
                )
                self.process_audio_stream(audio_chunks, context, output_definition)
            
            context.input_text = ''
            # 发送结束标记
            output = DataBundle(output_definition)
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
            context.submit_data(output, finish_stream=True)
            logger.info("speech end")

    def fish_speech(self, text: str, reffile: str, reftext: str, language: str, server_url: str) -> Iterator[bytes]:
        """
        调用 Fish TTS API 生成音频流
        """
        start = time.perf_counter()
        req = {
            'text': text,
            'reference_id': reffile,
            'reference_text': reftext,
            'language': language,
            'format': 'wav',
            'streaming': True,
            'use_memory_cache': 'on'
        }
        try:
            res = requests.post(
                f"{server_url}/v1/tts",
                json=req,
                stream=True,
                headers={
                    "content-type": "application/json",
                },
            )
            end = time.perf_counter()
            logger.info(f"fish_speech Time to make POST: {end - start}s")

            if res.status_code != 200:
                logger.error(f"Error: {res.text}")
                return

            first = True
            for chunk in res.iter_content(chunk_size=17640):  # 1764 44100*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"fish_speech Time to first chunk: {end - start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            logger.exception('fish_tts error')

    def process_audio_stream(self, audio_stream: Iterator[bytes], context: TTSContext, output_definition: DataBundleDefinition):
        """
        处理音频流并转换为所需的格式
        """
        import resampy
        
        data = b''
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                data += chunk
        
        if len(data) > 0:
            try:
                # 将音频数据转换为 numpy 数组
                # Fish TTS 返回的是 44100Hz 的 PCM 数据
                stream = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767
                
                # 重采样到目标采样率
                stream = resampy.resample(x=stream, sr_orig=44100, sr_new=self.sample_rate)
                
                # 调整形状为 (1, samples)
                output_audio = stream[np.newaxis, ...]
                
                # 创建输出数据包
                output = DataBundle(output_definition)
                output.set_main_data(output_audio)
                context.submit_data(output)
            except Exception as e:
                logger.exception(f'Error processing audio stream: {e}')

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        logger.info('destroy context')

    def filter_text(self, text):
        pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?，。！？ ]"  # 匹配不在范围内的字符
        filtered_text = re.sub(pattern, "", text)
        return filtered_text
