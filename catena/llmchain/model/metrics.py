""" 
该模块定义了一个Metrics类，用于记录和日志化模型的性能指标，包括：
- 输入和输出的token数量
- 生成响应的总时间
- 第一个token生成的时间
- 每秒生成的token数量
"""
from dataclasses import dataclass, field
from typing import Optional

from ...settings import settings
from ...catena_core.utils.timer import Timer
from ...cli.tools import info, info_condition

@dataclass
class Metrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: Optional[dict] = None
    completion_tokens_details: Optional[dict] = None
    time_to_first_token: Optional[float] = None
    response_timer: Timer = field(default_factory=Timer)

    def prinf(self, **kwargs):
        kwargs.update({"pre": False})
        with info_condition(settings.visualize.metrics, **kwargs):
            info("**************** METRICS OBJECT ****************")
            if self.time_to_first_token is not None:
                info(f"* Time to first token:         {self.time_to_first_token:.4f}s")
            info(f"* Time to generate response:   {self.response_timer.elapsed:.4f}s")
            info(f"* Tokens per second:           {self.output_tokens / self.response_timer.elapsed:.4f} tokens/s")
            info(f"* Input tokens:                {self.input_tokens or self.prompt_tokens}")
            info(f"* Output tokens:               {self.output_tokens or self.completion_tokens}")
            info(f"* Total tokens:                {self.total_tokens}")
            if self.prompt_tokens_details is not None:
                info(f"* Prompt tokens details:       {self.prompt_tokens_details}")
            if self.completion_tokens_details is not None:
                info(f"* Completion tokens details:   {self.completion_tokens_details}")
            info("**************** METRICS OBJECT ******************")