from dataclasses import dataclass, field
from typing import Optional

from catena_core.utils.timer import Timer
from catenasmith.cli_tools import info

@dataclass
class Metrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token: Optional[float] = None
    response_timer: Timer = field(default_factory=Timer)

    def log(self):
        info("**************** METRICS START ****************")
        if self.time_to_first_token is not None:
            info(f"* Time to first token:         {self.time_to_first_token:.4f}s")
        info(f"* Time to generate response:   {self.response_timer.elapsed:.4f}s")
        info(f"* Tokens per second:           {self.output_tokens / self.response_timer.elapsed:.4f} tokens/s")
        info(f"* Input tokens:                {self.input_tokens}")
        info(f"* Output tokens:               {self.output_tokens}")
        info(f"* Total tokens:                {self.total_tokens}")
        info("**************** METRICS END ******************")