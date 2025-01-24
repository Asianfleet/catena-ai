from time import time
from enum import Enum
from typing import Optional, Any, Dict
from dataclasses import dataclass

from ..settings import settings
from ..llmchain.message import Message
from ..catenasmith.cli_tools import info, info_condition


class ModelResponseEvent(str, Enum):
    """Events that can be sent by the Model.create_completion() method"""

    tool_call_started = "ToolCallStarted"
    tool_call_completed = "ToolCallCompleted"
    assistant_response = "AssistantResponse"


@dataclass
class ModelResponse:
    """Response returned by Model.create_completion()"""

    assistant_message: Optional[Message] = None
    content: Optional[str] = None
    parsed: Optional[Any] = None
    audio: Optional[Dict[str, Any]] = None
    tool_call: Optional[Dict[str, Any]] = None
    event: str = ModelResponseEvent.assistant_response.value
    created_at: int = int(time())
    
    def printf(self):
        with info_condition(settings.visualize.model_resp_metrics):
            info("**************** MODEL RESPONSE ****************")
            info(f"* Content: {self.content}")
            info(f"* Parsed: {self.parsed}")
            info(f"* Audio: {self.audio}")
            info(f"* Tool Call: {self.tool_call}")
            info(f"* Event: {self.event}")
            info(f"* Created At: {self.created_at}")
            info("**************** MODEL RESPONSE ******************")
