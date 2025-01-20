from time import time
from enum import Enum
from typing import Optional, Any, Dict

from dataclasses import dataclass


class ModelResponseEvent(str, Enum):
    """Events that can be sent by the Model.create_completion() method"""

    tool_call_started = "ToolCallStarted"
    tool_call_completed = "ToolCallCompleted"
    assistant_response = "AssistantResponse"


@dataclass
class ModelResponse:
    """Response returned by Model.create_completion()"""

    content: Optional[str] = None
    parsed: Optional[Any] = None
    audio: Optional[Dict[str, Any]] = None
    tool_call: Optional[Dict[str, Any]] = None
    event: str = ModelResponseEvent.assistant_response.value
    created_at: int = int(time())