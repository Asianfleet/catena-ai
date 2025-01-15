from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import (
    Any, 
    Dict, 
    Iterator, 
    List, 
    Literal,
    Optional, 
    Type, 
    Union,
    overload
)

class NextAction(str, Enum):
    CONTINUE = "continue"
    VALIDATE = "validate" 
    FINAL_ANSWER = "final_answer"

class RunEvent(str, Enum):
    RUN_START = "run_start"
    RUN_STEP = "run_step"
    RUN_RESPONSE = "run_response"
    RUN_END = "run_end"

@dataclass
class Message:
    content: str
    role: str = "user"

@dataclass
class ReasoningStep:
    thought: str
    action: str
    observation: str

class ExtraData(BaseModel):
    reasoning_steps: Optional[List[ReasoningStep]] = None

class RunResponse(BaseModel):
    content: Union[str, dict, None]
    event: RunEvent
    extra_data: Optional[ExtraData] = None

class Agent(BaseModel):
    response_model: Optional[Type[BaseModel]] = Field(None, alias="output_model")
    parse_response: bool = True
    structured_outputs: bool = False
    save_response_to_file: Optional[str] = None
    stream: Optional[bool] = None
    reasoning_max_steps: int = 10
    
    @property
    def is_streamable(self) -> bool:
        """Check if the agent supports streaming."""
        return True

    def _get_next_action(self, last_step: ReasoningStep) -> NextAction:
        """
        Determine the next action based on the last reasoning step.
        Can be overridden by subclasses for custom logic.
        """
        # Default implementation: Basic completion check
        if "final answer" in last_step.thought.lower():
            return NextAction.FINAL_ANSWER
        elif "validate" in last_step.thought.lower():
            return NextAction.VALIDATE
        return NextAction.CONTINUE

    def _think(self, context: str) -> str:
        """Generate the next thought. Can be overridden by subclasses."""
        return f"Thinking about: {context}"

    def _act(self, thought: str) -> str:
        """Determine the next action. Can be overridden by subclasses."""
        return f"Acting on: {thought}"

    def _observe(self, action: str) -> str:
        """Make an observation. Can be overridden by subclasses."""
        return f"Observed result of: {action}"

    def _run(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Iterator[RunResponse]:
        self.stream = stream and self.is_streamable
        
        # Convert message to string if needed
        if isinstance(message, (List, Dict)):
            message = str(message)
        elif isinstance(message, Message):
            message = message.content
        
        # Initialize reasoning steps
        reasoning_steps: List[ReasoningStep] = []
        
        # Emit run start event
        yield RunResponse(
            content=None,
            event=RunEvent.RUN_START,
            extra_data=ExtraData(reasoning_steps=reasoning_steps)
        )

        # Main reasoning loop
        step_count = 1
        next_action = NextAction.CONTINUE
        
        while next_action == NextAction.CONTINUE and step_count < self.reasoning_max_steps:
            # Generate thought
            thought = self._think(message if step_count == 1 else reasoning_steps[-1].observation)
            
            # Determine and take action
            action = self._act(thought)
            
            # Make observation
            observation = self._observe(action)
            
            # Create reasoning step
            step = ReasoningStep(thought=thought, action=action, observation=observation)
            reasoning_steps.append(step)
            
            if self.stream:
                yield RunResponse(
                    content=str(step),
                    event=RunEvent.RUN_STEP,
                    extra_data=ExtraData(reasoning_steps=reasoning_steps)
                )
            
            # Determine next action
            next_action = self._get_next_action(step)
            step_count += 1

        # Generate final response
        final_content = reasoning_steps[-1].observation if reasoning_steps else "No steps taken"
        
        if self.response_model and self.parse_response:
            final_content = self.response_model.parse_raw(final_content)

        yield RunResponse(
            content=final_content,
            event=RunEvent.RUN_END,
            extra_data=ExtraData(reasoning_steps=reasoning_steps)
        )

    @overload
    def run(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> RunResponse: ...

    @overload
    def run(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        stream: Literal[True] = True,
        **kwargs: Any,
    ) -> Iterator[RunResponse]: ...

    def run(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[RunResponse, Iterator[RunResponse]]:
        if stream and self.is_streamable:
            return self._run(message=message, stream=True, **kwargs)
        
        response_iterator = self._run(message=message, stream=False, **kwargs)
        if self.response_model is not None and self.parse_response:
            return next(response_iterator)
        return next(response_iterator)

    def print_response(self, message: Optional[Union[str, Message]] = None, **kwargs):
        _response_content = ""
        reasoning_steps = None

        if self.stream:
            for resp in self.run(message=message, stream=True, **kwargs):
                if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                    if resp.event == RunEvent.RUN_RESPONSE:
                        _response_content += resp.content
                    if resp.extra_data is not None and resp.extra_data.reasoning_steps is not None:
                        reasoning_steps = resp.extra_data.reasoning_steps
        else:
            run_response = self.run(message=message, stream=False, **kwargs)
            if isinstance(run_response, RunResponse):
                _response_content = str(run_response.content)
                if run_response.extra_data:
                    reasoning_steps = run_response.extra_data.reasoning_steps

        return _response_content, reasoning_steps

def init_agent(
    agent_type: Optional[str] = None,
    response_model: Optional[Type[BaseModel]] = None,
    **kwargs: Any
) -> Agent:
    """Factory method to create an Agent instance."""
    if agent_type is None:
        return Agent(response_model=response_model, **kwargs)
    
    # Add custom agent type initialization here
    raise ValueError(f"Unknown agent type: {agent_type}")

if __name__ == "__main__":
    # 创建基础 agent
    agent = init_agent()

    # 运行 agent（无流式输出）
    response = agent.run("你的问题")

    """ # 运行 agent（有流式输出）
    for step in agent.run("你的问题", stream=True):
        print(step) """
