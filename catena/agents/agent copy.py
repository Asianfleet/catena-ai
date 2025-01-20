from typing import Optional, Type, Iterator, Union, Any, Dict
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

class RunResponse(BaseModel):
    """Response model for Agent runs"""
    content: str
    metadata: Optional[Dict[str, Any]] = None

def init_agent(
    agent_type: Type["Agent"],
    **kwargs
) -> "Agent":
    """Factory method to initialize agents"""
    return agent_type(**kwargs)

class Agent(BaseModel):
    """Base Agent class with core functionality"""
    # Agent Response Settings
    response_model: Optional[Type[BaseModel]] = Field(None, alias="output_model")
    parse_response: bool = True
    structured_outputs: bool = False
    save_response_to_file: Optional[str] = None
    stream: Optional[bool] = None

    # Properties
    @property
    def is_streamable(self) -> bool:
        """Whether the agent supports streaming"""
        return True

    def _process_response(self, response: str) -> Union[BaseModel, str]:
        """Process the response according to settings"""
        if self.save_response_to_file:
            with open(self.save_response_to_file, 'w') as f:
                f.write(response)
        
        if self.response_model and self.parse_response:
            return self.response_model(content=response)
        return response

    def _generate_steps(self) -> Iterator[str]:
        """Default step generator"""
        yield "Thinking about the task..."
        yield "Processing information..."
        yield "Generating response..."
        yield "Final response ready."

    def _run(
        self,
        prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Iterator[RunResponse]:
        """Core execution logic with streaming support"""
        self.stream = stream and self.is_streamable
        
        # Execute steps
        accumulated_response = ""
        for step in self._generate_steps():
            if self.stream:
                yield RunResponse(
                    content=step,
                    metadata={"step": step}
                )
            accumulated_response += f"{step}\n"

        # Process final response
        final_response = self._process_response(accumulated_response)
        if isinstance(final_response, str):
            yield RunResponse(content=final_response)
        else:
            yield RunResponse(
                content=final_response.content,
                metadata=final_response.dict(exclude={'content'})
            )

    def run(
        self,
        prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[RunResponse, Iterator[RunResponse]]:
        """Main execution method with streaming control"""
        if self.response_model is not None and self.parse_response:
            run_response: RunResponse = next(
                self._run(prompt=prompt, stream=False, **kwargs)
            )
            return run_response
        else:
            if stream and self.is_streamable:
                resp = self._run(
                    prompt=prompt,
                    stream=True,
                    **kwargs
                )
                return resp
            else:
                resp = self._run(
                    prompt=prompt,
                    stream=False,
                    **kwargs
                )
                return next(resp)

    def print_response(
        self,
        prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> None:
        """Run and print results with streaming support"""
        result = self.run(prompt=prompt, stream=stream, **kwargs)
        if stream:
            for chunk in result:
                print(chunk.content)
        else:
            print(result.content)

class CustomAgent(Agent):
    """Example of a custom agent implementation"""
    def _generate_steps(self) -> Iterator[str]:
        """Custom step implementation"""
        yield "Custom step 1"
        yield "Custom step 2"
        yield "Custom final response"

if __name__ == "__main__":
    base_agent = init_agent(Agent)
    base_agent.print_response(stream=True)