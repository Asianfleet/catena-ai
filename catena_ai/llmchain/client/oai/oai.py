import os
from enum import Enum
from packaging import version
from pydantic import BaseModel, ConfigDict
from pydantic import Field, model_validator
from datetime import datetime
from typing import (
    Any, 
    Dict, 
    Iterator, 
    List,
    Literal, 
    Optional,
    Type, 
    Union,
    get_origin, 
    get_args
)

from ..base import Model
from ..metrics import Metrics
from ...response import ModelResponse
from ...message import Message, MessageBus
from ....error.modelerr import ModelError
from ....cli.tools import info, warning, error
from ....catena_core.alias.builtin import (
    NodeType as Ntype, 
    ModelProvider as Provider
)
from ....catena_core.callback import NodeCallback
from ....catena_core.tools.tool_registry import ToolRegistry as Tools
from ....catena_core.node.completion import NodeCompletion


try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.completion_usage import CompletionUsage
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
    from openai.types.chat.chat_completion_chunk import (
        ChatCompletionChunk,
        ChoiceDelta,
        ChoiceDeltaToolCall,
    )
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    MIN_OPENAI_VERSION = "1.52.0"

    # Check the installed openai version
    from openai import __version__ as installed_version

    if version.parse(installed_version) < version.parse(MIN_OPENAI_VERSION):
        warning(
            f"`openai` version must be >= {MIN_OPENAI_VERSION}, but found {installed_version}. "
            f"Please upgrade using `pip install --upgrade openai`."
        )
except (ModuleNotFoundError, ImportError):
    raise ImportError("`openai` not installed. Please install using `pip install openai`")



class OpenAIOrigin(Model):
    """
    用于与 OpenAI 模型交互的类。

    有关更多信息，请参阅：https://platform.openai.com/docs/api-reference/chat/create
    """

    model: str = "gpt-4o-mini"
    provider: Enum = Field(default=Provider.OpenAI, init=False)

    # 请求用参数
    store: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Any] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    modalities: Optional[List[str]] = None
    audio: Optional[Dict[str, Any]] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Any] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    user: Optional[str] = None
    top_p: Optional[float] = None
    extra_headers: Optional[Any] = None
    extra_query: Optional[Any] = None
    request_params: Optional[Dict[str, Any]] = None

    # 客户端初始化参数
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    default_headers: Optional[Any] = None
    default_query: Optional[Any] = None
    client_params: Optional[Dict[str, Any]] = None

    # OpenAI 客户端
    client: Optional[OpenAI] = None
    async_client: Optional[AsyncOpenAI] = None

    # 内部参数。不用于 API 请求
    # 是否使用结构化输出与此模型。
    structured_outputs: bool = False
    # 模型是否支持结构化输出。
    supports_structured_outputs: bool = True
    # 是否显示工具调用
    show_tool_calls: bool = False
    
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将模型实例转换为字典格式。

        返回：
            Dict[str, Any]: 包含模型基本信息的字典
        """
        _dict = self.model_dump(include={"name", "node_id", "provider", "metrics"})
        #if self.tools_builtin:
        #    _dict["tools_builtin"] = {k: v.to_dict() for k, v in self.tools_builtin.items()}
        if self.tools:
            _dict["tools"] = self.tools
        if self.tool_call_limit:
            _dict["tool_call_limit"] = self.tool_call_limit

        # 处理枚举类型的序列化
        from enum import Enum
        for key, value in _dict.items():
            if isinstance(value, Enum):
                _dict[key] = value.value

        return _dict
    
    @property
    def completion_args(self) -> Dict[str, Any]:
        """
        Returns keyword arguments for API requests.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
        request_params: Dict[str, Any] = {}
        if self.store is not None:
            request_params["store"] = self.store
        if self.frequency_penalty is not None:
            request_params["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias is not None:
            request_params["logit_bias"] = self.logit_bias
        if self.logprobs is not None:
            request_params["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            request_params["top_logprobs"] = self.top_logprobs
        if self.max_tokens is not None:
            request_params["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            request_params["max_completion_tokens"] = self.max_completion_tokens
        if self.modalities is not None:
            request_params["modalities"] = self.modalities
        if self.audio is not None:
            request_params["audio"] = self.audio
        if self.presence_penalty is not None:
            request_params["presence_penalty"] = self.presence_penalty
        if self.response_format is not None:
            request_params["response_format"] = self.response_format
        if self.seed is not None:
            request_params["seed"] = self.seed
        if self.stop is not None:
            request_params["stop"] = self.stop
        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.user is not None:
            request_params["user"] = self.user
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.extra_headers is not None:
            request_params["extra_headers"] = self.extra_headers
        if self.extra_query is not None:
            request_params["extra_query"] = self.extra_query
        if self.tools is not None:
            request_params["tools"] = self.tools
            if self.tool_choice is None:
                request_params["tool_choice"] = "auto"
            else:
                request_params["tool_choice"] = self.tool_choice
        if self.request_params is not None:
            request_params.update(self.request_params)
        return request_params
    
    def init_client(self) -> OpenAI:
        """
        初始化并返回 OpenAI 客户端实例。

        返回：
            OpenAI: 配置好的 OpenAI 客户端实例

        异常：
            ModelError: 如果未设置 OPENAI_API_KEY 环境变量
        """
        """ 获取 OpenAI 客户端 """
        if self.client:
            return self.client
        client_params: Dict[str, Any] = {}

        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ModelError("OPENAI_API_KEY not set. Please set the OPENAI_API_KEY environment variable.")

        if self.api_key is not None:
            client_params["api_key"] = self.api_key
        if self.organization is not None:
            client_params["organization"] = self.organization
        if self.base_url is not None:
            client_params["base_url"] = self.base_url
        if self.timeout is not None:
            client_params["timeout"] = self.timeout
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries
        if self.default_headers is not None:
            client_params["default_headers"] = self.default_headers
        if self.default_query is not None:
            client_params["default_query"] = self.default_query
        if self.client_params is not None:
            client_params.update(self.client_params)
 
        return OpenAI(**client_params)
    
    def update_usage_metrics(
        self, assistant_message: Message, metrics: Metrics, response_usage: Optional[CompletionUsage]
    ) -> None:
        """
        Update the usage metrics for the assistant message and the model.

        Args:
            assistant_message (Message): The assistant message.
            metrics (Metrics): The metrics.
            response_usage (Optional[CompletionUsage]): The response usage.
        """
        # Update time taken to generate response
        assistant_message.metrics["time"] = metrics.response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)
        if response_usage:
            prompt_tokens = response_usage.prompt_tokens
            completion_tokens = response_usage.completion_tokens
            total_tokens = response_usage.total_tokens

            if prompt_tokens is not None:
                metrics.input_tokens = prompt_tokens
                metrics.prompt_tokens = prompt_tokens
                assistant_message.metrics["input_tokens"] = prompt_tokens
                assistant_message.metrics["prompt_tokens"] = prompt_tokens
                self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + prompt_tokens
                self.metrics["prompt_tokens"] = self.metrics.get("prompt_tokens", 0) + prompt_tokens
            if completion_tokens is not None:
                metrics.output_tokens = completion_tokens
                metrics.completion_tokens = completion_tokens
                assistant_message.metrics["output_tokens"] = completion_tokens
                assistant_message.metrics["completion_tokens"] = completion_tokens
                self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + completion_tokens
                self.metrics["completion_tokens"] = self.metrics.get("completion_tokens", 0) + completion_tokens
            if total_tokens is not None:
                metrics.total_tokens = total_tokens
                assistant_message.metrics["total_tokens"] = total_tokens
                self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + total_tokens
            if response_usage.prompt_tokens_details is not None:
                if isinstance(response_usage.prompt_tokens_details, dict):
                    metrics.prompt_tokens_details = response_usage.prompt_tokens_details
                elif isinstance(response_usage.prompt_tokens_details, BaseModel):
                    metrics.prompt_tokens_details = response_usage.prompt_tokens_details.model_dump(exclude_none=True)
                assistant_message.metrics["prompt_tokens_details"] = metrics.prompt_tokens_details
                if metrics.prompt_tokens_details is not None:
                    for k, v in metrics.prompt_tokens_details.items():
                        self.metrics.get("prompt_tokens_details", {}).get(k, 0) + v
            if response_usage.completion_tokens_details is not None:
                if isinstance(response_usage.completion_tokens_details, dict):
                    metrics.completion_tokens_details = response_usage.completion_tokens_details
                elif isinstance(response_usage.completion_tokens_details, BaseModel):
                    metrics.completion_tokens_details = response_usage.completion_tokens_details.model_dump(
                        exclude_none=True
                    )
                assistant_message.metrics["completion_tokens_details"] = metrics.completion_tokens_details
                if metrics.completion_tokens_details is not None:
                    for k, v in metrics.completion_tokens_details.items():
                        self.metrics.get("completion_tokens_details", {}).get(k, 0) + v
    
    def format_assistant_message(
        self,
        response_message: ChatCompletionMessage,
        metrics: Metrics,
        response_usage: Optional[CompletionUsage],
    ) -> Message:
        """
        根据响应创建助理消息。

        - 参数：
            - response_message (ChatCompletionMessage)：响应消息。
            - Metrics：指标。
            - response_usage（可选[CompletionUsage]）：响应用法。

        - 返回：
            - 消息：assistant 消息。
        """
        assistant_message = Message(
            role=response_message.role or "assistant",
            content=response_message.content or "",
        )
        if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
            try:
                assistant_message.tool_calls = [t.model_dump() for t in response_message.tool_calls]
            except Exception as e:
                warning(f"Error processing tool calls: {e}")
        if hasattr(response_message, "audio") and response_message.audio is not None:
            try:
                assistant_message.audio = response_message.audio.model_dump()
            except Exception as e:
                warning(f"Error processing audio: {e}")

        # Update metrics
        self.update_usage_metrics(assistant_message, metrics, response_usage)
        return assistant_message
    
    def create_completion(self, messages: MessageBus) -> Union[ChatCompletion, ParsedChatCompletion]:
        """
        向 OpenAI API 发送请求。

        参数：
            messages (List[Message])：要发送到模型的消息列表。

        返回：
            ChatCompletion：来自 API 的聊天完成响应。
        """
        # 1、判断是否需要支持结构化输出
        if self.response_format is not None and self.structured_outputs:
            try:
                # 2、判断 response_format 是否为 BaseModel 的子类
                if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                    # 3、解析 API 响应
                    return self.init_client().beta.chat.completions.parse(
                        model=self.model,
                        messages=messages.model_message,  # type: ignore
                        **self.completion_args,
                    )
                else:
                    raise ValueError("response_format must be a subclass of BaseModel if structured_outputs=True")
            except Exception as e:
                raise e
        # 4、否则，直接返回 API 响应
        return self.init_client().chat.completions.create(
            model=self.model,
            messages=messages.model_message,  # type: ignore
            **self.completion_args
        )
    
    def create_completion_stream(self, messages: MessageBus) -> Iterator[ModelResponse]:
        """
        创建流式聊天完成响应。

        参数：
            messages (MessageBus): 要发送到模型的消息总线

        返回：
            Iterator[ModelResponse]: 模型响应迭代器
        """
        pass
    
    def get_tool_call_schema(self):
        self.tools = []
        
        def convert_json_format(type: Type) -> str:
            """ 将 Python 的类型转换为 JSON Schema 字符串 """
            if type == str:
                return "string"
            elif type == int:
                return "integer"
            elif type == float:
                return "number"
            elif type == bool:
                return "boolean"
            elif type == dict:
                return "object"
            elif type == list:
                return "array"
            elif type == Literal:
                return convert_json_format(get_args(type)[0])
        
        for tool in self.tools_builtin:
            toolcall_item = {"type": "function"}
            if hasattr(tool, "func_name"):
                function = tool.func_name
            else:
                raise ValueError("Invalid tool")
            toolcall_item["function"] = {
                "name": function,
                "description": tool.metadata["description"],    
                "parameters": {
                    "type": "object",
                    "properties":{}
                }
            }
            for (name, type), (_, desc) in zip(
               tool.metadata["parameters"].items(), tool.metadata["param_desc"].items() 
            ):
                toolcall_item["function"]["parameters"]["properties"].update({
                    name: {
                        "type": convert_json_format(type["type"]),
                        "description": desc
                    } 
                })
                if get_origin(type["default"]) is Literal:
                    toolcall_item["function"]["parameters"]["properties"][name].update(
                        {"enum": list(get_args(type["default"]))}
                    )
            self.tools.append(toolcall_item)
    
    # TODO: 让大模型筛选工具 
    def execute_tool_calls(
        self, messages: MessageBus, model_response: ModelResponse, tool_role: str
    ) -> Union[ModelResponse, None]:
        """执行工具调用并更新模型响应
        
        Args:
            messages: 消息总线
            model_response: 模型响应对象
            tool_role: 工具调用角色名称
            
        Returns:
            更新后的模型响应对象
        """
        assistant_message: Message = messages.latest
        if not assistant_message.tool_calls or len(assistant_message.tool_calls) == 0:
            # 没有工具调用时返回原始响应
            return None
        if model_response.content is None:
            model_response.content = ""
        model_response.content += "\nTools run:"
        
        for tool_call in assistant_message.tool_calls:
            try:
                # 验证工具调用参数
                if not tool_call.get("function"):
                    warning(f"Invalid tool call format: {tool_call}")
                    continue
                    
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                
                # 获取工具并执行
                tool = Tools.get_tool(name)
                if not tool:
                    warning(f"Tool not found: {name}")
                    continue
                import json
                args = json.loads(args)
                # 执行工具调用
                result = tool(**args)
                # 添加工具调用结果消息
                messages.add(
                    role=tool_role,
                    tool_call_id=tool_call["id"],
                    content=result,
                    metrics={
                        "tool_call_time": tool.time_elapsed,
                        "timestamp": datetime.now().isoformat()
                    } 
                )
                
                # 如果需要显示工具调用信息
                if self.show_tool_calls:
                    call_str = Tools.tool_call_metadata.get(
                        tool_call["id"], f"{name}({args})"
                    )
                    model_response.content += f"\n - {call_str}; return: {result}"
                    
            except Exception as e:
                error(f"Error executing tool call {tool_call['id']}: {e}")
                messages.add(
                    role=tool_role,
                    tool_call_id=tool_call["id"],
                    content=f"Error: {str(e)}",
                    metrics={"timestamp": datetime.now().isoformat()}
                )
                if self.show_tool_calls:
                    model_response.content += f"\n - Error executing {name}: {e}"
                    
        model_response.content += "\n"
        return model_response
    
    def generate_final_response(self, messages: MessageBus, model_response: ModelResponse) -> ModelResponse:
        """
        生成最终响应，处理工具调用后的结果。

        参数：
            messages (MessageBus): 消息总线，包含所有消息历史
            model_response (ModelResponse): 当前的模型响应对象

        返回：
            ModelResponse: 更新后的模型响应对象，包含工具调用结果和最终响应内容
        """
        last_message: Message = messages.latest
        if last_message.stop_after_tool_call:
            if (
                last_message.role == "assistant"
                and last_message.content is not None
                and isinstance(last_message.content, str)
            ):
                model_response.content += last_message.content
        else:
            response_after_tool_calls = self.response(messages)
            model_response.content += "Model Answer:\n"
            model_response.content += response_after_tool_calls.content
            if response_after_tool_calls.parsed is not None:
                # bubble up the parsed object, so that the final response has the parsed object
                # that is visible to the agent
                model_response.parsed = response_after_tool_calls.parsed
            if response_after_tool_calls.audio is not None:
                # bubble up the audio, so that the final response has the audio
                # that is visible to the agent
                model_response.audio = response_after_tool_calls.audio
        return model_response
        
    def response(self, messages: MessageBus) -> ModelResponse:
        """
        处理完整的模型响应流程，包括：
        1. 生成聊天完成响应
        2. 提取响应内容和使用数据
        3. 解析结构化输出（如果启用）
        4. 生成并添加助理消息
        5. 更新模型响应内容
        6. 处理工具调用
        7. 生成最终响应

        [参数]
         - messages (MessageBus): 包含所有消息历史的消息总线对象

        返回：
            ModelResponse: 包含以下内容的模型响应对象：
                - 响应内容
                - 解析后的结构化对象（如果启用）
                - 音频数据（如果有）
                - 助理消息
                - 使用指标

        处理流程：
            1. 启动计时器并生成聊天完成响应
            2. 提取响应消息和使用数据
            3. 如果启用结构化输出，尝试解析响应
            4. 格式化助理消息并添加到消息总线
            5. 更新模型响应内容和音频数据
            6. 处理工具调用（如果有）
            7. 返回最终的模型响应对象
        """
        model_response = ModelResponse()
        model_metrics = Metrics()
        
        # 1、生成 ChatCompletion，同时计时
        model_metrics.response_timer.start()
        response: Union[ChatCompletion, ParsedChatCompletion] = self.create_completion(messages)
        model_metrics.response_timer.stop()
        
        # 2、提取 ChatCompletion 的内容以及使用数据（token 花费量等）
        response_message: ChatCompletionMessage = response.choices[0].message
        response_usage: Optional[CompletionUsage] = response.usage
        
        # 3、解析结构输出（如果开启结构化输出）
        try:
            if (
                self.response_format is not None
                and self.structured_outputs
                and issubclass(self.response_format, BaseModel)
            ):
                parsed_object = response_message.parsed  # type: ignore
                if parsed_object is not None:
                    model_response.parsed = parsed_object
        except Exception as e:
            warning(f"Error retrieving structured outputs: {e}")
        # 4、生成 assistant 消息
        assistant_message = self.format_assistant_message(response_message, model_metrics, response_usage)
        # 5、将 assistant 消息添加到消息列表中
        messages.add(assistant_message)
        model_response.assistant_message = assistant_message
        # 在控制台展示
        # -*- Log response and metrics
        assistant_message.printf(level="Code")
        model_metrics.prinf(level="Code")
        
        # 6、使用 assistant 消息内容和音频更新模型响应
        if assistant_message.content is not None:
            # add the content to the model response
            model_response.content = assistant_message.get_content_string()
        if assistant_message.audio is not None:
            # add the audio to the model response
            model_response.audio = assistant_message.audio

        # 7、处理工具调用
        tool_role = "tool"
        toolcall_results = self.execute_tool_calls(
            messages=messages,
            model_response=model_response,
            tool_role=tool_role,
        )

        # 如果有工具调用结果，生成最终响应
        if toolcall_results:
            return self.generate_final_response(
                messages=messages,
                model_response=model_response
            )
        
        #info("---------- OpenAI Response End ----------")
        return model_response

    async def acreate_completion(self, messages: MessageBus) -> ModelResponse:
        """
        异步创建聊天完成响应。

        参数：
            messages (MessageBus): 要发送到模型的消息总线

        返回：
            ModelResponse: 包含模型响应的对象
        """
        pass
    
    async def acreate_completion_stream(self, messages: MessageBus) -> Any:
        """
        异步创建流式聊天完成响应。

        参数：
            messages (MessageBus): 要发送到模型的消息总线

        返回：
            Any: 模型响应迭代器
        """
        pass
    
    def reset(self) -> None:
        """重置OpenAI模型为初始状态"""
        # 重置请求参数
        self.store = None
        self.metadata = None
        self.frequency_penalty = None
        self.logit_bias = None
        self.logprobs = None
        self.top_logprobs = None
        self.max_tokens = None
        self.max_completion_tokens = None
        self.modalities = None
        self.audio = None
        self.presence_penalty = None
        self.response_format = None
        self.seed = None
        self.stop = None
        self.temperature = None
        self.user = None
        self.top_p = None
        self.extra_headers = None
        self.extra_query = None
        self.request_params = None

        # 重置客户端初始化参数
        self.api_key = None
        self.organization = None
        self.base_url = None
        self.timeout = None
        self.max_retries = None
        self.default_headers = None
        self.default_query = None
        self.client_params = None

        # 重置客户端实例
        self.client = None
        self.async_client = None

        # 重置内部参数
        self.structured_outputs = False
        self.supports_structured_outputs = True
        
        # 重置继承自基类的工具相关属性
        self.tools = None
        self.tool_choice = None
        self.tool_call_limit = None

    def operate(self, input: NodeCompletion) -> NodeCompletion:
        """
        执行模型操作并返回完成结果。

        参数：
            input (NodeCompletion): 包含输入数据的节点完成对象

        返回：
            NodeCompletion: 包含模型响应和回调信息的节点完成对象
        """
        model_message: MessageBus = input.main_data
        model_response = self.response(model_message)
        model_completion = NodeCompletion(
            main_data=model_response,
            callback=NodeCallback(
                source=self.nid,
                target=Ntype.MEM,
                name="update_memory",
                main_input=model_response.assistant_message
            ),
            extra_data=input.extra_data
        )
        
        return model_completion
    
    
if __name__ == "__main__":
    # python -m catena_ai.llmchain.model.oai.oai
    oai_model = OpenAIOrigin()
    #cpl = oai_model.create_completion(messages=MessageBus([Message(role="user", content="你好")]))
    class ResponseFormat(BaseModel):
            answer: str   
            reason: str
                 
    oai_model.response_format = ResponseFormat
    oai_model.structured_outputs = True
    messages = MessageBus([Message(
        role="user", 
        content="星球是什么形状的？请按给定格式输出"
    )])
    print(messages.model_message)
    #response = oai_model.create_completion(messages)
    #event = response.choices[0].message.parsed
    #print(event)
    model_response = oai_model.response(messages)
    model_response.printf()
