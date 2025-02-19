import unittest

from catena_ai.llmchain.memory import InfMemory
from catena_ai.llmchain.client.oai import OpenAIOrigin, OpenAI_C
from catena_ai.llmchain.prompt import ModelPrompt
from catena_ai.agents.tools import tool

class TestChain(unittest.TestCase):
    
    def setUp(self):
        oai_model = OpenAIOrigin()
        oai_model_c = OpenAI_C()
        inf_memory = InfMemory()
        model_prompt = ModelPrompt()
        self.model = {
            "oai": oai_model,
            "oai_c": oai_model_c,
        }
        self.memory = {
            "inf": inf_memory,
        }
        self.prompt = {
            "mp": model_prompt,
        }
        
    def test_prompt_model(self):
        chain = self.prompt["mp"] >> self.model["oai"]
        
        chain.operate("Hello, world!")
        
    def test_prompt_memory_model(self):
        proc = self.prompt["mp"] >> self.memory["inf"] >> self.model["oai"]
        chain = "hello" >> proc
        
        completion = chain.operate()
        print(completion)
        self.assertIsNotNone(completion)

    def test_prompt_memory_model_tool(self):
        proc = self.prompt["mp"] >> self.memory["inf"] >> self.model["oai"]
        chain = "今天齐河天气怎么样" >> proc
        
        @tool
        def get_weather(city: str) -> str:
            return "齐河天气晴朗" 
        
        get_weather("qihe")
        
        completion = chain.operate()
        print(completion)
        self.assertIsNotNone(completion)

        
    def test_image_prompt(self):
        self.model["oai_c"].model = "qwen2.5-vl-72b-instruct"
        self.model["oai_c"].base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model["oai_c"].api_key = "sk-925b0170f7cd4ffb8fda54faedc05214"
        
        temlpate = ModelPrompt.from_template("你需要描述提供的图片的内容")
        
        chain = temlpate >> self.model["oai_c"]
        
        import base64
        with open("/home/legion4080/Programing/catena/assets/catena.jpg", "rb") as img:
            img_base64 = base64.b64encode(img.read()).decode("utf-8")
        
        desc = chain.operate(img_base64).main_data
        print(desc)