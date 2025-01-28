import unittest

from catena_ai.llmchain.memory import InfMemory
from catena_ai.llmchain.model.oai import OpenAIOrigin
from catena_ai.llmchain.prompt import ModelPrompt
from catena_ai.agents.tools import tool

class TestChain(unittest.TestCase):
    
    def setUp(self):
        oai_model = OpenAIOrigin()
        inf_memory = InfMemory()
        model_prompt = ModelPrompt()
        self.model = {
            "oai": oai_model,
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