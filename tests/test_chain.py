import unittest

from catena.llmchain.memory import InfMemory
from catena.llmchain.model.oai import OpenAIOrigin
from catena.llmchain.prompt import ModelPrompt

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