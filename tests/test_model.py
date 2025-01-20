import unittest

from pydantic import BaseModel
from catena.llmchain.model.oai.oai import OpenAIOrigin
from catena.llmchain.message import Message

class TestOpenAIOrigin(unittest.TestCase):
    def setUp(self):
        self.oai_model = OpenAIOrigin()
        
    def test_response(self):
        # 测试正常情况下的response调用
        messages = [Message(role="user", content="你好")]
        response = self.oai_model.create_completion(messages)
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)
        
        # 测试结构化输出报错
        with self.assertRaises(ValueError):
            
            self.oai_model.structured_outputs = True
            self.oai_model.response_format = dict  # 使用dict作为示例
            response = self.oai_model.create_completion(messages)
            self.assertIsNotNone(response)

        # 测试结构化输出正常
        class ResponseFormat(BaseModel):
            answer: str   
            reason: str
                 
        self.oai_model.response_format = ResponseFormat
        self.oai_model.structured_outputs = True
        messages = [Message(
            role="user", 
            content="星球是什么形状的？请按给定格式输出"
        )]
        response = self.oai_model.create_completion(messages)
        event = response.choices[0].message.parsed
        self.assertIsNotNone(event)
