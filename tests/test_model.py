import unittest

from pydantic import BaseModel
from catena.llmchain.model.oai.oai import OpenAIOrigin
from catena.llmchain.message import Message, MessageBus

class TestOpenAIOrigin(unittest.TestCase):
    def setUp(self):
        self.oai_model = OpenAIOrigin()
        
        class ResponseFormat(BaseModel):
            answer: str   
            reason: str
        
        self.ResponseFormat = ResponseFormat
        self.hello_messages = MessageBus(
            [Message(role="user", content="你好")]
        )
        self.question_messages = MessageBus(
            [Message(role="user", content="星球是什么形状的？")]
        )
        
    def test_completion(self):
        """ 测试 create_completion 函数 """
        # 测试消息：你好
        completion = self.oai_model.create_completion(self.hello_messages)
        self.assertIsNotNone(completion)
        self.assertIsNotNone(completion.choices)
        self.assertGreater(len(completion.choices), 0)
        
        # 结构化输出
        self.oai_model.response_format = self.ResponseFormat
        self.oai_model.structured_outputs = True
        # 测试消息：星球是什么形状的？
        completion = self.oai_model.create_completion(self.question_messages)
        event = completion.choices[0].message.parsed
        self.assertIsNotNone(event)

    def test_response(self):
        """ 测试 response 函数 """
        model_response = self.oai_model.response(self.question_messages)
        self.assertIsNotNone(model_response)      
        self.oai_model.structured_outputs = False
        model_response = self.oai_model.response(self.question_messages)
        self.assertIsNotNone(model_response)  
        
    def test_error_structure(self):
        # 测试结构化输出报错
        self.oai_model.reset()
        with self.assertRaises(ValueError):
            
            self.oai_model.structured_outputs = True
            self.oai_model.response_format = dict  # 使用dict作为示例
            self.oai_model.create_completion(self.question_messages)    