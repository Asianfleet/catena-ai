import unittest
from catena_ai.llmchain.message import Message, MessageBus, MessageRole

class TestMessage(unittest.TestCase):
    def setUp(self):
        self.message = Message(
            role="user",
            content="你好"
        )
        self.bus = MessageBus()

    def test_message_initialization(self):
        # 测试Message初始化
        self.assertEqual(self.message.role, "user")
        self.assertEqual(self.message.content, "你好")
        self.assertIsNone(self.message.tool_call)
        self.assertIsNone(self.message.tool_call_builtin)
        self.assertIsNone(self.message.tool_call_id)
        self.assertIsNone(self.message.audio)
        self.assertIsNone(self.message.images)
        self.assertIsNone(self.message.videos)
        self.assertIsNone(self.message.context)
        self.assertIsNone(self.message.metrics)
        self.assertIsNotNone(self.message.created_at)

    def test_message_to_dict(self):
        # 测试Message序列化
        message_dict = self.message.to_dict()
        self.assertEqual(message_dict["role"], "user")
        self.assertEqual(message_dict["content"], "你好")

    def test_message_bus(self):
        # 测试MessageBus更新和获取最新消息
        self.bus.add(self.message)
        self.assertEqual(len(self.bus), 1)
        self.assertEqual(self.bus.latest, self.message)

        # 测试通过kwargs更新
        self.bus.add(role="assistant", content="你好，有什么可以帮您？")
        self.assertEqual(len(self.bus), 2)
        self.assertEqual(self.bus.latest.role, "assistant")
        self.assertEqual(self.bus.latest.content, "你好，有什么可以帮您？")

    def test_message_role(self):
        # 测试不同角色消息的创建
        system_msg = MessageRole.system("系统消息")
        self.assertEqual(system_msg["role"], "system")
        self.assertEqual(system_msg["content"], "系统消息")

        assistant_msg = MessageRole.assistant("助手消息")
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertEqual(assistant_msg["content"], "助手消息")

        user_msg = MessageRole.user("用户消息")
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(user_msg["content"], "用户消息")

        context_msg = MessageRole.context("上下文消息")
        self.assertEqual(context_msg["role"], "system")
        self.assertTrue(context_msg["content"].startswith("上下文信息："))

if __name__ == "__main__":
    unittest.main()