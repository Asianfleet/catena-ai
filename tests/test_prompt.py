import unittest

from catena_ai.llmchain.prompt import ModelPrompt

class TestPrompt(unittest.TestCase):
    def test_load_template(self):
        template = ModelPrompt.load_prompt("comic.elements_complete")
        self.assertEqual(template.message[0]["role"], "system")