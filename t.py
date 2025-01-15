from llmchain.prompt.prompt import LLMPrompt
from llmchain.model.models import Qwen, OpenAIC
from llmchain.parser.parser import LLMOutputParser
from llmchain.memory.memory import InfMemory
from catena_core.settings import settings

import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_64 = encode_image("/home/legion4080/AIPJ/MYXY/test/12.18/background.png")
# prompt = LLMPrompt.from_template("utils.analyze_image")

str_tem = """ 
假如你是一个经济评论员，请针对给出的材料进行分析，按要求回答问题。
问题：{task}
材料：{context}
""" 

prompt = LLMPrompt.from_template(str_tem)

context = """ 
2025年政策将更加积极。中央经济工作会议指出，“更积极的财政政策”+“适度宽松的货币政策”，预计力度为过去十年之最；
“大力提振消费，提高投资效益，全方位扩大国内需求”；“积极发展首发经济、冰雪经济、银发经济”；
“持续用力推动房地产市场止跌回稳”；“高质量完成国有企业改革深化提升行动，出台民营经济促进法”。
"""
question = "“更积极的财政政策”+“适度宽松的货币政策”具体怎样影响房地产市场止跌回稳？"

settings.debug.configure(
    enable_func_level_debug=False, 
    enable_chain_visualize=True
)
# pipe = LLMPrompt("你好") >> Qwen()
qwen = OpenAIC(
    model="qwen-turbo"
)
#pipe = "你好" >> LLMPrompt >> Qwen >> LLMOutputParser
#pipe = "你好" >> LLMPrompt >> qwen >> LLMOutputParser
#pipe = LLMPrompt("你好") >> qwen >> LLMOutputParser
""" pipe = (
    {"image": image_64, "query": "描述这幅图片"} 
    | prompt 
    | OpenAIC("qwen-vl-max-latest") 
    | LLMOutputParser()
) """
pipe = {"context": context, "task": question} >> prompt >> qwen
pipe1 = LLMPrompt >> InfMemory >> Qwen
print("\n结果：", pipe1.operate("你好"), "\n\n")
print("\n结果：", pipe1.operate("什么是Python"))