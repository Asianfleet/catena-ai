import os
from typing import List
from openai import OpenAI

from catena.settings import settings


def init_minimal_client(provider: str = "OpenAI"):
    if provider == "OpenAI":
        return OpenAI()
    else:
        raise ValueError(f"Unknown model {provider}")
    

def minimal_llm_response(
    model: str = "gpt-4o-mini",
    messages: List = None,
    **kwargs
) -> OpenAI:
    
    client = init_minimal_client()
    if kwargs:
        response_args = settings.minimal_llm.config.__container__.update(kwargs)
    else:
        response_args = settings.minimal_llm.config.__container__
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **response_args
    )
    return response.choices[0].message.content
print(str(minimal_llm_response.__code__))
