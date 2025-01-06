import numpy as np
import os
import base64
from openai import OpenAI
import PIL
import torch
from io import BytesIO
import anthropic

def gen_openai(API_Key, Prompt):
    # Initialize OpenAI client
    client = OpenAI(api_key=API_Key)
    # ChatGPT completions
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Write a haiku about programming."
                    }],
            }
        ],
    )
    return response.choices[0].message.content

def gen_claude(API_Key, Prompt):
    #Initialize Claude client
    client = anthropic.Anthropic(api_key=API_Key)

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": Prompt}
    ])

    return message.content