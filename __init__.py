from typing import Any, Dict, Tuple
from .utils import gen_openai, gen_claude

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

def replace_placeholders(template: str, data: dict) -> str:
        try:
            return template.format(**data)
        except KeyError as e:
                # Handle missing keys in the dictionary
                raise KeyError(f"Missing key in data: {e}")

class LLMConcat:
    def __contains__(self, key):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("String",)
    FUNCTION = "run"
    CATEGORY = "5x00"

    @classmethod
    def IS_CHANGED(cls):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, dict]:
        return {
            "required": {
                "API_Key": ("STRING", {"default": ""}),
                "Service" : (["OpenAI", "Claude"],),
                "Prompt" : ("STRING", {"default": "Describe the image."}),
            },
            "optional": {},
        }

    def run(self, Prompt, API_Key, Service, **kwargs) -> tuple:
        # Process the values
        result = replace_placeholders(Prompt, kwargs)
        caption = ""
        if API_Key is not None:
            if Service == "OpenAI":
                caption = gen_openai(API_Key, result)
            if Service == "Claude":
                caption = gen_claude(API_Key, result)
        else:
             caption = result
        return (caption,)

NODE_CLASS_MAPPINGS = {
    "LLMConcate": LLMConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMConcate": "Concat w/ LLM",
}