from typing import Any, Dict, Tuple

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

class LLMConcat:

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("String",)
    FUNCTION = "concat"
    CATEGORY = "5x00"

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, dict]:
        return {
            "required": {
                "Prompt" : ("STRING", {"multiline": True, "default": "{string_1} wearing {string_2} standing in front of a {string_3}"}),
                "API_Key" : ("STRING", {}),
                "Service" : (["OpenAI", "Claude"],),
            },
            "optional": {}
        }

    def run(self, **kw) -> Tuple[Any | None]:
        value = None
        if len(values := kw.values()) > 0:
            value = next(iter(values))
        return (value,)

NODE_CLASS_MAPPINGS = {
    "LLMConcate": LLMConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMConcate": "Concat w/ LLM",
}