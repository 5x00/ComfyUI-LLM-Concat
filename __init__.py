from typing import Any, Dict, Tuple

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

class LLMConcat:

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("String",)
    FUNCTION = "run"
    CATEGORY = "5x00"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, dict]:
        return {
            "required": {},
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