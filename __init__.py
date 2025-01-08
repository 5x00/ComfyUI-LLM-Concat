from typing import Any, Dict, Tuple
from .utils import gen_openai, gen_claude
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor
import comfy.model_management as mm
import folder_paths

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
        
class Prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Prompt": ("STRING", {"default": "Eg. {string_1} standing in a beautiful {string_2} scene.", "multiline": True}),              
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "5x00/Prompt Plus/Utils"

    def run(self, Prompt):
        return (Prompt,)

class TriggerToPromptAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "run"
    CATEGORY = "5x00/Prompt Plus"

    def run(self, **kwargs) -> tuple:
        # Process the values
        api_model = kwargs['model']
        api_key = api_model['api_key']
        service = api_model['api']
        result = replace_placeholders(kwargs['prompt'], kwargs)
        caption = ""
        if api_key is not None:
            if service == "OpenAI":
                caption = gen_openai(api_key, result)
            if service == "Claude":
                caption = gen_claude(api_key, result)
        else:
             caption = result

        caption = caption.rstrip("\"")
        caption = caption.lstrip("\"")
        return (caption,)
    
class LoadAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "api" : (["OpenAI", "Claude"],),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("APIMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_api"
    CATEGORY = "5x00/Prompt Plus/Utils"

    def load_api(self, api, api_key):
        _model = {
            'api': api, 
            'api_key': api_key
        }

        return (_model,)
    
class LoadCustomModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        'microsoft/Phi-3.5-mini-instruct',
                    ],
                    {
                        "default": 'microsoft/Phi-3.5-mini-instruct'
                    }
                ),
                "precision": (
                    ['fp16', 'bf16', 'fp32'],
                    {
                        "default": 'bf16'
                    }
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("CUSTOMMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "5x00/Prompt Plus/Utils"

    def load_model(self, model, precision):
        device = mm.get_torch_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)
        
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model, local_dir=model_path, local_dir_use_symlinks=False)
            
        print(f"Loading custom model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        custom_model = {
            'model': model, 
            'tokenizer': tokenizer,
            'dtype': dtype
        }

        return (custom_model,)

class TriggerToPromptCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {         
                "max_new_tokens": ("INT", {"default": 500, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.1, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),
                "do_sample": ("BOOLEAN", {"default": False}),                
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_text",)
    FUNCTION = "run"
    CATEGORY = "5x00/Prompt Plus"

    def run(self, max_new_tokens, temperature, top_p, do_sample, **kwargs):
        model = kwargs['model']
        prompt = kwargs['prompt']
        tokenizer = model['tokenizer']

        model = model['model']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        fixed_prompt = replace_placeholders(prompt, kwargs)

        messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": fixed_prompt},
            ]
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": temperature,
            "do_sample": do_sample,
        }

        output = pipe(messages, **generation_args)
        output_text = output[0]['generated_text']
        output_text = output_text.rstrip("\"")
        output_text = output_text.lstrip(" \"")
        return (output_text,)
    
class TriggerToPromptSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {                     
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_text",)
    FUNCTION = "run"
    CATEGORY = "5x00/Prompt Plus"

    def run(self, **kwargs):

        prompt = replace_placeholders(kwargs['prompt'], kwargs)
        return (prompt,)

class LoadFlorenceModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        'thwri/CogFlorence-2.2-Large',
                    ],
                    {
                        "default": 'thwri/CogFlorence-2.2-Large'
                    }
                ),
                "precision": (
                    ['fp16', 'bf16', 'fp32'],
                    {
                        "default": 'fp16'
                    }
                ),
                "attention": (
                    ['flash_attention_2', 'sdpa', 'eager'],
                    {
                        "default": 'sdpa'
                    }
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("VLMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "5x00/Prompt Plus/Utils"

    def load_model(self, model, precision, attention):
        device = mm.get_torch_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "VLM", model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading Florence2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model, local_dir=model_path, local_dir_use_symlinks=False)
            
        print(f"Using {attention} for attention")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("thwri/CogFlorence-2.2-Large", trust_remote_code=True)

        _model = {
            'model': model, 
            'processor': processor,
            'dtype': dtype
        }

        return (_model,)

class RunCustomVLM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("VLMODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "run_model"
    CATEGORY = "5x00/Prompt Plus"

    def run_model(self, image, model, prompt, max_new_tokens, num_beams, do_sample):
        device = mm.get_torch_device()
        processor = model['processor']
        model = model['model']
        dtype = model['dtype']
        model.to(device)

        image = image.permute(0, 3, 1, 2)
        image_pil = F.to_pil_image(image[0])
        inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(device)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        print(parsed_answer)

        return (parsed_answer,)

class RunAPIVLM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Image" : ("IMAGE", {}), 
                "Prompt" : ("STRING", {"multiline": True, "default": "Describe the image."}),
                "API_Key" : ("STRING", {}),
                "Service" : (["OpenAI", "Claude"],),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "create_caption"
    CATEGORY = "5x00/Prompt Plus"

    def create_caption(self, Image, Prompt, API_Key, Service):
        print("Generating caption for the image...")
        caption = ""
        if Service == "OpenAI":
            caption = gen_openai(API_Key, Image, Prompt)
        if Service == "Claude":
            caption = gen_claude(API_Key, Image, Prompt)
        print(f"Caption generated: {caption}")
        return (caption,)

NODE_CLASS_MAPPINGS = {
    "LoadAPI" : LoadAPI,
    "TriggerToPromptAPI": TriggerToPromptAPI,
    "LoadCustomModel": LoadCustomModel,
    "TriggerToPromptCustom": TriggerToPromptCustom,
    "TriggerToPromptSimple": TriggerToPromptSimple,
    "LoadFlorenceModel": LoadFlorenceModel,
    "RunCustomVLM": RunCustomVLM,
    "RunAPIVLM" : RunAPIVLM,
    "Prompt": Prompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TriggerToPromptAPI": "Trigger2Prompt [ChatGPT/Claude]",
    "TriggerToPromptCustom" : "Trigger2Prompt [Custom]",
    "TriggerToPromptSimple" : "Trigger2Prompt [Simple]",
    "LoadAPI" : "Load API Model",
    "LoadCustomModel": "Load Custom Model",
    "LoadFlorenceModel": "Load Florence Model",
    "RunCustomVLM": "VLM2Prompt [Custom]",
    "RunAPIVLM": "VLM2Prompt [ChatGPT/Claude]",
    "Prompt": "Multiline Prompt",
}
