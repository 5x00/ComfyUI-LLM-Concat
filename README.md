# ComfyUI-Prompt-Plus
Prompt Plus is a collection of LLM and VLM nodes that make prompting easier for image and video generation.

⭐ Trigger Words to Prompt / Trigger2Prompt

⭐ Guided VLM Prompter / VLM2Prompt

## Nodes
### Trigger2Prompt
Let's use combine any number of trigger words and combine them into a single prompt through a guided approach. This can be useful if you're running a VLM node or captioner on multiple images and need to combine the data into a prompt.
There are three variations to this node > [API] uses ChatGPT/Claude for processing, [Custom] uses Phi-3.5-mini for local processing, and [Simple] let's you parse the string as it is without any processing.

![image](https://github.com/user-attachments/assets/5f3cf352-15d6-4a15-b8e6-26c329b8bbda)

### VLM2Prompt
Let's you use a VLM along with a guidance prompt to generate captions. This can be useful if you want to generate prompts from images with some level of tweaking.
There are two variations to this node > [API] uses ChatGPT/Claude for processing, [Custom] uses Florence for local processing

### Utililty
Includes a few nodes that support Trigger2Prompt and Guided VLM.

## Installation

- git clone this repository into ComfyUI/custom_nodes
- pip install -r requirements.txt

## Example Workflow


Download the "flux_example.json" to test the full workflow.

