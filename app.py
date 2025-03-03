import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import torch
from diffusers import DiffusionPipeline
from io import BytesIO
import base64
import os


class InferlessPythonModel:
    def initialize(self):
        # Configure the pipeline
        model_id = "cagliostrolab/animagine-xl-3.1"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = DiffusionPipeline.from_pretrained(
                        model_id, 
                        torch_dtype=torch.float16, 
                        use_safetensors=True, 
                    )
        self.pipe.to('cuda')

    def infer(self, inputs):
        prompt = inputs["prompt"]
        negative_prompt = inputs["negative_prompt"]
        
        image = self.pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            width=832,
            height=1216,
            guidance_scale=7,
            num_inference_steps=28
        ).images[0]

        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        return { "generated_image_base64" : img_str }
        
    def finalize(self):
        self.pipe = None
