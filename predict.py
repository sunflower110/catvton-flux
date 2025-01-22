import os
import subprocess
import time

from cog import BasePredictor, Input, Path, Secret
from diffusers.utils import load_image, check_min_version
from diffusers import FluxFillPipeline
from diffusers import FluxTransformer2DModel
import numpy as np
import torch
from torchvision import transforms

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load part of the model into memory to make running multiple predictions efficient"""
        self.dtype = torch.bloat16
        self.try_on_transformer = FluxTransformer2DModel.from_pretrained("xiaozaa/catvton-flux-beta", 
            torch_dtype=self.dtype)
        self.try_off_transformer = FluxTransformer2DModel.from_pretrained("xiaozaa/cat-tryoff-flux", 
            torch_dtype=self.dtype)
        
    def predict(self,
                hf_token: Secret(description="Hugging Face API token. Create a write token at https://huggingface.co/settings/token. You also need to approve the Flux Dev terms."),
                image: Path = Input(description="Image file path", default="https://github.com/nftblackmagic/catvton-flux/raw/main/example/person/1.jpg"),
                mask: Path = Input(description="Mask file path", default="https://github.com/nftblackmagic/catvton-flux/blob/main/example/person/1_mask.png?raw=true"),
                try_on: bool = Input(True, description="Try on or try off"),
                garment: Path = Input(description="Garment file path", default="https://github.com/nftblackmagic/catvton-flux/raw/main/example/garment/00035_00.jpg"),
                num_steps: int = Input(50, description="Number of steps to run the model for"),
                guidance_scale: float = Input(30, description="Guidance scale for the model"),
                seed: int = Input(0, description="Seed for the model"),
                width: int = Input(576, description="Width of the output image"),
                height: int = Input(768, description="Height of the output image")):
                
        
        size = (width, height)
        if try_on:
            self.transformer = self.try_on_transformer
        else:
            self.transformer = self.try_off_transformer

        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=self.transformer,
            torch_dtype=self.dtype,
            token=hf_token
        ).to("cuda")

        self.pipe.transformer.to(self.dtype)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # For RGB images
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        i = load_image(image).convert("RGB").resize(size)
        m = load_image(mask).convert("RGB").resize(size)
        g = load_image(garment).convert("RGB").resize(size)

        # Transform images using the new preprocessing
        image_tensor = transform(i)
        mask_tensor = mask_transform(m)[:1]  # Take only first channel
        garment_tensor = transform(g)

        # Create concatenated images
        inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)  # Concatenate along width
        garment_mask = torch.zeros_like(mask_tensor)

        if try_on:
            extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)
        else:
            extended_mask = torch.cat([1 - garment_mask, mask_tensor], dim=2)

        prompt = f"The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; " \
                f"[IMAGE1] Detailed product shot of a clothing" \
                f"[IMAGE2] The same cloth is worn by a model in a lifestyle setting."
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            height=size[1],
            width=size[0] * 2,
            image=inpaint_image,
            mask_image=extended_mask,
            num_inference_steps=num_steps,
            generator=generator,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            prompt=prompt,
        ).images[0]

        # Split and save results
        width = size[0]
        garment_result = result.crop((0, 0, width, size[1]))
        try_result = result.crop((width, 0, width * 2, size[1]))
        
        return garment_result, try_result
