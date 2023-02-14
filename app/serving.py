import os
import torch
import base64
import logging
from mega import StableDiffusionMegaPipeline


from io import BytesIO
from PIL import Image
from typing import Dict
import torch

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from together_worker.fast_inference import FastInferenceInterface
from together_web3.together import TogetherWeb3, TogetherClientOptions
from together_web3.computer import ImageModelInferenceChoice, RequestTypeImageModelInference

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "DEBUG"))

def parse_tags(input: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    if not input:
        return tags
    for word in input.split():
        if not word:
            continue
        kv = word.split("=")
        tags[kv[0]] = "=".join(kv[1:])
    return tags

class FastStableDiffusion(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        args = args if args is not None else {}
        super().__init__(model_name, args)
        self.pipe = StableDiffusionMegaPipeline.from_pretrained(
            os.environ.get("MODEL", "runwayml/stable-diffusion-v1-5"),
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token=args.get("auth_token"),
        )
        self.format = args.get("format", "JPEG")
        self.device = args.get("device", "cuda")
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.avail_modes = ["text2img", "inpaint", "img2img", "safe_text2img", "semantic_guidance"]

    def dispatch_request(self, args, env) -> Dict:
        try:
            mode = args[0].get("mode", "text2img")
            if mode not in self.avail_modes:
                raise Exception(f"Invalid mode: {mode}")
            prompt = args[0]["prompt"]
            seed = args[0].get("seed")
            generator = torch.Generator(self.device).manual_seed(seed) if seed else None

            if mode in ['inpaint', 'img2img']:
                image_input = args[0].get("image_base64")
                image_input = Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")
                image_input.thumbnail((768, 768))
            if mode == 'text2img':
                output = self.pipe.text2img(
                    prompt if isinstance(prompt, list) else [prompt],
                    generator=generator,
                    height=args[0].get("height", 512),
                    width=args[0].get("width", 512),
                    num_images_per_prompt=args[0].get("n", 1),
                    num_inference_steps=args[0].get("steps", 50),
                    guidance_scale=args[0].get("guidance_scale", 7.5),
                )
            elif mode == 'safe_text2img':
                output = self.pipe.safe_text2img(
                    prompt if isinstance(prompt, list) else [prompt],
                    generator=generator,
                    height=args[0].get("height", 512),
                    width=args[0].get("width", 512),
                    num_images_per_prompt=args[0].get("n", 1),
                    num_inference_steps=args[0].get("steps", 50),
                    guidance_scale=args[0].get("guidance_scale", 7.5),
                )
            elif mode == 'inpaint':
                output = self.pipe.inpaint(
                    prompt if isinstance(prompt, list) else [prompt],
                    image_input,
                    # mask image required, leave it for now
                    generator=generator,
                    num_images_per_prompt=args[0].get("n", 1),
                    num_inference_steps=args[0].get("steps", 50),
                    guidance_scale=args[0].get("guidance_scale", 7.5),
                )
            elif mode == 'img2img':
                output = self.pipe.img2img(
                    prompt if isinstance(prompt, list) else [prompt],
                    image_input,
                    generator=generator,
                    height=args[0].get("height", 512),
                    width=args[0].get("width", 512),
                    num_images_per_prompt=args[0].get("n", 1),
                    num_inference_steps=args[0].get("steps", 50),
                    guidance_scale=args[0].get("guidance_scale", 7.5),
                )
            elif mode == "semantic_guidance":
                print(args[0])
                editing_prompt = args[0].get("editing_prompt", None)
                
                reverse_editing_direction = args[0].get("reverse_editing_direction", False)
                edit_guidance_scale = args[0].get("edit_guidance_scale", 500)
                edit_warmup_steps = args[0].get("edit_warmup_steps", 10)
                edit_cooldown_steps = args[0].get("edit_cooling_steps", None)
                edit_threshold = args[0].get("edit_threshold", None)
                edit_momentum_scale = args[0].get("edit_momentum", 0.1)
                edit_mom_beta = args[0].get("edit_mom_beta", 0.4)
                edit_weight = args[0].get("edit_weight", None)
                sem_guidance = args[0].get("sem_guidance", None)
                output = self.pipe.semantic_guidance(
                    prompt if isinstance(prompt, list) else [prompt],
                    generator=generator,
                    height=args[0].get("height", 512),
                    width=args[0].get("width", 512),
                    num_images_per_prompt=args[0].get("n", 1),
                    num_inference_steps=args[0].get("steps", 50),
                    guidance_scale=args[0].get("guidance_scale", 7.5),
                    editing_prompt=editing_prompt,
                    reverse_editing_direction=reverse_editing_direction,
                    edit_guidance_scale=edit_guidance_scale,
                    edit_warmup_steps=edit_warmup_steps,
                    edit_cooldown_steps=edit_cooldown_steps,
                    edit_threshold=edit_threshold,
                    edit_momentum_scale=edit_momentum_scale,
                    edit_mom_beta=edit_mom_beta,
                    edit_weight=edit_weight,
                    sem_guidance=sem_guidance,
                )
            choices = []
            for image in output.images:
                buffered = BytesIO()
                image.save(buffered, format=args[0].get("format", self.format))
                img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
                choices.append(ImageModelInferenceChoice(img_str))
            return {
                "result_type": RequestTypeImageModelInference,
                "choices": choices
            }
        except Exception as e:
            logging.exception(e)
            return {
                "result_type": "error",
                "value": str(e),
            }

if __name__ == "__main__":
    coord_url = os.environ.get("COORD_URL", "localhost")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=os.environ.get("COORD_HTTP_URL", f"http://{coord_url}:8092"),
        websocket_url=os.environ.get("COORD_WS_URL", f"ws://{coord_url}:8093/websocket"),
    )
    fip = FastStableDiffusion(
        model_name=os.environ.get("SERVICE", "universal-diffusion"),
        args={
            "auth_token": os.environ.get("AUTH_TOKEN"),
            "coordinator": coordinator,
            "device": os.environ.get("DEVICE", "cuda"),
            "format": os.environ.get("FORMAT", "JPEG"),
            "gpu_num": 1 if torch.cuda.is_available() else 0,
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_mem": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "group_name": os.environ.get("GROUP", "group1"),
            "worker_name": os.environ.get("WORKER", "worker1"),
        }
    )
    fip.start()