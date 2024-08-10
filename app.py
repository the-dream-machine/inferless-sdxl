import base64
from diffusers import DiffusionPipeline,EulerDiscreteScheduler
from io import BytesIO
import torch

class InferlessPythonModel:
  def initialize(self):
    scheduler_kwargs = {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "interpolation_type": "linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "steps_offset": 1,
        "timestep_spacing": "leading",
        "trained_betas": None,
        "use_karras_sigmas": False
    }
    
    self.base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        scheduler=EulerDiscreteScheduler(**scheduler_kwargs),
    ).to("cuda")
    
    self.refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=self.base.text_encoder_2,
        vae=self.base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        scheduler=EulerDiscreteScheduler(**scheduler_kwargs),
    ).to("cuda")
    
    # This section attempts to compile the UNet models within the pipelines
    # using `torch.compile` for potentially faster inference. 
    # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
    # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

  def infer(self, inputs):
    prompt = inputs["prompt"]
    
    image = self.base(
        prompt=prompt,
        negative_prompt="low quality, low resolution, greyscale, multiple fingers, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, deformed toes standing still, posing",
        num_inference_steps=20,
        denoising_end=0.8,
        output_type="latent",
    ).images
      
    image = self.refiner(
        prompt=prompt,
        num_inference_steps=20,
        denoising_start=0.8,
        image=image,
    ).images[0]

    buff = BytesIO()
    image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return {"generated_image_base64": img_str.decode('utf-8')}

  def finalize(self,args):
    self.base = None
    self.refiner = None
