# pyright: basic
# %% [markdown]
r"""
3. Завдання щодо генерації або стилізації зображень (на вибір)
Вирішіть завдання перенесення стилю або генерації зображень (архітектура за вашим вибором: GAN/DCGAN/VAE/Diffusion).
"""

# %%
import torch
from diffusers import StableDiffusionPipeline

# %%
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# %%
prompt = "A fantasy landscape with mountains and a lake, digital art"
image = pipeline(prompt).images[0]
image.show()
