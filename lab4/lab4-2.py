# pyright: basic
# %% [markdown]
r"""
Недождій Олексій ФФ-31мн

Lab 4. Advanced Nets
2. Проведіть експерименти з моделями бібліотеки Hugging Face (раніше - Hugging Face Transformers, https://huggingface.co/) за допомогою (наприклад) Pipeline модуля
"""

# %%
from transformers import pipeline

# %%
pipe = pipeline(model="benjamin/gpt2-large-wechsel-ukrainian")

# %%
user_input = input("Ask model something: ")
output = pipe(user_input, return_full_text=True)
print(output)

# %%
