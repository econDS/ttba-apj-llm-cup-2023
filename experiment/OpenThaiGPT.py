# Databricks notebook source
from transformers import pipeline
import torch

pipe = pipeline(model="Adun/openthaigpt-1.0.0-beta-7b-ckpt-hf", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

# COMMAND ----------

def create_promp_tempate(prompt: str) -> str:
    return f"""
    ### Instruction
    {prompt}
    
    ### Answer
    """

# COMMAND ----------

prompt = "นายกรัฐมนตรีของไทยคือใคร ในปี 2562"
response = pipe(create_promp_tempate(prompt), max_new_tokens=200, temperature=0.1, top_p=0.3, repetition_penalty=1.15)
print(response[0]["generated_text"])

# COMMAND ----------

prompt = "ถ้าวันนี้คือวันจันทร์ พรุ่งนี้คือวันอะไร"
response = pipe(create_promp_tempate(prompt), max_new_tokens=200, temperature=0.1, top_p=0.3, repetition_penalty=1.15)
print(response[0]["generated_text"])

# COMMAND ----------

prompt = "ขั้นตอนการทำผัดกะเพราหมูกรอบ"
response = pipe(create_promp_tempate(prompt), max_new_tokens=200, temperature=0.1, top_p=0.3, repetition_penalty=1.15)
print(response[0]["generated_text"])

# COMMAND ----------

prompt = "ประวัติศาสตร์ของประเทศไทย"
response = pipe(create_promp_tempate(prompt), max_new_tokens=200, temperature=0.1, top_p=0.3, repetition_penalty=1.15)
print(response[0]["generated_text"])
