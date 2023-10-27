# Databricks notebook source
from transformers import pipeline
import torch

pipe = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

# COMMAND ----------

response = pipe("History of Thailand")
print(response[0]["generated_text"])

# COMMAND ----------

response = pipe("Who is the prime minister of Thailand?")
print(response[0]["generated_text"])

# COMMAND ----------

response = pipe("วิธีทำกะเพราหมูกรอบ")
print(response[0]["generated_text"])
