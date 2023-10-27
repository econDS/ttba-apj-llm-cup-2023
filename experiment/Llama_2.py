# Databricks notebook source
# !pip install auto-gptq==0.4.2
!pip install auto-gptq

# COMMAND ----------

from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
# model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# COMMAND ----------

use_triton = False
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

# COMMAND ----------

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.3,
    repetition_penalty=1.15
)

# COMMAND ----------

response = pipe("History of Thailand")
print(response[0]["generated_text"])

# COMMAND ----------

response = pipe("Who is the prime minister of Thailand")
print(response[0]["generated_text"])

# COMMAND ----------

response = pipe("วิธีทำกะเพราหมูกรอบ")
print(response[0]["generated_text"])
