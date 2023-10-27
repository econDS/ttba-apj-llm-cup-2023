# Databricks notebook source
!pip install -q google-generativeai

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

import pprint
import google.generativeai as palm

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gen an API key
# MAGIC https://developers.generativeai.google/tutorials/setup

# COMMAND ----------

from dotenv import load_dotenv
import os
load_dotenv()
PALM_API_KEY = os.environ.get("PALM_API_KEY")

# COMMAND ----------

palm.configure(api_key=PALM_API_KEY)

# COMMAND ----------

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name
print(model)

# COMMAND ----------

prompt = "History of Thailand"

completion = palm.generate_text(
    model=model,
    prompt=prompt,
    temperature=0,
    # The maximum length of the response
    max_output_tokens=800,
)

print(completion.result)

# COMMAND ----------

prompt = "Who is the priminister of Thailand?"

completion = palm.generate_text(
    model=model,
    prompt=prompt,
    temperature=0,
    # The maximum length of the response
    max_output_tokens=800,
)

print(completion.result)

# COMMAND ----------

prompt = "วิธีทำกะเพราหมูกรอบ"

completion = palm.generate_text(
    model=model,
    prompt=prompt,
    temperature=0,
    # The maximum length of the response
    max_output_tokens=800,
)

print(completion.result)
