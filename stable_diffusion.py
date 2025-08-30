import os
import boto3
import json
import base64


prompt = """
Provide me with a 4K HD image of a beach, featuring a blue sky in the rainy season and a cinematic look.
"""
prompt_template = [{"text": prompt,"weight": 1}]
bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "text_prompts": prompt_template,
    "cfg_scale": 10,
    "seed": 0,
    "steps": 50,
    "width": 1024,
    "height": 1024
}

body = json.dumps(payload)
model_id = "stability.stable-diffusion-xl-v0"

response = bedrock.invoke_model(
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
    body=body
)

response_body = json.loads(response.get("body").read())
print(response_body)

artifact = response_body.get("artifacts")[0]
encoded_img = artifact.get("base64").encode("UTF-8")
decoded_img = base64.b64decode(encoded_img)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/img.png"

with open(file_name, "wb") as f:
    f.write(decoded_img)
