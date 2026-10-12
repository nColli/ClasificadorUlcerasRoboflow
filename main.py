from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

API_URL: str = os.getenv('API_URL') or 'https://serverless.roboflow.com'
API_KEY = os.getenv('API_KEY')
API_MODEL = os.getenv('API_MODEL')

client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY,
).configure(InferenceConfiguration(api_key_transport="header"))

pil_image = Image.open("images.jpeg")
results = client.infer(pil_image, model_id=API_MODEL)
print(results)