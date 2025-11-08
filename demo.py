from model.image_encoder import ImageEncoder
from PIL import Image
import requests
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_encoder = ImageEncoder()
image_feature_extractor=image_encoder.get_feature_extractor() 
feature_extractor = image_feature_extractor(images=image, return_tensors="pt")
print(feature_extractor['pixel_values'])