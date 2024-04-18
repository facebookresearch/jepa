from vjepa_encoder.vision_encoder import JepaEncoder

encoder = JepaEncoder.load_model(
    "logs/params-encoder.yaml"
)

import numpy
import torch
img = numpy.random.random(size=(360, 480, 3))

x = torch.rand((32, 3, 256, 900))

print("Input Img:", img.shape)
embedding = encoder.embed_image(img)

print(embedding)
print(embedding.shape)


embedding = encoder.embed_image(x)
print(embedding)
print(embedding.shape)