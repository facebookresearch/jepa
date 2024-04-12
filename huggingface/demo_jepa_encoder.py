from vjepa_encoder.vision_encoder import JepaEncoder

encoder = JepaEncoder.load_model(
    "logs/params-encoder.yaml"
)

import numpy
img = numpy.random.random(size=(360, 480, 3))

print("Input Img:", img.shape)
embedding = encoder.embed_image(img)

print(embedding)
print(embedding.shape)