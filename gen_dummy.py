from PIL import Image
import numpy as np

# Create a dummy image (224x224 RGB image, completely black representing no DR features)
img_array = np.zeros((224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save('dummy_img.jpg')
print("Dummy image saved!")
