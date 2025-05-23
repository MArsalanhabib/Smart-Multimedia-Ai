
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
import numpy as np

model = resnet50.ResNet50(weights='imagenet')

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet50.preprocess_input(x)
    preds = model.predict(x)
    return resnet50.decode_predictions(preds, top=3)[0]
