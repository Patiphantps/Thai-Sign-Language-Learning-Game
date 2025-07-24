import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model

def load_models():
    return {
        "life": load_model("model/life1_model.h5"),
        "greeting": load_model("model/greeting1_model.h5"),
        "family": load_model("model/family1_model.h5"),
        "feeling": load_model("model/feeling1_model.h5")
    }

def preprocess_image(frame):
    # ตัดตรงกลาง
    height, width, _ = frame.shape
    x1 = (width - 250) // 2
    y1 = (height - 250) // 2
    cropped = frame[y1:y1+250, x1:x1+250]

    # resize + normalize
    resized = cv2.resize(cropped, (224, 224))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)
