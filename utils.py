import cv2
import numpy as np
from PIL import Image

def load_image(path):
    return cv2.imread(path)

def save_image(path, img):
    cv2.imwrite(path, img)

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
