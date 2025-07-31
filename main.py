import cv2
import json
import sys
import os
from omr import find_document_contour, four_point_transform, preprocess_image, extract_omr_answers
from ocr import extract_text_and_boxes
from translate import translate_text
import utils

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(image_path):
    config = load_config()
    image = utils.load_image(image_path)
    doc_cnt = find_document_contour(image)
    if doc_cnt is not None:
        warped = four_point_transform(image, doc_cnt)
    else:
        warped = image
    text_blocks = extract_text_and_boxes(warped, config.get('languages', ['ar', 'en']))
    results = []
    import numpy as np
    def to_native(obj):
        if isinstance(obj, (np.integer, int, float)):
            return int(obj)
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [to_native(x) for x in obj]
        return obj
    for block in text_blocks:
        if any('\u0600' <= c <= '\u06FF' for c in block['text']):
            # Use deep-translator for better translation quality, fallback to googletrans if needed
            try:
                translation = translate_text(block['text'], engine='deep-translator')
            except Exception:
                translation = translate_text(block['text'], engine='googletrans')
            bbox = block['bbox']
            bbox_py = to_native(bbox)
            results.append({
                'arabic_text': block['text'],
                'english_translation': translation,
                'bounding_box': bbox_py
            })
    for item in results:
        print(f"Arabic: {item['arabic_text']}")
        print(f"English: {item['english_translation']}")
        print(f"Bounding Box: {item['bounding_box']}")
        print('-' * 40)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py <image_path>')
        sys.exit(1)
    main(sys.argv[1])
