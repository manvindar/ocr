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
    import re
    arabic_blocks = []
    for block in text_blocks:
        # Remove blocks that are too short, contain no Arabic, or are likely icons/symbols
        text = block['text'].strip()
        # Only keep if at least 2 Arabic words and mostly Arabic letters
        arabic_letters = re.findall(r'[\u0600-\u06FF]', text)
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
        # Heuristic: at least 2 words, at least 60% of chars are Arabic, and length > 4
        if len(arabic_words) >= 2 and len(arabic_letters) / max(len(text),1) > 0.6 and len(text) > 4:
            try:
                translation = translate_text(text, engine='deep-translator')
            except Exception:
                translation = translate_text(text, engine='googletrans')
            bbox = block['bbox']
            bbox_py = to_native(bbox)
            arabic_blocks.append({
                'arabic_text': text,
                'english_translation': translation,
                'bounding_box': bbox_py
            })
    # Print detailed info as before
    for item in arabic_blocks:
        print(f"Arabic: {item['arabic_text']}")
        print(f"English: {item['english_translation']}")
        print(f"Bounding Box: {item['bounding_box']}")
        print('-' * 40)
    # Print clean line-by-line translation as requested
    print("\nLine by line translation:\n")
    for idx, item in enumerate(arabic_blocks, 1):
        print(f"{idx}. {item['english_translation']}")
        print(f"{item['arabic_text']}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py <image_path>')
        sys.exit(1)
    main(sys.argv[1])
