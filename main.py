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

def process_image(image_path, config):
    import numpy as np
    import re
    image = utils.load_image(image_path)
    doc_cnt = find_document_contour(image)
    if doc_cnt is not None:
        warped = four_point_transform(image, doc_cnt)
    else:
        warped = image
    text_blocks = extract_text_and_boxes(warped, config.get('languages', ['ar', 'en']))
    def to_native(obj):
        if isinstance(obj, (np.integer, int, float)):
            return int(obj)
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [to_native(x) for x in obj]
        return obj
    arabic_blocks = []
    for block in text_blocks:
        text = block['text'].strip()
        arabic_letters = re.findall(r'[\u0600-\u06FF]', text)
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
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
    return arabic_blocks

def main(input_path):
    config = load_config()
    import glob
    import os
    all_results = []
    if os.path.isdir(input_path):
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
        image_files.sort()
        for img_path in image_files:
            print(f"\nProcessing: {img_path}\n{'='*40}")
            arabic_blocks = process_image(img_path, config)
            for item in arabic_blocks:
                print(f"Arabic: {item['arabic_text']}")
                print(f"English: {item['english_translation']}")
                print(f"Bounding Box: {item['bounding_box']}")
                print('-' * 40)
            print("\nLine by line translation:\n")
            for idx, item in enumerate(arabic_blocks, 1):
                print(f"{idx}. {item['english_translation']}")
                print(f"{item['arabic_text']}")
            all_results.append({'image': img_path, 'results': arabic_blocks})
    else:
        arabic_blocks = process_image(input_path, config)
        for item in arabic_blocks:
            print(f"Arabic: {item['arabic_text']}")
            print(f"English: {item['english_translation']}")
            print(f"Bounding Box: {item['bounding_box']}")
            print('-' * 40)
        print("\nLine by line translation:\n")
        for idx, item in enumerate(arabic_blocks, 1):
            print(f"{idx}. {item['english_translation']}")
            print(f"{item['arabic_text']}")
        all_results.append({'image': input_path, 'results': arabic_blocks})
    # Optionally, return all_results or write to a file

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py <image_path>')
        sys.exit(1)
    main(sys.argv[1])
