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
    processed = preprocess_image(warped)
    omr_results = extract_omr_answers(processed, config)
    text_blocks = extract_text_and_boxes(warped, config.get('languages', ['ar', 'en']))
    questions = []
    q_num = 1
    for block in text_blocks:
        if any('\u0600' <= c <= '\u06FF' for c in block['text']):
            translation = translate_text(block['text'], config.get('translation_engine', 'googletrans'))
            bbox = block['bbox']
            questions.append({
                'question_number': q_num,
                'arabic_text': block['text'],
                'english_translation': translation,
                'bounding_box': bbox
            })
            q_num += 1
    output = {
        'omr_results': omr_results,
        'extracted_questions': questions
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py <image_path>')
        sys.exit(1)
    main(sys.argv[1])
