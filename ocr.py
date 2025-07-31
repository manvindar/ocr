import easyocr

def extract_text_and_boxes(image, languages=['ar', 'en']):
    reader = easyocr.Reader(languages, gpu=False)
    results = reader.readtext(image)
    text_blocks = []
    for (bbox, text, conf) in results:
        text_blocks.append({
            'text': text,
            'bbox': bbox,
            'confidence': conf
        })
    return text_blocks
