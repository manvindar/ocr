# OMR & Arabic OCR Application

This application processes images of Arabic OMR sheets and documents, extracts marked answers, performs high-accuracy OCR for Arabic text, and provides English translations.

## Usage

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

2. Run the application:
   ```sh
   python main.py <path_to_image>
   ```

3. Output is printed as JSON with OMR results and extracted questions.

## Configuration
- Edit `config.json` to adjust OMR layout, language, and translation engine.

## Modules
- `main.py`: Entry point
- `omr.py`: OMR processing
- `ocr.py`: OCR extraction
- `translate.py`: Translation
- `utils.py`: Helpers

## Requirements
- Python 3.8+
- See `requirements.txt`
# ocr
