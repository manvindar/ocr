from googletrans import Translator
from deep_translator import GoogleTranslator

def translate_text(text, engine='googletrans'):
    if engine == 'googletrans':
        translator = Translator()
        result = translator.translate(text, src='ar', dest='en')
        return result.text
    elif engine == 'deep-translator':
        return GoogleTranslator(source='ar', target='en').translate(text)
    else:
        raise ValueError('Unsupported translation engine')
