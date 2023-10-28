from google.cloud import translate_v2
from ratelimit import limits, sleep_and_retry
import html
import logging

class Translator():
    def __init__(self) -> None:
        self.client = translate_v2.Client(target_language="en")

    @sleep_and_retry
    @limits(calls=15, period=1)
    def translate(self, text):
        logging.info(f"translating {text[:20]}")
        result = self.client.translate(text)
        return html.unescape(result["translatedText"])
    