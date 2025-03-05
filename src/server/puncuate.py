import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()


def setup_gemini_api(api_key: str):
    """Initialize the Gemini API client with the given API key."""
    genai.configure(api_key=api_key)


def add_punctuation(sentence: str) -> str:
    """Send a sentence to Gemini API and get a properly punctuated response."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = (
        "You are an AI that trasnalates Nepali sentences into English sentences. "
        "Your task is to take a Nepali sentence and return "
        "a properly translated english sentence. "
        "Ensure the sentence seems natural and is suitable for text-to-speech (TTS) processing. "
        "Return ONLY the corrected sentence without any explanations.\n\n"
        f"Input: {sentence}\n"
        "Output:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def main(input_sentence: str) -> str:
    API_KEY = os.getenv("GEMINI_API_KEY")
    setup_gemini_api(API_KEY)

    output_sentence = add_punctuation(input_sentence)
    return output_sentence


# print(main("नमस्ते तपाईंलाई कस्तो छ?"))  # "Hello, how are you?"
