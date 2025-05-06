import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()


def setup_gemini_api(api_key: str):
    """Initialize the Gemini API client with the given API key."""
    genai.configure(api_key=api_key)


def translate(sentence: str) -> str:
    """Send a sentence to Gemini API and get a properly punctuated response."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "You are an AI that trasnalates Nepali sentences into English sentences. "
        "Your task is to take a Nepali sentence and return "
        "a properly translated english sentence without punctuations and make all words in lowercase. "
        "Ensure the sentence seems natural and is suitable for text-to-speech (TTS) processing. "
        "Return ONLY the corrected sentence without any explanations and punctuations and keep words in lowercase.\n\n"
        f"Input: {sentence}\n"
        "Output:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def add_punctuation(sentence: str) -> str:
    """Send a sentence to Gemini API and get a properly punctuated response."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "You are an AI that punctuates the English sentences. "
        "Your task is to take a normal English sentence and return "
        "a properly punctuated english sentence."
        "Ensure the sentence seems natural and is suitable for text-to-speech (TTS) processing. "
        "Return ONLY the punctuated sentence.\n\n"
        f"Input: {sentence}\n"
        "Output:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def translation(input_sentence: str) -> str:
    API_KEY = os.getenv("GEMINI_API_KEY")
    setup_gemini_api(API_KEY)

    output_sentence = translate(input_sentence)
    return output_sentence


def punctuate(input_sentence: str) -> str:
    API_KEY = os.getenv("GEMINI_API_KEY")
    setup_gemini_api(API_KEY)

    output_sentence = add_punctuation(input_sentence)
    return output_sentence


# print(main("नमस्ते तपाईंलाई कस्तो छ?"))  # "Hello, how are you?"
