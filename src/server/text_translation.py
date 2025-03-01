# import torch
# import os
# import gc
# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# # Set global memory optimization settings
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# _model = None
# _tokenizer = None

# def load_model(model_name="facebook/mbart-large-50-many-to-many-mmt", force_cpu=False):
#     """Load model with memory optimizations and caching"""
#     global _model, _tokenizer
    
#     if _model is not None and _tokenizer is not None:
#         return _tokenizer, _model
    
#     # Clear memory before loading
#     gc.collect()
#     torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
#     # Determine device
#     device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(f"Loading translation model on {device}...")
    
#     # Load tokenizer
#     tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
#     tokenizer.src_lang = "nep_NEP"
    
#     # Load model with memory optimizations
#     if device == "cuda":
#         # Half precision for GPU
#         model = MBartForConditionalGeneration.from_pretrained(
#             model_name, 
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True
#         ).to(device)
#     else:
#         # Regular precision for CPU
#         model = MBartForConditionalGeneration.from_pretrained(
#             model_name,
#             low_cpu_mem_usage=True
#         ).to(device)
    
#     # Cache model and tokenizer
#     _tokenizer = tokenizer
#     _model = model
    
#     return tokenizer, model

# def convert_discrete_to_continuous(discrete_sequence):
#     continuous_sentence = "".join(discrete_sequence.split(" "))
#     return continuous_sentence

# def translate_nepali_to_english(continuous_sentence):
#     """Translate with memory optimization and fallbacks"""
#     target_lang = "en_XX"
    
#     # Chunk text if it's too long
#     max_length = 512  # Typical maximum for mBART
    
#     if len(continuous_sentence) > max_length:
#         return translate_long_text(continuous_sentence, target_lang)
    
#     try:
#         # Try with GPU first
#         tokenizer, model = load_model(force_cpu=False)
#         device = next(model.parameters()).device
        
#         inputs = tokenizer(continuous_sentence, return_tensors="pt").to(device)
        
#         # Use memory-efficient generation settings
#         with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
#             with torch.no_grad():
#                 generated_tokens = model.generate(
#                     **inputs,
#                     forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
#                     max_length=512,
#                     num_beams=2,  # Reduce beam size to save memory
#                     length_penalty=1.0
#                 )
        
#         english_translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
#         # Clear memory
#         del inputs, generated_tokens
#         torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
#         return english_translation
        
#     except RuntimeError as e:
#         # If we run out of memory, fall back to CPU
#         if "CUDA out of memory" in str(e):
#             print("CUDA out of memory. Falling back to CPU...")
#             torch.cuda.empty_cache()
#             gc.collect()
            
#             # Force CPU processing
#             tokenizer, model = load_model(force_cpu=True)
#             inputs = tokenizer(continuous_sentence, return_tensors="pt")
            
#             with torch.no_grad():
#                 generated_tokens = model.generate(
#                     **inputs,
#                     forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
#                     max_length=512,
#                     num_beams=1  # Greedy search to save memory
#                 )
            
#             english_translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
#             return english_translation
#         else:
#             raise e

# def translate_long_text(text, target_lang):
#     """Translate long text in chunks"""
#     # Split text into chunks
#     chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    
#     # Translate each chunk
#     translations = []
#     for chunk in chunks:
#         # Force CPU to avoid memory issues with long texts
#         tokenizer, model = load_model(force_cpu=True)
#         inputs = tokenizer(chunk, return_tensors="pt")
        
#         with torch.no_grad():
#             generated_tokens = model.generate(
#                 **inputs,
#                 forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
#                 max_length=512,
#                 num_beams=1
#             )
        
#         translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
#         translations.append(translation)
    
#     # Combine translations
#     full_translation = ' '.join(translations)
#     return full_translation

import requests
import streamlit as st

def translate_nepali_to_english(api_url, ref_text):
    """Calls the API to translate Nepali text to English."""
    data = ref_text
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get('transcription')
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None
