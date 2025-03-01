# import torch
# import torchaudio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
# from pydub import AudioSegment
# import librosa
# import numpy as np
# import scipy.io.wavfile as wavfile

# def load_model():
#     # config_path = './audio_vocab.json'
#     config_path = 'src/server/audio_vocab.json'
#     UNK_TOKEN = '__UNK__'
#     PAD_TOKEN = '__PAD__'
#     WORD_DELIMITER = '|'
#     tokenizer = Wav2Vec2CTCTokenizer(config_path, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, word_delimiter_token=WORD_DELIMITER)
#     feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
#                                                 padding_value=0.0, do_normalize=True,
#                                                 return_attention_mask=True)

#     processor = Wav2Vec2Processor(
#         feature_extractor=feature_extractor,
#         tokenizer=tokenizer
#     )

#     model = Wav2Vec2ForCTC.from_pretrained("Strange18/wav2vec2-nepali-asr")
#     return model, processor
# def preprocess_audio(input_path, target_sr=16000):
#     output_path = input_path
#     if not input_path.endswith(".wav"):
#         audio = AudioSegment.from_file(input_path)
#         audio.export(output_path, format="wav")
#         input_path = output_path
    
#     audio, sr = librosa.load(input_path, sr=target_sr)
    
#     noise_sample = audio[:int(0.5 * sr)]
#     noise_mean = np.mean(noise_sample)
#     audio_denoised = audio - noise_mean  
    
#     max_amplitude = np.max(np.abs(audio_denoised))
#     if max_amplitude > 0:
#         audio_normalized = audio_denoised / max_amplitude
#     else:
#         audio_normalized = audio_denoised

#     audio_trimmed, _ = librosa.effects.trim(audio_normalized, top_db=20)

#     wavfile.write(output_path, target_sr, (audio_trimmed * 32767).astype(np.int16))
#     return output_path


# def transcribe_audio_file(file_path):
#     output_file_path = preprocess_audio(file_path)
#     waveform, sample_rate = torchaudio.load(output_file_path)
#     target_sample_rate = 16000  
#     model, processor = load_model()

#     waveform = waveform.squeeze().numpy()  
#     input_values = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt")

#     input  = {key: value for key, value in input_values.items()}
#     with torch.no_grad():
#         logits = model(**input).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.decode(predicted_ids[0])
#     return transcription

################################################################################################### Used up one


# import torch
# import torchaudio
# import gc
# import os
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
# from pydub import AudioSegment
# import librosa
# import numpy as np
# import scipy.io.wavfile as wavfile

# # Set global memory optimization settings
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# def load_model():
#     # Clear CUDA cache before loading model
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # config_path = './audio_vocab.json'
#     config_path = 'src/server/audio_vocab.json'
#     UNK_TOKEN = '__UNK__'
#     PAD_TOKEN = '__PAD__'
#     WORD_DELIMITER = '|'
#     tokenizer = Wav2Vec2CTCTokenizer(config_path, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, word_delimiter_token=WORD_DELIMITER)
#     feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
#                                                 padding_value=0.0, do_normalize=True,
#                                                 return_attention_mask=True)

#     processor = Wav2Vec2Processor(
#         feature_extractor=feature_extractor,
#         tokenizer=tokenizer
#     )

#     # Check if CUDA is available and set appropriate device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Use half precision to reduce memory usage
#     if device == torch.device('cuda'):
#         torch.cuda.set_per_process_memory_fraction(0.6)  # Use only 60% of GPU memory
#         model = Wav2Vec2ForCTC.from_pretrained(
#             "Strange18/wav2vec2-nepali-asr",
#             torch_dtype=torch.float16  # Use half precision
#         )
#     else:
#         model = Wav2Vec2ForCTC.from_pretrained("Strange18/wav2vec2-nepali-asr")
    
#     model = model.to(device)
#     return model, processor, device

# def preprocess_audio(input_path, target_sr=16000):
#     output_path = input_path
#     if not input_path.endswith(".wav"):
#         audio = AudioSegment.from_file(input_path)
#         audio.export(output_path, format="wav")
#         input_path = output_path
    
#     audio, sr = librosa.load(input_path, sr=target_sr)
    
#     noise_sample = audio[:int(0.5 * sr)]
#     noise_mean = np.mean(noise_sample)
#     audio_denoised = audio - noise_mean  
    
#     max_amplitude = np.max(np.abs(audio_denoised))
#     if max_amplitude > 0:
#         audio_normalized = audio_denoised / max_amplitude
#     else:
#         audio_normalized = audio_denoised

#     audio_trimmed, _ = librosa.effects.trim(audio_normalized, top_db=20)

#     wavfile.write(output_path, target_sr, (audio_trimmed * 32767).astype(np.int16))
#     return output_path

# def process_audio_in_chunks(waveform, sample_rate, model, processor, device, chunk_size_seconds=10):
#     """Process longer audios in chunks to avoid memory issues"""
#     chunk_size = chunk_size_seconds * sample_rate
#     total_length = len(waveform)
#     transcription = ""
    
#     for i in range(0, total_length, chunk_size):
#         # Get chunk and ensure we don't exceed array bounds
#         end_idx = min(i + chunk_size, total_length)
#         chunk = waveform[i:end_idx]
        
#         # Process chunk
#         input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        
#         # Move to appropriate device
#         input_values = {k: v.to(device) for k, v in input_values.items()}
        
#         # Use mixed precision for inference
#         with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
#             with torch.no_grad():
#                 logits = model(**input_values).logits
        
#         # Get transcription
#         predicted_ids = torch.argmax(logits, dim=-1)
#         chunk_transcription = processor.decode(predicted_ids[0])
#         transcription += chunk_transcription + " "
        
#         # Clear memory after each chunk
#         del logits, predicted_ids, input_values
#         torch.cuda.empty_cache()
#         gc.collect()
    
#     return transcription.strip()

# def transcribe_audio_file(file_path):
#     try:
#         output_file_path = preprocess_audio(file_path)
#         waveform, sample_rate = torchaudio.load(output_file_path)
#         waveform = waveform.squeeze().numpy()
        
#         # Load model with memory optimizations
#         model, processor, device = load_model()
        
#         try:
#             # Try processing the entire audio first
#             return transcribe_full_audio(waveform, sample_rate, model, processor, device)
#         except RuntimeError as e:
#             # If we run out of memory, fall back to chunk processing
#             if "CUDA out of memory" in str(e):
#                 print("CUDA out of memory. Falling back to chunk processing...")
#                 torch.cuda.empty_cache()
#                 gc.collect()
                
#                 # If still on GPU, try again with chunks
#                 if device.type == 'cuda':
#                     return process_audio_in_chunks(waveform, sample_rate, model, processor, device)
#                 # If already on CPU, we have a different issue
#                 else:
#                     raise e
#             else:
#                 raise e
#         finally:
#             # Always clean up
#             del model
#             torch.cuda.empty_cache()
#             gc.collect()
            
#     except Exception as e:
#         print(f"Error in transcription: {str(e)}")
#         # Last resort: try on CPU
#         try:
#             # Force CPU processing
#             os.environ['CUDA_VISIBLE_DEVICES'] = ''
#             model, processor, device = load_model()
#             return process_audio_in_chunks(waveform, sample_rate, model, processor, device, chunk_size_seconds=5)
#         except Exception as cpu_e:
#             print(f"CPU fallback also failed: {str(cpu_e)}")
#             raise cpu_e

# def transcribe_full_audio(waveform, sample_rate, model, processor, device):
#     """Process the entire audio at once if memory allows"""
#     input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
#     input_values = {k: v.to(device) for k, v in input_values.items()}
    
#     with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
#         with torch.no_grad():
#             logits = model(**input_values).logits
    
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.decode(predicted_ids[0])
    
#     return transcription

#################################################################################################

import requests
import streamlit as st

def transcribe_audio_file(api_url, audio_file_path):
    """Calls the API to transcribe Nepali speech to text."""
    with open(audio_file_path, 'rb') as audio_file:
        files = {
            'audio_file': (audio_file_path, audio_file, 'audio/wav')
        }
        headers = {
            'accept': 'application/json'
        }

        response = requests.post(api_url, headers=headers, files=files)

    if response.status_code == 200:
        return response.json().get("transcription", "Transcription failed")
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None
