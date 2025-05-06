# import torch
# import os
# import logging
# import gc
# import subprocess
# import sys
# from f5_tts.api import F5TTS

# def text_to_audio_synthesis(ref_text, gen_text, ref_audio, output_path='output.wav'):
#     """
#     Synthesize speech using F5-TTS with a reference audio and text
    
#     Args:
#         ref_text (str): The text corresponding to the reference audio
#         gen_text (str): The new text to synthesize in the same voice
#         ref_audio (str): Path to the reference audio file
#         output_path (str): Path to save the synthesized audio
        
#     Returns:
#         str: Path to the generated audio file
#     """
#     # Create a completely separate process for CPU-only operation
#     # This ensures no GPU memory is used at all
#     script_path = os.path.join(os.path.dirname(__file__), "cpu_tts_helper.py")
    
#     # Create the helper script if it doesn't exist
#     if not os.path.exists(script_path):
#         with open(script_path, "w") as f:
#             f.write("""
# import os
# import sys
# import torch
# import gc
# from f5_tts.api import F5TTS

# def main():
#     # Ensure we're using CPU only
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
#     # Parse arguments
#     if len(sys.argv) != 6:
#         print("Usage: python cpu_tts_helper.py ref_audio ref_text gen_text output_path checkpoint_path")
#         return 1
        
#     ref_audio = sys.argv[1]
#     ref_text = sys.argv[2]
#     gen_text = sys.argv[3]
#     output_path = sys.argv[4]
#     checkpoint_path = sys.argv[5]
    
#     # Create directories if needed
#     os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
#     # Process in chunks if needed
#     if len(gen_text) > 100:
#         # Split text into manageable chunks
#         import re
#         sentences = split_text(gen_text)
#         process_chunks(ref_audio, ref_text, sentences, output_path, checkpoint_path)
#     else:
#         # Process directly
#         model = F5TTS(
#             ckpt_file=checkpoint_path,
#             vocoder_name="vocos",
#             device="cpu"
#         )
        
#         model.infer(
#             ref_file=ref_audio,
#             ref_text=ref_text,
#             gen_text=gen_text,
#             file_wave=output_path,
#             remove_silence=True
#         )
    
#     return 0

# def split_text(text):
#     # Simple sentence splitting
#     import re
#     sentences = re.split(r'(?<=[.!?])\\s+', text)
    
#     # Further split if needed
#     max_len = 80
#     result = []
#     for sentence in sentences:
#         if len(sentence) <= max_len:
#             result.append(sentence)
#         else:
#             # Split on natural breaks
#             parts = re.split(r'(?<=[,;:])\\s+', sentence)
#             if max(len(p) for p in parts) <= max_len:
#                 result.extend(parts)
#             else:
#                 # Last resort: split by character count
#                 for i in range(0, len(sentence), max_len):
#                     result.append(sentence[i:i+max_len])
    
#     return result

# def process_chunks(ref_audio, ref_text, sentences, output_path, checkpoint_path):
#     import tempfileo
#     from pydub import AudioSegment
    
#     temp_files = []
    
#     # Create model once
#     model = F5TTS(
#         ckpt_file=checkpoint_path,
#         vocoder_name="vocos",
#         device="cpu"
#     )
    
#     # Process each chunk
#     with tempfile.TemporaryDirectory() as temp_dir:
#         for i, sentence in enumerate(sentences):
#             if not sentence.strip():
#                 continue
                
#             temp_file = os.path.join(temp_dir, f"chunk_{i}.wav")
#             print(f"Processing chunk {i+1}/{len(sentences)}")
            
#             try:
#                 model.infer(
#                     ref_file=ref_audio,
#                     ref_text=ref_text,
#                     gen_text=sentence,
#                     file_wave=temp_file,
#                     remove_silence=True
#                 )
#                 temp_files.append(temp_file)
#             except Exception as e:
#                 print(f"Error on chunk {i+1}: {str(e)}")
#                 continue
        
#         # Combine audio files
#         combine_audio(temp_files, output_path)

# def combine_audio(audio_files, output_file):
#     from pydub import AudioSegment
    
#     if not audio_files:
#         print("No audio files to combine")
#         return
        
#     combined = AudioSegment.empty()
#     for file in audio_files:
#         try:
#             audio = AudioSegment.from_file(file)
#             combined += audio
#         except Exception as e:
#             print(f"Error combining audio: {str(e)}")
            
#     combined.export(output_file, format="wav")

# if __name__ == "__main__":
#     sys.exit(main())
# """)
    
#     # Validate inputs
#     if not os.path.exists(ref_audio):
#         raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
    
#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
#     # Path to the checkpoint
#     checkpoint_path = "D:/Major_Project/Checkpoints/F5-TTS/model_last_reduced.pt"
    
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
#     # Clear GPU memory before launching the subprocess
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         gc.collect()
    
#     try:
#         # Run the CPU-only helper script as a separate process
#         print("Starting speech synthesis in a separate CPU-only process...")
        
#         result = subprocess.run([
#             sys.executable,
#             script_path,
#             ref_audio,
#             ref_text,
#             gen_text,
#             output_path,
#             checkpoint_path
#         ], check=True)
        
#         if result.returncode == 0:
#             print(f"Successfully generated audio at {output_path}")
#             return output_path
#         else:
#             raise RuntimeError(f"Speech synthesis process failed with code {result.returncode}")
            
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Speech synthesis process error: {str(e)}")
#         raise RuntimeError(f"Speech synthesis failed: {str(e)}")
#     except Exception as e:
#         logging.error(f"Error in speech synthesis: {str(e)}")
#         raise

import requests
import streamlit as st

def text_to_audio_synthesis(api_url, ref_text, gen_text, audio_file_path, output_audio_path):
    """Calls the API to generate English audio and saves the output."""
    with open(audio_file_path, 'rb') as audio_file:
        files = {
            'audio_file': (audio_file_path, audio_file, 'audio/wav')
        }
        data = {
            'ref_text': ref_text,
            'gen_text': gen_text
        }
        headers = {
            'accept': 'application/json'
        }

        response = requests.post(api_url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        with open(output_audio_path, 'wb') as f:
            f.write(response.content)
        return output_audio_path
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None