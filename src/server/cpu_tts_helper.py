
import os
import sys
import torch
import gc
from f5_tts.api import F5TTS

def main():
    # Ensure we're using CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Parse arguments
    if len(sys.argv) != 6:
        print("Usage: python cpu_tts_helper.py ref_audio ref_text gen_text output_path checkpoint_path")
        return 1
        
    ref_audio = sys.argv[1]
    ref_text = sys.argv[2]
    gen_text = sys.argv[3]
    output_path = sys.argv[4]
    checkpoint_path = sys.argv[5]
    
    # Create directories if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Process in chunks if needed
    if len(gen_text) > 100:
        # Split text into manageable chunks
        import re
        sentences = split_text(gen_text)
        process_chunks(ref_audio, ref_text, sentences, output_path, checkpoint_path)
    else:
        # Process directly
        model = F5TTS(
            ckpt_file=checkpoint_path,
            vocoder_name="vocos",
            device="cpu"
        )
        
        model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=output_path,
            remove_silence=True
        )
    
    return 0

def split_text(text):
    # Simple sentence splitting
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Further split if needed
    max_len = 80
    result = []
    for sentence in sentences:
        if len(sentence) <= max_len:
            result.append(sentence)
        else:
            # Split on natural breaks
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            if max(len(p) for p in parts) <= max_len:
                result.extend(parts)
            else:
                # Last resort: split by character count
                for i in range(0, len(sentence), max_len):
                    result.append(sentence[i:i+max_len])
    
    return result

def process_chunks(ref_audio, ref_text, sentences, output_path, checkpoint_path):
    import tempfile
    from pydub import AudioSegment
    
    temp_files = []
    
    # Create model once
    model = F5TTS(
        ckpt_file=checkpoint_path,
        vocoder_name="vocos",
        device="cpu"
    )
    
    # Process each chunk
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            temp_file = os.path.join(temp_dir, f"chunk_{i}.wav")
            print(f"Processing chunk {i+1}/{len(sentences)}")
            
            try:
                model.infer(
                    ref_file=ref_audio,
                    ref_text=ref_text,
                    gen_text=sentence,
                    file_wave=temp_file,
                    remove_silence=True
                )
                temp_files.append(temp_file)
            except Exception as e:
                print(f"Error on chunk {i+1}: {str(e)}")
                continue
        
        # Combine audio files
        combine_audio(temp_files, output_path)

def combine_audio(audio_files, output_file):
    from pydub import AudioSegment
    
    if not audio_files:
        print("No audio files to combine")
        return
        
    combined = AudioSegment.empty()
    for file in audio_files:
        try:
            audio = AudioSegment.from_file(file)
            combined += audio
        except Exception as e:
            print(f"Error combining audio: {str(e)}")
            
    combined.export(output_file, format="wav")

if __name__ == "__main__":
    sys.exit(main())
