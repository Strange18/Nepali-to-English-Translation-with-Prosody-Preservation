import torch
import os
import logging
import gc
from f5_tts.api import F5TTS

def text_to_audio_synthesis(ref_text, gen_text, ref_audio, output_path='output.wav'):
    """
    Synthesize speech using F5-TTS with a reference audio and text
    
    Args:
        ref_text (str): The text corresponding to the reference audio
        gen_text (str): The new text to synthesize in the same voice
        ref_audio (str): Path to the reference audio file
        output_path (str): Path to save the synthesized audio
        
    Returns:
        str: Path to the generated audio file
    """
    # Set PyTorch memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,garbage_collection_threshold:0.8'
    
    # Validate inputs
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Path to the checkpoint
    checkpoint_path = "D:/Major_Project/Checkpoints/F5-TTS/model_15000.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Optimize GPU memory before starting
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print memory status
        free_memory, total_memory = torch.cuda.mem_get_info()
        print(f"GPU memory: {free_memory/1e9:.2f}GB free of {total_memory/1e9:.2f}GB total")
        
        # Set memory fraction to avoid using all GPU memory
        torch.cuda.set_per_process_memory_fraction(0.7)
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available. Using CPU.")
    
    try:
        # Process text in manageable chunks if it's long
        max_chars = 50  # Smaller chunks for GPU processing
        
        if len(gen_text) > max_chars:
            print(f"Text is too long ({len(gen_text)} chars). Processing in chunks...")
            return process_long_text(ref_text, gen_text, ref_audio, output_path, checkpoint_path, device, max_chars)
        
        # For shorter text, process normally
        print(f"Processing short text ({len(gen_text)} chars) on {device}")
        
        # Create model with GPU optimization
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):  # Use mixed precision on GPU
            tts_model = F5TTS(
                ckpt_file=checkpoint_path, 
                vocoder_name="vocos", 
                device=device
            )
            
            tts_model.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=gen_text,
                file_wave=output_path,
                remove_silence=True
            )
        
        # Clean up
        del tts_model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return output_path
        
    except torch.cuda.OutOfMemoryError:
        # If we run out of memory on GPU, fall back to CPU with CUDA totally disabled
        print("GPU out of memory. Falling back to CPU...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save the current CUDA visibility state
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        try:
            # Completely disable CUDA visibility
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Force CPU device
            device = "cpu"
            print("Switching to CPU for this synthesis")
            
            # Reinitialize PyTorch to recognize the environment change
            torch.utils.backcompat.broadcast_warning.enabled = True
            torch.utils.backcompat.keepdim_warning.enabled = True
            torch.backends.cudnn.enabled = False
            
            # Create CPU-only model
            tts_model = F5TTS(
                ckpt_file=checkpoint_path, 
                vocoder_name="vocos", 
                device=device
            )
            
            # Process on CPU
            tts_model.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=gen_text,
                file_wave=output_path,
                remove_silence=True
            )
            
            return output_path
        finally:
            # Restore original CUDA visibility
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
            
    except Exception as e:
        logging.error(f"Error in audio synthesis: {str(e)}")
        raise

def process_long_text(ref_text, gen_text, ref_audio, output_path, checkpoint_path, device, max_chars):
    """Process longer text by breaking it into chunks"""
    import tempfile
    from pydub import AudioSegment
    
    # Break text into sentences or chunks
    sentences = split_into_sentences(gen_text)
    temp_files = []
    
    # Initialize model only once
    print(f"Initializing TTS model on {device}")
    with torch.cuda.amp.autocast(enabled=(device=="cuda")):
        tts_model = F5TTS(
            ckpt_file=checkpoint_path, 
            vocoder_name="vocos", 
            device=device
        )
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Generate temp filename
            temp_file = os.path.join(temp_dir, f"chunk_{i}.wav")
            
            try:
                print(f"Processing chunk {i+1}/{len(sentences)}: '{sentence[:20]}...'")
                
                # Check GPU memory status before each chunk
                if device == "cuda":
                    free_memory, _ = torch.cuda.mem_get_info()
                    # If low memory, do garbage collection
                    if free_memory < 500_000_000:  # Less than 500MB free
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Process with mixed precision on GPU
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    tts_model.infer(
                        ref_file=ref_audio,
                        ref_text=ref_text,
                        gen_text=sentence,
                        file_wave=temp_file,
                        remove_silence=True
                    )
                temp_files.append(temp_file)
                print(f"✓ Chunk {i+1} completed")
            except torch.cuda.OutOfMemoryError:
                print(f"Out of GPU memory on chunk {i+1}. Falling back to CPU for this chunk.")
                # Switch to CPU for this chunk
                torch.cuda.empty_cache()
                gc.collect()
                
                # Process on CPU
                backup_device = "cpu"
                backup_model = F5TTS(
                    ckpt_file=checkpoint_path, 
                    vocoder_name="vocos", 
                    device=backup_device
                )
                
                backup_model.infer(
                    ref_file=ref_audio,
                    ref_text=ref_text,
                    gen_text=sentence,
                    file_wave=temp_file,
                    remove_silence=True
                )
                
                temp_files.append(temp_file)
                del backup_model
                gc.collect()
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                continue
                
        # Combine the audio files
        print(f"Combining {len(temp_files)} audio chunks")
        combine_audio_files(temp_files, output_path)
    
    # Cleanup
    del tts_model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return output_path

def split_into_sentences(text):
    """Split text into sentences for easier processing"""
    # Simple sentence splitting on period, question mark, and exclamation point
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If sentences are still too long, split them further
    max_len = 50  # Reduced for GPU-friendly chunks
    result = []
    for sentence in sentences:
        if len(sentence) <= max_len:
            result.append(sentence)
        else:
            # Split on commas or other natural breaks
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            if max(len(p) for p in parts) <= max_len:
                result.extend(parts)
            else:
                # Last resort: split by character count
                for i in range(0, len(sentence), max_len):
                    result.append(sentence[i:i+max_len])
    
    return result

def combine_audio_files(audio_files, output_file):
    """Combine multiple audio files into one"""
    from pydub import AudioSegment
    
    if not audio_files:
        raise ValueError("No audio files to combine")
    
    combined = AudioSegment.empty()
    for file in audio_files:
        try:
            audio = AudioSegment.from_file(file)
            combined += audio
        except Exception as e:
            print(f"Error combining audio file {file}: {str(e)}")
    
    combined.export(output_file, format="wav")