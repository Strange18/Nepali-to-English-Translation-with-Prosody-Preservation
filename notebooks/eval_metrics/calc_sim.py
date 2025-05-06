import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
import numpy as np
from scipy.spatial.distance import cosine

def extract_speaker_embedding(audio_path, model, device="cpu"):
    """
    Extract speaker embedding from an audio file using a pre-trained speaker recognition model.
    
    Args:
        audio_path (str): Path to the audio file (.wav format recommended)
        model (SpeakerRecognition): Pre-trained speaker recognition model
        device (str): Device to run the model on ("cpu" or "cuda")
    
    Returns:
        np.ndarray: Speaker embedding vector
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure the audio is mono and resampled to 16kHz (required by most speaker models)
    if waveform.shape[0] > 1:  # Convert stereo to mono if needed
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
    
    # Extract embedding using the pre-trained model
    embedding = model.encode_batch(waveform.to(device))
    return embedding.squeeze().cpu().detach().numpy()

def calculate_speaker_similarity(audio1_path, audio2_path, device="cpu"):
    """
    Calculate speaker similarity (cosine similarity) between two audio files.
    
    Args:
        audio1_path (str): Path to the first audio file
        audio2_path (str): Path to the second audio file
        device (str): Device to run the model on ("cpu" or "cuda")
    
    Returns:
        float: Cosine similarity score (between -1 and 1, where 1 indicates identical speakers)
    """
    # Load pre-trained speaker recognition model (e.g., ECAPA-TDNN)
    model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    
    # Extract embeddings for both audio files
    embedding1 = extract_speaker_embedding(audio1_path, model, device)
    embedding2 = extract_speaker_embedding(audio2_path, model, device)
    
    # Ensure embeddings have the same length
    min_length = min(len(embedding1), len(embedding2))
    embedding1 = embedding1[:min_length]
    embedding2 = embedding2[:min_length]
    
    # Calculate cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# Example usage
if __name__ == "__main__":
    # Specify paths to two audio files (e.g., .wav files of different speakers)
    audio1_path = "ref1.wav"
    audio2_path = "gen1.wav"
    
    # Set device (use "cuda" if GPU is available, otherwise "cpu")
    device = "cpu"  # Change to "cuda" if you have a GPU and torch.cuda.is_available()
    
    # Calculate speaker similarity
    similarity_score = calculate_speaker_similarity(audio1_path, audio2_path, device)
    print(f"Speaker Similarity Score (Cosine Similarity): {similarity_score:.4f}")
    
    # Interpretation: 
    # - A score close to 1 indicates the speakers are very similar.
    # - A score close to 0 or negative indicates the speakers are dissimilar.
