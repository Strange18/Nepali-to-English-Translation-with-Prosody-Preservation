import librosa
import numpy as np

def extract_mfcc(audio_path, sample_rate=16000, n_mfcc=13):
    """
    Extract MFCCs (Mel-Frequency Cepstral Coefficients) from an audio file.
    MFCCs are used as a proxy for mel-cepstral coefficients for MCD calculation.
    
    Args:
        audio_path (str): Path to the audio file (.wav format recommended)
        sample_rate (int): Sample rate of the audio (default: 16kHz)
        n_mfcc (int): Number of MFCC coefficients to extract (default: 13)
    
    Returns:
        np.ndarray: MFCC matrix (frames × n_mfcc)
    """
    # Load the audio file
    waveform, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    
    return mfcc.T  # Transpose to get frames × n_mfcc

def calculate_mcd(reference_mfcc, estimate_mfcc):
    """
    Calculate Mel-Cepstral Distortion (MCD) between two sets of MFCCs.
    
    Args:
        reference_mfcc (np.ndarray): MFCCs of the reference audio (frames × n_mfcc)
        estimate_mfcc (np.ndarray): MFCCs of the estimated/processed audio (frames × n_mfcc)
    
    Returns:
        float: MCD value in decibels (dB)
    """
    # Ensure both MFCC matrices have the same number of coefficients
    n_mfcc = min(reference_mfcc.shape[1], estimate_mfcc.shape[1])
    reference_mfcc = reference_mfcc[:, :n_mfcc]
    estimate_mfcc = estimate_mfcc[:, :n_mfcc]
    
    # Ensure both MFCC matrices have the same number of frames
    min_frames = min(reference_mfcc.shape[0], estimate_mfcc.shape[0])
    reference_mfcc = reference_mfcc[:min_frames, :]
    estimate_mfcc = estimate_mfcc[:min_frames, :]
    
    # Calculate the difference between MFCCs for each frame
    diff = reference_mfcc - estimate_mfcc
    
    # Calculate MCD using the formula: MCD = 10 / ln(10) * sqrt(2 * sum((c1 - c2)^2))
    # where c1 and c2 are the cepstral coefficients
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=1)))
    mcd = 10.0 / np.log(10) * mcd  # Convert to dB
    
    return mcd

# Example usage
if __name__ == "__main__":
    # Specify paths to two audio files (e.g., .wav files of reference and processed speech)
    reference_audio_path = "ref1.wav"
    estimate_audio_path = "gen1.wav"
    
    # Extract MFCCs for both audio files
    reference_mfcc = extract_mfcc(reference_audio_path)
    estimate_mfcc = extract_mfcc(estimate_audio_path)
    
    # Calculate MCD
    mcd_value = calculate_mcd(reference_mfcc, estimate_mfcc)
    print(f"Mel-Cepstral Distortion (MCD): {mcd_value:.2f} dB")
    
    # Interpretation:
    # - Lower MCD values indicate greater similarity between the audio signals.
    # - Typical MCD values for high-quality speech synthesis are below 5-6 dB, but this depends on the application.
