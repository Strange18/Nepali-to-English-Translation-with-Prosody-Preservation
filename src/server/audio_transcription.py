import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from pydub import AudioSegment
import librosa
import numpy as np
import scipy.io.wavfile as wavfile

def load_model():
    config_path = './audio_vocab.json'
    UNK_TOKEN = '__UNK__'
    PAD_TOKEN = '__PAD__'
    WORD_DELIMITER = '|'
    tokenizer = Wav2Vec2CTCTokenizer(config_path, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, word_delimiter_token=WORD_DELIMITER)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
                                                padding_value=0.0, do_normalize=True,
                                                return_attention_mask=True)

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    model = Wav2Vec2ForCTC.from_pretrained("Strange18/wav2vec2-nepali-asr")
    return model, processor
def preprocess_audio(input_path, target_sr=16000):
    output_path = input_path
    if not input_path.endswith(".wav"):
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        input_path = output_path
    
    audio, sr = librosa.load(input_path, sr=target_sr)
    
    noise_sample = audio[:int(0.5 * sr)]
    noise_mean = np.mean(noise_sample)
    audio_denoised = audio - noise_mean  
    
    max_amplitude = np.max(np.abs(audio_denoised))
    if max_amplitude > 0:
        audio_normalized = audio_denoised / max_amplitude
    else:
        audio_normalized = audio_denoised

    audio_trimmed, _ = librosa.effects.trim(audio_normalized, top_db=20)

    wavfile.write(output_path, target_sr, (audio_trimmed * 32767).astype(np.int16))
    return output_path


def transcribe_audio_file(file_path):
    output_file_path = preprocess_audio(file_path)
    waveform, sample_rate = torchaudio.load(output_file_path)
    target_sample_rate = 16000  
    model, processor = load_model()

    waveform = waveform.squeeze().numpy()  
    input_values = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt")

    input  = {key: value for key, value in input_values.items()}
    with torch.no_grad():
        logits = model(**input).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription






