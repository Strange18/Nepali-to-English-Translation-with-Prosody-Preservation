import torch
import torchaudio
import sounddevice as sd
import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from safetensors import safe_open
import os
import numpy as np

model_name_or_path = '/mnt/45b9faff-45f3-43f2-903f-9b92a9a6338c/Major_Project/Checkpoints/50-epoch-kaggle-3-Aug/my_dataset/wav2vec2-nepali-asr/checkpoint-15750/'  
config_path = '/mnt/45b9faff-45f3-43f2-903f-9b92a9a6338c/Major_Project/Checkpoints/50-epoch-kaggle-3-Aug/my_dataset/vocab.json'
checkpoint = '/mnt/45b9faff-45f3-43f2-903f-9b92a9a6338c/Major_Project/Checkpoints/50-epoch-kaggle-3-Aug/my_dataset/wav2vec2-nepali-asr/checkpoint-15750/model.safetensors'

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

with safe_open(checkpoint, framework="pt") as f:
    model_state_dict = {key: torch.tensor(f.get_tensor(key)) for key in f.keys()}
model = Wav2Vec2ForCTC.from_pretrained("Strange18/wav2vec2-nepali-asr")
model.eval()

def transcribe_audio_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    target_sample_rate = 16000  

    if sample_rate != target_sample_rate:
        from torchaudio.transforms import Resample
        resample = Resample(sample_rate, target_sample_rate)
        resampled_waveform = resample(waveform)
    else:
        resampled_waveform = waveform
    waveform = resampled_waveform.squeeze().numpy()  
    input_values = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt")

    input  = {key: value for key, value in input_values.items()}
    with torch.no_grad():
        logits = model(**input).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    os.system('clear')
    print(f'Transcription: {transcription}')
    print('\n'*5)

audio_buffer = []

def audio_callback(indata, frames, time, status):
    global audio_buffer
    
    if status:
        print(status)
    
    audio_buffer.append(indata.copy())

def transcribe_real_time():
    global audio_buffer

    sample_rate = 16000
    block_size = 16000  # 1 second of audio at 16kHz
    record_time = 5  # seconds
    buffer_size = sample_rate * record_time

    input_device = sd.default.device[0]
    print(input_device)
    
    try:
        while True:
            audio_buffer.clear()  # Clear buffer before each recording
            stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=block_size, device=input_device)
            with stream:
                print('Listening...')
                sd.sleep(record_time * 1000)  # sleep for record_time seconds

            # Concatenate all recorded blocks
            audio_data = np.concatenate(audio_buffer, axis=0)
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            
            # If stereo, convert to mono
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.mean(dim=1)
            
            input_values = processor(audio_tensor, return_tensors="pt", sampling_rate=16000).input_values
            
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            
            print(f'Transcription: {transcription}')
            
            # Ask user if they want to continue or exit
            continue_transcription = input("Press Enter to continue or type 'exit' to stop: ")
            if continue_transcription.lower() == 'exit':
                break
    except KeyboardInterrupt:
        print("Transcription interrupted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio from a file or in real time.")
    parser.add_argument("--file", type=str, help="Path to the audio file to transcribe.")
    parser.add_argument("--realtime", action="store_true", help="Enable real-time transcription.")
    
    args = parser.parse_args()
    
    if args.file:
        transcribe_audio_file(args.file)
    elif args.realtime:
        transcribe_real_time()
    else:
        print("Please specify either --file or --realtime.")
