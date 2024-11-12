from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import scipy.io.wavfile as wavfile
import torch

def text_to_audio_synthesis(text, output_path='output.wav'):
    
    device = torch.device('cuda')
        
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
    model = models
    model[0] = model[0].to(device)
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    
    sample = TTSHubInterface.get_model_input(task, text)
    sample['net_input']['src_tokens']  = sample['net_input']['src_tokens'].to(device)

    wav, rate = TTSHubInterface.get_prediction(task, model[0], generator, sample)
    wavfile.write(output_path, rate, wav.cpu().numpy())
    return output_path


