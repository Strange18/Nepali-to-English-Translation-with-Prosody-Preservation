from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_model(model_name="facebook/mbart-large-50-many-to-many-mmt"):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer.src_lang = "nep_NEP"  
    return tokenizer, model
       

def convert_discrete_to_continuous(discrete_sequence):
    continuous_sentence = "".join(discrete_sequence.split(" "))
    return continuous_sentence

def translate_nepali_to_english(continuous_sentence):
    target_lang = "en_XX"  
    tokenizer, model = load_model()
    inputs = tokenizer(continuous_sentence, return_tensors="pt")
    
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    
    english_translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return english_translation

