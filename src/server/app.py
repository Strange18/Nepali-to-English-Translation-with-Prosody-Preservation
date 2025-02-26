from audio_transcription import transcribe_audio_file
from text_translation import translate_nepali_to_english
from audio_synthesis import text_to_audio_synthesis

import streamlit as st
import tempfile
import jiwer
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

st.set_page_config(
        page_title="Home",
)
st.title("Nepali-to-English Emotion Transfer in Audio")


uploaded_file = st.file_uploader("Upload your Nepali audio recording", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    col1, col2 = st.columns(2)

    with col1:
        st.header("User Input")
        actual_transcription = st.text_area("Enter the actual Nepali transcription of the audio")
        actual_translation = st.text_area("Enter the actual English translation")

    with col2:
        st.header("Generated Output")
        if st.button("Generate English Audio"):
            with st.spinner("Processing..."):
                try:
                    generated_transcription = transcribe_audio_file(temp_file_path)
                    st.write("Nepali Transcription of the Audio:")
                    st.write(generated_transcription)
                    
                    generated_translation = translate_nepali_to_english(generated_transcription)
                    st.write("Translated To English:")
                    st.write(generated_translation)
                    
                    # output_audio_path = text_to_audio_synthesis(generated_translation)
                    output_audio_path = text_to_audio_synthesis(generated_transcription, generated_translation, temp_file_path)
                    # output_audio_path = "output.wav"
                    st.write("Normal Generated Speech")
                    st.audio(output_audio_path, format="audio/wav")

                    with st.expander("Show Evaluation Metrics"):
                        st.header("Metrics")
                        
                        if actual_transcription:
                            wer = jiwer.wer(actual_transcription, generated_transcription)
                            cer = jiwer.cer(actual_transcription, generated_transcription)
                            st.write(f"Word Error Rate (WER): {wer:.2f}")
                            st.write(f"Character Error Rate (CER): {cer:.2f}")

                        if actual_translation:
                            bleu_score = sentence_bleu(
                                [actual_translation.split()], generated_translation.split()
                            )
                            st.write(f"BLEU Score: {bleu_score:.2f}")

                            sacrebleu_score = corpus_bleu([generated_translation], [[actual_translation]]).score
                            st.write(f"SacreBLEU Score: {sacrebleu_score:.2f}")

                            rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                            rouge_scores = rouge.score(actual_translation, generated_translation)
                            st.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.2f}")
                            st.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.2f}")
                            st.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.2f}")
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
