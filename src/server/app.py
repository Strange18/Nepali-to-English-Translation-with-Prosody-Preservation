from audio_transcription import transcribe_audio_file
from text_translation import translate_nepali_to_english
from audio_synthesis import text_to_audio_synthesis

import requests
import streamlit as st
import tempfile
import jiwer
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from puncuate import translation, punctuate
from st_audiorec import st_audiorec

st.set_page_config(
    page_title="Home",
)
st.title("Nepali-to-English Translation with Prosody Preservation")


tab1, tab2 = st.tabs(["Upload Audio File", "Record Audio"])

# Audio processing container
audio_container = st.empty()
temp_file_path = None

# File Upload Tab
with tab1:
    uploaded_file = st.file_uploader(
        "Upload your Nepali audio recording", type=["wav", "mp3", "m4a"], key="uploader"
    )
    if uploaded_file is not None:
        audio_container.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

# Record Audio Tab
with tab2:
    # Using streamlit-webrtc for audio recording
    st.write("Click to start recording (max 30 seconds)")
    audio_recording = st_audiorec()
    if audio_recording is not None:
        audio_container.audio(audio_recording, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_recording)
            temp_file_path = temp_file.name

# Processing section (appears after either upload or recording)
if temp_file_path is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.header("User Input")
        actual_transcription = st.text_area(
            "Enter the actual Nepali transcription of the audio"
        )
        actual_translation = st.text_area("Enter the actual English translation")

    with col2:
        st.header("Generated Output")
        if st.button("Generate English Audio"):
            with st.spinner("Processing..."):
                try:
                    api_url = "https://8000-01jn43ebh87qgh1nckz11tq51j.cloudspaces.litng.ai/transcribe"  # Update with your actual API URL
                    generated_transcriptions = transcribe_audio_file(
                        api_url, temp_file_path
                    )
                    generated_transcription, transcription_time_taken = (
                        generated_transcriptions["transcription"],
                        generated_transcriptions["time_taken"],
                    )
                    st.write("Nepali Transcription of the Audio:")
                    st.write(generated_transcription)
                    st.write(f"The time taken is {transcription_time_taken}")

                    api_url = "https://8000-01jn43ebh87qgh1nckz11tq51j.cloudspaces.litng.ai/translate"
                    generated_translations = translate_nepali_to_english(
                        api_url, generated_transcription
                    )

                    generated_translation = generated_translations["transcriptions"]
                    translation_time_taken = generated_translations["time_taken"]

                    # generated_translation = translation(generated_transcription)
                    # st.write("Translated To English:")
                    st.write(generated_translation)
                    st.write(f"The time taken is {translation_time_taken}")

                    # generated_translation = punctuate(generated_translation)

                    # generated_translation = f".....{generated_translation}....."

                    # API parameters
                    api_url = "https://8000-01jn43ebh87qgh1nckz11tq51j.cloudspaces.litng.ai/tts"
                    output_audio_path = "output.wav"

                    # Call API for text-to-speech synthesis
                    audio_file_generated = text_to_audio_synthesis(
                        api_url,
                        generated_transcription,
                        generated_translation,
                        temp_file_path,
                        output_audio_path,
                    )

                    if audio_file_generated:
                        st.write("Generated English Speech:")
                        st.audio(audio_file_generated, format="audio/wav")

                    # # output_audio_path = text_to_audio_synthesis(generated_translation)
                    # output_audio_path = text_to_audio_synthesis(generated_transcription, generated_translation, temp_file_path)
                    # output_audio_path = "output.wav"
                    # st.write("Normal Generated Speech")
                    # st.audio(output_audio_path, format="audio/wav")

                    with st.expander("Show Evaluation Metrics"):
                        st.header("Metrics")

                        if actual_transcription:
                            wer = jiwer.wer(
                                actual_transcription, generated_transcription
                            )
                            cer = jiwer.cer(
                                actual_transcription, generated_transcription
                            )
                            st.write(f"Word Error Rate (WER): {wer:.2f}")
                            st.write(f"Character Error Rate (CER): {cer:.2f}")

                        if actual_translation:
                            bleu_score = sentence_bleu(
                                [actual_translation.split()],
                                generated_translation.split(),
                            )
                            st.write(f"BLEU Score: {bleu_score:.2f}")

                            sacrebleu_score = corpus_bleu(
                                [generated_translation], [[actual_translation]]
                            ).score
                            st.write(f"SacreBLEU Score: {sacrebleu_score:.2f}")

                            rouge = rouge_scorer.RougeScorer(
                                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                            )
                            rouge_scores = rouge.score(
                                actual_translation, generated_translation
                            )
                            st.write(
                                f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.2f}"
                            )
                            st.write(
                                f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.2f}"
                            )
                            st.write(
                                f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.2f}"
                            )

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
