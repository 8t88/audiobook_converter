


from IPython.display import Audio
import nltk
import numpy as np
import os


def get_model_weights():
    #if tts_model == "bark"
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from bark.api import semantic_to_waveform
    from bark.generation import (
        generate_text_semantic,
        preload_models,
    )

    preload_models()


def convert_text_to_audio(paragraphs_sentences, speaker="v2/en_speaker_6", gen_temp=0.6, gen_p=0.01):

    silence_sentence = np.zeros(int(0.1 * SAMPLE_RATE))  # .15 second of silence for sentence break
    silence_paragraph = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence for paragraph break
    #num_pieces_run = 0

    pieces = []
    for paragraph_sentences in paragraphs_sentences[0:4]:
        for sentence in paragraph_sentences:
    
    
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=speaker,
                temp=gen_temp,
                min_eos_p=0.9,  # this controls how likely the generation is to end
            )
    
            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER,) 
            pieces += [audio_array, silence_sentence.copy()]
        pieces += [silence_paragraph.copy()]
    result = np.concatenate(pieces)
    write_wav(save_location + "/test.wav", SAMPLE_RATE, save_location)
    
