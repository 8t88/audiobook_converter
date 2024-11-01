import os
import sys
import argparse
from scipy.io.wavfile import write as write_wav
import nltk
import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.api import semantic_to_waveform
from bark.generation import (
    generate_text_semantic,
    preload_models,
)


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='name of the file to save the result as')
    parser.add_argument('-m', '--model_name',  type=str, default='bark',
                        help='which TTS model to use. Options include: bark') #to add: XTTS, deepgram
    parser.add_argument('-w', '--ckpt', type=str,
                        help='full path to the model checkpoint file')

    return parser

def get_clean_text(book_filepath):
    try:
        file = open(book_filepath, "r")
        book_text = file.read().split('#license#')[0].strip()
        file.close()
    except:
        print("could not read file")
        sys.exit(1)
    return book_text

def format_paragraphs(book_text):

    paragraphs_breaks = [item.strip() for item in book_text.split('\n\n')]

    paragraphs = []
    for paragraph in paragraphs_breaks:
        repl = [x.strip() for x in paragraph.split('\n')]
        paragraphs.append(" ".join(repl))

    paragraphs_sentences = []
    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)
        paragraphs_sentences.append(sentences)
        
    return paragraphs_sentences

def create_booktext_iterable(booktext_filepath):
    booktext = get_clean_text(booktext_filepath)
    segmented_booktext = format_paragraphs(booktext)
    return segmented_booktext

def load_and_setup_model(model_name):
    print("loading model: " + model_name)
    if model_name not in ('bark', 'xtts'):
        return "error: requested model not recognized"
        
    if(model_name == 'bark'):
        preload_models()
        print("model loaded")
    else:
        print("model not loaded")


def run_model_and_save_output(paragraphs_sentences, result_file_name, model_name):

    silence_sentence = np.zeros(int(0.1 * SAMPLE_RATE))  # .15 second of silence for sentence break
    silence_paragraph = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence for paragraph break
    save_location = os.path.dirname(os.getcwd()) + '/data/' #improve later
    #num_pieces_run = 0

    pieces = []
    for paragraph_sentences in paragraphs_sentences[0:4]:
        for sentence in paragraph_sentences:
            audio_array = convert_text_to_audio(sentence, model_name)
            #add sentence-pause between each sentence
            pieces += [audio_array, silence_sentence.copy()]
        #add paragraph-pause between each paragraph
        pieces += [silence_paragraph.copy()]
    result = np.concatenate(pieces)
    write_wav(save_location + result_file_name, SAMPLE_RATE, result)
    print("conversion complete, saved at " + save_location + result_file_name)
    

def convert_text_to_audio(sentence, model_name, speaker="v2/en_speaker_6", gen_temp=0.6, gen_p=0.01):

    if(model_name=='bark'):    
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=speaker,
            temp=gen_temp,
            min_eos_p=0.9,  # this controls how likely the generation is to end
        )
        wav = semantic_to_waveform(semantic_tokens, history_prompt=speaker) 

    elif(model_name=='tts'):
        config = XttsConfig()
        config.load_json("/path/to/xtts/config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
        model.cuda()
        wav = model.synthesize(sentence,
                config,
                speaker_wav="/data/TTS-public/_refclips/3.wav",
                gpt_cond_len=3,
                language="en")
        
    return wav
    
def main():

    parser = argparse.ArgumentParser(description='text to audio inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # print("input: " + args.input)
    # print("output: " + args.output)
    # print("model: " + args.model_name)
    load_and_setup_model(args.model_name)
    paragraph_sentences = create_booktext_iterable(args.input)
    run_model_and_save_output(paragraph_sentences, result_file_name=args.output, model_name=args.model_name)
    
    
if __name__ == '__main__':
    main()

