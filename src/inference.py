import os
import sys
import argparse
from scipy.io.wavfile import write as write_wav


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
    # parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    # parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    # parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
    #                     help='Sampling rate')

    return parser

def get_clean_text(book_filepath):
    file = open(book_filepath, "r")
    book_text = file.read().split('#license#')[0].strip()
    file.close()
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
    if model_name not in ('bark'):
        return "error: requested model not recognized"
        
    if(model_name == "bark"):
        from bark import SAMPLE_RATE, generate_audio, preload_models
        from bark.api import semantic_to_waveform
        from bark.generation import (
            generate_text_semantic,
            preload_models,
        )
        preload_models()


def convert_text_to_audio(paragraphs_sentences, result_file_name, speaker="v2/en_speaker_6", gen_temp=0.6, gen_p=0.01, ):

    #separate conditions for bark vs tts in the future? Right now just running bark
    silence_sentence = np.zeros(int(0.1 * SAMPLE_RATE))  # .15 second of silence for sentence break
    silence_paragraph = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence for paragraph break
    save_location = "./data/"
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
    
            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=speaker) 
            pieces += [audio_array, silence_sentence.copy()]
        pieces += [silence_paragraph.copy()]
    result = np.concatenate(pieces)
    write_wav(save_location + result_file_name, SAMPLE_RATE, save_location)
    

def main():

    parser = argparse.ArgumentParser(description='text to audio inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # print("input: " + args.input)
    # print("output: " + args.output)
    # print("model: " + args.model_name)
    load_and_setup_model(args.model_name)
    paragraph_sentences = create_booktext_iterable(args.input)
    convert_text_to_audio(paragraph_sentences, result_file_name=args.output)


if __name__ == '__main__':
    main()

