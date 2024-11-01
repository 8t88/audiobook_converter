## Audiobook Generator
Experimenting with TTS models to create audiobooks.
This code allows you to input a book as a text file and creates a wav file of a human voice reading the, using one of the open-source tts models.
Models currently available:
    - bark
    - XTTS
    
## Installation
clone the repo and install the requirements.txt file

## Usage
navigate to the src folder and run the inference.py file, with the paramters of
 - input (required): the .txt file to be converted
 - output (required): the name of the .wav file the audiobook will be saved as


    example
    ```python inference.py -i /path/to/file.txt -o wavresult.wav```