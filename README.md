## Audiobook Generator
Experimenting with TTS models to create audiobooks.
This code allows you to input a book as a text file and creates a wav file of a human voice reading the book, using one of the open-source tts models.  
Models currently available:  
 - bark  
 - XTTS
    
## Installation
clone the repo and install the requirements

    ```git pull https://github.com/8t88/audiobook_converter.git```  
    ```cd audiobook_converter```  
    ```pip install -e .```  

## Usage
navigate to the src folder and run the inference.py file, with the parameters of:
 - input (required): the .txt file to be converted
 - output (required): the name of the .wav file which the audiobook will be saved


example:  
    ```python inference.py -i /path/to/file.txt -o wavresult.wav``` 