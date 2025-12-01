#Tennyson and Matt's Code Combined - Working 

#pre-requisites 
#pip install transformers
#pip install torch
#pip install hf_xet
#python 3.13 

import torch
import transformers
import os

print(f' GPU?: {torch.cuda.is_available()}') 
#torch only supports nvidia gpus #:madface: and the AMD methods aren't as well supported
#gpu would help with processing times (if available on your pc)

from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)

#file_path = r"C:\Users\Tennyson\Downloads\TestingFolder\alexandria_ted.mp3"


#actual transcription
print("BEGINNING TRANSCRIPTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
transcription = pipe(r"C:\Users\matts\Downloads\greece.mp3", return_timestamps=True, language='en')

#processing for the 13 minute video took 4:06 on my laptop 
print("FINISHED TRANSCRIPTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# print(transcription) this output has timestamps which we do not want but is required for long video

cleanOutput = [] 
for entry in transcription: #this code removes timestamps and makes it only a textual output 
    if 'text' in transcription:
        cleanOutput.append(transcription['text'])

print(cleanOutput)

#MATT CODE FINAL

print("BEGINNING SUMMARIZATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

from transformers import pipeline, logging

logging.set_verbosity_error() #hides info and warnings - leading to a cleaner output


summarizer = pipeline(
    "summarization",
    model="t5-base",
    tokenizer="t5-base"
)

input_txt = cleanOutput[0]

summary = summarizer(
    "summarize: " + input_txt,
    max_length=100,
    min_length=30,
    num_beams = 5
)[0]["summary_text"]

#fix the output formatting 

#summary.strip
#tokens = summary.split(".")

# summaryClean = []
# for n in tokens: 
#     n = n.strip
#     n = n.capitalize()
#     n = n + "."
#     summaryClean.append(n)


print(summary)