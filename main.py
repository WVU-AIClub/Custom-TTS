import os, shutil
# !pip install pydub
from pydub import AudioSegment
# Install ffmpeg. On Windows go to ffmpeg.ord and add to your system's path

import torch
# !pip install torch

import torchaudio
# !pip install torchaudio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import librosa, soundfile as sf
# !pip install librosa


def convert_to_wav(input_file, output_file):
    # Load the audio file
    print(input_file)

    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format='wav')

    print(f"Converted {input_file} to {output_file}")

def rename_files(folder_path):

    # Create the "numbered_files" directory, if it doesn't exist
    output_folder = "numbered_files"
    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(folder_path)

    print(f'Folder path: {folder_path}')
    print(f'Files found: {len(files)}')

    for index, file in enumerate(files, start=1):
        old_path = os.path.join(folder_path, file)

        if os.path.isfile(old_path):
            new_file_name = f'{index}.{file.split(".")[-1]}'
            new_path = os.path.join(output_folder, new_file_name)

        shutil.copy(old_path, new_path)
        print(f'Renamed: {file} -> {new_file_name}')


def transcribe(folder_path):
    output_file = os.path.join(folder_path, "lists.txt")

    
    wav_files_range = range(1,len(os.listdir(folder_path)) + 1)
    file_and_transcripts = []

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")


    for i in wav_files_range:
        wav_file = os.path.join(folder_path, f"{i}.wav")

        if os.path.exists(wav_file):
            # Recognize the speech in the .wav file
            try:
                waveform, sample_rate = torchaudio.load(wav_file)
                
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)

                waveform = waveform.squeeze(0)
                # Process the waveform
                input_values = processor(
                    waveform, return_tensors="pt", sampling_rate=16000).input_values
                print(f"Input values shape: {input_values.shape}")


                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcript = processor.decode(predicted_ids[0])

            except FileNotFoundError:
                print(f"File not found: {wav_file}")
                continue

            file_and_transcripts.append(f"/contents/TTS-TT2/wavs/{i}.wav|{transcript}")
        else:
            print(f"File not found: {wav_file}")
            
    with open(output_file, "w") as f:
        for line in file_and_transcripts:
            f.write(f"{line}\n")

    print(f"File '{output_file}' created successfully.")

def preprocess():
    input_path = 'numbered_files'
    output_path = 'output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        filepath = os.path.join(input_path, filename) # should be file_path
        y, sr = librosa.load(filepath, sr=22050)

        #trim silence
        trimmed_audio, _ = librosa.effects.trim(y, top_db=20)
        normalized_audio = librosa.util.normalize(trimmed_audio)

        output_filepath = os.path.join(output_path, filename)
        sf.write(output_filepath, normalized_audio, sr, subtype='PCM_16')



if __name__ == '__main__':
    raw_folder_path = 'raw_recordings'
    wav_folder_path = 'wav_recordings'
    num_folder_path = 'numbered_files'

    # # Not always needed
    # for file in os.listdir(raw_folder_path):
    #     convert_to_wav(raw_folder_path+"/"+file, wav_folder_path+"/"+f'{file.split(".")[0]}.wav')

    # # Step 1
    # rename_files(wav_folder_path)

    # Step 2 ## BROKEN HEELLPPP!!!!
    transcribe(num_folder_path)

    # Step 3
    preprocess()