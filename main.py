import os, glob, shutil
import torch                        # !pip install torch
import torchaudio                   # !pip install torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor # !pip install transformers
import librosa, soundfile as sf     # !pip install librosa
from pydub import AudioSegment      # !pip install pydub
import taglib                       # !pip install pytaglib

# Install ffmpeg. On Windows go to ffmpeg.ord and add to your system's path
# This is the only way to get pydub to work

def convert_audio(input_folder: str, output_folder: str) -> None:
    """ Converts audio files from a given input folder to WAV format,
    renames them sequentially, and saves them to an output folder.
    Args:
        input_folder (str): The path to the folder contianing the raw audio files
        output_folder (str):    The path wherffe the converted and renamed WAV files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f'Input folder: {input_folder}')
    print(f'Output folder: {output_folder}')

    # Get all files from the input folder
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    print(f'Files found: {len(files)}')

    for index, file in enumerate(files, start=1):
        input_file_path = os.path.join(input_folder, file)

        # Determine the new file name and path
        output_file_name = f'{index}.wav'
        output_file_path = os.path.join(output_folder, output_file_name)

        try:
            # Convert the file to WAV format
            audio = AudioSegment.from_file(input_file_path)
            audio.export(output_file_path, format='wav')
            print(f"Converted and renamed: {file} -> {output_file_name}")

        except Exception as e:
            print(f"Error processing {file}: {e}")


def transcribe(folder_path: str, transcript_file: str="metadata.csv") -> None:
    ''' Transcribes the data and then saves a lists of the transcript'''
    output_file = os.path.join(folder_path, transcript_file)
    file_and_transcripts = []

    try:
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Failed to load model or processor: {e}")
        return

    wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")), key=lambda x: \
                       int(os.path.basename(x).split('.')[0]))
    
    if not wav_files:
        print(f"No .wav files found in {folder_path}. Exiting...")
        return
    
    for wav_file in wav_files:
        try:
            # Load and preprocess the audio
            waveform, sample_rate = torchaudio.load(wav_file)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            waveform = waveform.squeeze(0)

            # Process with the model
            input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = processor.decode(predicted_ids[0])

            # Watch this LINE!!
            temp = os.path.basename(wav_file)
            file_and_transcripts.append(f"/contents/TTS-TT2/wavs/{temp}|{transcript}")
            print(f"Transcribed {temp}.")

        except Exception as e:
            print(f"Skipping {wav_file} due to an error: {e}")
            continue
            
    with open(output_file, "w") as f:
        for line in file_and_transcripts:
            f.write(f"{line}\n")

    print(f"File '{output_file}' created successfully.")


def process_audio_files(input_folder:str, output_folder:str='processed_files') -> None:
    ''' Preprocesses .wav files from the input folder, updates their metadata, and saves
    them to a single output folder.
    
    Args:
        input_folder (str): The folder containing the raw .wav files.
        output_folder (str): The folder to save the final processed files.
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)

            # Preprocessing
            y, sr = librosa.load(filepath, sr=22050)
            trimmed_audio, _ = librosa.effects.trim(y, top_db=20)
            normalized_audio = librosa.util.normalize(trimmed_audio)

            output_filepath = os.path.join(output_folder, filename)
            sf.write(output_filepath, normalized_audio, sr, subtype='PCM_16')
            
            # Metadata update on the newly saved file
            try:
                file_number = int(os.path.splitext(filename)[0])
            except ValueError:
                print(f"Skipping metadata update for '{filename}': filename is not a number.")
                continue

            with taglib.File(output_filepath, save_on_exit=True) as audio:
                audio.tags['TITLE'] = [str(file_number)]
                audio.tags['TRACKNUMBER'] = [str(file_number)]

            print(f"Processed '{filename}'. Updated title and track number to '{file_number}'.")

    print("\nAll .wav files have been processed and saved to the output folder.")


if __name__ == '__main__':
    raw_folder_path = 'raws'
    wav_folder_path = 'wavs'

    # Step 0
    # Check if audio file is too large and needs to be converted into smaller chunks

    # Step 1
    convert_audio(raw_folder_path, wav_folder_path)

    # Step 2
    transcribe(wav_folder_path)

    # Step 3
    process_audio_files(wav_folder_path)