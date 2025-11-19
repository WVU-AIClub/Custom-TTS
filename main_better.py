import os, shutil
import librosa, soundfile as sf     # !pip install librosa
from pydub import AudioSegment      # !pip install pydub
import taglib                       # !pip install pytaglib
import whisper                      # !pip install openai-whisper

SAMPLE_RATE_HZ = 22050

def check_voice_folder(voice_folder: str) -> None:
    """ Checks if the required 'voice_folder' and its 'raws' subfolder
    If they do, it ensures the creation of 'splits', 'wavs', and 'output'
    for the voice processing workflow
    Args:
        voice_folder: The path to the main voice project folder.

    Raises: 
        SystemExit: If the main folder or the required 'raws' subfolder
        does not exist, instruct the user on how to proceed.
    """
    raws_path = voice_folder + "/raws"

    # Checks if the main voice folder AND data in there
    files = [f for f in os.listdir(raws_path) if os.path.isfile(os.path.join(raws_path, f))]
    if len(files) == 0:
        quit(f"Please enter data into the '{raws_path}'.")

    # Since 'raws' exists, ensure the other required subfolder are there
    required_subfolder = ["/splits", "/wavs", "/outputs"]
    for folder_name in required_subfolder:
        folder_path = voice_folder + folder_name
        os.makedirs(folder_path, exist_ok=True)

    print("All folders checked")
    

def split_long_audio(input_folder: str, output_folder: str, chunk_length_ms: int=30000) -> None:
    """ Checks audio files in a folder and splits any over the specified duration
    into smaller chunks.
    
    Args:
        input_folder (str): The Path to the folder containing the audio files.
        output_folder (str): The Path where the split audio chunks will be saved.
        chunk_length_ms (int): The duration of each chunk in milliseconds (default is 30 secs)
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for file in files:
        input_file_path = os.path.join(input_folder, file)
        try:
            audio = AudioSegment.from_file(input_file_path)

            # Check if the audio file is longer than the chunk length
            if len(audio) > chunk_length_ms:
                print(f"Splitting long audio file: {file}")
                file_name_without_ext = os.path.splitext(file)[0]
                chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
                for i, chunk in enumerate(chunks):
                    chunk_name = f"{file_name_without_ext}_part{i+1}.wav"
                    chunk_path = os.path.join(output_folder, chunk_name)
                    chunk.export(chunk_path, format="wav")
                print(f"Split {file} into {len(chunks)} chunks.")
            
            else:
                # If the file is not long, just copy it to the output folder
                print(f"Copying short audio file: {file}")
                shutil.copy(input_file_path, output_folder)

        except Exception as e:
            print(f"Error processing {file}: {e}")
    

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
    model = whisper.load_model("base")
    
    wav_files = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
    wav_files = sorted(wav_files, key=lambda x: int(os.path.splitext(x)[0]))

    transcript_dict = {}
    with open(transcript_file, 'w') as transcript:
        for wav_file in wav_files:
            print(f"Transcribing {wav_file}")
            result = model.transcribe(f"{folder_path}/{wav_file}")
            transcribed_text = result['text'].strip()

            # Check if this transcript already exists
            if transcribed_text in transcript_dict:
                print(f"Duplicate transcript found:")
                print(f"Current file: {wav_file}")
                print(f"Duplicate of: {transcript_dict[transcribed_text]}")
                print(f"Transcript: {transcribed_text}")
                print("---")
            else:
                # Add the transcript to the dictionary
                transcript_dict[transcribed_text] = wav_file

            # Write the result to the transcript file without space after '|' TODO: Needs to make new line each time7
            transcript.write(f"{wav_file}|{transcribed_text}")


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
            y, sr = librosa.load(filepath, sr=SAMPLE_RATE_HZ)
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
    voice_folder = 'Peter'

    raw_folder_path = voice_folder + '/raws'
    split_folder_path = voice_folder + '/splits'
    wav_folder_path = voice_folder + '/wavs'
    metadata_path = voice_folder + '/metadata.csv'
    output_path = voice_folder + '/outputs'

    # Step -1
    check_voice_folder(voice_folder)

    # Step 0
    split_long_audio(raw_folder_path, split_folder_path)

    # Step 1
    convert_audio(split_folder_path, wav_folder_path)

    # Step 2
    transcribe(wav_folder_path, metadata_path)

    # Step 3
    process_audio_files(wav_folder_path, output_path)