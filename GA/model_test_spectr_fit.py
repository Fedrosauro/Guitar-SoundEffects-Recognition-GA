import warnings
import logging
import time

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress logs below ERROR level
logging.getLogger().setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import random
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH, inference
from pedalboard import Pedalboard, Reverb, Chorus, Distortion, Delay, Phaser, Compressor, Gain, Clipping
from pedalboard.io import AudioFile
from midi2audio import FluidSynth 
from pydub import AudioSegment
import tempfile
import concurrent.futures
import os
from pathlib import Path
from utils import *
from copy import deepcopy
from scipy.optimize import curve_fit
import itertools
from skimage.metrics import structural_similarity as ssim

def generate_random_instrument_midi(midi_file, program, min_duration, max_duration, min_note, max_note):
    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI()
    
    # Create an Instrument instance for a distortion instrument
    instrument = pretty_midi.Instrument(program=program)
    
    # Determine the length of the MIDI in seconds
    duration = random.randint(min_duration, max_duration)
    min_note_number = pretty_midi.note_name_to_number(min_note)
    max_note_number = pretty_midi.note_name_to_number(max_note)
    
    # Generate random notes
    current_time = 0.0
    while current_time < duration:
        note_number = random.randint(min_note_number, max_note_number)
        
        note_duration = random.uniform(0.1, 1.0)
        
        # Ensure that the note ends before the total duration
        note_end_time = min(current_time + note_duration, duration)
        
        # Create a Note instance and add it to the instrument instrument
        note = pretty_midi.Note(
            velocity=random.randint(60, 127),  # Random velocity
            pitch=note_number,
            start=current_time,
            end=note_end_time
        )
        instrument.notes.append(note)
        
        # Update the current time
        current_time = note_end_time
    
    # Add the instrument instrument to the PrettyMIDI object
    midi_data.instruments.append(instrument)
    
    # Write out the MIDI data to a file
    midi_data.write(midi_file)
    
    return midi_data
    
    
def mp3_to_midi_w_return(audio_path, midi_path, onset_threshold, frame_threshold):
    _, midi_data, __ = inference.predict(
        audio_path,    
        model_or_model_path = ICASSP_2022_MODEL_PATH, 
        onset_threshold = onset_threshold, # 0.6 note segmentation 1) how easily a note should be split into two. (split notes <- ..0.5.. -> merge notes)
        frame_threshold = frame_threshold, # 0.6 model confidence threshold 2) the model confidence required to create a note. (more notes <- ..0.3.. -> fewer notes)
    )

    for instrument in midi_data.instruments:
        instrument.program = 30 #distortion guitar program
                
    midi_data.write(midi_path)
    
    return midi_data

def midi_to_mp3_w_path_remov(midi_file, audio_path, fs):
    #convert MIDI to WAV using FluidSynth
    wav_file = midi_file.replace('.midi', '.wav').replace('.mid', '.wav')
    fs.midi_to_audio(midi_file, wav_file)

    #convert WAV to MP3 using pydub
    sound = AudioSegment.from_wav(wav_file)
    sound.export(audio_path, format="mp3")
    
    Path(wav_file).unlink(missing_ok=True)

    #print(f"Conversion complete: {audio_path}")
    
def spectrogram_ssim_fitness(audio_1, audio_2):
    y1, _ = librosa.load(audio_1, sr=None)
    y2, _ = librosa.load(audio_2, sr=None)

    # Compute spectrograms
    S1 = np.abs(librosa.stft(y1))
    S2 = np.abs(librosa.stft(y2))

    # Pad the smaller spectrogram to match the shape of the larger one
    if S1.shape != S2.shape:
        target_shape = (
            max(S1.shape[0], S2.shape[0]),
            max(S1.shape[1], S2.shape[1])
        )
        S1 = pad_to_shape(S1, target_shape)
        S2 = pad_to_shape(S2, target_shape)

    # Normalize the spectrograms
    S1_norm = (S1 - S1.min()) / (S1.max() - S1.min())
    S2_norm = (S2 - S2.min()) / (S2.max() - S2.min())

    # Compute SSIM
    ssim_score = ssim(S1_norm, S2_norm, data_range=1.0)
    return ssim_score

def process_data_generation_lines(fs, program, min_note, max_note, min_duration, max_duration, effects, effect_structure, effects_map, onset_threshold, frame_threshold):   
    # Create temporary MIDI and MP3 files
    fd_midi_init, temp_midi_init_name = tempfile.mkstemp(suffix='.mid', dir="../temp_files/")
    fd_midi_gen, temp_midi_gen_name = tempfile.mkstemp(suffix='.mid', dir="../temp_files/")
    fd_mp3_init, temp_mp3_init_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")
    fd_mp3_gen, temp_mp3_gen_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")
    fd_effected_audio_name, temp_effected_audio_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")  # Temporary file for effected audio

    # Close file descriptors as we only need the file paths
    os.close(fd_midi_init)
    os.close(fd_midi_gen)
    os.close(fd_mp3_init)
    os.close(fd_mp3_gen)
    os.close(fd_effected_audio_name)

    # Dictionary to hold the data for this individual run
    result_data = {
        "original_midi_data": [],
        "generated_midi_data": [],
        "individual": [], 
        "ssim_fitness": [],
        "onset_threshold": [],
        "frame_threshold": [],
    }
    
    # Generate the MIDI data and apply effects
    initial_midi_data = generate_random_instrument_midi(temp_midi_init_name, program, min_duration, max_duration, min_note, max_note)
    midi_to_mp3_w_path_remov(temp_midi_init_name, temp_mp3_init_name, fs)
        
    individual = create_individual(effects, effect_structure)
    board = Pedalboard([])
    for effect_key, params in individual.items():
        effect_class = globals()[effects_map[effect_key]]
        board.append(effect_class(**params))
            
    # Apply effects and create the effected audio

    create_effected_audio_for_parallelization(board, temp_mp3_init_name, temp_effected_audio_name)   
    generated_midi_data = mp3_to_midi_w_return(temp_effected_audio_name, temp_midi_gen_name, onset_threshold, frame_threshold)
    midi_to_mp3_w_path_remov(temp_midi_gen_name, temp_mp3_gen_name, fs)
    
    ssim_fitness = spectrogram_ssim_fitness(temp_mp3_init_name, temp_mp3_gen_name)

    result_data["individual"].append(individual)
    result_data["original_midi_data"].append(initial_midi_data.instruments[0].notes)
    if generated_midi_data.instruments:
        result_data["generated_midi_data"].append(generated_midi_data.instruments[0].notes)
        result_data["ssim_fitness"].append(ssim_fitness)
    else:
        result_data["generated_midi_data"].append(generated_midi_data.instruments)
        result_data["ssim_fitness"].append(0)
        
    result_data["onset_threshold"].append(onset_threshold)
    result_data["frame_threshold"].append(frame_threshold)
    
    # Clean up temporary files
    Path(temp_midi_init_name).unlink(missing_ok=True)
    Path(temp_midi_gen_name).unlink(missing_ok=True)
    Path(temp_mp3_init_name).unlink(missing_ok=True)
    Path(temp_mp3_gen_name).unlink(missing_ok=True)
    Path(temp_effected_audio_name).unlink(missing_ok=True)

    return result_data    
    
def test_model_db_creation_parallel(data, n_data, csv_filename, soundfont, program, min_note, max_note, min_duration, max_duration, effects, effect_structure, effects_map, onset_threshold, frame_threshold):
    param_combinations = list(itertools.product(
        onset_threshold,  
        frame_threshold,  
    ))
    
    fs = FluidSynth(soundfont)

    n_workers = 32
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        
        for params in param_combinations:
            onset_threshold, frame_threshold = params
            
            # Use ProcessPoolExecutor for parallel processing
            for _ in range(n_data):  # Generate n_data for each combination
                futures.append(
                    executor.submit(
                        process_data_generation_lines, fs, program, min_note, max_note, min_duration, max_duration, 
                        effects, effect_structure, effects_map, onset_threshold, frame_threshold
                    )
                )
                            
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            result_data = future.result()

            # Extend result_data to data dictionary
            data["original_midi_data"].extend(result_data["original_midi_data"])
            data["generated_midi_data"].extend(result_data["generated_midi_data"])
            data["individual"].extend(result_data["individual"])
            data["ssim_fitness"].extend(result_data["ssim_fitness"])
            data["onset_threshold"].extend(result_data["onset_threshold"])
            data["frame_threshold"].extend(result_data["frame_threshold"])

    # Once all the data is collected, create the DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f"CSV file '{csv_filename}' created successfully.")
    
    
data = {
    "original_midi_data": [],
    "generated_midi_data": [],
    "individual": [], 
    "ssim_fitness": [],
    "onset_threshold": [],
    "frame_threshold": [],
}

#number of data generaterd
n_data = 2

#file names
csv_filename = "../results/dataset_test_model_spectr_fit22.csv"
soundfont = '../audio2midi2audio/FluidR3_GM.sf2'  # Path to your SoundFont file

#guitar specifications
program = 30
min_note = 'E2'
max_note = 'E6' # Generate a random note between MIDI note 40 (E2) and 88 (E6)
min_duration = 10
max_duration = 20

onset_threshold = np.arange(0.0, 1.05, 0.05)
frame_threshold = np.arange(0.0, 1.05, 0.05)

#effects used
n_effects = 6
effects = [i for i in range(n_effects)]
effect_structure = {
    0: { "rate_hz": ('float', (1.0, 20.0)), },# Chorus
    1: { "delay_seconds": ('float', (1.0, 5.0)), },# Delay
    2: { "drive_db": ('float', (1.0, 20.0)), },# Distortion
    3: { "gain_db": ('float', (-10.0, 10.0)) },# Gain
    4: { "depth": ('float', (0.2, 0.6)), },# Phaser
    5: { "wet_level": ('float', (0.2, 0.6)), },# Reverb
}
effects_map = {
    0: 'Chorus',
    1: 'Delay',
    2: 'Distortion',
    3: 'Gain',
    4: 'Phaser',
    5: 'Reverb',
}

# Test with the parameter combinations
if __name__ == '__main__':
    start = time.time()
    test_model_db_creation_parallel(data, n_data, csv_filename, soundfont, program, min_note, max_note, min_duration, max_duration, effects, effect_structure, effects_map, onset_threshold, frame_threshold)
    print(f"Total time: {time.time() - start}")