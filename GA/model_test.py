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

def midi_to_mp3_w_path_remov(midi_file, audio_path, soundfont):
    #convert MIDI to WAV using FluidSynth
    fs = FluidSynth(soundfont)
    wav_file = midi_file.replace('.midi', '.wav').replace('.mid', '.wav')
    fs.midi_to_audio(midi_file, wav_file)

    #convert WAV to MP3 using pydub
    sound = AudioSegment.from_wav(wav_file)
    sound.export(audio_path, format="mp3")
    
    Path(wav_file).unlink(missing_ok=True)

    #print(f"Conversion complete: {audio_path}")

def process_data_generation_lines(soundfont, program, min_note, max_note, min_duration, max_duration, effects, effect_structure, effects_map, threshold, fan_out, max_distance_atan, onset_threshold, frame_threshold, max_key_distance):   
    # Create temporary MIDI and MP3 files
    fd_midi_init, temp_midi_init_name = tempfile.mkstemp(suffix='.mid', dir="../temp_files/")
    fd_midi_gen, temp_midi_gen_name = tempfile.mkstemp(suffix='.mid', dir="../temp_files/")
    fd_mp3_init, temp_mp3_init_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")
    fd_mp3_gen, temp_mp3_gen_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")
    
    # Close file descriptors as we only need the file paths
    os.close(fd_midi_init)
    os.close(fd_midi_gen)
    os.close(fd_mp3_init)
    os.close(fd_mp3_gen)
    
    # Dictionary to hold the data for this individual run
    result_data = {
        "original_midi_data": [],
        "generated_midi_data": [],
        "individual": [], 
        "dissimilarity": [],
        "threshold": [],
        "fan_out": [],
        "max_distance_atan": [],
        "onset_threshold": [],
        "frame_threshold": [],
        "max_key_distance": []
    }
    
    # Generate the MIDI data and apply effects
    initial_midi_data = generate_random_instrument_midi(temp_midi_init_name, program, min_duration, max_duration, min_note, max_note)
    midi_to_mp3_w_path_remov(temp_midi_init_name, temp_mp3_init_name, soundfont)
    
    peaks_original = z_score_peaks_calculation(temp_mp3_init_name, threshold)
    hashes0 = generate_hashes(peaks_original, fan_out)
    hash_table = create_database(hashes0)
    
    peaks_copy_original = z_score_peaks_calculation(temp_mp3_init_name, threshold)
    hashes1 = generate_hashes(peaks_copy_original, fan_out)
    
    individual = create_individual(effects, effect_structure)
    board = Pedalboard([])
    for effect_key, params in individual.items():
        effect_class = globals()[effects_map[effect_key]]
        board.append(effect_class(**params))
        
    # Apply effects and create the effected audio
    fd_effected_audio_name, temp_effected_audio_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")  # Temporary file for effected audio
    os.close(fd_effected_audio_name)

    create_effected_audio_for_parallelization(board, temp_mp3_init_name, temp_effected_audio_name)   
    generated_midi_data = mp3_to_midi_w_return(temp_effected_audio_name, temp_midi_gen_name, onset_threshold, frame_threshold)
    midi_to_mp3_w_path_remov(temp_midi_gen_name, temp_mp3_gen_name, soundfont)
    
    peaks_temp = z_score_peaks_calculation(temp_mp3_gen_name, threshold)
    hashes_temp = generate_hashes(peaks_temp, fan_out)
    
    copy_hash_table = deepcopy(hash_table)
    or_time_pairs = search_database(copy_hash_table, hashes1, max_key_distance)
    if len(or_time_pairs) >= 2: #case for which the list of keys is not empty
        or_times, or_sample_times = zip(*or_time_pairs)
        max_x = max(or_times)
        
        popt1, _ = curve_fit(linear_func, or_times, or_sample_times)
        m1, q1 = popt1
        
        copy_hash_table = deepcopy(hash_table)
        time_pairs_temp = search_database(copy_hash_table, hashes_temp, max_key_distance)
        
        if len(time_pairs_temp) >= 2: #case for which some keys have found match in the list of keys
            temp_times, sample_times = zip(*time_pairs_temp)
            if (len(temp_times) / len(or_times)) >= 0.1: #there is at least some confidence
                popt, _ = curve_fit(linear_func, temp_times, sample_times)
                m2, q2 = popt
                x_intersection =  (q2 - q1)/(m1 - m2)
                intersection_too_far = x_intersection >= max_x * 2 or x_intersection <= -max_x * 0.5
                
                diss = dissimilarity_lines_difference_angle_correction(m1, q1, m2, q2, intersection_too_far, max_distance_atan) * (len(or_times) / len(temp_times))
            else:
                diss = 1000.0
        else:
            diss = 1000.0
    else:
        diss = 1000.0

    result_data["dissimilarity"].append(diss)
    result_data["individual"].append(individual)
    result_data["original_midi_data"].append(initial_midi_data.instruments[0].notes)
    if generated_midi_data.instruments:
        result_data["generated_midi_data"].append(generated_midi_data.instruments[0].notes)
    else:
        result_data["generated_midi_data"].append(generated_midi_data.instruments)
    result_data["threshold"].append(threshold)
    result_data["fan_out"].append(fan_out)
    result_data["max_distance_atan"].append(max_distance_atan)
    result_data["onset_threshold"].append(onset_threshold)
    result_data["frame_threshold"].append(frame_threshold)
    result_data["max_key_distance"].append(max_key_distance)
    
    # Clean up temporary files
    Path(temp_midi_init_name).unlink(missing_ok=True)
    Path(temp_midi_gen_name).unlink(missing_ok=True)
    Path(temp_mp3_init_name).unlink(missing_ok=True)
    Path(temp_mp3_gen_name).unlink(missing_ok=True)
    Path(temp_effected_audio_name).unlink(missing_ok=True)

    return result_data
    
    
def test_model_db_creation_parallel(data, n_data, csv_filename, soundfont, program, min_note, max_note, min_duration, max_duration, effects, effect_structure, effects_map, threshold, fan_out, max_distance_atan, onset_threshold, frame_threshold, max_key_distance):
    param_combinations = list(itertools.product(
        [threshold],
        [fan_out],
        [max_distance_atan],
        [onset_threshold],  
        [frame_threshold],  
        [max_key_distance]
    ))
    
    n_workers = os.cpu_count() - 1
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        
        for params in param_combinations:
            threshold, fan_out, max_distance_atan, onset_threshold, frame_threshold, max_key_distance = params
            
            # Use ProcessPoolExecutor for parallel processing
            for _ in range(n_data):  # Generate n_data for each combination
                futures.append(
                    executor.submit(
                        process_data_generation_lines, soundfont, program, min_note, max_note, min_duration, max_duration, 
                        effects, effect_structure, effects_map, threshold, fan_out, max_distance_atan, onset_threshold, frame_threshold, max_key_distance
                    )
                )
                            
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            result_data = future.result()

            # Extend result_data to data dictionary
            data["original_midi_data"].extend(result_data["original_midi_data"])
            data["generated_midi_data"].extend(result_data["generated_midi_data"])
            data["individual"].extend(result_data["individual"])
            data["dissimilarity"].extend(result_data["dissimilarity"])
            data["threshold"].extend(result_data["threshold"])
            data["fan_out"].extend(result_data["fan_out"])
            data["max_distance_atan"].extend(result_data["max_distance_atan"])
            data["onset_threshold"].extend(result_data["onset_threshold"])
            data["frame_threshold"].extend(result_data["frame_threshold"])
            data["max_key_distance"].extend(result_data["max_key_distance"])

    # Once all the data is collected, create the DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f"CSV file '{csv_filename}' created successfully.")
    
    
data = {
    "original_midi_data": [],
    "generated_midi_data": [],
    "individual": [], 
    "dissimilarity": [],
    "threshold": [],
    "fan_out": [],
    "max_distance_atan": [],
    "onset_threshold": [],
    "frame_threshold": [],
    "max_key_distance": []
}

#number of data generaterd
n_data = 600

#file names
csv_filename = "../results/df_model_test_with_best_values.csv"
soundfont = '../audio2midi2audio/FluidR3_GM.sf2'  # Path to your SoundFont file

#guitar specifications
program = 30
min_note = 'E2'
max_note = 'E6' # Generate a random note between MIDI note 40 (E2) and 88 (E6)
min_duration = 5
max_duration = 20

#EXTERNAL PARAMS
threshold = 1.5 #best value found
fan_out = 10 #best value found
max_distance_atan = 80 #best value found
onset_threshold = 0.5  # standard value that is used by the library (no need to be changed)
frame_threshold = 0.3  # standard value that is used by the library (no need to be changed)
max_key_distance = 95 #best value found

#threshold_values = np.arange(2, 3.5, 0.5)
#fan_out_values = np.arange(10, 30, 10)
#max_distance_atan_values = np.arange(90, 110, 10)
#onset_threshold_values = np.arange(0.4, 0.8, 0.2)
#frame_threshold_values = np.arange(0.4, 0.8, 0.2)
#max_key_distance_values = np.arange(20, 60, 20)

#effects used
n_effects = 6
effects = [i for i in range(n_effects)]
effect_structure = {
    0: { "rate_hz": ('float', (0.0, 100.0)), },# Chorus
    1: { "delay_seconds": ('float', (0.0, 10.0)), },# Delay
    2: { "drive_db": ('float', (0.0, 50.0)), },# Distortion
    3: { "gain_db": ('float', (-50.0, 50.0)) },# Gain
    4: { "depth": ('float', (0.0, 1.0)), },# Phaser
    5: { "wet_level": ('float', (0.0, 1.0)), },# Reverb
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
    test_model_db_creation_parallel(data, n_data, csv_filename, soundfont, program, min_note, max_note, min_duration, max_duration, effects, effect_structure, effects_map, threshold, fan_out, max_distance_atan, onset_threshold, frame_threshold, max_key_distance)
    print(f"Total time: {time.time() - start}")