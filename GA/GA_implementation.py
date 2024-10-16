import warnings
import logging
import pandas as pd

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress logs below ERROR level
logging.getLogger().setLevel(logging.ERROR)

from utils import *


#################################################
#      EFFECTS TO TEST VARIUS INSTRUMENTS
#################################################
n_effects = 9
effects = [i for i in range(n_effects)]

effect_structure = {
    0: {"rate_hz": ('float', (0.0, 100.0))},  # Chorus
    1: {"delay_seconds": ('float', (0.0, 10.0))},  # Delay
    2: {"drive_db": ('float', (0.0, 50.0))},  # Distortion
    3: {"gain_db": ('float', (-50.0, 50.0))},  # Gain
    4: {"depth": ('float', (0.0, 1.0))},  # Phaser
    5: {"wet_level": ('float', (0.0, 1.0))},  # Reverb
    6: {"threshold_db": ('float', (-60.0, 0.0))},  # Compressor
    7: {"threshold_db": ('float', (-60.0, 0.0))},  # Limiter
    8: {"threshold_db": ('float', (-60.0, 0.0))},  # Clipping
}

effects_map = {
    0: 'Chorus',
    1: 'Delay',
    2: 'Distortion',
    3: 'Gain',
    4: 'Phaser',
    5: 'Reverb',
    6: 'Compressor',
    7: 'Limiter',
    8: 'Clipping',
}

#################################################
#      EFFECTS TO TEST GUITAR EFFECTS REC.
#################################################

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


onset_threshold = 0.5
frame_threshold = 0.3
threshold = 2.2
fan_out = 25
max_distance_atan = 20 
max_key_distance = 50 

instrument_program = 30
constellation_map_alg = z_score_peaks_calculation
fitness = fitness_lines_difference_for_parallel_comp

pop_size = 10
p_mutation = 0.8
p_crossover = 0.5
p_pop_item = 0.5
p_add_new_effect = 0.5
n_iter = 50
t_size = 2

soundfont = '../audio2midi2audio/FluidR3_GM.sf2'  # Path to your SoundFont file
clear_audio_path = 'clear_audio.mp3'
midi_path = 'clear_midi.mid'

df = pd.read_csv('../results/dataset_audios_guitar_aggr_mut_low_setup.csv')

def dataset_GA_execution_general(df, 
                            onset_threshold,
                            frame_threshold,
                            threshold,
                            fan_out,
                            max_distance_atan,
                            max_key_distance,
                            instrument_program,
                            constellation_map_alg,
                            fitness,
                            pop_size,
                            p_mutation,
                            p_crossover,
                            p_pop_item,
                            p_add_new_effect,
                            n_iter,
                            t_size,
                            soundfont,
                            clear_audio_path,
                            midi_path,
                            effects,
                            effect_structure,
                            effects_map):
    temp_data = []

    for _, row in df.iterrows():
        desired_audio_path = '../dataset_creation/audios/' + row["audio_name"]
        #desired_audio_path = '../dataset_creation/test_audios/78.mp3'
        print(desired_audio_path)
        mp3_to_midi(desired_audio_path, midi_path, onset_threshold, frame_threshold, instrument_program)
        midi_to_mp3(midi_path, clear_audio_path, soundfont)
        
        best_invdivid = GA_lines_comp(clear_audio_path, 
                                        desired_audio_path,
                                        threshold,
                                        fan_out,
                                        max_distance_atan,
                                        max_key_distance,
                                        constellation_map_alg,
                                        fitness,
                                        pop_size,
                                        p_mutation,
                                        p_crossover,
                                        p_pop_item,
                                        p_add_new_effect,
                                        n_iter,
                                        t_size,
                                        effects,
                                        effect_structure,
                                        effects_map)    
        print(best_invdivid)
        break
        temp_data.append(best_invdivid)
        df['GA_effects'] = pd.Series(temp_data)
        df.to_csv('../results/dataset_audios.csv', index=False)

if __name__ == '__main__':
    dataset_GA_execution_general(df, 
                            onset_threshold,
                            frame_threshold,
                            threshold,
                            fan_out,
                            max_distance_atan,
                            max_key_distance,
                            instrument_program,
                            constellation_map_alg,
                            fitness,
                            pop_size,
                            p_mutation,
                            p_crossover,
                            p_pop_item,
                            p_add_new_effect,
                            n_iter,
                            t_size,
                            soundfont,
                            clear_audio_path,
                            midi_path,
                            effects,
                            effect_structure,
                            effects_map)