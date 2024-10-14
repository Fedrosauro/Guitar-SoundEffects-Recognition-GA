import warnings
import logging
import pandas as pd

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress logs below ERROR level
logging.getLogger().setLevel(logging.ERROR)

from utils import *

df = pd.read_csv('../results/dataset_audios.csv')

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
threshold = 1.5
fan_out = 58
max_distance_atan = 80 
max_key_distance = 95 

instrument_program = 30
constellation_map_alg = z_score_peaks_calculation
fitness = fitness_lines_difference_for_parallel_comp

pop_size = 500
p_mutation = 0.6
p_crossover = 0.4
p_pop_item = 0.5
p_add_new_effect = 0.5
n_iter = 15
t_size = 5

soundfont = '../audio2midi2audio/FluidR3_GM.sf2'  # Path to your SoundFont file
clear_audio_path = 'clear_audio.mp3'
midi_path = 'clear_midi.mid'

def dataset_GA_execution(df, 
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
        print(desired_audio_path)
        #desired_audio_path = '../dataset_creation/test_audios/8.mp3'
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
        temp_data.append(best_invdivid)
        df['GA_effects'] = pd.Series(temp_data)
        df.to_csv('../results/dataset_audios.csv', index=False)

if __name__ == '__main__':
    dataset_GA_execution(df, 
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