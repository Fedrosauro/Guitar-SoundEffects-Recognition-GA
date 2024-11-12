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
'''
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
}'''

#################################################
#      EFFECTS TO TEST GUITAR EFFECTS REC.
#################################################

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

onset_threshold = 0.6
frame_threshold = 0.4

instrument_program = 30 #guitar 30, violin 40
fitness = fitness_spectrograms_comp

pop_size = 250 #150
p_mutation = 0.8
p_crossover = 0.6
p_pop_item = 0.5
p_add_new_effect = 0.5
n_iter = 15  #25 #15 * 250
t_size = 5

soundfont = '../audio2midi2audio/FluidR3_GM.sf2'  # Path to your SoundFont file
clear_audio_path = 'clear_audio.mp3'
midi_path = 'clear_midi.mid'    

df = pd.read_csv('../results/dataset_audios_guitar_low_ranges_w_model_spectr_fit_opt_model_params.csv')
df['GA_effects'] = pd.Series('')        
df['similarity_indv'] = pd.Series('')        
df['similarity_indvs'] = pd.Series('')        
df['best_fit'] = pd.Series('')      
df['best_fitnesses'] = pd.Series('')      
df['GA_bests'] = pd.Series('')      
df.to_csv('../results/dataset_audios_guitar_low_ranges_w_model_spectr_fit_opt_model_params.csv', index=False)


def dataset_GA_execution_general(df, 
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
                            effects_map,
                            onset_threshold, 
                            frame_threshold,
                            instrument_program):
    
    temp_data_ga_effects = []
    temp_data_ga_fitness = []
    temp_data_ga_bests_saved = []
    temp_data_ga_best_fitnesses = []

    for _, row in df.iterrows():
        desired_audio_path = '../dataset_creation/audios_low_ranges_w_clean/' + row["audio_name"]
        
        name_parts = row["audio_name"].split('.')
        #clean_audio_path = f"../dataset_creation/audios_low_ranges_w_clean/{name_parts[0]}_clean.{name_parts[1]}"
        #desired_audio_path = '../dataset_creation/test_audios/78.mp3'
        print(desired_audio_path)
        #print(clean_audio_path)
        mp3_to_midi(desired_audio_path, midi_path, onset_threshold, frame_threshold, instrument_program)
        midi_to_mp3(midi_path, clear_audio_path, soundfont)
        
        best, best_individs, best_fitnesses = GA_spectro_comp(clear_audio_path, 
                                        desired_audio_path,
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
        print(f"{best[0]}, {best[1]}")
        temp_data_ga_effects.append(best[0])
        temp_data_ga_fitness.append(best[1])
        temp_data_ga_bests_saved.append(best_individs)
        temp_data_ga_best_fitnesses.append(best_fitnesses)
        df['GA_effects'] = pd.Series(temp_data_ga_effects)        
        df['GA_bests'] = pd.Series(temp_data_ga_bests_saved)        
        df['best_fit'] = pd.Series(temp_data_ga_fitness)        
        df['best_fitnesses'] = pd.Series(temp_data_ga_best_fitnesses)      
        df.to_csv('../results/dataset_audios_guitar_low_ranges_w_model_spectr_fit_opt_model_params.csv', index=False)

if __name__ == '__main__':
    dataset_GA_execution_general(df, 
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
                            effects_map,
                            onset_threshold, 
                            frame_threshold,
                            instrument_program)