import concurrent.futures
from multiprocessing import cpu_count
import warnings
import logging
import pandas as pd

from utils import *

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

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

onset_threshold = 0.5
frame_threshold = 0.3
threshold = 2.2
fan_out = 25
max_distance_atan = 50 
max_key_distance = 50 

instrument_program = 30 #guitar 30, violin 40
constellation_map_alg = z_score_peaks_calculation
fitness = fitness_lines_difference_for_parallel_comp

pop_size = 1 
p_mutation = 0.8
p_crossover = 0.6
p_pop_item = 0.5
p_add_new_effect = 0.5
n_iter = 5 #15 * 250
t_size = 1

soundfont = '../audio2midi2audio/FluidR3_GM.sf2'  # Path to your SoundFont file
clear_audio_path = 'clear_audio.mp3'
midi_path = 'clear_midi.mid' 

def process_single_file(args):
    row, params = args
    
    name_parts = row["audio_name"].split('.')
    desired_audio_path = f'../dataset_creation/audios_low_ranges_w_clean/{row["audio_name"]}'
    clean_audio_path = f"../dataset_creation/audios_low_ranges_w_clean/{name_parts[0]}_clean.{name_parts[1]}"
    
    print(f"Processing: {desired_audio_path}")
    
    best_invdivid, best_fit = GA_lines_comp(clean_audio_path, 
                                    desired_audio_path,
                                    params['threshold'],
                                    params['fan_out'],
                                    params['max_distance_atan'],
                                    params['max_key_distance'],
                                    params['constellation_map_alg'],
                                    params['fitness'],
                                    params['pop_size'],
                                    params['p_mutation'],
                                    params['p_crossover'],
                                    params['p_pop_item'],
                                    params['p_add_new_effect'],
                                    params['n_iter'],
                                    params['t_size'],
                                    params['effects'],
                                    params['effect_structure'],
                                    params['effects_map'])
    
    return {
        'audio_name': row["audio_name"],
        'GA_effects': best_invdivid,
        'best_fit': best_fit
    }

def dataset_GA_execution_general(df, **params):
    # Determine optimal number of processes (leave one core free)
    n_workers = 5
    
    # Prepare list of arguments for each file
    rows_to_process = [(row, params) for _, row in df.iloc[::-1].iterrows()]
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_single_file, args)
            for args in rows_to_process
        ]
        
        # Use tqdm to show progress
        from tqdm import tqdm
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
                
                # Update DataFrame and save intermediate results
                for res in results:
                    mask = df['audio_name'] == res['audio_name']
                    df.loc[mask, 'GA_effects'] = res['GA_effects']
                    df.loc[mask, 'best_fit'] = res['best_fit']
                df.to_csv('../results/dataset_audios_guitar_low_ranges_pumped.csv', index=False)
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
    
    return results

if __name__ == '__main__':
    # Load DataFrame
    df = pd.read_csv('../results/dataset_audios_guitar_low_ranges_pumped.csv')
    df['GA_effects'] = pd.Series('')        
    df['similarity_indv'] = pd.Series('')        
    df['best_fit'] = pd.Series('')        
    df.to_csv('../results/dataset_audios_guitar_low_ranges_pumped.csv', index=False)

    # Package parameters
    params = {
        'onset_threshold': onset_threshold,
        'frame_threshold': frame_threshold,
        'threshold': threshold,
        'fan_out': fan_out,
        'max_distance_atan': max_distance_atan,
        'max_key_distance': max_key_distance,
        'instrument_program': instrument_program,
        'constellation_map_alg': constellation_map_alg,
        'fitness': fitness,
        'pop_size': pop_size,
        'p_mutation': p_mutation,
        'p_crossover': p_crossover,
        'p_pop_item': p_pop_item,
        'p_add_new_effect': p_add_new_effect,
        'n_iter': n_iter,
        't_size': t_size,
        'soundfont': soundfont,
        'clear_audio_path': clear_audio_path,
        'midi_path': midi_path,
        'effects': effects,
        'effect_structure': effect_structure,
        'effects_map': effects_map
    }
    
    # Run parallel processing
    results = dataset_GA_execution_general(df, **params)