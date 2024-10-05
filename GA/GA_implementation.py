from utils import *

df = pd.read_csv('../results/dataset_audios.csv')

note_segmentation = 0.6
model_confidence = 0.6
instrument_program = 30
constellation_map_alg = z_score_peaks_calculation
fitness = fitness_lines_difference_for_parallel_comp
threshold = 2
fan_out = 20
pop_size = 500
p_pop_item = 0.6
p_add_new_effect = 0.5
n_iter = 20
t_size = 8
soundfont = '../audio2midi2audio/FluidR3_GM.sf2'  # Path to your SoundFont file
clear_audio_path = 'clear_audio.mp3'
midi_path = 'clear_midi.mid'

def dataset_GA_execution(df, 
                         note_segmentation,
                         model_confidence,
                         instrument_program,
                         constellation_map_alg,
                         fitness,
                         threshold,
                         fan_out,
                         pop_size,
                         p_pop_item,
                         p_add_new_effect,
                         n_iter,
                         t_size,
                         soundfont,
                         clear_audio_path,
                         midi_path):
    temp_data = []

    for _, row in df.iterrows():
        #desired_audio_path = '../dataset_creation/test_audios/' + row["audio_name"]
        desired_audio_path = '../dataset_creation/test_audios/41.mp3'
        mp3_to_midi(desired_audio_path, midi_path, note_segmentation, model_confidence, instrument_program)
        midi_to_mp3(midi_path, clear_audio_path, soundfont)
        
        best_invdivid = GA_lines_comp(clear_audio_path, desired_audio_path, constellation_map_alg, threshold, fan_out, fitness, pop_size, p_pop_item, p_add_new_effect, n_iter, t_size)    
        print(best_invdivid)
        #temp_data.append(best_invdivid)
        #df['GA_effects'] = pd.Series(temp_data)
        #df.to_csv('../results/dataset_audios.csv', index=False)

if __name__ == '__main__':
    dataset_GA_execution(df, 
                    note_segmentation,
                    model_confidence,
                    instrument_program,
                    constellation_map_alg,
                    fitness,
                    threshold,
                    fan_out,
                    pop_size,
                    p_pop_item,
                    p_add_new_effect,
                    n_iter,
                    t_size,
                    soundfont,
                    clear_audio_path,
                    midi_path)