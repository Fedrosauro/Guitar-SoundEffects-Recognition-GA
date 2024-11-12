import random
from pedalboard.io import AudioFile
import librosa
from statistics import NormalDist
from scipy.ndimage import label as label_features
from scipy.ndimage import maximum_position as extract_region_maximums
import numpy as np
from collections import defaultdict
import math
from pedalboard import Pedalboard, Reverb, Chorus, Distortion, Delay, Phaser, Compressor, Gain, Clipping, Limiter, HighpassFilter, LowpassFilter
from scipy.optimize import curve_fit
from copy import deepcopy
from basic_pitch import ICASSP_2022_MODEL_PATH, inference
from midi2audio import FluidSynth 
from pydub import AudioSegment
import shutil
import tempfile
import concurrent.futures
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim


def mp3_to_midi(audio_path, midi_path, note_segmentation, model_confidence, instrument_program):
    _, midi_data, __ = inference.predict(
        audio_path,    
        model_or_model_path = ICASSP_2022_MODEL_PATH, 
        onset_threshold = note_segmentation, # 0.6 note segmentation 1) how easily a note should be split into two. (split notes <- ..0.5.. -> merge notes)
        frame_threshold = model_confidence, # 0.3 model confidence threshold 2) the model confidence required to create a note. (more notes <- ..0.3.. -> fewer notes)
    )

    for instrument in midi_data.instruments:
        instrument.program = instrument_program #distortion guitar program 30

    midi_data.write(midi_path)
    
def midi_to_mp3(midi_file, audio_path, soundfont):
    #convert MIDI to WAV using FluidSynth
    fs = FluidSynth(soundfont)
    wav_file = midi_file.replace('.midi', '.wav').replace('.mid', '.wav')
    fs.midi_to_audio(midi_file, wav_file)

    #convert WAV to MP3 using pydub
    sound = AudioSegment.from_wav(wav_file)
    sound.export(audio_path, format="mp3")
        
    #print(f"Conversion complete: {audio_path}")

def create_individual(effects, effect_structure):
    n_effects_chosen = random.randint(1, len(effects))
    selected_effects = random.sample(effects, n_effects_chosen)
    
    individ = {}
    for effect in selected_effects:
        if effect in effect_structure:
            structure = effect_structure[effect]
            individ[effect] = {
                param: round(random.uniform(range_[0], range_[1]), 2) 
                for param, (_, range_) in structure.items()
            }
    return individ

def create_effected_audio(board, file_path):
    with AudioFile(file_path) as f:
        output_file = file_path[:-4] + "_output.mp3"
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)
    return output_file

def create_effected_audio_for_parallelization(board, file_path, output_file):
    with AudioFile(file_path) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

def z_score_peaks_calculation(file_path, threshold):
    y, _ = librosa.load(path=file_path, sr=None)

    x = librosa.stft(y)
    x = librosa.amplitude_to_db(abs(x))
    
    flattened = np.matrix.flatten(x)
    filtered = flattened[flattened > np.min(flattened)]

    ndist = NormalDist(np.mean(filtered), np.std(filtered))
    zscore = np.vectorize(lambda x: ndist.zscore(x))
    zscore_matrix = zscore(x)

    mask_matrix = zscore_matrix > threshold
    labelled_matrix, num_regions = label_features(mask_matrix)
    label_indices = np.arange(num_regions) + 1

    peak_positions = extract_region_maximums(zscore_matrix, labelled_matrix, label_indices)

    peaks = [[x, y] for y, x in peak_positions]
    return peaks

def create_database(hashes):
    database = defaultdict(list)
    for hash_value, time_offset in hashes:
        database[hash_value].append(time_offset)
    return database

def generate_hashes(constellation_map, fan_out=10):
    hashes = []
    for anchor in constellation_map:
        for target in constellation_map:
            if target[0] > anchor[0]:  #ensure target is after anchor in time
                delta_t = target[0] - anchor[0]
                freq1, freq2 = anchor[1], target[1]
                hash_value = (freq1, freq2, delta_t)
                hashes.append((hash_value, anchor[0]))  # (hash_value, time_offset)
                
                #limit the fan-out to a certain number of target points
                if len(hashes) >= fan_out:
                    break
    return hashes

def find_closest_key(target_key, dictionary):
    keys_array = np.array(list(dictionary.keys()))
    distances = np.linalg.norm(keys_array - target_key, axis=1)
    closest_index = np.argmin(distances)
    closest_key = tuple(keys_array[closest_index])
    closest_distance = distances[closest_index] 
    return closest_key, closest_distance

def search_database(database, sample_hashes, max_key_distance):
    match_offsets = []
    for hash_value, sample_offset in sample_hashes:
        closest_key, closest_distance =  find_closest_key(hash_value, database)
        if closest_distance < max_key_distance:
            for track_offset in database[closest_key]:
                match_offsets.append((track_offset, sample_offset))
                database[closest_key] = database[closest_key][1:]
                break
    return match_offsets

def dissimilarity_lines_difference_angle_correction(m1, q1, m2, q2, intersection_too_far, d_max):
    if m1 == m2 or intersection_too_far:
        if q1 == q2:
            return 0.0
        else:
            return angle_equivalent(d = (abs(q2 - q1) / math.sqrt(1 + m1**2)), d_max = d_max) #distance of the 2 parallel lines
    else: #lines intersect
        if m1 * m2 == -1:
            return 90.0
        else:
            tan_theta = abs((m1 - m2) / (1 + m1 * m2))
            theta_radians = math.atan(tan_theta)
            theta_degrees = math.degrees(theta_radians)
            return theta_degrees
        
def angle_equivalent(d, d_max):
    return (180 / np.pi) * np.arctan(d / d_max)
        
def linear_func(x, a, b):
    return a * x + b

def fitness_lines_difference_for_parallel_comp(individual, clear_audio_path, hash_table, m1, q1, original_or_times, constellation_map_alg, threshold, fan_out, max_key_distance, max_distance_atan, effects_map):
    fd_mp3_clear_name, temp_mp3_clear_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")
    fd_effected_audio_name, temp_effected_audio_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")  # Temporary file for effected audio

    os.close(fd_mp3_clear_name)
    os.close(fd_effected_audio_name)
    
    shutil.copy(clear_audio_path, temp_mp3_clear_name)
    
    board = Pedalboard([])
    for effect_key, params in individual.items():
        effect_class = globals()[effects_map[effect_key]]
        board.append(effect_class(**params))
    
    create_effected_audio_for_parallelization(board, temp_mp3_clear_name, temp_effected_audio_name)   
    new_peaks = constellation_map_alg(temp_effected_audio_name, threshold)
    new_hashes = generate_hashes(new_peaks, fan_out)
    copy_hash_table = deepcopy(hash_table)
    time_pairs = search_database(copy_hash_table, new_hashes, max_key_distance)
    
    if len(time_pairs) >= 2:
        or_times, sample_times = zip(*time_pairs)
        if (len(or_times) / len(original_or_times)) >= 0.1: #there is at least some confidence
            popt1, _ = curve_fit(linear_func, or_times, sample_times)
            m2, q2 = popt1
            
            max_x = max(original_or_times)
            x_intersection =  (q2 - q1)/(m1 - m2)
            diss = dissimilarity_lines_difference_angle_correction(m1, q1, m2, q2, x_intersection >= max_x * 2 or x_intersection <= -max_x * 0.5, max_distance_atan)
        else:
            Path(temp_mp3_clear_name).unlink(missing_ok=True)
            Path(temp_effected_audio_name).unlink(missing_ok=True)
            return individual, 1000.0
    else:
        Path(temp_mp3_clear_name).unlink(missing_ok=True)
        Path(temp_effected_audio_name).unlink(missing_ok=True)
        return individual, 1000.0
        
    Path(temp_mp3_clear_name).unlink(missing_ok=True)
    Path(temp_effected_audio_name).unlink(missing_ok=True)
    
    return individual, diss * (len(original_or_times) / len(or_times))

def pad_to_shape(spectrogram, target_shape):
    pad_height = target_shape[0] - spectrogram.shape[0]
    pad_width = target_shape[1] - spectrogram.shape[1]
    return np.pad(spectrogram, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

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

def fitness_spectrograms_comp(individual, clear_audio_path, desired_audio_path, effects_map):
    fd_mp3_clear_name, temp_mp3_clear_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")
    fd_effected_audio_name, temp_effected_audio_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")  # Temporary file for effected audio
    fd_desired_audio_name, temp_desired_audio_name = tempfile.mkstemp(suffix='.mp3', dir="../temp_files/")  # Temporary file for effected audio

    os.close(fd_mp3_clear_name)
    os.close(fd_effected_audio_name)
    os.close(fd_desired_audio_name)
    
    shutil.copy(clear_audio_path, temp_mp3_clear_name)
    shutil.copy(desired_audio_path, temp_desired_audio_name)
    
    board = Pedalboard([])
    for effect_key, params in individual.items():
        effect_class = globals()[effects_map[effect_key]]
        board.append(effect_class(**params))
    
    create_effected_audio_for_parallelization(board, temp_mp3_clear_name, temp_effected_audio_name)   
    ssim_fitness = spectrogram_ssim_fitness(temp_desired_audio_name, temp_effected_audio_name)
        
    Path(temp_mp3_clear_name).unlink(missing_ok=True)
    Path(temp_effected_audio_name).unlink(missing_ok=True)
    Path(temp_desired_audio_name).unlink(missing_ok=True)
    
    return individual, ssim_fitness

def parallel_fitness_calculation_lines_comp(pop, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, fit, max_key_distance, max_distance_atan, effects_map):
    n_workers = 8
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(fit, ind, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, max_key_distance, max_distance_atan, effects_map)
            for ind in pop
        ]

        for future in concurrent.futures.as_completed(futures):
            ind_fitness = future.result() 
            results.append(ind_fitness)

    return results

def parallel_fitness_calculation_spectro_comp(pop, clear_audio_path, desired_audio_path, fit, effects_map):
    n_workers = 8
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(fit, ind, clear_audio_path, desired_audio_path, effects_map)
            for ind in pop
        ]

        for future in concurrent.futures.as_completed(futures):
            ind_fitness = future.result() 
            results.append(ind_fitness)

    return results

def mutation(individual, p_pop_item, p_add_new_effect, effects, effect_structure):
    if not individual: 
        effect = random.randint(0, len(effects) - 1)
        structure = effect_structure[effect]
        new_gene = {
            effect: 
            {param: round(random.uniform(range_[0], range_[1]), 2) for param, (_, range_) in structure.items()}
        }
        return new_gene 
    
    offspring = deepcopy(individual)
    items = list(offspring.items())
    
    available_effects = set(effects) - set(offspring.keys())
    if not available_effects:
        if random.random() < p_pop_item:
            items.pop(random.randrange(len(items))) 
            return dict(items)
        else:
            return dict(items)
        
    effect = random.choice(list(available_effects))
    structure = effect_structure[effect]
    
    if random.random() < p_pop_item:
        items.pop(random.randrange(len(items))) 
        return dict(items)
    
    #randomly decide between replacing an existing effect or adding a new one
    if random.random() < p_add_new_effect:
        new_gene = (
            effect, 
            {param: round(random.uniform(range_[0], range_[1]), 2) 
             for param, (_, range_) in structure.items()}
        )
        items.append(new_gene)
    else:
        index = random.randint(0, len(items) - 1)
        new_gene = (
            effect, 
            {param: round(random.uniform(range_[0], range_[1]), 2) 
             for param, (_, range_) in structure.items()}
        )
        items[index] = new_gene
    
    return dict(items)  
    
def aggressive_mutation(individual, p_pop_item, p_add_new_effect, effects, effect_structure):
    n_mut = random.randint(1, len(effect_structure))
    offspring = deepcopy(individual)
    
    for _ in range(n_mut):  
        if not offspring: 
            effect = random.randint(0, len(effects) - 1)
            structure = effect_structure[effect]
            new_gene = {
                effect: 
                {param: round(random.uniform(range_[0], range_[1]), 2) for param, (_, range_) in structure.items()}
            }
            offspring = new_gene
            continue

        items = list(offspring.items())
        
        if random.random() < p_pop_item and len(items) > 1:
            items.pop(random.randrange(len(items))) 
            offspring = dict(items)
            continue
        
        available_effects = set(effects) - set(offspring.keys())
        
        if not available_effects:  # No available effects to add, just remove if p_pop_item > random
            if random.random() < p_pop_item and len(items) > 0:
                items.pop(random.randrange(len(items))) 
                offspring = dict(items)
            continue
        
        effect = random.choice(list(available_effects))
        structure = effect_structure[effect]
        
        # Decide between replacing an existing effect or adding a new one
        if random.random() < p_add_new_effect:  # Add a new effect
            new_gene = (
                effect, 
                {param: round(random.uniform(range_[0], range_[1]), 2) 
                 for param, (_, range_) in structure.items()}
            )
            items.append(new_gene)
        else:  # Replace an existing effect
            index = random.randint(0, len(items) - 1)
            new_gene = (
                effect, 
                {param: round(random.uniform(range_[0], range_[1]), 2) 
                 for param, (_, range_) in structure.items()}
            )
            items[index] = new_gene
        
        offspring = dict(items)

    return offspring

def inner_mutation(individual, effect_structure):
    if not individual: 
        return individual
    offspring = deepcopy(individual)
    key = random.choice(list(offspring.keys()))
    structure = effect_structure[key]
    offspring[key] = {
        param: round(random.uniform(range_[0], range_[1]), 2) 
        for param, (_, range_) in structure.items()
    }
    return offspring

def crossover(parent_1, parent_2):
    offspring_1 = deepcopy(parent_1)
    offspring_2 = deepcopy(parent_2)
    
    set1, set2 = set(parent_1.keys()), set(parent_2.keys())
    common_keys = list(set1 & set2) #find common and different keys
    different_keys = list(set1 ^ set2)

    #modify offspring based on common elements
    if common_keys:
        index_1 = random.randint(0, len(common_keys))
        for i in range(index_1):
            key = common_keys[i]
            offspring_1[key], offspring_2[key] = offspring_2[key], offspring_1[key]

    #modify offspring based on symmetric difference elements
    if different_keys:
        index_2 = random.randint(0, len(different_keys))
        for j in range(index_2):
            key = different_keys[j]
            if key in offspring_1:
                offspring_2[key] = offspring_1.pop(key)
            else:
                offspring_1[key] = offspring_2.pop(key)

    return offspring_1, offspring_2

def tournament_selection_with_precomputed_fitness_min(pop_with_fitness, t_size):
    tournament = random.choices(pop_with_fitness, k=t_size)

    best_individual, best_fitness = min(tournament, key=lambda x: x[1])  # x[1] is the fitness value

    return best_individual

def tournament_selection_with_precomputed_fitness_max(pop_with_fitness, t_size):
    tournament = random.choices(pop_with_fitness, k=t_size)

    best_individual, best_fitness = max(tournament, key=lambda x: x[1])  # x[1] is the fitness value

    return best_individual

def init_population(pop_size, effects, effect_structure):
    pop = []
    for _ in range(pop_size):
        pop.append(create_individual(effects, effect_structure))
    return pop

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def similarity_array(individ, effect_structure, n_effects):
    sim_array = np.zeros(n_effects)

    for effect_id, params in individ.items():
        for param_name, param_value in params.items():
            
            if effect_id == 3 and param_name == 'gain_db':
                param_value += 10  # Apply the translation
                
            _, (min_val, max_val) = effect_structure[effect_id][param_name]
            # Normalize the parameter value
            normalized_value = normalize(param_value, min_val, max_val)
            # Assign the normalized value to the corresponding position in B
            sim_array[effect_id] = normalized_value

    # Convert B to a numpy array A
    return np.array(sim_array)

def similarity_calculation(sim_array1, sim_array2, sim_max):
    sim = np.sqrt(np.sum(np.square(sim_array1 - sim_array2)))
    return (sim_max - sim)/ sim_max

def shared_fitness_spectro_comp(pop_with_fitness, alpha, beta, sigma):
    ones = np.ones(6)  
    zeros = np.zeros(6)
    sim_max = np.sqrt(np.sum(np.square(ones - zeros))) #similarità euclidea

    effect_structure = {
        0: { "rate_hz": ('float', (0.0, 20.0)) },  # Chorus
        1: { "delay_seconds": ('float', (0.0, 5.0)) },  # Delay
        2: { "drive_db": ('float', (0.0, 20.0)) },  # Distortion
        3: { "gain_db": ('float', (0, 20.0)) },  # Gain
        4: { "depth": ('float', (0.0, 0.6)) },  # Phaser
        5: { "wet_level": ('float', (0.0, 0.6)) },  # Reverb
    }

    shared_fitness_values = []
    for x, raw_fitness in pop_with_fitness:
        penalty_sum = sum(
            1 - ((1 - similarity_calculation(similarity_array(x, effect_structure, len(effect_structure)), 
                                        similarity_array(y, effect_structure, len(effect_structure)), sim_max)) / sigma) ** alpha
            if (1 - similarity_calculation(similarity_array(x, effect_structure, len(effect_structure)),
                                      similarity_array(y, effect_structure, len(effect_structure)), sim_max)) <= sigma
            else 0
            for y, _ in pop_with_fitness
        )
        scaled_fitness = (raw_fitness ** beta) / penalty_sum
        shared_fitness_values.append((x, scaled_fitness))
    return shared_fitness_values

def shared_fitness_lines_comp(pop_with_fitness, alpha, beta, sigma):
    ones = np.ones(6)  
    zeros = np.zeros(6)
    sim_max = np.sqrt(np.sum(np.square(ones - zeros))) #similarità euclidea

    effect_structure = {
        0: { "rate_hz": ('float', (0.0, 20.0)) },  # Chorus
        1: { "delay_seconds": ('float', (0.0, 5.0)) },  # Delay
        2: { "drive_db": ('float', (0.0, 20.0)) },  # Distortion
        3: { "gain_db": ('float', (0, 20.0)) },  # Gain
        4: { "depth": ('float', (0.0, 0.6)) },  # Phaser
        5: { "wet_level": ('float', (0.0, 0.6)) },  # Reverb
    }

    shared_fitness_values = []
    for x, raw_fitness in pop_with_fitness:
        penalty_sum = sum(
            1 - ((1 - similarity_calculation(similarity_array(x, effect_structure, len(effect_structure)), 
                                        similarity_array(y, effect_structure, len(effect_structure)), sim_max)) / sigma) ** alpha
            if (1 - similarity_calculation(similarity_array(x, effect_structure, len(effect_structure)),
                                      similarity_array(y, effect_structure, len(effect_structure)), sim_max)) <= sigma
            else 0
            for y, _ in pop_with_fitness
        )
        scaled_fitness = (raw_fitness ** beta) * penalty_sum
        shared_fitness_values.append((x, scaled_fitness))
    return shared_fitness_values

def GA_lines_comp(clear_audio_path, desired_audio_path, threshold, fan_out, max_distance_atan, max_key_distance, constellation_map_alg, fit, pop_size, p_mutation, p_crossover, p_pop_item, p_add_new_effect, n_iter, t_size, effects, effect_structure, effects_map):
  pop = init_population(pop_size, effects, effect_structure)
  
  peaks_original = constellation_map_alg(desired_audio_path, threshold)
  hashes0 = generate_hashes(peaks_original, fan_out)
  hash_table = create_database(hashes0)

  peaks_copy_original = constellation_map_alg(desired_audio_path, threshold)
  hashes1 = generate_hashes(peaks_copy_original, fan_out)
  copy_hash_table = deepcopy(hash_table)
  time_pairs = search_database(copy_hash_table, hashes1, max_key_distance)
  or_times, sample_times = zip(*time_pairs)
  if not or_times:
      return "error", "error"
  popt1, _ = curve_fit(linear_func, or_times, sample_times)
  m1, q1 = popt1
  
  best = {}
  best_saves = []
  best_fitnesses = []
  
  alpha = 1.0
  beta = 3.0
  sigma = 0.408248290463863 #distance between [0. 0. 0. 0. 0. 0.] and [1. 0. 0. 0. 0. 0.]

  j = 0
  for i in range(0, n_iter):
    print(f"Iteration: {i}")
    
    pop_with_fitness = parallel_fitness_calculation_lines_comp(
        pop, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, fit, max_key_distance, max_distance_atan, effects_map
    )
    
    pop_with_shared_fitness = shared_fitness_spectro_comp(pop_with_fitness, alpha, beta, sigma)
    
    print(f'TOURNAMENT SELECTION')
    selected = [
        tournament_selection_with_precomputed_fitness_min(pop_with_shared_fitness, t_size) 
        for _ in range(pop_size)
    ]  
         
    pairs = [[selected[i], selected[(i + 1) % len(selected)]] for i in range(len(selected))]    
    
    print(f'CROSSOVER AND MUTATION')
    offsprings_cross = []
    for x, y in pairs:
        if random.random() < p_crossover:
            of1, of2 = crossover(x, y)
            offsprings_cross.append(random.choice([of1, of2]))
        else:
            offsprings_cross.append(random.choice([x, y]))
                
    offsprings = []
    for x in offsprings_cross:
        if random.random() < p_mutation:
            of1 = mutation(x, p_pop_item, p_add_new_effect, effects, effect_structure)
            of1_deeper_mut = inner_mutation(of1, effect_structure)
            offsprings.append(of1_deeper_mut)
        else:
            offsprings.append(x)      
                      
    pop = deepcopy(offsprings)
    #pop = init_population(pop_size, effects, effect_structure)
    
    print(f'CANDIDATE BEST')
    pop_with_fitness = parallel_fitness_calculation_lines_comp(
        pop, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, fit, max_key_distance, max_distance_atan, effects_map
    )
    
    best_candidate, best_candidate_fitness = min(pop_with_fitness, key=lambda x: x[1])  # x[1] is the fitness value
    best_so_far, best_so_far_fitness = fit(best, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, max_key_distance, max_distance_atan, effects_map)
    
    print(f"Best candidate found: {best_candidate}")
    print(f"\nCandidate fitness: {best_candidate_fitness} , best fitness so far: {best_so_far_fitness}")
    if best_candidate_fitness < best_so_far_fitness:
      best = best_candidate
      j = i
      if best_candidate_fitness <= 0.1:
          best_fitnesses.append(best_candidate_fitness)
          best_saves.append(best)
          print(f"Best fitness at generation {j}: {fit(best, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, max_key_distance, max_distance_atan, effects_map)}\n")
          return fit(best, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, max_key_distance, max_distance_atan, effects_map), best_saves, best_fitnesses
              
    if best_candidate_fitness < best_so_far_fitness:
        best_fitnesses.append(best_candidate_fitness)
    else:
        best_fitnesses.append(best_so_far_fitness)
    best_saves.append(best)
    print(f"Best fitness at generation {j}: {fit(best, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, max_key_distance, max_distance_atan, effects_map)}\n")

  return fit(best, clear_audio_path, hash_table, m1, q1, or_times, constellation_map_alg, threshold, fan_out, max_key_distance, max_distance_atan, effects_map), best_saves, best_fitnesses

def GA_spectro_comp(clear_audio_path, desired_audio_path, fit, pop_size, p_mutation, p_crossover, p_pop_item, p_add_new_effect, n_iter, t_size, effects, effect_structure, effects_map):
  pop = init_population(pop_size, effects, effect_structure) 
  
  best = {}
  best_saves = []
  best_fitnesses = []
  
  alpha = 1.0
  beta = 3.0
  sigma = 0.408248290463863 #distance between [0. 0. 0. 0. 0. 0.] and [1. 0. 0. 0. 0. 0.]

  j = 0
  for i in range(0, n_iter):
    print(f"Iteration: {i}")
    
    pop_with_fitness = parallel_fitness_calculation_spectro_comp(
        pop, clear_audio_path, desired_audio_path, fit, effects_map
    )
    
    pop_with_shared_fitness = shared_fitness_lines_comp(pop_with_fitness, alpha, beta, sigma)
    
    print(f'TOURNAMENT SELECTION')
    selected = [
        tournament_selection_with_precomputed_fitness_max(pop_with_shared_fitness, t_size) 
        for _ in range(pop_size)
    ]  
         
    pairs = [[selected[i], selected[(i + 1) % len(selected)]] for i in range(len(selected))]    
    
    print(f'CROSSOVER AND MUTATION')
    offsprings_cross = []
    for x, y in pairs:
        if random.random() < p_crossover:
            of1, of2 = crossover(x, y)
            offsprings_cross.append(random.choice([of1, of2]))
        else:
            offsprings_cross.append(random.choice([x, y]))
                
    offsprings = []
    for x in offsprings_cross:
        if random.random() < p_mutation:
            of1 = mutation(x, p_pop_item, p_add_new_effect, effects, effect_structure)
            of1_deeper_mut = inner_mutation(of1, effect_structure)
            offsprings.append(of1_deeper_mut)
        else:
            offsprings.append(x)      
                      
    pop = deepcopy(offsprings)
    #pop = init_population(pop_size, effects, effect_structure)
    
    print(f'CANDIDATE BEST')
    pop_with_fitness = parallel_fitness_calculation_spectro_comp(
        pop, clear_audio_path, desired_audio_path, fit, effects_map
    )
        
    best_candidate, best_candidate_fitness = max(pop_with_fitness, key=lambda x: x[1])  # x[1] is the fitness value
    best_so_far, best_so_far_fitness = fit(best, clear_audio_path, desired_audio_path, effects_map)
    
    print(f"Best candidate found: {best_candidate}")
    print(f"\nCandidate fitness: {best_candidate_fitness} , best fitness so far: {best_so_far_fitness}")
    if best_candidate_fitness > best_so_far_fitness:
      best = best_candidate
      j = i
      if best_candidate_fitness >= 0.999991:
          best_fitnesses.append(best_candidate_fitness)
          best_saves.append(best)
          print(f"Best fitness at generation {j}: {fit(best, clear_audio_path, desired_audio_path, effects_map)}\n")
          return fit(best, clear_audio_path, desired_audio_path, effects_map), best_saves, best_fitnesses
              
    if best_candidate_fitness > best_so_far_fitness:
        best_fitnesses.append(best_candidate_fitness)
    else:
        best_fitnesses.append(best_so_far_fitness)
    best_saves.append(best)
    print(f"Best fitness at generation {j}: {fit(best, clear_audio_path, desired_audio_path, effects_map)}\n")

  return fit(best, clear_audio_path, desired_audio_path, effects_map), best_saves, best_fitnesses