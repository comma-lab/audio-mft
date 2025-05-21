import numpy as np
import pickle
import os
import glob
import pandas as pd
from scipy.signal import medfilt
import scipy
import math
import librosa
import soundfile
import essentia.standard as es

class BespokeFeatures:
    class Melody:
        """
        Melody processing class with static methods for melody extraction and analysis.
        Based on Melodia melody extraction files. See https://www.justinsalamon.com/melody-extraction.html
        """
        @staticmethod
        def get_melody_from_file(melodia_file, stop_sec=None):
            """
            Loads melody frequencies from file.

            :param melodia_file: Path to the melody file.
            :param stop_sec: Time in seconds to stop reading the file (max length).
            :return: List of frequencies.
            """
            # If the file doesnt exist return blank list
            if not os.path.exists(melodia_file):
                return []
            
            # Load data
            data = np.loadtxt(melodia_file, delimiter=',')
            times, freqs = (data[:, 0], data[:, 1])

            # Truncate if a max length is given
            if stop_sec is not None:
                stop_idx = np.where(times < stop_sec)[0]
                times, freqs = times[stop_idx], freqs[stop_idx]

            # Remove frequencies less than or equal to 0
            freqs[freqs <= 0] = np.nan

            return freqs

        @staticmethod
        def hz2midi(hz):
            """
            Converts frequency values in Hz to MIDI values.
            Adapted from https://github.com/justinsalamon/audio_to_midi_melodia.git

            :param hz: Array of frequency values in Hz.
            :return: Array of MIDI values.
            """
            hz_nonneg = hz.copy()
            idx = (hz_nonneg <= 0)
            hz_nonneg[idx] = 1
            midi = 69 + 12 * np.log2(hz_nonneg / 440.)
            midi[idx] = 0
            midi = np.round(midi)
            return midi
        
        @staticmethod
        def midi_to_notes(midi, fs, hop, smooth, minduration):
            """
            Converts MIDI values to notes sequences.
            Adapted from https://github.com/justinsalamon/audio_to_midi_melodia.git

            :param midi: Array of MIDI values.
            :param fs: Sampling rate.
            :param hop: Hop size.
            :param smooth: Smoothing duration.
            :param minduration: Minimum note duration.
            :return: List of notes with onset, duration, and pitch.
            """
            # Apply smoothing
            if (smooth > 0):
                filter_duration = smooth
                filter_size = int(filter_duration * fs / float(hop))
                if filter_size % 2 == 0:
                    filter_size += 1
                midi_filt = medfilt(midi, filter_size)
            else:
                midi_filt = midi


            notes = []
            p_prev = 0
            duration = 0
            onset = 0
            for n, p in enumerate(midi_filt):
                if p == p_prev:
                    duration += 1
                else:
                    # treat 0 as silence
                    if p_prev > 0:
                        # add note
                        duration_sec = duration * hop / float(fs)
                        # add notes that meet length threshold
                        if duration_sec >= minduration:
                            onset_sec = onset * hop / float(fs)
                            notes.append([onset_sec, duration_sec, p_prev])
                    # New note
                    onset = n
                    duration = 1
                    p_prev = p

            # Add final note
            if p_prev > 0:
                duration_sec = duration * hop / float(fs)
                onset_sec = onset * hop / float(fs)
                notes.append([onset_sec, duration_sec, p_prev])

            return notes

        @staticmethod
        def get_note_diffs(midi_notes):
            """
            Computes differences between consecutive notes. Original code.

            :param midi_notes: List of notes with onset, duration, and pitch.
            :return: Tuple containing note differences, mean pitch, and standard deviation of pitch.
            """

            note_diff = []
            # initialise values with the first note
            weighted_sum = midi_notes[0][2] * midi_notes[0][1]
            total_time = midi_notes[0][1]
            pitches = [midi_notes[0][2]]

            # loop through midi notes comparing each note with next note
            for i in range(len(midi_notes) - 1):
                diff = midi_notes[i][2] - midi_notes[i + 1][2]
                weighted_sum += midi_notes[i + 1][2] * midi_notes[i + 1][1]
                total_time += midi_notes[i + 1][1]
                
                note_diff.append(diff)
                pitches.append(midi_notes[i + 1][2])
            # Calculate duration weighted mean
            mean_pitch = weighted_sum / total_time
            # Calculate overall standard deviation
            std_dev = np.std(pitches)
            return note_diff, mean_pitch, std_dev

        @staticmethod
        def melody_stats(note_diffs):
            """
            Computes statistical features of the melody. Original code.

            :param note_diffs: List of note differences.
            :return: Tuple containing mean absolute difference, standard deviation of absolute differences, histogram of intervals, and direction.
            """
            # Calculate the overall direction of a melody
            # Eg ascending or decending
            direction = 0
            for diff in note_diffs:
                if diff > 0:
                    direction += 1
                elif diff < 0:
                    direction -= 1
            direction /= len(note_diffs)

            # Calculate stats based on the absolute valuye of the note difference
            note_diff_abs = [abs(diff) for diff in note_diffs]
            mean_diff_abs = np.mean(note_diff_abs)
            std_diff_abs = np.std(note_diff_abs)

            # Create histogram of note intervals normalised to octave
            intervals = [diff % 12 for diff in note_diffs]
            hist_interval, _ = np.histogram(intervals, bins=np.arange(0, 13))
            hist_interval = hist_interval.astype(float)
            hist_interval /= hist_interval.max()

            return mean_diff_abs, std_diff_abs, hist_interval, direction
        
        @staticmethod
        def get_melody_features(filename, smooth=0.25, minduration=0.1, stop_sec=None, fs = 48000, hop = 128):
            """
            Extracts melody features from an audio file. Original code.

            :param filename: Path to the melody file.
            :param smooth: Smoothing duration.
            :param minduration: Minimum note duration.
            :param stop_sec: Time in seconds to stop reading the file.
            :return: Dictionary of melody features.
            """

            melody = BespokeFeatures.Melody.get_melody_from_file(filename, stop_sec)
            
            melody_midi = BespokeFeatures.Melody.hz2midi(melody)
            melody_midi = [0 if math.isnan(val) else val for val in melody_midi]
            midi_notes = BespokeFeatures.Melody.midi_to_notes(melody_midi, fs, hop, smooth, minduration)
            note_diffs, mean_height, pitch_range = BespokeFeatures.Melody.get_note_diffs(midi_notes)
            mean_diff_abs, std_diff_abs, hist_interval, direction = BespokeFeatures.Melody.melody_stats(note_diffs)
            return {
                "melody.pitch_range": pitch_range,
                "melody.direction": direction,
                "melody.duration_weighted_mean_pitch_height": mean_height,
                "melody.diff_abs_mean": mean_diff_abs,
                "melody.diff_abs_std": std_diff_abs,
                "melody.hist_interval": hist_interval,
            }

    # Timbre Class
    class Timbre:
        """
        Timbre processing class with static methods for timbre feature extraction.
        """
        @staticmethod
        def mel_spectrogram(y, sr, win_len_sec=0.04, nmels=40):
            """
            Computes a mel-spectrogram from an audio signal.

            :param y: Audio signal.
            :param sr: Sampling rate.
            :param nmels: Number of mel bands.
            :return: Mel-spectrogram and the corresponding sampling rate.
            """
            # convert window length to samples
            # hop = window length / 8
            win1 = int(round(win_len_sec * sr))
            hop1 = int(round(win1 / 8.))

            #calculate mel
            nfft1 = int(2 ** np.ceil(np.log2(win1)))
            D = np.abs(librosa.stft(y, n_fft=nfft1, hop_length=hop1, win_length=win1, window=scipy.signal.hamming)) ** 2
            melspec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=nmels, fmax=8000)
            melsr = sr / float(hop1)
            return melspec, melsr

        @staticmethod
        def calc_mfccs(melspec, sr):
            """
            Computes MFCCs from a mel-spectrogram.

            :param melspec: Mel-spectrogram.
            :param sr: Sampling rate.
            :return: Array of MFCCs.
            """
            mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(melspec), n_mfcc=13)[1:, :]
            return mfccs

        @staticmethod
        def calc_delta_mfccs(mfccs):
            """
            Computes delta MFCCs from MFCCs.

            :param mfccs: Array of MFCCs.
            :return: Array of delta MFCCs.
            """
            delta_mfccs = librosa.feature.delta(mfccs)
            return delta_mfccs

        @staticmethod
        def get_timbre_features(filename):
            """
            Extracts timbre features from an audio file.

            :param filename: Path to the audio file.
            :return: Dictionary of timbre features.
            """
            y, sr = BespokeFeatures.load_audiofile(filename, sr=None)
            melspec, melsr = BespokeFeatures.Timbre.mel_spectrogram(y, sr)
            mfccs = BespokeFeatures.Timbre.calc_mfccs(melspec, sr)
            delta_mfccs = BespokeFeatures.Timbre.calc_delta_mfccs(mfccs)
            mfcc_mean, mfcc_std = BespokeFeatures.mean_std(mfccs, melsr)
            delta_mfcc_mean, delta_mfcc_std = BespokeFeatures.mean_std(delta_mfccs, melsr)
            return {
                "timbre.mfcc_mean": mfcc_mean, 
                "timbre.mfcc_std": mfcc_std, 
                "timbre.delta_mfcc_mean": delta_mfcc_mean, 
                "timbre.delta_mfcc_std": delta_mfcc_std
            }
        

    # Harmony Class
    class Harmony:
        """
        Harmony processing class with static methods for extracting harmony-related features from audio files.
        Some code inspired by https://github.com/mpanteli/music-outliers.git
        """

        @staticmethod
        def get_cqt(y, sr, hop_ms=5):
            """
            Computes the Constant-Q Transform (CQT) of an audio signal.

            :param y: Audio time series.
            :param sr: Sampling rate of the audio.
            :param hop_ms: Hop length in milliseconds.
            :return: CQT matrix.
            """  
            hop_length = sr * hop_ms // 1000
            C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'), hop_length=hop_length, n_bins=72))
            return C

        @staticmethod
        def get_chroma(y, sr, hop_ms=5):
            """
            Computes the chromagram of an audio signal.

            :param y: Audio time series.
            :param sr: Sampling rate of the audio.
            :param hop_ms: Hop length in milliseconds.
            :return: Chromagram matrix and adjusted sampling rate.
            """
            hop_length = sr * hop_ms // 1000
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), hop_length=hop_length)
            ssr = sr / float(hop_length)
            return chroma, ssr

        @staticmethod
        def get_harmony_features(filename):
            """
            Extracts harmony features from an audio file, including the chromagram mean and standard deviation.

            :param filename: Path to the audio file.
            :return: Dictionary containing chromagram mean and standard deviation.
            """
            y, sr = BespokeFeatures.load_audiofile(filename, sr=None)
            chroma, ssr = BespokeFeatures.Harmony.get_chroma(y, sr)
            root_note = chroma.sum(axis=1).argmax()
            chroma_inv = np.concatenate((chroma[root_note:, :], chroma[:root_note, :]), axis=0)
            mean, std = BespokeFeatures.mean_std(chroma_inv, ssr)
            return {
                "harmony.chromagram_mean": mean, 
                "harmony.chromagram_std": std
            }

    
    @staticmethod
    def load_audiofile(filename, sr=None):
        """
        Loads an audio file using soundfile and converts it to mono.

        :param filename: Path to the audio file.
        :param sr: Sampling rate for the audio file. If None, the file's original sampling rate is used.
        :return: Audio time series (y) and sampling rate (sr).
        """
        # librosa load was creating issues with some mp3 files so using soundfile
        y, sr = soundfile.read(filename, samplerate=sr)
        y = librosa.to_mono(np.transpose(y))
        return y, sr

    @staticmethod
    def mean_std(array, sr, win2sec=8, hop2sec=0.5):
        """
        Computes the mean and standard deviation of the array over a sliding window.
        Adapted form https://github.com/mpanteli/music-outliers.git
        
        :param array: Input array to compute statistics on.
        :param sr: Sampling rate of the audio.
        :param win2sec: Window length in seconds.
        :param hop2sec: Hop length in seconds.
        :return: Mean and standard deviation arrays.
        """
        win2 = int(round(win2sec * sr))
        hop2 = int(round(hop2sec * sr))
        nbins, origframes = array.shape
        nframes = int(1 + np.floor((origframes - win2) / float(hop2)))
        array_ave = np.empty((nbins, nframes))
        array_std = np.empty((nbins, nframes))
        for i in range(nframes):
            array_ave[:, i] = np.mean(array[:, (i * hop2):(i * hop2 + win2)], axis=1)
            array_std[:, i] = np.std(array[:, (i * hop2):(i * hop2 + win2)], axis=1)
        array_ave = np.mean(array_ave, axis=1)
        array_std = np.mean(array_std, axis=1)
        return array_ave, array_std
    

class EssentiaFeatures:
    @staticmethod
    def get_essentia_features(filename, select_features_headers=None):
        """
        Extracts features from an audio file using Essentia and selects specific features based on headers.

        :param filename: Path to the audio file.
        :param select_features_headers: Dictionary specifying which features to extract. If None, defaults are used.
        :return: Dictionary of selected features.
        """
        
        select_features = {}
        # if no feature headers are specified, use default
        if select_features_headers == None:
            select_features_headers = {}
            select_features_headers["timbre"] = ['lowlevel.melbands_flatness_db.mean', 'lowlevel.melbands_flatness_db.stdev', 'lowlevel.melbands_kurtosis.mean', 'lowlevel.melbands_kurtosis.stdev', 'lowlevel.melbands_skewness.mean', 'lowlevel.melbands_skewness.stdev', 'lowlevel.melbands_spread.mean', 'lowlevel.melbands_spread.stdev', 'lowlevel.spectral_centroid.mean', 'lowlevel.spectral_centroid.stdev', 'lowlevel.spectral_complexity.mean', 'lowlevel.spectral_complexity.stdev', 'lowlevel.spectral_contrast_coeffs.mean', 'lowlevel.spectral_contrast_coeffs.stdev', 'lowlevel.spectral_contrast_valleys.mean', 'lowlevel.spectral_contrast_valleys.stdev', 'lowlevel.spectral_flux.mean', 'lowlevel.spectral_flux.stdev', 'lowlevel.zerocrossingrate.mean', 'lowlevel.zerocrossingrate.stdev']
            select_features_headers["dynamics"] = ['lowlevel.dynamic_complexity', 'lowlevel.loudness_ebu128.integrated']
            select_features_headers["rhythm"] = ['rhythm.beats_loudness.mean', 'rhythm.beats_loudness.stdev', 'rhythm.bpm', 'rhythm.bpm_histogram_first_peak_bpm', 'rhythm.bpm_histogram_first_peak_weight', 'rhythm.bpm_histogram_second_peak_bpm', 'rhythm.bpm_histogram_second_peak_weight', 'rhythm.danceability', 'rhythm.onset_rate']
            select_features_headers["harmony"] = ['lowlevel.pitch_salience.mean', 'lowlevel.pitch_salience.stdev', 'tonal.chords_histogram', 'tonal.key_krumhansl.key', 'tonal.key_krumhansl.scale', 'tonal.key_krumhansl.strength']
        
        # Mapping for keys
        key_mapping = {
            'C': 0,
            'C#': 1, 'Db': 1,
            'D': 2,
            'D#': 3, 'Eb': 3,
            'E': 4,
            'F': 5,
            'F#': 6, 'Gb': 6,
            'G': 7,
            'G#': 8, 'Ab': 8,
            'A': 9,
            'A#': 10, 'Bb': 10,
            'B': 11
        }
        
        # Compute all essentia features
        # calulate 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
        features, _ = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                        rhythmStats=['mean', 'stdev'],
                                        tonalStats=['mean', 'stdev'])(filename)
        
        #  Select only the desired features
        for category in select_features_headers:
            select_features[category] = {key: features[key] for key in select_features_headers[category]}

        # convert key in to one hot
        # scale into int
        key_one_hot = np.zeros(12)
        key_one_hot[key_mapping[select_features["harmony"]["tonal.key_krumhansl.key"]]] = 1
        select_features["harmony"]["tonal.key_krumhansl.key"] = key_one_hot
        select_features["harmony"]["tonal.key_krumhansl.scale"] = 0 if select_features["harmony"]["tonal.key_krumhansl.scale"] == 'major' else 1

        return select_features

class DatasetConstructor:
    @staticmethod
    def get_file_paths(audio_folder, file_extension="mp3"):
        """
        Retrieves the file paths of all audio files in the specified folder with the given file extension.

        :param audio_folder: Path to the folder containing audio files.
        :param file_extension: Extension of the audio files to search for.
        :return: List of file paths to audio files.
        """
        #remove / at rhe end of the folder path
        if audio_folder[-1]=="/":
            audio_folder=audio_folder[-1]
        # remove . at the start of the extension
        if file_extension[1]==".":
            file_extension=file_extension[1:]
        
        # get all paths
        path = audio_folder + r"/*." + file_extension
        audio_filepaths = glob.glob(path)
        return audio_filepaths
    
    @staticmethod
    def extract_from_folder(audio_folder, file_extension="mp3", save_dataset=True, dataset_filename="dataset.pkl", save_checkpoint=True, checkpoint_interval=100):
        """
        Extracts features from all audio files in a folder and optionally saves the dataset and checkpoints.

        :param audio_folder: Path to the folder containing audio files.
        :param file_extension: Extension of the audio files to search for.
        :param save_dataset: Whether to save the final dataset.
        :param dataset_filepath: Path to save the dataset file.
        :param save_checkpoint: Whether to save checkpoints (large datasets take long time so this can be useful).
        :param checkpoint_interval: Interval at which to save checkpoints.
        :return: Dictionary containing extracted features for all files.
        """
        audio_filepaths = DatasetConstructor.get_file_paths(audio_folder, file_extension="mp3")

        if len(audio_filepaths) == 0:
            raise RuntimeError("No audio files found in folder.")
        
        print(f"Number of files to analyse: {len(audio_filepaths)}\n")

        master_features = {}
        count = 0
        # for each audio file
        for audio_filepath in audio_filepaths:
            count+=1
            print(f"File: {count}:")
            # extract
            file_id = ''.join(os.path.basename(audio_filepath).split(".")[:-1])
            master_features[file_id] = DatasetConstructor.extract_from_file(audio_filepath)
            
            # save checkpoint
            if save_dataset and save_checkpoint and (count % checkpoint_interval == 0):
                print("Save checkpoint reached.")
                DatasetConstructor.save_dataset(dataset_filename, master_features)
                
        # save
        if save_dataset:
            DatasetConstructor.save_dataset(dataset_filename, master_features)
        
        return master_features
        
    @staticmethod
    def save_dataset(file_path, features):
        """
        Saves the dataset to a file.

        :param file_path: Path to save the dataset file.
        :param features: Dictionary containing the dataset features.
        """
        print("Saving dataset...\n")
        with open(file_path, 'wb') as fp:
                    pickle.dump(features, fp)

    @staticmethod
    def extract_from_file(audio_filepath):
        """
        Extracts various features from an audio file.

        :param audio_filepath: Path to the audio file.
        :return: Dictionary containing the extracted features.
        """
        bespoke_features = {}
        try:
            print("Getting harmony features...")
            bespoke_features["harmony"] = BespokeFeatures.Harmony.get_harmony_features(audio_filepath)
            print("Getting timbre features...")
            bespoke_features["timbre"] = BespokeFeatures.Timbre.get_timbre_features(audio_filepath)
            melodia_file = ''.join(audio_filepath.split(".")[:-1]) + "_vamp_mtg-melodia_melodia_melody.csv"
            print("Getting melody features...")
            bespoke_features["melody"] = BespokeFeatures.Melody.get_melody_features(melodia_file)
            print("Getting essentia features...\n")
            essentia_features = EssentiaFeatures.get_essentia_features(audio_filepath)
            print()
        except:
            print("Failed to find features for this file.")
            return {}
        # Combine the dictionaries
        features = {key: {**bespoke_features.get(key, {}), **essentia_features.get(key, {})} for key in set(bespoke_features) | set(essentia_features)}

        return features
    
    @staticmethod
    def expand_dict_columns(df):
        """
        Expands columns containing dictionaries into multiple columns.

        :param df: DataFrame containing the dataset.
        :return: DataFrame with expanded dictionary columns.
        """
        # Iterate through each column
        for col in df.columns:
            # Check if the column contains dictionaries
            if df[col].apply(lambda x: isinstance(x, dict)).any():
                # Expand the dictionary into separate columns
                expanded_cols = df[col].apply(pd.Series)
                
                # Rename the expanded columns to include the original column name
                expanded_cols = expanded_cols.add_prefix(f"{col}_")
                
                # Drop the original column and concatenate the new expanded columns
                df = df.drop(columns=[col]).join(expanded_cols)
        
        return df

    @staticmethod
    def expand_np_array_columns(df):
        """
        Expands columns containing numpy arrays into multiple float columns.

        :param df: DataFrame containing the dataset.
        :return: DataFrame with expanded numpy array columns.
        """
        new_columns = []
        new_column_names = []

        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                # Find the maximum length of the arrays in this column
                max_len = df[col].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0).max()
                
                # Expand each numpy array in this column into separate columns
                for i in range(max_len):
                    new_column_names.append(f"{col}_{i}")
                    new_columns.append(df[col].apply(lambda x: x[i] if isinstance(x, np.ndarray) and len(x) > i else np.nan))
            else:
                # Keep the column as it is
                new_column_names.append(col)
                new_columns.append(df[col])
        # Create a new DataFrame with the expanded columns
        expanded_df = pd.concat(new_columns, axis=1)
        expanded_df.columns = new_column_names
        return expanded_df
    
    @staticmethod
    def dict2df(dataset_dict, included_categories=[], save_dataset=True, dataset_filename="dataset.csv"):
        """
        Converts a nested dictionary into a DataFrame and optionally saves it as a CSV file.

        :param dataset_dict: Dictionary containing the dataset.
        :param included_categories: List of categories to include in the DataFrame. If empty, all categories are included.
        :param save_dataset: Whether to save the DataFrame as a CSV file.
        :param dataset_filename: Path to save the CSV file.
        :return: DataFrame containing the dataset.
        """
    
        # Flatten the nested dictionary
        flattened_dataset = []

        for file_id, features in dataset_dict.items():
            flattened_row = {'file_id': file_id}
            for category, feature_dict in features.items():
                if category in included_categories or included_categories == []:
                    flattened_row.update(feature_dict)
            flattened_dataset.append(flattened_row)

        # Create DataFrame
        dataset_df = pd.DataFrame(flattened_dataset)

        # Make all columns individual values
        dataset_df = DatasetConstructor.expand_dict_columns(dataset_df)
        dataset_df = DatasetConstructor.expand_np_array_columns(dataset_df)

        # Save
        if save_dataset:
            dataset_df.to_csv(dataset_filename, index=False)
            
        return dataset_df
    