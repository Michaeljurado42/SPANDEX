import os
import glob
import torch
import numpy as np
import re
import soundfile as sf
from typing import Tuple, Dict, Any
import pickle
import random
class DNSAudio:
    """Aduio dataset loader for DNS.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    """
    def __init__(self, root: str = './', subsample = None) -> None:
        self.root = root
        self.root = root
        pickle_file_name = root.replace("/", "_") + "_glob_result.pickle"
        
        # Check if the pickle file exists, and load it if it does
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as handle:
                self.noisy_files = pickle.load(handle)
        else:
            print("Does not exist in cache")
            self.noisy_files = glob.glob(root + 'noisy/**.wav')
            # Save the result to the pickle file
            with open(pickle_file_name, 'wb') as handle:
                pickle.dump(self.noisy_files, handle)

        if subsample != None:
            np.random.seed(42)
            self.noisy_files = np.random.choice(self.noisy_files, subsample, replace=False)

        self.file_id_from_name = re.compile('fileid_(\d+)')
        self.snr_from_name = re.compile('snr(-?\d+)')
        self.target_level_from_name = re.compile('tl(-?\d+)')
        self.source_info_from_name = re.compile('^(.*?)_snr')

    def _get_filenames(self, n: int) -> Tuple[str, str, str, Dict[str, Any]]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = self.root + f'clean/clean_fileid_{file_id}.wav'
        noise_file = self.root + f'noise/noise_fileid_{file_id}.wav'
        snr = int(self.snr_from_name.findall(filename)[0])
        target_level = int(self.target_level_from_name.findall(filename)[0])
        source_info = self.source_info_from_name.findall(filename)[0]
        metadata = {'snr': snr,
                    'target_level': target_level,
                    'source_info': source_info}
        return noisy_file, clean_file, noise_file, metadata

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                        np.ndarray,
                                        np.ndarray,
                                        Dict[str, Any]]:
        noisy_file, clean_file, noise_file, metadata = self._get_filenames(n)
        
        try:
            noisy_audio, sampling_frequency = sf.read(noisy_file)
            clean_audio, _ = sf.read(clean_file)
            noise_audio, _ = sf.read(noise_file)
        except Exception as e:
            error_message = (f"Error reading files for index {n}: {e}\n"
                            f"Noisy file: {noisy_file}\n"
                            f"Clean file: {clean_file}\n"
                            f"Noise file: {noise_file}\n")
            with open("error_log.txt", "a") as error_log:
                error_log.write(error_message)
            random_index = random.randint(0, self.__len__() - 1)
            return self.__getitem__(random_index)

        num_samples = 30 * sampling_frequency  # 30 sec data
        metadata['fs'] = sampling_frequency

        if len(noisy_audio) > num_samples:
            noisy_audio = noisy_audio[:num_samples]
        else:
            noisy_audio = np.concatenate([noisy_audio,
                                        np.zeros(num_samples - len(noisy_audio))])
        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio,
                                        np.zeros(num_samples - len(clean_audio))])
        if len(noise_audio) > num_samples:
            noise_audio = noise_audio[:num_samples]
        else:
            noise_audio = np.concatenate([noise_audio,
                                        np.zeros(num_samples - len(noise_audio))])
        return noisy_audio, clean_audio, noise_audio, metadata

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.noisy_files)

    # def __getitem__(self, n: int) -> Tuple[np.ndarray,
    #                                        np.ndarray,
    #                                        np.ndarray,
    #                                        Dict[str, Any]]:
    #     """Gets the nth sample from the dataset.

    #     Parameters
    #     ----------
    #     n : int
    #         Index of the dataset sample.

    #     Returns
    #     -------
    #     np.ndarray
    #         Noisy audio sample.
    #     np.ndarray
    #         Clean audio sample.
    #     np.ndarray
    #         Noise audio sample.
    #     Dict
    #         Sample metadata.
    #     """
    #     noisy_file, clean_file, noise_file, metadata = self._get_filenames(n)
    #     noisy_audio, sampling_frequency = sf.read(noisy_file)
    #     clean_audio, _ = sf.read(clean_file)
    #     noise_audio, _ = sf.read(noise_file)
    #     num_samples = 30 * sampling_frequency  # 30 sec data
    #     metadata['fs'] = sampling_frequency

    #     if len(noisy_audio) > num_samples:
    #         noisy_audio = noisy_audio[:num_samples]
    #     else:
    #         noisy_audio = np.concatenate([noisy_audio,
    #                                       np.zeros(num_samples
    #                                                - len(noisy_audio))])
    #     if len(clean_audio) > num_samples:
    #         clean_audio = clean_audio[:num_samples]
    #     else:
    #         clean_audio = np.concatenate([clean_audio,
    #                                       np.zeros(num_samples
    #                                                - len(clean_audio))])
    #     if len(noise_audio) > num_samples:
    #         noise_audio = noise_audio[:num_samples]
    #     else:
    #         noise_audio = np.concatenate([noise_audio,
    #                                       np.zeros(num_samples
    #                                                - len(noise_audio))])
    #     return noisy_audio, clean_audio, noise_audio, metadata

if __name__ == '__main__':
    train_set = DNSAudio(
        root='../../data/MicrosoftDNS_4_ICASSP/training_set/')
    validation_set = DNSAudio(
        root='../../data/MicrosoftDNS_4_ICASSP/validation_set/')
