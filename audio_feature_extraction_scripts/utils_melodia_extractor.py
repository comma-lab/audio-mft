# You need Vamp installed on your machine for this to work and to download the sonic annotator to your computer.

import os
import glob
# Path to the folder containing the audio files you want to analyse
path = r"/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/en_200/*.mp3"
file_paths = glob.glob(path)
sonic_annotator_path = "/Users/benjaminheyderman/Downloads/sonic-annotator-1.6-macos/_sonic-annotator"
for file_path in file_paths:
    # Run Vamp plugin on each file
    # Saves the melodia information in the same folder as the audio file
    os.system(f"{sonic_annotator_path} -d vamp:mtg-melodia:melodia:melody {file_path} -w csv")
