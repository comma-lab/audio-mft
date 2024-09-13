import os
import glob

input_directory = "/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/*/"
extension = "webm"

# Get a list of all .webm files in the input directory
webm_files = glob.glob(os.path.join(input_directory, f"*.{extension}"))

sonic_annotator_path = "/Users/benjaminheyderman/Downloads/sonic-annotator-1.6-macos/_sonic-annotator"

# Loop over each file in the list
for input_file in webm_files:
    output_file = input_file.replace(extension, "mp3")
    
    # Construct the ffmpeg command
    if not os.path.exists(output_file):
        command = f"ffmpeg -i \"{input_file}\" -q:a 0 \"{output_file}\""
    
        # Execute the convert
        os.system(command)
    '''
    # Extract the melody
    os.system(f"{sonic_annotator_path} -d vamp:mtg-melodia:melodia:melody {output_file} -w csv")
    '''
