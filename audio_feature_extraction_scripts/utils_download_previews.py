
from requests import get
import csv

def get_preview(preview_url, index):
    """
    Download preview from the given URL and save it.

    Parameters:
    preview_url (str): The URL of the audio preview to download.
    index (int): The index to use in the filename for saving the audio file.

    Returns:
    None
    """
    # Get the preview from the URL
    preview = get(preview_url)

    # Save file using padded index as file name eg 000001.mp3
    file_name = f"/Users/benjaminheyderman/Documents/QM Final Project Research/Track-Preview-Scrape/200_audio/{str(index).zfill(6)}.mp3"
    open(file_name, "wb").write(preview.content)

# For each item in the dataset, download the preview
index = 0
with open('MFT_human_annotated_lyrics_prev.csv', 'r') as input_csvfile:
        input_db = csv.reader(input_csvfile)
        header = next(input_db)
        for line in input_db:
                preview_url = line[-1]
                if preview_url != "":
                    get_preview(preview_url, index)
                index+=1