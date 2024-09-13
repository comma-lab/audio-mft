from lyricsgenius import Genius
import pandas as pd
import re
import time

def get_lyrics(artist, title):
    """
    Retrieve and clean lyrics using lyricsgenius (python library for interacting with the Genius API).

    Parameters:
    artist (str): The artist of the song.
    title (str): The title of the song.

    Returns:
    str: The cleaned lyrics of the song, or None if not found.
    """

    # Initialize retry counter
    retried = 0
    while True:
        try:
            # Search for song
            song = genius.search_song(title=title, artist=artist)
            print()
            break  # Break the loop if the song is found
        except:
            # If search fails increment retry counter and wait for 1 second
            retried += 1
            time.sleep(1)
            # If the number of retries reaches 10, print a message and exit the function
            if retried == 10:
                print("Tried 10 times, exiting...")
                return None

    # If no song is found, return None
    if song is None:
        print("Song not found...")
        return None
    
    else:
        # Check if the found song artist matches the given artist
        # Genis' search function commonly returns the wrong lyrics if it does not have the correct ones
        if song.title.lower() != title.lower():
            print("Wrong match...")
            return None
        
        # Get the lyrics of the song
        song_lyrics = song.lyrics
        
        # Split the lyrics into lines and filter out adverts
        lines = song_lyrics.split('\n')
        filtered_lines = [line for line in lines if "You might also like" not in line]
        
        # Join the filtered lines back into a single string
        cleaned_lyrics = '\n'.join(filtered_lines)
        
        # remove metadata prefix
        index = cleaned_lyrics.find("Lyrics")
        if index != -1:
            cleaned_lyrics = cleaned_lyrics[index + len("Lyrics"):].strip()

        # Remove other unwanted characters
        cleaned_lyrics = cleaned_lyrics.replace("Embed", "")
        cleaned_lyrics = cleaned_lyrics.replace('"', "")
        cleaned_lyrics = cleaned_lyrics.replace("'", "")
        
        # Remove any text within square brackets
        # eg. [chorus x2]
        cleaned_lyrics = re.sub(r'\[.*?\]', '', cleaned_lyrics)
        
        return cleaned_lyrics

db_file = 'dataset.csv'
songs_df = pd.read_csv(db_file)

# Add lyrics column if not present
if 'lyrics' not in songs_df.columns:
    songs_df['lyrics'] = ''

# Initialise Genius
GENIUS_API_TOKEN = '****** TOKEN HERE ******'
genius = Genius(GENIUS_API_TOKEN)

save_every = 50

# Find the index of the last written row
# This can crash due to rate limiting so I included this so you can pick up after
last_saved_index = songs_df[songs_df['lyrics'] != ''].last_valid_index()
if last_saved_index is None:
    last_saved_index = -1

print(f"Starting from index {last_saved_index + 1}")

# for each item in the dataset, get the lyrics
for idx in range(last_saved_index + 1, len(songs_df)):
    artist = songs_df.at[idx, 'artist']
    title = songs_df.at[idx, 'title']
    lyrics = get_lyrics(artist, title)
    songs_df.at[idx, 'lyrics'] = lyrics
    
    # Save checkpoints
    if (idx + 1) % save_every == 0:
        songs_df.to_csv(db_file, index=False)
        print(f"Saved progress at index {idx + 1}")

# Save final dataframe
songs_df.to_csv(db_file, index=False)
print("Final save completed.")