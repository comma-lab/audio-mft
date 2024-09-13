from dotenv import load_dotenv
import os
import time
import spotifyscrape
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get Spotify API credentials from environment variables
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# Get the Spotify API token
token = spotifyscrape.spotifyscrape.get_token(client_id, client_secret)

# Load the dataset into a DataFrame
db_file = 'dataset.csv'
songs_df = pd.read_csv(db_file)

# Add a 'preview' column if it doesn't already exist
if 'preview' not in songs_df.columns:
    songs_df['preview'] = ''

# Distance between save check points
save_every = 50

# Find the index of the last written row
# This can crash due to rate limiting so I included this so you can pick up after
last_written = songs_df[songs_df['preview'] != ''].last_valid_index()
if last_written is None:
    last_written = -1

# Print the starting point
print(f"Starting from index {last_written + 1}")

# Loop through the DataFrame starting from the last written index + 1
for index in range(last_written + 1, len(songs_df)):
    artist = songs_df.at[index, 'artist']
    title = songs_df.at[index, 'title']
    print(f"Loading File {index}")
    print(f"Artist Name: {artist}")
    print(f"Song Name: {title}")
    
    # Search for the track on Spotify
    spotify_result = spotifyscrape.spotifyscrape.search_track(token, artist, title)

    # Check if a result was found
    if len(spotify_result) > 0:
        # Choose the top result
        spotify_result = spotify_result[0]
        # Save the preview URL to the DataFrame
        songs_df.at[index, 'preview'] = spotify_result["preview_url"]
    else:
        print("Failed to find song")
    
    print()
    time.sleep(1)  # Rate limiting delay

    # Save the DataFrame to a CSV file at checkpoints
    if (index + 1) % save_every == 0:
        songs_df.to_csv(db_file, index=False)
        print(f"Saved progress at index {index + 1}")

# Save the remaining data
songs_df.to_csv(db_file, index=False)
print("Final save completed.")