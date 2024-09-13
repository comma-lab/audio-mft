
# Collection of functions I wrote/adapted from Spotify documentation
# All can be used in slightly different ways to scrape information from Spotify

import base64
from requests import post,get
import json

class spotifyscrape:
    @staticmethod
    def get_token(client_id, client_secret):
        """
        Get Spotify access Token.

        Parameters:
        client_id (str): spotify client ID.
        client_secret (str): spotify client secret code.

        Returns:
        token (str): Access token required for API requests.
        """
        auth_string = client_id+":"+client_secret
        auth_bytes = auth_string.encode("utf-8")
        auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
        
        url = "https://accounts.spotify.com/api/token"

        headers = {
            "Authorization":"Basic "+auth_base64,
            "Content-Type":"application/x-www-form-urlencoded"
        }

        data = {"grant_type":"client_credentials"}
        result = post(url, headers=headers, data=data)
        json_result = json.loads(result.content)
        token = json_result["access_token"]
        return token
    @staticmethod
    def get_auth_header(token):
        """
        Generate an authorization header for API requests.

        Parameters:
        token (str): Spotify access token.

        Returns:
        dict: Authorization header.
        """
        return{"Authorization":"Bearer "+token}

    @staticmethod
    def search_track(token, artist_name, track_name):
        """
        Search for a track on Spotify using artist and track name.

        Parameters:
        token (str): The access token required for API requests.
        artist_name (str): The name of the artist.
        track_name (str): The name of the track.

        Returns:
        json_result (list): A list containing search results for the track.
        """

        url = "https://api.spotify.com/v1/search"
        header = spotifyscrape.get_auth_header(token)
        artist_name = artist_name.replace(" ", "%20")
        track_name = track_name.replace(" ", "%20")
        track_name = track_name.replace("'", "")
        track_name = track_name.replace("-", " ")
        track_name = track_name.replace("#", "")
        track_name = track_name.replace("&", "")

        query = f"?q=artist%3A{artist_name}%20track%3A{track_name}"
        if len(query)>80:
            query=query[:80]
        query = query+"&type=track&limit=1"
        
        query_url  = url + query
        
        result = get(query_url, headers=header)
        json_result = json.loads(result.content)["tracks"]["items"]
        
        return json_result

    @staticmethod
    def get_audio_features(token, spotify_id):
        """
        Retrieve audio features for a track by its Spotify ID.

        Parameters:
        token (str): Spotify access token.
        spotify_id (str): The Spotify ID of the track.

        Returns:
        json_result (dict): A dictionary containing audio features of the track.
        """
        url = "https://api.spotify.com/v1/audio-features/"
        header = spotifyscrape.get_auth_header(token)
        
        query_url  = url + spotify_id
        result = get(query_url, headers=header)
        json_result = json.loads(result.content)

        return json_result

    @staticmethod
    def get_track(token, spotify_id):
        """
        Retrieve track details by its Spotify ID.

        Parameters:
        token (str): Spotify access token.
        spotify_id (str): The Spotify ID of the track.

        Returns:
        json_result (dict): A dictionary containing details of the track.
        """
        
        header = spotifyscrape.get_auth_header(token)
        query_url = f"https://api.spotify.com/v1/tracks/{spotify_id}"

        result = get(query_url, headers=header)
    
        json_result = json.loads(result.content)

        return json_result

    @staticmethod
    def get_multiple_tracks(token, spotify_ids):
        """
        Retrieve details for multiple tracks by their Spotify IDs. 
        Helps to avoid rate limiting by combining API calls

        Parameters:
        token (str): Spotify access token.
        spotify_ids (str): A comma separated string of Spotify IDs.

        Returns:
        json_result (list): A list containing details of the tracks.
        """
        
        header = spotifyscrape.get_auth_header(token)
        
        query_url = f"https://api.spotify.com/v1/tracks?ids={spotify_ids}"
        
        result = get(query_url, headers=header)
        
        json_result = json.loads(result.content)

        if "tracks" in json_result:
            return json_result['tracks']
        else:
            exit()

    @staticmethod
    def get_preview(token, spotify_id, file_name):
        """
        Download the preview audio of a track by its Spotify ID and save it to a file.

        Parameters:
        token (str): Spotify access token.
        spotify_id (str): The Spotify ID of the track.
        file_name (str): The name of the file to save the audio.

        Returns:
        None
        """
        preview_url = spotifyscrape.get_track(token, spotify_id)["preview_url"]

        if preview_url != None:
            preview = get(preview_url)
            open(file_name, "wb").write(preview.content)

    @staticmethod
    def get_multiple_previews(token, spotify_ids, track_indexes):
        """
        Download the preview audios for multiple tracks by their Spotify IDs and save them to files.

        Parameters:
        token (str): Spotify access token.
        spotify_ids (str): A comma separated string of Spotify IDs.
        track_indexes (list): A list of indexes corresponding to each track.

        Returns:
        None
        """
        tracks = spotifyscrape.get_multiple_tracks(token, spotify_ids)
        i = 0
        for track in tracks:
            if track is not None:
                preview_url = track["preview_url"] 
                
                if preview_url != None:
                    preview = get(preview_url)
                    file_name = "files/AudioPreview_"+str(track_indexes[i]) + ".mp3"
                    open(file_name, "wb").write(preview.content)
            i+=1
