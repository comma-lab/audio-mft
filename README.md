# Overview
This repository contains data and code used in the paper "_Predicting Moral Values in Lyrics Through Audio_" submitted to the 2025 Content-Based Multimedia Indexing (CBMI) conference.

## Human-Annotated Lyrics

- We utilise a dataset of 200 English language song lyrics, annotated with 10 moral values (virtue/vice polarities treated as separate labels) by two bilingual annotators [link here](https://github.com/vjosapreniqi/ismir-mft-values/blob/main/Lyrics_Data/MFT_human_annotated_lyrics.csv).

## Audio Excerpts

- The preview URLs are gathered using the `utils_get_previews` script, and the previews themselves are downloaded using the `utils_download_previews` script.

## Feature Extraction and Dataset Construction

The `utils_construct_dataset` script extracts features and constructs the dataset, using the `audiofeatureextractor` class:
- A combination of custom-designed, [Essentia](https://essentia.upf.edu/index.html), and [MELODIA](https://www.upf.edu/web/mtg/melodia) features are extracted.
- The extracted features are saved in a dictionary format, categorized by type for easier filtering or elimination.
- The class includes functionality to convert these dictionaries into Pandas DataFrames, making them ready for use with XGBoost.

## Lyrics Moral Predictions from Audio

- The `moral-foundations-predictions.ipynb` notebook is used to predict moral foundations based on the extracted audio features.

## Contributors

- **Audio Feature Extraction**: Ben Heyderman
- **Model Development**: Ben Heyderman
- **Model Curation**: Charalampos Saitis, Johan Pauwels, and Vjosa Preniqi
- **Annotation of Multi-language Moral Lyrics**: Supervised by Charalampos Saitis and Vjosa Preniqi 
