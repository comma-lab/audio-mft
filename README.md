# Overview
This repository contains data and code used in the paper "_Predicting Moral Values in Lyrics Through Audio_" submitted to the 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP).

## Human-Annotated MFT Datasets

- We utilise a dataset of 200 English-language song lyrics, annotated with 10 moral values (with MFT polarities treated as separate labels), tagged by two bilingual annotators [link here][https://github.com/vjosapreniqi/ismir-mft-values/blob/main/Lyrics_Data/MFT_human_annotated_lyrics.csv].

- We introduce a multi-language dataset consisting of 600 songs in six different languages, manually annotated with five moral value labels (grouping MFT virtues and vices by foundation).


## Artificial Dataset Creation:
- To create a bigger training dataset, we annotated automatically 34k Wasabi song lyrics with 10 Moral foundations using [MoralBERT][https://github.com/vjosapreniqi/ismir-mft-values/tree/main/BertModels]. 
- Steps required were downloading Lyrics and Audio features from audio previews.

1. **Lyrics**: Lyrics can be scraped from Genius using the `utils_get_lyrics` script located in the supporting scripts folder. Note that lyrics are also available through the WASABI API (which is not included in the GitHub version), offering an alternative method.
2. **Previews**: The preview URLs are gathered using the `utils_get_previews` script, and the previews themselves are downloaded using the `utils_download_previews` script.

## Feature Extraction and Dataset Construction

The `utils_construct_dataset` script extracts features and constructs the dataset, using the `audiofeatureextractor` class:
- A combination of custom-designed features (some developed from previous work) and those provided by the Essentia library is extracted.
- The extracted features are saved in a dictionary format, categorized by type for easier filtering or elimination.
- The class includes functionality to convert these dictionaries into Pandas DataFrames, making them ready for use with XGBoost.

## Lyrics Moral Predictions from Audio

The `moral-foundations-predictions.ipynb` notebook is used to predict moral foundations based on the extracted audio features.

## Contributors

- **Audio Feature Extraction**: Ben Heyderman
- **Model Development**: Ben Heyderman
- **Model Curation**: Charalampos Saitis, Johan Pauwels, and Vjosa Preniqi
- **Annotation of Multi-language Moral Lyrics**: Supervised by Charalampos Saitis and Vjosa Preniqi 
