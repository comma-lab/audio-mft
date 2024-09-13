from audiofeatureextractor import DatasetConstructor
import glob
import os

path = "/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations"

paths = ['/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/tr_100', 
         '/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/gr_100', 
         '/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/ru_100', 
         '/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/fr_100', 
         '/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/es_100']

for path in paths:

    # Extract features and save dataset in dictionary format
    dataset_dict = DatasetConstructor.extract_from_folder(path)

    # Convert to dataframe
    dataset_df = DatasetConstructor.dict2df(dataset_dict, dataset_filename=f"dataset_{os.path.basename(path)}.csv")

    print(dataset_df.head())
