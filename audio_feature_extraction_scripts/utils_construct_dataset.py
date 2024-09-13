from audiofeatureextractor import DatasetConstructor

# Extract features and save dataset in dictionary format
dataset_dict = DatasetConstructor.extract_from_folder("/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/en_200")

# Convert to dataframe
dataset_df = DatasetConstructor.dict2df(dataset_dict)

print(dataset_df.head())
