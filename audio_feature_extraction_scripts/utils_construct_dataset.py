from audiofeatureextractor import DatasetConstructor

# Extract features and save dataset in dictionary format
dataset_dict = DatasetConstructor.extract_from_folder("../datasets/en_200")

# Convert to dataframe
dataset_df = DatasetConstructor.dict2df(dataset_dict)

print(dataset_df.head())
