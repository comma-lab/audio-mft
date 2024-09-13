import pandas as pd
import os


dataset_path = '/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/dataset_en200.csv'
metadata_path = '/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/en_200-ids.csv'

# Load the datasets
dataset = pd.read_csv(dataset_path)
metadata = pd.read_csv(metadata_path)


# Merge the datasets on file_id and youtube_id
merged_df = pd.merge(dataset, 
                     metadata[['youtube_id', 'is_care', 'is_harm', 'is_fairness', 'is_cheating', 
                                 'is_loyalty', 'is_betrayal', 'is_authority', 'is_subversion', 
                                 'is_purity', 'is_degradation']],
                     left_on='file_id', right_on='youtube_id', how='left')

# Remove 'is_'
merged_df.rename(columns={
    'is_care': 'care', 
    'is_harm': 'harm',
    'is_fairness': 'fairness',
    'is_cheating': 'cheating',
    'is_loyalty': 'loyalty',
    'is_betrayal': 'betrayal',
    'is_authority': 'authority',
    'is_subversion': 'subversion',
    'is_purity': 'purity',
    'is_degradation': 'degradation'
}, inplace=True)

# Save the result to a new CSV file
merged_df.to_csv(f"/Users/benjaminheyderman/Documents/QM_Final_Project_Research/Track-Preview-Scrape/final/MoralAnnotations/m_{os.path.basename(dataset_path)}", index=False)