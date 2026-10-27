# DSAI
# Smart Building Energy Demand Prediction

DSAI 310-2, American University of Bahrain
Salman (F2400286) and Rashid (F2400308)

Predicts hourly building electricity demand from the ASHRAE Great Energy Predictor III
(GEPIII) dataset, comparing linear regression, random forest and LightGBM.

## Requirements

```
pip install pandas numpy pyarrow
```

## Data

Download the competition data from Kaggle
(https://www.kaggle.com/c/ashrae-energy-prediction) and place these three files in `data/raw/`:

- `train.csv`
- `weather_train.csv`
- `building_metadata.csv`

The data folder is not committed to GitHub because the files are large.

## Run data preparation

```
python prepare_data.py data/raw
```

This filters to electricity readings for 2016, merges readings, building metadata and
weather, removes invalid readings, builds the features and creates a chronological
train/validation/test split.

Outputs are written to `data/processed/`:

- `electricity_2016_features.parquet`: the cleaned, feature-engineered table
- `prep_report.txt`: summary counts (rows, missing values, rows removed)

## Project status

- Task 1 (literature review and introduction): done
- Task 2 (data preparation): done
- Tasks 3 to 5 (exploratory analysis, models, evaluation): in progress
