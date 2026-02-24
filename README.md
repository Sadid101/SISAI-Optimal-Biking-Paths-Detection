# SISAI Final Project MVP Group H

University of Oulu, SISAI Course 2026.

## Project Overview
This project predicts hourly pedestrian traffic in Oulu, Finland, by correlating historical foot traffic data from EcoCounter with meteorological data from the Finnish Meteorological Institute (FMI).

The system utilizes a Linear Regression model to analyze how environmental factors (temperature) and temporal factors (hour of day, day of week, season) impact urban mobility.

## Environment Setup

### Prerequisites
* **Python 3.8+**
* A virtual environment is highly recommended to keep dependencies isolated.

### Installation
1. **Clone the repository:**
```bash
git clone https://github.com/Sadid101/SISAI-Optimal-Biking-Paths-Detection.git
cd SISAI-Optimal-Biking-Paths-Detection
```
2. **Create and Activate Virtual Environment:**
# Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```
# macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3.  **Install Dependencies:**
```bash
pip install -r requirements.txt
```

## Data Pipeline Overview

The project is built around a clear and reproducible data pipeline:

### Pedestrian data collection
Historical pedestrian count data is fetched from Oulu’s EcoCounter system and split into **training**, **validation**, and **testing** datasets.

### Weather data collection
Historical hourly temperature data is fetched from the Finnish Meteorological Institute (FMI) and split using the same ratios as the pedestrian data.

### Dataset synchronization
Weather and pedestrian datasets are aligned by timestamp to ensure consistency and to prevent mismatches caused by missing sensor data.

### Model training and evaluation
A baseline **Linear Regression** model is trained using the prepared datasets and evaluated on unseen data.


## How to Run the Project

Follow the steps below to run the full data pipeline and baseline model.  
All scripts are located in the `src/` directory.

### 1. Fetch pedestrian data
Downloads historical pedestrian traffic data from Oulu’s EcoCounter system and splits it into training, validation, and testing datasets.
```bash
python src/pedestrianDataFetcher.py
```
### 2. Build historical weather datasets
Fetches historical hourly temperature data from the Finnish Meteorological Institute (FMI) and creates matching training, validation, and testing datasets.
```bash
python src/weather_dataset_builder.py
```
### 3. Synchronize datasets
Aligns pedestrian and weather datasets by timestamp to ensure consistency and prevent errors caused by missing or mismatched records.
```bash
python src/sync_datasets.py
```
### 4. Run the baseline model
Loads the prepared datasets, trains a Linear Regression model, evaluates its performance, and prints a sample prediction.
```bash
python src/app.py
```

## Script Descriptions

### `app.py`
- Main entry point of the project  
- Handles data loading, feature engineering, model training, evaluation, and example prediction  

### `pedestrianDataFetcher.py`
- Fetches historical pedestrian count data from Oulu’s open GraphQL API  
- Converts timestamps to local time  
- Splits the data into training, validation, and testing sets  

### `weather_dataset_builder.py`
- Fetches historical hourly weather data from the Finnish Meteorological Institute (FMI)  
- Focuses on temperature as the primary feature  
- Builds aligned datasets for training, validation, and testing  

### `forecast_fetcher.py`
- Fetches 24-hour weather forecasts from FMI  
- Included for future extensions and real-time prediction use cases  
- Not required for running the current baseline model  

### `sync_datasets.py`
- Synchronizes weather and pedestrian datasets using weather timestamps as the master timeline  
- Prevents training errors caused by missing or mismatched data 

## Model Details

- **Model:** Linear Regression  
- **Target:** Hourly pedestrian count  

### Features
- Temperature  
- Hour of day (cyclical encoding)  
- Day of week  
- Month / season indicator  
- Weekend indicator  

The model is intentionally simple and interpretable, serving as a baseline for future improvements.

## Data Sources

- **Finnish Meteorological Institute (FMI):** Open weather data  
- **City of Oulu:** EcoCounter pedestrian traffic data  

All data used in this project is publicly available and anonymized.