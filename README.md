# SISAI Final Project MVP Group H

University of Oulu, SISAI Course 2026.

## Project Overview
This project predicts hourly pedestrian traffic in Oulu, Finland, by correlating historical foot traffic data from EcoCounter with meteorological data from the Finnish Meteorological Institute (FMI).

The system utilizes a Random Forest model to analyze how environmental factors (temperature) and temporal factors (hour of day, day of week) impact urban mobility.

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
- **Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```
- **macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3.  **Install Dependencies:**
```bash
pip install -r requirements.txt
```

## How the AI Component Works

The core of this project is a Random Forest Regressor that predicts pedestrian density.

1. **Cyclical Feature Engineering:** To help the AI understand time, the "Hour of Day" is transformed into Sine and Cosine components. This ensures the model recognizes that 23:00 and 00:00 are chronologically adjacent, preventing the "reset" error typical in linear 0–23 numbering.

2. **Weather Integration:** The model utilizes live temperature data fetched from the FMI API as a primary driver for urban mobility.

3. **Data Synchronization:** The system dynamically aligns disparate datasets from Oulu's traffic sensors and FMI weather stations into a unified timeline, using interpolation to bridge gaps in sensor data.


## How to Run the Project

Follow the steps below to run the full data pipeline and baseline model.  
The main script is located in the `src/` directory, and all helper scripts are located in `src/utils/`.

### 1. Fetch pedestrian data
Downloads historical pedestrian traffic data from Oulu’s EcoCounter system and splits it into training, validation, and testing datasets.
```bash
python src/utils/pedestrianDataFetcher.py
```
### 2. Build historical weather datasets
Fetches historical hourly temperature data from the Finnish Meteorological Institute (FMI) and creates matching training, validation, and testing datasets.
```bash
python src/utils/weather_dataset_builder.py
```
### 3. Synchronize datasets
Aligns pedestrian and weather datasets by timestamp to ensure consistency and prevent errors caused by missing or mismatched records.
```bash
python src/utils/sync_datasets.py
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

### `utils/pedestrianDataFetcher.py`
- Fetches historical pedestrian count data from Oulu’s open GraphQL API  
- Converts timestamps to local time  
- Splits the data into training, validation, and testing sets  

### `utils/weather_dataset_builder.py`
- Fetches historical hourly weather data from the Finnish Meteorological Institute (FMI).
- Data Cleaning: Includes built-in logic to refetch and repair missing historical values.

### `utils/forecast_fetcher.py`
- Fetches live 7-day weather forecasts from FMI to allow the model to predict future traffic. 

### `utils/sync_datasets.py`
- Aligns weather and pedestrian datasets by timestamp and handles missing data via interpolation.


## Model Details

- **Model:** Random Forest
- **Target:** Hourly pedestrian count  

### Features

- **Temperature:** Real-time and historical Celsius values.
- **Time (Cyclical):** Hour_sin and Hour_cos encoding.
- **Temporal Flags:** Weekday, Weekend indicator, and Month.  

The model is intentionally simple and interpretable, serving as a baseline for future improvements.

## Data Sources

- **Finnish Meteorological Institute (FMI):** Open weather data  
- **City of Oulu:** EcoCounter pedestrian traffic data  

All data used in this project is publicly available and anonymized.