# SISAI Final Project MVP Group H

University of Oulu, SISAI Course 2026.

## Project Overview
This project predicts hourly pedestrian traffic in Oulu, Finland, by correlating historical foot traffic data from EcoCounter with meteorological data from the Finnish Meteorological Institute (FMI).

The system utilizes a Random Forest regression model to analyze how environmental factors (temperature) and temporal factors (hour of day, day of week) impact urban mobility.

## Environment Setup

### Prerequisites
* **Python 3.8+**
* Git
* Internet connection (for fetching API data)
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

## Main Libraries Used
* **pandas** – data processing
* **numpy** – numerical operations
* **scikit-learn** – machine learning models (Random Forest, Linear Regression)
* **matplotlib** – visualization
* **pathlib / json** – file handling


## Running the Application

The main entry point of the project is:
```bash
python src/app.py
```
After running the command above, follow the prompts displayed in the console:

1. **Select a start and end date**
   - You can choose a prediction range of up to 7 days.
   - If a default value is shown, you may press Enter to use it.

2. **Select a monitoring location**
   - A list of available sites will be printed.
   - Enter the number corresponding to the location you want to use.

3. **Confirm data usage (if prompted)**
   - If historical data for the selected site already exists locally, you will be asked whether to reuse it or fetch fresh data.
   - If no data exists, it will be fetched automatically.

Once these steps are completed, the application will:
- Ensure historical weather data exists (fetch if missing).
- Update the latest weather forecast from the FMI API.
- Train the prediction model using historical data.
- Generate pedestrian traffic predictions for the selected time range.
- Display the results in the console.
- Show a visualization graph.
- Ask whether you want to save the predictions as a JSON file in the `results/` directory.

No additional scripts need to be executed manually. The application handles all required data fetching, preprocessing, and prediction steps automatically.


## AI Component

The system uses supervised machine learning to predict hourly pedestrian counts.

### Model
- **Default model:** Random Forest Regressor
- **Alternative option:** Linear Regression (implemented for comparison)

The model is trained each time the application runs using historical pedestrian data combined with historical weather data.

Random Forest was selected as the default model because it handles non-linear relationships more effectively than linear regression, which is important when modeling human mobility behavior under varying weather conditions.

### Features Used
The model uses the following input features:

- Temperature (°C)
- Month
- Weekday
- Weekend indicator
- Hour of day (cyclical encoding using sine and cosine transformation)

Cyclical encoding ensures that 23:00 and 00:00 are treated as adjacent hours rather than numerically distant values.

The target variable is the hourly pedestrian count.


## Data Sources

### Finnish Meteorological Institute (FMI)
Open weather data via WFS API  
[FMI Open Data WFS](https://opendata.fmi.fi/wfs)

### City of Oulu
EcoCounter pedestrian traffic data via GraphQL API  
[Oulu Traffic GraphQL API](https://api.oulunliikenne.fi/proxy/graphql)

All data used in this project is publicly available and anonymized.

## Developer Scripts (Internal Tools)

The `src/utils/` directory contains helper scripts used internally by the application. These are intended for development, debugging, and experimentation purposes.

Examples include:
- Fetching historical pedestrian data from the API
- Building historical weather datasets
- Synchronizing pedestrian and weather datasets
- Fetching forecast data separately

These scripts are automatically utilized by `app.py` and are not required for regular users. However, they can be executed independently by developers who want to inspect or rebuild specific parts of the data pipeline.

## Notes
- This is a console-based application.
- The system is a decision-support prototype and does not automatically execute maintenance actions.
- Developed for educational purposes as part of the SISAI course.