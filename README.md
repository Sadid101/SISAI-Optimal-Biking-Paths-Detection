# SISAI Final Project MVP Group H

University of Oulu, SISAI Course 2026.

## Project Overview
This project predicts hourly pedestrian traffic in Oulu, Finland, by correlating historical foot traffic data from EcoCounter with meteorological data from the Finnish Meteorological Institute (FMI).

The system utilizes a Random Forest regression model to analyze how environmental factors (temperature) and temporal factors (hour of day, day of week) impact urban mobility.

## Environment Setup

### Prerequisites
* Python 3.8+
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
* **pandas** – tabular data processing
* **numpy** – numerical operations and feature transformations
* **scikit-learn** – machine learning models (Random Forest, Linear Regression)
* **requests** – API communication (FMI + Oulu GraphQL)
* **matplotlib** – charting in console mode
* **streamlit** – web UI for interactive prediction workflow
* **plotly** – interactive charts in the web UI
* **openpyxl** – Excel export support from the web UI
* **pathlib / json** – file handling and JSON read/write


## Running the UI (Recommended)

The project now includes a Streamlit-based UI:

```bash
python -m streamlit run src/streamlit_app.py
```

Then open: `http://localhost:8501`

### Windows

**Option A (quick start script):**

```bat
run_streamlit.bat
```

or PowerShell:

```powershell
.\run_streamlit.ps1
```

**Option B (manual):**

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run src/streamlit_app.py
```

**Optional (console-based app):**

```powershell
python src/app.py
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run src/streamlit_app.py
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run src/streamlit_app.py
```

### What the UI provides

- Searchable site selection dropdown
- Date/time range selection (up to 7 days)
- Model selection (Random Forest or Linear Regression)
- Prediction table and interactive Plotly visualization
- Export options (CSV / JSON / Excel)


## Running the Console App (Legacy)

Console mode is still available:

```bash
python src/app.py
```

After running the command above, follow the prompts in the terminal to:

1. Select start and end date/time
2. Select a monitoring location
3. Choose whether to reuse or refetch local data (if prompted)

The app then fetches/validates data, trains the model, generates predictions, and shows the plot.


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
- The project now supports both **Streamlit UI** and **console** execution modes.
- Internet connection is required for fresh API fetches; cached/local datasets are reused when available.
- The system is a decision-support prototype and does not automatically execute maintenance actions.
- Developed for educational purposes as part of the SISAI course.