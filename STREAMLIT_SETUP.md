# 🚶 Streamlit UI Setup & Run Guide

## Quick Start (5 minutes)

### Step 0: Ensure you have checked the main README and have the prerequisites set up (Python, Pip)

### Step 1: Install Additional Dependencies

Open PowerShell in your project root directory and run:

```Windows (powershell) or Macos (terminal)
python -m pip install -r requirements.txt

```

This installs:

- `streamlit` - Web UI framework
- `plotly` - Interactive charts
- `openpyxl` - Excel export support

### Step 2: Run the Streamlit App

From the project root directory, run:

```Windows (powershell) or Macos (terminal)
streamlit run src/streamlit_app.py

OR alternatively directly with python:
python3 -m streamlit run src/streamlit_app.py

```

**Expected Output:**

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 3: Open in Browser

Automatically opens at `http://localhost:8501` or manually visit that URL

---

## 🎯 What You'll See

### Left Sidebar

- **📍 Location Selection** - Search and select from 47 sensor sites
- **📅 Date Range** - Pick dates up to 7 days in advance
- **⏰ Time Selection** - Choose start/end times (hourly)
- **🤖 Model Selection** - Random Forest or Linear Regression
- **🔮 Generate Predictions** - Click to run the analysis

### Main Area

Once you click "Generate Predictions", you'll see:

1. **📊 Summary Statistics**
   - Average pedestrians
   - Peak hour and count
   - Average temperature
   - Total hours predicted

2. **📈 Interactive Chart**
   - Dual-axis chart (pedestrians + temperature)
   - Hover to see exact values
   - Zoom and pan enabled
   - Toggle data series on/off

3. **📋 Data Table**
   - Detailed hourly predictions
   - View 10, 25, 50, 100 rows or all data
   - Sortable and searchable

4. **📥 Export Options**
   - Download as CSV
   - Download as JSON
   - Download as Excel (if openpyxl installed)

---

## 🔧 Troubleshooting

### Issue: "Module not found: streamlit"

**Solution:**

```powershell
python -m pip install streamlit plotly
```

### Issue: "port 8501 already in use"

**Solution:** Use a different port

```powershell
streamlit run src/streamlit_app.py --server.port 8502
```

### Issue: "Data not found" error

**Solutions:**

- The app will automatically fetch missing data when you click "Generate Predictions"
- Ensure internet connection for API calls
- Check that `rawData/` and `src/data/` directories exist

### Issue: "Chart not displaying"

**Solution:** Ensure plotly is installed

```powershell
python -m pip install plotly --upgrade
```

---

## 📊 Features

✅ **Beautiful Modern UI** - Clean, professional interface
✅ **Site Search** - Fuzzy search across 47 locations
✅ **Interactive Charts** - Zoom, pan, hover details
✅ **Multiple Exports** - CSV, JSON, Excel formats
✅ **Real-time Data** - Automatic forecast updates
✅ **Two ML Models** - Choose accuracy vs speed
✅ **Responsive Design** - Works on desktop/tablet
✅ **Status Messages** - Track data loading progress

---

## 🎨 Customization Options

### Change Default Model

Edit `streamlit_app.py` line with:

```python
model_type = st.radio(
    "Choose prediction model:",
    ["Linear Regression", "Random Forest"],  # Change order here
    ...
)
```

### Adjust Chart Colors

In `streamlit_app.py`, modify chart colors:

```python
line=dict(color='#1f77b4', width=2),  # Blue for pedestrians
line=dict(color='#ff7f0e', width=2, dash='dash'),  # Orange for temperature
```

### Change Max Date Range

In `streamlit_app.py`:

```python
max_date = min_date + timedelta(days=14)  # Change 7 to 14 for 2 weeks
```

---

## 💡 Tips & Tricks

1. **Speed Up Predictions**
   - Select a shorter date range
   - Use Linear Regression instead of Random Forest
   - Select a single day for quick testing

2. **Better Insights**
   - Predict for a full week to see patterns
   - Use Random Forest for highest accuracy
   - Export data and compare multiple days

3. **Data Export**
   - CSV is best for Excel analysis
   - JSON is best for API integration
   - Excel is best for presentations

---

## File Structure

```
src/
├── streamlit_app.py          ← Run this file
├── app.py                    ← Original console app
├── utils/
│   ├── forecast_fetcher.py
│   ├── pedestrianDataFetcher.py
│   └── weather_dataset_builder.py
└── data/
    ├── pedestrians/
    │   ├── default/
    │   └── sites/
    └── weather/
        └── default/
```

---

## Common Commands

```powershell
# Run the Streamlit app (main interface)
streamlit run src/streamlit_app.py

# Run the original console app (for testing)
python src/app.py

# Install all dependencies
python -m pip install -r requirements.txt

# Update Streamlit to latest version
python -m pip install --upgrade streamlit

# Clear Streamlit cache
streamlit cache clear

# Run on specific port
streamlit run src/streamlit_app.py --server.port 8502
```

---

## Performance Notes

- **First run:** May take 2-5 minutes as it fetches historical data
- **Subsequent runs:** Much faster due to caching
- **Random Forest:** Takes 1-2 minutes to train (first time per site)
- **Linear Regression:** Takes 10-30 seconds to train
- **Predictions:** 1-3 seconds for typical date range

---

## Support & Help

If you encounter issues:

1. Check console output for error messages
2. Ensure all requirements are installed
3. Verify data files exist in `src/data/`
4. Try clearing Streamlit cache: `streamlit cache clear`
5. Restart the app

---

**Happy Predicting! 🚶📊**

Created for SISAI Final Project - Optimal Biking Paths Detection
