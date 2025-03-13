# Ozone Prediction Model Development

This project implements and extends ozone concentration prediction models based on the methodology described in the research paper "Analysis and prediction of atmospheric ozone concentrations using machine learning."

## Project Structure

```
ozone-fossil-prediction/
├── data/               # Data directory
│   ├── interim/       # Intermediate data
│   ├── processed/     # Processed data ready for modeling
│   └── raw/          # Raw data downloads
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks
├── results/           # Model outputs and figures
│   ├── figures/      # Visualizations and plots
│   └── models/       # Saved model metrics and parameters
└── src/              # Source code
    ├── data/         # Data processing scripts
    ├── features/     # Feature engineering code
    ├── models/       # Model implementations
    └── visualization/# Visualization code
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

The project uses data from the following sources:

1. **NABEL Data (Baseline Model)**
   - Source: Switzerland's National Air Pollution Monitoring Network
   - URL: https://www.bafu.admin.ch/bafu/de/home/themen/luft/zustand/daten/datenabfrage-nabel.html
   - Station: Lugano-Università
   - Period: 2016-2023
   - Variables: Ozone concentration, radiation

2. **Extended Model Data** (To be implemented)
   - Global energy consumption data
   - Atmospheric ozone concentration data at different heights

## Usage

### Baseline Model

1. Download NABEL data and place it in `data/raw/`

2. Process the data:
```bash
python src/data/nabel_data_loader.py
```

3. Train and evaluate the baseline model:
```bash
python src/models/baseline_model.py
```

Results will be saved in the `results/` directory.

## Model Details

### Baseline Model
- Simple linear regression using scikit-learn
- Input: Daily average radiation (W/m²)
- Output: Daily average ozone concentration (µg/m³)
- Model equation: [O₃] = 0.350 · Radiation + 33.962 µg/m³

#### Performance Metrics
- MAE: 17.15 µg/m³
- RMSE: 21.57 µg/m³
- R²: 0.735

The baseline model explains approximately 73.5% of the variance in ozone concentrations using only radiation data. This demonstrates a strong relationship between solar radiation and ozone formation, providing a solid foundation for more sophisticated models.

#### Visualizations
The model generates several visualizations in `results/figures/`:
- Scatter plot showing the relationship between radiation and ozone concentration
- Residual plot for analyzing prediction errors
- Histogram showing the distribution of residuals
- Time series plot comparing actual vs predicted ozone concentrations

### Extended Model (To be implemented)
- Multiple regression with fossil fuel consumption data
- Prediction at different atmospheric heights
- Regional predictions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Add your license information here]