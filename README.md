# Motorlytics: training pipeline

Put every file in your project root (`D:\Ml project\`), next to the `Data\` folder.

```
pip install -r requirements.txt
```

## Run order

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python 01_fetch_weather.py` | **Live API** (Open-Meteo): hourly Ann Arbor weather, Oct 2017 to Dec 2018 | `Data/weather_annarbor_api.csv` (~11k rows) |
| 1b | `python 01b_fetch_envirocar.py --tracks 300` | **Live API** (enviroCar): real OBD-II tracks from other drivers, used as unseen test data | `Data/envirocar_rows.csv`, `Data/processed/envirocar_trips.parquet` |
| 2 | `python 02_build_datasets.py` | Builds both training sets from the **raw** VED files (first test with `--max-files 3`) | `Data/processed/trips.parquet`, `rows.parquet` |
| 3 | `python 03_train.py --task trip --external Data/processed/envirocar_trips.parquet` | Main model: trip L/100 km. Feature selection, 3 models, 10-fold grouped CV, tests, held-out test, error analysis, external validation | `models/`, `Data/processed/results_trip/` |
| 3b | `python 03_train.py --task row --folds 10` | Second model: instantaneous fuel rate (L/h) | `Data/processed/results_row/` |
| 4 | `python 04_driving_styles.py` | K-Means eco / normal / aggressive profiles, and the eco profile used for tips | `models/driving_style.joblib` |
| 5 | `python 05_predict_trip.py my_log.csv --disp 1.0 --cyl 3 --cls Car --petrol <PKR/L>` | Viva demo: prediction, style, tips and PKR cost for an **unseen** trip log | `prediction.json` |
| 6 | `python 06_make_plots.py` | 16 report figures, plus the numbers behind each figure | `plots/*.png`, `plots/plot_notes.txt` |

`motorlytics_features.py` holds all the feature code. The builder, training and the app all import
it, so a trip is processed the same way during training and in the app.

## The two training sets

**trips.parquet (main).** One row per trip, target `l_per_100km` = total MAF-derived fuel ÷ distance.
This is the number a user sees, and it averages out sensor noise. About 30 features:
- driving behaviour: average and moving speed, speed std, idle share, stops/km, hard accelerations
  and brakes per km, RPA (relative positive acceleration), time shares in speed bands, distance, duration
- engine (OBD only): mean RPM, 90th-percentile RPM, share of time above 3000 RPM, RPM per km/h, load
- context: weather (API), outside-air temperature, hour, weekend, month
- vehicle: displacement, cylinders, turbo, weight, class (one-hot)

**rows.parquet (second).** Sampled time-steps, target `fuel_Lph`, with rolling and lag features computed
inside each trip.

Both use **petrol ICE cars only** (hybrids and diesel removed via the static file).

## Cleaning and outliers (for the report)
- Logging gaps over 5 s are excluded from integration; acceleration is clipped to ±8 m/s².
- Trips under 1 km or 2 min, or with under 80% MAF coverage, are dropped.
- L/100 km is kept within 2 to 40; then a per-vehicle robust z-score (median/MAD) over 6 is removed.
- Missing values are median-imputed inside each model pipeline (LightGBM handles NaN natively).

## Things to check on the real data
1. **`VED_BASE_DATE`** in `motorlytics_features.py` (2017-11-01). Confirm in the VED README how `DayNum`
   is defined; a wrong base shifts the weather join.
2. The builder prints its **column mapping** for the first file. Check that speed, MAF and RPM are matched.
3. If `03_train.py` is slow on the row task, lower `--max-rows`.
4. Swap any model in `make_models()` for one from your course outline (KNN, Decision Tree, SVR) if needed.