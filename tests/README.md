# Tests

Headless tests of the Streamlit app and the prediction core. Each test runs against its **own temporary
database and calibration store** (set through the `MOTORLYTICS_DB` and `MOTORLYTICS_CALIB` environment
variables in `_helpers.py`), so your real app data is never touched.

**Requirements:** the project environment (`pip install -r requirements.txt`) and trained models in `models/`
(run `03_train.py` and `04_driving_styles.py` first). Some checks use the internet (Wikipedia car photos,
enviroCar, Open-Meteo); without it those checks fall back or are reported as NOTE instead of failing.

## Run

From the project folder:

```
.venv\Scripts\python.exe tests\run_all.py                  (Windows)
python tests/run_all.py                                    (other systems, environment active)
python tests/run_all.py test_cli test_samples              (only some tests)
python tests/test_flow.py                                  (one test, full output)
```

`run_all.py` prints PASSED / FAILED per file and every failed check, and exits with code 1 if anything failed.

## What each test checks

| Test | Checks |
|---|---|
| `test_flow.py` | Register (salted hash) → add car → petrol price → upload 3 MAF trips (car gets calibrated) → every page renders → fill-up form → 4th trip predicted calibrated with a route → logout |
| `test_cars.py` | Catalogue suggestions (Suzuki Cultus 2018), saved details, masked registration, edit keeps the calibration key, CNG warning, car profile numbers equal a direct calculation, delete needs confirmation, safe database migration |
| `test_photos.py` | Credited Wikipedia example photo (internet), own photo resized and shown first, invalid file rejected, removal, body-type drawing for unknown models |
| `test_pakwheels.py` | PakWheels match for a Suzuki Alto, mixed-trip clamp (unit test), city ≤ mixed ≤ highway km/L on the synthetic Alto trips, own calibration replaces the PakWheels factor, badges and sidebar |
| `test_samples.py` | Every file in `samples/` uploads; measured fuel equals the training pipeline's value; the raw VED file uses the full model tier |
| `test_cli.py` | `05_predict_trip.py` on the samples: tiers, measured figure, calibration and fill-up from the command line |
| `test_quick_estimate.py` | Quick estimate: style and road-type ordering, factor sources (none / PakWheels / own calibration), claimed km/L, nothing saved until the Save button, page renders |
| `test_fetch.py` | "Fetch a real trip": an unseen enviroCar track with real MAF that uploads (internet), cached track ids are skipped, fallback to a bundled sample when the fetch fails |

Streamlit's `AppTest` cannot drive the file-upload widget, so uploads use `app_core.ingest_logs()`, the same
function the Upload trip page calls.

## Other files

- `_helpers.py`: temp database setup, `Checks` (PASS/FAIL collector), `upload()`.
- `fixtures/SYNTHETIC_alto_*.csv`: **made-up** Suzuki Alto trips in Lahore (speed + GPS, no measured fuel), made by
  `fixtures/make_synthetic_alto.py`. They show what the app does with a small Pakistani car; they cannot show
  whether a prediction is right.
- `demo_account.py`: **not a test.** Creates the demo account `demo@motorlytics.test` / `demo12345` with three cars
  and trips in the app's real database (or `--db other.db`), so you can click through the app.
