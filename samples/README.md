# Sample trip logs

Demo logs for the **Upload trip** page (and for `05_predict_trip.py`). Every file here was checked to upload
without errors. When you add the car in the app, use the engine size and cylinders listed below, so the
prediction matches the car that recorded the trip.

## Files

| File | Source | Car (engine) | Road type | Distance | Avg speed | Measured fuel (from MAF) |
|---|---|---|---|---|---|---|
| `gps_trip1.csv` | enviroCar | Opel Corsa 2011 (1.2 L, 4 cyl) | mixed | 7.60 km | 35.9 km/h | 5.44 L/100 km |
| `gps_trip2.csv` | enviroCar | Opel Corsa 2011 (1.2 L, 4 cyl) | mixed | 2.20 km | 38.7 km/h | 4.40 L/100 km |
| `gps_trip3.csv` | enviroCar | Opel Corsa 2011 (1.2 L, 4 cyl) | city | 2.83 km | 15.9 km/h | 10.09 L/100 km |
| `gps_trip4.csv` | enviroCar | Opel Corsa 2011 (1.2 L, 4 cyl) | mixed | 3.11 km | 43.9 km/h | 3.97 L/100 km |
| `ved_city_vehicle575.csv` | VED, held-out test vehicle | VED vehicle 575 (2.4 L, 4 cyl) | city | 4.91 km | 12.6 km/h | 13.44 L/100 km |
| `ved_mixed_vehicle530.csv` | VED, held-out test vehicle | VED vehicle 530 (2.5 L, 4 cyl) | mixed | 10.32 km | 45.1 km/h | 9.05 L/100 km |
| `ved_highway_vehicle157.csv` | VED, held-out test vehicle | VED vehicle 157 (3.5 L, 6 cyl) | highway | 12.68 km | 72.3 km/h | 7.97 L/100 km |
| `envirocar_city_mazda_cx5_2013.csv` | enviroCar validation set | Mazda CX-5 2013 (2.0 L, 4 cyl) | city | 4.24 km | 22.3 km/h | 8.81 L/100 km |
| `envirocar_mixed_opel_corsa_d_2013.csv` | enviroCar validation set | Opel Corsa D 2013 (1.4 L, 4 cyl) | mixed | 11.81 km | 43.6 km/h | 6.61 L/100 km |
| `envirocar_highway_bmw_1_2013.csv` | enviroCar validation set | BMW 1 Series 2013 (1.5 L, 4 cyl) | highway | 31.53 km | 73.4 km/h | 6.19 L/100 km |

**Where they come from**

- **`gps_trip1-4`**: four consecutive tracks of one Opel Corsa 2011 from the enviroCar open data (September 2016,
  Germany), the same files used in the app tests. Upload trips 1-3 together, then trip 4, to see the per-car
  calibration take effect.
- **`ved_*`**: real trips from the Vehicle Energy Dataset (VED, Ann Arbor, 2017-18). The three vehicles are in the
  **held-out test set** of `03_train.py` (`GroupShuffleSplit`, 20 % of vehicles, seed 42), so the models never
  saw them during training. VED trip keys: `575_1942`, `530_773`, `157_3318`. Engine sizes are from
  `VED_Static_Data_ICE&HEV.xlsx`.
- **`envirocar_*`**: real tracks from the enviroCar external-validation set (`01b_fetch_envirocar.py`, real MAF
  only, cleaned as in `09_generalisation.py`), from cars other than the Corsa above. enviroCar track ids:
  `57ecf98ee4b0f05fc22efcda` (Mazda), `57dacd31e4b0f05fc1f3fc86` (Corsa D), `57ede8cce4b01fb1bfbc59bc` (BMW).

The city / mixed / highway labels use the app's road types by average speed: city below 30 km/h, mixed 30-60,
highway above 60. Each VED and enviroCar trip is a typical trip for its group, chosen near the median fuel use.
"Measured" is fuel worked out from the MAF sensor (petrol, stoichiometric air-fuel ratio 14.7), which is the
app's real-world figure.

## Columns the uploader accepts

One row per sample, ideally one per second. Column names are matched case-insensitively by pattern, and the
first matching column is used. Other columns are ignored.

| Meaning | Required? | Unit | Accepted column names (examples) |
|---|---|---|---|
| Time | **required** | ISO date-time (`2016-09-29T16:16:24Z`, `2026-03-10 09:00:00`) or epoch seconds / milliseconds | `time`, `timestamp`, `datetime`, `date time`, `Device Time`, `GPS Time` |
| Speed | **required** | **km/h** (not mph) | `speed`, `Vehicle Speed[km/h]`, `Speed (OBD)(km/h)`, `OBD speed`, `gps_speed`, `GPS Speed` |
| Latitude, longitude | optional: route map | degrees | `lat` / `latitude`, `lon` / `lng` / `longitude` |
| Mass air flow (MAF) | optional: **measured** fuel and per-car calibration | g/s | `maf`, `MAF[g/sec]`, `Mass Air Flow Rate(g/s)`, `air flow rate` |
| Engine RPM | optional: engine-sensor model tiers | rev/min | `rpm`, `Engine RPM[RPM]`, `Engine RPM(rpm)`, `engine speed` |
| Engine load | optional: the full model tier | % | `load`, `engine load`, `eLoad`, `Absolute Load[%]` |
| Outside air temperature | optional | °C | `oat`, `OAT[DegC]`, `ambient air temp`, `outside ...` |
| Fuel rate | optional: measured fuel only if there is no MAF | L/h | `fuel rate ...`, `consumption` |

Raw VED files (`VehId`, `Trip`, `DayNum`, `Timestamp(ms)`, ...) are also accepted as they are; the app then
picks the engine-sensor models automatically. The `ved_*` samples here are converted to the plain format
above, so the app treats them like a phone log (speed-only model unless the car is calibrated).

**How a log is read**

- Trips are split where the log pauses for more than 5 minutes; a file can hold several trips.
- Trips shorter than 1 km or 1 minute are ignored.
- Gaps longer than 5 seconds are left out of the distance and fuel totals.
- Times ending in `Z` are UTC and are shown as given (the samples are not converted to local time).
- Logs with only time and speed work: the app then shows a predicted figure and no route.

## Data sources and licences

- **enviroCar** (`gps_trip*`, `envirocar_*`): open data from the enviroCar citizen-science project
  (https://envirocar.org), published under the Open Database License (ODbL). Contains information from enviroCar,
  made available under the ODbL.
- **Vehicle Energy Dataset (VED)** (`ved_*`): G. S. Oh, D. J. Leblanc, H. Peng, "Vehicle Energy Dataset (VED),
  A Large-scale Dataset for Vehicle Energy Consumption Research", IEEE T-ITS, 2020;
  https://github.com/gsoh/VED. See that repository for its licence terms.
