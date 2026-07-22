# BoomBikes Demand Prediction — Data Dictionary

This document serves as the schema reference for the BoomBikes daily rental demand dataset.

---

## 1. Core Target Variable

| Variable | Type | Description | Values / Units |
| :--- | :--- | :--- | :--- |
| **`cnt`** | **Integer (Target)** | Total daily bike rental demand | Count of rentals (casual + registered) |

---

## 2. Exogenous Calendar Features

| Variable | Type | Description | Values / Units |
| :--- | :--- | :--- | :--- |
| `dteday` | Date | Calendar date of observation | DD-MM-YYYY |
| `season` | Categorical | Seasons in chronological order | 1: Spring, 2: Summer, 3: Fall, 4: Winter |
| `yr` | Binary | Year indicator | 0: 2018, 1: 2019 |
| `mnth` | Categorical | Month of year | 1 (January) to 12 (December) |
| `holiday` | Binary | Public holiday flag | 0: Regular day, 1: Public holiday |
| `weekday` | Categorical | Day of the week | 0 (Sunday) to 6 (Saturday) |
| `workingday` | Binary | Mapped commuter business day flag | 0: Weekend or Holiday, 1: Business day |

---

## 3. Meteorological (Weather) Features

| Variable | Type | Description | Values / Units |
| :--- | :--- | :--- | :--- |
| `weathersit` | Categorical | Weather severity score | 1: Clear / Few clouds / Partly cloudy<br>2: Mist + Cloudy / Broken clouds<br>3: Light Snow / Light Rain + Thunderstorm<br>4: Heavy Rain + Ice Pallets + Thunderstorm |
| `temp` | Numeric | Normalized daily temperature | Measured in degrees Celsius (°C) |
| `atemp` | Numeric | Normalized "feels-like" temperature | Measured in degrees Celsius (°C) |
| `hum` | Numeric | Normalized daily relative humidity | Percentage (%) |
| `windspeed` | Numeric | Normalized daily wind speed | Kilometers per hour (km/h) |

---

## 4. Metadata and Target-Leaking Fields (Excluded from Modeling)

| Variable | Type | Description | Reason for Exclusion |
| :--- | :--- | :--- | :--- |
| `instant` | Integer | Row sequence identifier | Arbitrary database index with no predictive power |
| `casual` | Integer | Count of casual non-registered rentals | Target leak: part of the target construction (`casual` + `registered` = `cnt`) |
| `registered` | Integer | Count of registered user rentals | Target leak: part of the target construction (`casual` + `registered` = `cnt`) |

---

## 5. Engineered Features (Derived in Pipeline)

| Variable | Type | Description | Formula / Logic |
| :--- | :--- | :--- | :--- |
| `comfort_index` | Numeric | Outdoor comfort metric combining temperature and humidity | `temp - 0.5 * (hum / 100)` |
| `is_weekend` | Binary | Flag indicating weekend days | `weekday.isin([0, 6])` |
| `quarter` | Categorical | Quarter of the calendar year | 1: Jan-Mar, 2: Apr-Jun, 3: Jul-Sep, 4: Oct-Dec |
| `is_warm_season` | Binary | Flag for high-temperature seasons | `season.isin([2, 3])` |
| `is_bad_weather` | Binary | Flag for high severity weather situations | `weathersit >= 3` |
| `temp_yr` | Numeric | Multiplicative interaction of temperature and year | `temp * yr` |
| `day_of_year` | Integer | Day of year index | 1 to 365 (extracted from `dteday`) |
