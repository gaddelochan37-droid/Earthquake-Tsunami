# Earthquake-Tsunami

## Problem Statement
Predict whether an earthquake will generate a tsunami based on seismic data using 6 different classification models.

## Dataset Description
- **Source**: Kaggle - Global Earthquake-Tsunami Risk Assessment Dataset
- **Instances**: 782
- **Features**: 12
  - magnitude: Earthquake magnitude (Richter scale)
  - cdi: Community Decimal Intensity (0-9)
  - mmi: Modified Mercalli Intensity (1-9)
  - sig: Significance score (0-2000)
  - nst: Number of seismic stations used
  - dmin: Distance to nearest station (degrees)
  - gap: Azimuthal gap (degrees)
  - depth: Earthquake depth (km)
  - latitude: Latitude coordinate
  - longitude: Longitude coordinate
  - Year: Year of occurrence (2001-2022)
  - Month: Month of occurrence (1-12)
- **Target**: tsunami (0=No Tsunami, 1=Tsunami)
- **Type**: Binary Classification
