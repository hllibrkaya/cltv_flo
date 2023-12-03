# FLO - BG-NBD and Gamma-Gamma for CLTV Prediction And Customer Segmentation

## Overview to Business Problem

FLO aims to establish a roadmap for sales and marketing activities. To enable the company to make medium to long-term
plans, it is essential to predict the potential value that existing customers will bring to the company in the future.

## Project Structure

- **cltv.py**: Python script containing functions for preparing cltv dataset.
- **research.py**: Python script containing obs./researchs from raw dataset.
- **cltv_flo.csv**: Excel file containing the cltv scores and segments.
- **dataset/flo_data_20k.csv**: Excel file containing the dataset.

## Requirements

Make sure to install the required packages before running the script:

```bash
pip install -r requirements.txt
```

The required packages are:

- pandas==2.1.1
- matplotlib==3.8.2
- lifetimes==0.11.3

## Functionality

1. Data Preparation

- The `data_prep` function in the script handles outlier values and creates additional features for CLTV modeling
    - Outlier replacement with thresholds
    - Feature creation

2. CLTV Modeling

- The `create_cltv` function prepares the data and calculates CLTV features. It includes:
    - Recency calculation
    - Frequency calculation
    - Monetary calculation
- The modelling function uses the **BetaGeoFitter** and **GammaGammaFitter** models to predict future sales and average
  customer
  value. It calculates CLTV based on the specified number of months.

3. Customer Segmentation
    - The cltv_final function segments customers based on their CLTV scores. It uses the qcut method to create segments
      with the specified count and labels.

4. Other functions
    - There are several functions in the code, for grabbing columns, detecting outliers, dataset summary etc.

## Usage

1. **Clone the repository:**

```bash
   git clone https://github.com/hllibrkaya/cltv_flo.git
   cd cltv_flo
```

2. **Install the required packages:**

```bash
pip install -r requirements.txt
```

3. **Run the script**

```bash
python cltv.py
```

## Note

Ensure that the dataset (`dataset/flo_data_20k.csv`) is present in the project directory before running the script.
