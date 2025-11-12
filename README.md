# 🐊 UF Athletics Data Collection & Cleaning - Capstone Project

This project automatically collects, cleans, and merges data on University of Florida athletics teams (SEC sports only), focusing on championship seasons and inter-team performance trends.

## Project Structure

```
capstone project/
├── UF_Athletics_Data_Cleaning.ipynb    # Main Jupyter notebook
├── requirements.txt                     # Python package dependencies
├── cleaned_data/                        # Output directory for CSV files
│   ├── master_seasons_cleaned.csv      # Main cleaned and merged dataset
│   ├── sportsref_cfb_raw.csv           # Raw football data
│   └── sportsref_cbb_raw.csv           # Raw basketball data
└── README.md                            # This file
```

## Data Sources

1. [FloridaGators.com - Official Overview of Championships](https://floridagators.com/sports/2015/12/10/_overview_)
2. [SEC Sports - University of Florida](https://www.secsports.com/school/university-of-florida)
3. [Sports-Reference College Football](https://www.sports-reference.com/cfb/schools/florida/)
4. [Sports-Reference College Basketball](https://www.sports-reference.com/cbb/schools/florida/)
5. [Wikipedia - Florida Gators](https://en.wikipedia.org/wiki/Florida_Gators) (cross-check only)

**Note:** Lacrosse and any non-SEC sports are excluded.

## Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Open and run the Jupyter notebook: `UF_Athletics_Data_Cleaning.ipynb`

## Output Files

All cleaned data is saved in the `cleaned_data/` folder:
- **master_seasons_cleaned.csv**: Main dataset with cleaned and normalized data for all sports
- **sportsref_cfb_raw.csv**: Raw football data from Sports-Reference
- **sportsref_cbb_raw.csv**: Raw basketball data from Sports-Reference

## Usage

Simply run all cells in the notebook. The notebook will:
1. Install required packages if missing
2. Scrape data from the sources listed above
3. Clean and normalize the data
4. Save CSV files to the `cleaned_data/` folder

