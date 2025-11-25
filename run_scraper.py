#!/usr/bin/env python3
"""Run the data scraping and cleaning to generate CSV files"""

import os, re, time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

OUTPUT_DIR = 'cleaned_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; UFDataCollector/1.0)'}

sources = {
    'sportsref_cfb': 'https://www.sports-reference.com/cfb/schools/florida/',
    'sportsref_cbb': 'https://www.sports-reference.com/cbb/schools/florida/',
    'sportsref_wbb': 'https://www.sports-reference.com/cbb/schools/florida/women/',
    'wikipedia_baseball': 'https://en.wikipedia.org/wiki/List_of_Florida_Gators_baseball_seasons',
    'wikipedia_softball': 'https://en.wikipedia.org/wiki/List_of_Florida_Gators_softball_seasons',
    'wikipedia_tennis': 'https://en.wikipedia.org/wiki/Florida_Gators_men%27s_tennis',
    'wikipedia_wtennis': 'https://en.wikipedia.org/wiki/Florida_Gators_women%27s_tennis',
    'wikipedia_soccer': 'https://en.wikipedia.org/wiki/Florida_Gators_women%27s_soccer',
    'wikipedia_volleyball': 'https://en.wikipedia.org/wiki/Florida_Gators_women%27s_volleyball'
}

def read_html_tables(url, match=None, header=0):
    try:
        print(f'Trying pandas.read_html on {url}')
        kwargs = {'header': header}
        if match is not None:
            kwargs['match'] = match
        dfs = pd.read_html(url, **kwargs)
        return dfs
    except Exception as e:
        print('Primary read_html failed:', e)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            html = resp.text
            kwargs = {'header': header}
            if match is not None:
                kwargs['match'] = match
            dfs = pd.read_html(StringIO(html), **kwargs)
            return dfs
        except Exception as e2:
            print('Fallback failed:', e2)
            return []

def extract_main_table(dfs):
    if not dfs: 
        print('Warning: No tables found')
        return pd.DataFrame()
    df = max(dfs, key=lambda x: x.shape[0])
    if df.empty:
        print('Warning: Largest table is empty')
        return pd.DataFrame()
    
    first_row = df.iloc[0].astype(str).str.lower()
    if any(term in ' '.join(first_row.values) for term in ['year', 'season', 'rk']):
        df.columns = df.iloc[0].astype(str)
        df = df.iloc[1:].reset_index(drop=True)
    
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[~df.iloc[:, 0].astype(str).str.contains('rk|year|season', case=False, na=False)]
    df = df.dropna(how='all').reset_index(drop=True)
    
    return df

def normalize_season_to_4digit(s):
    """Convert season to 4-digit year format (e.g., 2024-25 -> 2025)"""
    s = str(s).strip()
    
    # If in format like 2024-25 or 2024-2025, return the second year
    m = re.match(r'(\d{4})-(\d{2,4})', s)
    if m:
        year2_str = m.group(2)
        if len(year2_str) == 2:
            # 2024-25 -> 2025
            year2 = int(year2_str)
            # Handle century
            if year2 < 26:
                return str(2000 + year2)
            else:
                return str(1900 + year2)
        else:
            # 2024-2025 -> 2025
            return year2_str
    
    # If just a 4-digit year, return as is
    m = re.match(r'(\d{4})$', s)
    if m:
        return s
    
    # If in format like 24-25, convert to 2025
    m = re.match(r'(\d{2})-(\d{2})', s)
    if m:
        year2 = int(m.group(2))
        if year2 < 26:
            return str(2000 + year2)
        else:
            return str(1900 + year2)
    
    return s

def clean_sportsref(df, sport):
    """Clean data and return: year, wins, losses, win%, national_championship"""
    if df.empty:
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'win%', 'national_championship'])
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    
    df['wins'] = pd.NA
    df['losses'] = pd.NA
    
    # Find wins column - look for first 'w' or 'wins' column only
    wins_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower == 'w' or col_lower == 'wins':
            wins_col = c
            break
    
    if wins_col is not None:
        df['wins'] = pd.to_numeric(df[wins_col], errors='coerce')
    else:
        print(f"Warning: Could not find wins column. Available columns: {list(df.columns)}")
        df['wins'] = pd.NA
    
    # Find losses column - look for first 'l' or 'losses' column only
    losses_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower == 'l' or col_lower == 'losses':
            losses_col = c
            break
    
    if losses_col is not None:
        df['losses'] = pd.to_numeric(df[losses_col], errors='coerce')
    else:
        print(f"Warning: Could not find losses column. Available columns: {list(df.columns)}")
        df['losses'] = pd.NA
    
    # Find season/year column
    season_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower in ['season', 'year', 'yr']:
            season_col = c
            break
    
    if season_col:
        df['year'] = df[season_col].astype(str)
    else:
        df['year'] = df.index.astype(str)
    
    # Store original year for duplicate detection
    df['_original_year'] = df['year'].copy()
    
    # Normalize year to 4-digit format
    df['year'] = df['year'].apply(normalize_season_to_4digit)
    
    # Remove duplicates - keep the first occurrence (usually the main season record)
    before_dup = len(df)
    df = df.drop_duplicates(subset=['year'], keep='first').copy()
    print(f"  Removed {before_dup - len(df)} duplicate years")
    
    # Calculate win percentage
    df['win%'] = df.apply(
        lambda row: (float(row['wins']) / (float(row['wins']) + float(row['losses'])) * 100) 
        if pd.notna(row['wins']) and pd.notna(row['losses']) and (float(row['wins']) + float(row['losses'])) > 0 
        else pd.NA, 
        axis=1
    )
    
    df['win%'] = pd.to_numeric(df['win%'], errors='coerce')
    df['win%'] = df['win%'].round(3)
    
    # Check for championships
    notes_cols = [c for c in df.columns if any(term in str(c).lower() for term in ['note', 'tourney', 'champ', 'ncaa', 'result', 'postseason'])]
    df['national_championship'] = 'no'
    for c in notes_cols:
        if c in df.columns:
            champ_mask = df[c].astype(str).str.contains('won.*ncaa.*tournament.*national.*final|national.*champion', case=False, na=False, regex=True)
            df.loc[champ_mask, 'national_championship'] = 'yes'
    
    # Filter out rows with no valid wins/losses
    df = df[df['wins'].notna() & df['losses'].notna()].copy()
    
    if df.empty:
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'win%', 'national_championship'])
    
    # Sort by year descending (most recent first)
    df['_sort_year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.sort_values('_sort_year', ascending=False).drop(['_sort_year', '_original_year'], axis=1).reset_index(drop=True)
    
    return df[['year', 'wins', 'losses', 'win%', 'national_championship']]

def clean_wikipedia_season_table(df, sport_name, championship_years=None):
    """Clean Wikipedia data - returns: year, wins, losses, win%, national_championship"""
    if df.empty:
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'win%', 'national_championship'])
    df = df.copy()
    
    # Remove header rows
    if 'season' in df.columns:
        initial_len = len(df)
        df = df[df['season'].astype(str) != 'Florida Gators'].copy()
        df = df[~df['season'].astype(str).str.contains('Florida Gators|Season|Total', case=False, na=False)].copy()
        if 'wins' in df.columns:
            df = df[df['wins'].astype(str) != 'Florida Gators'].copy()
        print(f"  Removed header rows: {initial_len} -> {len(df)}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    
    # Handle duplicate columns
    if df.columns.duplicated().any():
        new_cols = []
        seen = {}
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
    
    # Find columns
    wins_col = None
    losses_col = None
    
    # Look for overall columns first (softball/soccer/volleyball structure)
    if 'overall' in df.columns:
        wins_col = 'overall'
    if 'overall.1' in df.columns:
        losses_col = 'overall.1'
    
    # Fallback to standard columns
    if wins_col is None:
        for c in df.columns:
            col_lower = str(c).lower().strip()
            if col_lower in ['wins', 'w']:
                wins_col = c
                break
    if losses_col is None:
        for c in df.columns:
            col_lower = str(c).lower().strip()
            if col_lower in ['losses', 'l']:
                losses_col = c
                break
    
    # Filter out "No games played" and cancelled seasons
    before_filter = len(df)
    if wins_col and wins_col in df.columns:
        df = df[~df[wins_col].astype(str).str.contains('No games played|canceled|cancelled', case=False, na=False)].copy()
    if losses_col and losses_col in df.columns:
        df = df[~df[losses_col].astype(str).str.contains('No games played|canceled|cancelled', case=False, na=False)].copy()
    print(f"  After filtering 'No games played': {before_filter} -> {len(df)}")
    
    # Set year column
    if 'season' in df.columns:
        df['year'] = df['season'].astype(str)
    elif 'year' in df.columns:
        df['year'] = df['year'].astype(str)
    else:
        df['year'] = df.index.astype(str)
    
    # Filter valid years (4-digit numbers only)
    before_year_filter = len(df)
    df = df[df['year'].astype(str).str.match(r'^\d{4}$', na=False)].copy()
    print(f"  After filtering valid years: {before_year_filter} -> {len(df)}")
    
    # Convert to numeric
    if wins_col and wins_col in df.columns:
        col_data = df[wins_col]
        if isinstance(col_data, pd.DataFrame):
            df['wins'] = pd.to_numeric(col_data.iloc[:, 0], errors='coerce')
        else:
            df['wins'] = pd.to_numeric(col_data, errors='coerce')
    else:
        df['wins'] = pd.NA
        print(f"  Warning: Could not find wins column")
    
    if losses_col and losses_col in df.columns:
        col_data = df[losses_col]
        if isinstance(col_data, pd.DataFrame):
            df['losses'] = pd.to_numeric(col_data.iloc[:, 0], errors='coerce')
        else:
            df['losses'] = pd.to_numeric(col_data, errors='coerce')
    else:
        df['losses'] = pd.NA
        print(f"  Warning: Could not find losses column")
    
    # Preserve original season for championship detection
    df['_original_season'] = df['year'].copy()
    
    # Remove duplicates - keep row with most games
    df['_total_games'] = df['wins'].fillna(0) + df['losses'].fillna(0)
    df = df.sort_values('_total_games', ascending=False, na_position='last')
    before_dup_removal = len(df)
    df = df.drop_duplicates(subset=['year'], keep='first').copy()
    df = df.drop('_total_games', axis=1)
    print(f"  After removing duplicates: {before_dup_removal} -> {len(df)} rows")
    
    # Calculate win percentage
    df['win%'] = df.apply(
        lambda row: (float(row['wins']) / (float(row['wins']) + float(row['losses'])) * 100) 
        if pd.notna(row['wins']) and pd.notna(row['losses']) and (float(row['wins']) + float(row['losses'])) > 0 
        else pd.NA, 
        axis=1
    )
    
    df['win%'] = pd.to_numeric(df['win%'], errors='coerce')
    df['win%'] = df['win%'].round(3)
    
    # Mark championships
    df['national_championship'] = 'no'
    if championship_years:
        df.loc[df['_original_season'].isin([str(y) for y in championship_years]), 'national_championship'] = 'yes'
    df = df.drop('_original_season', axis=1)
    
    # Final filter
    before_final_filter = len(df)
    df = df[df['wins'].notna() & df['losses'].notna()].copy()
    print(f"  After final wins/losses filter: {before_final_filter} -> {len(df)}")
    
    if df.empty:
        print(f"  Warning: {sport_name} dataframe is empty after all filtering")
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'win%', 'national_championship'])
    
    # Sort by year descending
    df['_sort_year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.sort_values('_sort_year', ascending=False).drop('_sort_year', axis=1).reset_index(drop=True)
    
    return df[['year', 'wins', 'losses', 'win%', 'national_championship']]

# Main execution
print('='*60)
print('Scraping Football and Basketball from Sports-Reference...')
print('='*60)

cfb_dfs = read_html_tables(sources['sportsref_cfb'])
cbb_dfs = read_html_tables(sources['sportsref_cbb'])

cfb_raw = extract_main_table(cfb_dfs)
cbb_raw = extract_main_table(cbb_dfs)

if not cfb_raw.empty:
    cfb_raw.to_csv(f'{OUTPUT_DIR}/sportsref_cfb_raw.csv', index=False)
    print(f'✓ Football raw: {len(cfb_raw)} rows saved')
if not cbb_raw.empty:
    cbb_raw.to_csv(f'{OUTPUT_DIR}/sportsref_cbb_raw.csv', index=False)
    print(f'✓ Basketball raw: {len(cbb_raw)} rows saved')

print('\n' + '='*60)
print('Cleaning Football and Basketball...')
print('='*60)

cfb_raw_clean = pd.read_csv(f'{OUTPUT_DIR}/sportsref_cfb_raw.csv')
cleaned_cfb = clean_sportsref(cfb_raw_clean, 'football')
if not cleaned_cfb.empty:
    cleaned_cfb.to_csv(f'{OUTPUT_DIR}/football.csv', index=False)
    print(f'\n✓ Football: {len(cleaned_cfb)} seasons saved')
    print(f'  Date range: {cleaned_cfb["year"].min()} → {cleaned_cfb["year"].max()}')
    print(f'  National Championships: {(cleaned_cfb["national_championship"] == "yes").sum()}')

cbb_raw_clean = pd.read_csv(f'{OUTPUT_DIR}/sportsref_cbb_raw.csv')
cleaned_cbb = clean_sportsref(cbb_raw_clean, 'basketball')
if not cleaned_cbb.empty:
    cleaned_cbb.to_csv(f'{OUTPUT_DIR}/basketball.csv', index=False)
    print(f'\n✓ Basketball: {len(cleaned_cbb)} seasons saved')
    print(f'  Date range: {cleaned_cbb["year"].min()} → {cleaned_cbb["year"].max()}')
    print(f'  National Championships: {(cleaned_cbb["national_championship"] == "yes").sum()}')

print('\n' + '='*60)
print("Scraping Women's Basketball from Sports-Reference...")
print('='*60)

wbb_dfs = read_html_tables(sources['sportsref_wbb'])
wbb_raw = extract_main_table(wbb_dfs)

if not wbb_raw.empty:
    wbb_raw.to_csv(f'{OUTPUT_DIR}/sportsref_wbb_raw.csv', index=False)
    print(f'✓ Women\'s Basketball raw: {len(wbb_raw)} rows saved')
    
    wbb_raw_clean = pd.read_csv(f'{OUTPUT_DIR}/sportsref_wbb_raw.csv')
    cleaned_wbb = clean_sportsref(wbb_raw_clean, 'womens_basketball')
    if not cleaned_wbb.empty:
        cleaned_wbb.to_csv(f'{OUTPUT_DIR}/womens_basketball.csv', index=False)
        print(f'\n✓ Women\'s Basketball: {len(cleaned_wbb)} seasons saved')
        print(f'  Date range: {cleaned_wbb["year"].min()} → {cleaned_wbb["year"].max()}')
        print(f'  National Championships: {(cleaned_wbb["national_championship"] == "yes").sum()}')

print('\n' + '='*60)
print('Scraping Baseball from Wikipedia...')
print('='*60)

baseball_dfs = read_html_tables(sources['wikipedia_baseball'])
if baseball_dfs:
    baseball_raw = extract_main_table(baseball_dfs)
    if not baseball_raw.empty:
        baseball_raw.to_csv(f'{OUTPUT_DIR}/wikipedia_baseball_raw.csv', index=False)
        print(f'✓ Baseball raw: {len(baseball_raw)} rows saved')
        
        cleaned_baseball = clean_wikipedia_season_table(baseball_raw, 'baseball', championship_years=[2017])
        if not cleaned_baseball.empty:
            cleaned_baseball.to_csv(f'{OUTPUT_DIR}/baseball.csv', index=False)
            print(f'\n✓ Baseball: {len(cleaned_baseball)} seasons saved')
            print(f'  Date range: {cleaned_baseball["year"].min()} → {cleaned_baseball["year"].max()}')
            print(f'  National Championships: {(cleaned_baseball["national_championship"] == "yes").sum()}')

print('\n' + '='*60)
print('Scraping Softball from Wikipedia...')
print('='*60)

softball_dfs = read_html_tables(sources['wikipedia_softball'])
if softball_dfs:
    softball_raw = extract_main_table(softball_dfs)
    if not softball_raw.empty:
        softball_raw.to_csv(f'{OUTPUT_DIR}/wikipedia_softball_raw.csv', index=False)
        print(f'✓ Softball raw: {len(softball_raw)} rows saved')
        
        cleaned_softball = clean_wikipedia_season_table(softball_raw, 'softball', championship_years=[2014, 2015])
        if not cleaned_softball.empty:
            cleaned_softball.to_csv(f'{OUTPUT_DIR}/softball.csv', index=False)
            print(f'\n✓ Softball: {len(cleaned_softball)} seasons saved')
            print(f'  Date range: {cleaned_softball["year"].min()} → {cleaned_softball["year"].max()}')
            print(f'  National Championships: {(cleaned_softball["national_championship"] == "yes").sum()}')

print('\n' + '='*60)
print("Scraping Men's Tennis from Wikipedia...")
print('='*60)

tennis_dfs = read_html_tables(sources['wikipedia_tennis'])
if tennis_dfs:
    tennis_raw = extract_main_table(tennis_dfs)
    if not tennis_raw.empty:
        tennis_raw.to_csv(f'{OUTPUT_DIR}/wikipedia_tennis_raw.csv', index=False)
        print(f'✓ Tennis raw: {len(tennis_raw)} rows saved')
        
        # Men's tennis championships
        cleaned_tennis = clean_wikipedia_season_table(tennis_raw, 'tennis', 
            championship_years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021])
        if not cleaned_tennis.empty:
            cleaned_tennis.to_csv(f'{OUTPUT_DIR}/tennis.csv', index=False)
            print(f'\n✓ Tennis: {len(cleaned_tennis)} seasons saved')
            print(f'  Date range: {cleaned_tennis["year"].min()} → {cleaned_tennis["year"].max()}')
            print(f'  National Championships: {(cleaned_tennis["national_championship"] == "yes").sum()}')

print('\n' + '='*60)
print("Scraping Women's Tennis from Wikipedia...")
print('='*60)

wtennis_dfs = read_html_tables(sources['wikipedia_wtennis'])
if wtennis_dfs:
    wtennis_raw = extract_main_table(wtennis_dfs)
    if not wtennis_raw.empty:
        wtennis_raw.to_csv(f'{OUTPUT_DIR}/wikipedia_wtennis_raw.csv', index=False)
        print(f'✓ Women\'s Tennis raw: {len(wtennis_raw)} rows saved')
        
        # Women's tennis championships
        cleaned_wtennis = clean_wikipedia_season_table(wtennis_raw, "women's tennis",
            championship_years=[1992, 1994, 1996, 1998, 2003, 2011, 2012, 2017])
        if not cleaned_wtennis.empty:
            cleaned_wtennis.to_csv(f'{OUTPUT_DIR}/womens_tennis.csv', index=False)
            print(f'\n✓ Women\'s Tennis: {len(cleaned_wtennis)} seasons saved')
            print(f'  Date range: {cleaned_wtennis["year"].min()} → {cleaned_wtennis["year"].max()}')
            print(f'  National Championships: {(cleaned_wtennis["national_championship"] == "yes").sum()}')

print('\n' + '='*60)
print("Scraping Women's Soccer from Wikipedia...")
print('='*60)

soccer_dfs = read_html_tables(sources['wikipedia_soccer'])
if soccer_dfs:
    soccer_raw = extract_main_table(soccer_dfs)
    if not soccer_raw.empty:
        soccer_raw.to_csv(f'{OUTPUT_DIR}/wikipedia_soccer_raw.csv', index=False)
        print(f'✓ Soccer raw: {len(soccer_raw)} rows saved')
        
        cleaned_soccer = clean_wikipedia_season_table(soccer_raw, 'soccer', championship_years=[])
        if not cleaned_soccer.empty:
            cleaned_soccer.to_csv(f'{OUTPUT_DIR}/soccer.csv', index=False)
            print(f'\n✓ Soccer: {len(cleaned_soccer)} seasons saved')
            print(f'  Date range: {cleaned_soccer["year"].min()} → {cleaned_soccer["year"].max()}')
            print(f'  National Championships: {(cleaned_soccer["national_championship"] == "yes").sum()}')

print('\n' + '='*60)
print("Scraping Women's Volleyball from Wikipedia...")
print('='*60)

volleyball_dfs = read_html_tables(sources['wikipedia_volleyball'])
if volleyball_dfs:
    volleyball_raw = extract_main_table(volleyball_dfs)
    if not volleyball_raw.empty:
        volleyball_raw.to_csv(f'{OUTPUT_DIR}/wikipedia_volleyball_raw.csv', index=False)
        print(f'✓ Volleyball raw: {len(volleyball_raw)} rows saved')
        
        cleaned_volleyball = clean_wikipedia_season_table(volleyball_raw, 'volleyball', championship_years=[])
        if not cleaned_volleyball.empty:
            cleaned_volleyball.to_csv(f'{OUTPUT_DIR}/volleyball.csv', index=False)
            print(f'\n✓ Volleyball: {len(cleaned_volleyball)} seasons saved')
            print(f'  Date range: {cleaned_volleyball["year"].min()} → {cleaned_volleyball["year"].max()}')
            print(f'  National Championships: {(cleaned_volleyball["national_championship"] == "yes").sum()}')

print('\n' + '='*60)
print('FINAL DATA SUMMARY')
print('='*60)

sports_list = [
    ('football', 'Football'),
    ('basketball', 'Basketball'),
    ('womens_basketball', "Women's Basketball"),
    ('baseball', 'Baseball'),
    ('softball', 'Softball'),
    ('tennis', "Men's Tennis"),
    ('womens_tennis', "Women's Tennis"),
    ('soccer', "Women's Soccer"),
    ('volleyball', "Women's Volleyball")
]

for filename, sport_name in sports_list:
    csv_path = f'{OUTPUT_DIR}/{filename}.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f'\n📊 {sport_name}:')
        print(f'  Total seasons: {len(df)}')
        print(f'  Columns: {list(df.columns)}')
        if not df.empty:
            print(f'  Date range: {df["year"].min()} → {df["year"].max()}')
            print(f'  National Championships: {(df["national_championship"] == "yes").sum()}')

print('\n✅ Done! All CSV files saved to cleaned_data/ folder')