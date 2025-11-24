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
    'wikipedia_baseball': 'https://en.wikipedia.org/wiki/List_of_Florida_Gators_baseball_seasons',
    'wikipedia_softball': 'https://en.wikipedia.org/wiki/List_of_Florida_Gators_softball_seasons'
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

def normalize_season_to_yryr(s):
    """Convert season to 2-digit yr-yr format (e.g., 2024 -> 24-25)"""
    s = str(s).strip()
    
    # If already in 2-digit yr-yr format (25-26), return as is
    m_2digit = re.match(r'(\d{2})-(\d{2})$', s)
    if m_2digit:
        return s
    
    # If in 4-digit yr-yr format (2024-25 or 2024-2025), convert to 2-digit
    m_4digit = re.match(r'(\d{4})-(\d{2,4})', s)
    if m_4digit:
        year1 = int(m_4digit.group(1))
        year2_str = m_4digit.group(2)
        if len(year2_str) == 2:
            # Already 2-digit second year, just convert first
            return f'{year1 % 100:02d}-{year2_str}'
        elif len(year2_str) == 4:
            # Both 4-digit, convert both
            year2 = int(year2_str)
            return f'{year1 % 100:02d}-{year2 % 100:02d}'
    
    # If just a 4-digit year, convert to 2-digit yr-yr format
    # For baseball, the year refers to the spring season, so 2017 = 2016-2017 academic year = 16-17
    m_year = re.match(r'(\d{4})$', s)
    if m_year:
        year = int(m_year.group(1))
        # For spring sports, year 2017 = 2016-2017 academic year = 16-17
        prev_year = (year - 1) % 100
        return f'{prev_year:02d}-{year % 100:02d}'
    
    # If just a 2-digit year, assume it's the first year
    m_2year = re.match(r'(\d{2})$', s)
    if m_2year:
        year = int(m_2year.group(1))
        next_year = (year + 1) % 100
        return f'{year:02d}-{next_year:02d}'
    
    return s

def clean_sportsref(df, sport):
    """Clean data and return: year, wins, losses, win%, national_championship"""
    if df.empty:
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'win%', 'national_championship'])
    df = df.copy()
    
    # Flatten multi-level column names if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    
    # Initialize wins and losses columns
    df['wins'] = pd.NA
    df['losses'] = pd.NA
    
    # Try to find and populate wins column (case-insensitive search)
    # Look for 'w' or 'wins' - prefer the first 'w' column (overall wins, not conference wins)
    wins_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        # Match 'w' exactly or 'wins', but not 'w.1' (which is conference wins)
        if col_lower == 'w' or col_lower == 'wins':
            wins_col = c
            break
    
    if wins_col is not None:
        # Simple direct assignment
        df['wins'] = pd.to_numeric(df[wins_col], errors='coerce')
    else:
        print(f"Warning: Could not find wins column. Available columns: {list(df.columns)}")
        df['wins'] = pd.NA
    
    # Try to find and populate losses column (case-insensitive search)
    # Look for 'l' or 'losses' - prefer the first 'l' column
    losses_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        # Match 'l' exactly or 'losses', but not 'l.1' (which is conference losses)
        if col_lower == 'l' or col_lower == 'losses':
            losses_col = c
            break
    
    if losses_col is not None:
        # Simple direct assignment
        df['losses'] = pd.to_numeric(df[losses_col], errors='coerce')
    else:
        print(f"Warning: Could not find losses column. Available columns: {list(df.columns)}")
        df['losses'] = pd.NA
    
    # Handle season column (case-insensitive search)
    season_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower in ['season', 'year', 'yr']:
            season_col = c
            break
    
    if season_col:
        df['year'] = df[season_col].astype(str)
    else:
        # Try to use index if it looks like a year
        df['year'] = df.index.astype(str)
    
    # Normalize year to 2-digit yr-yr format (e.g., 24-25)
    df['year'] = df['year'].apply(normalize_season_to_yryr)
    
    # Calculate win percentage
    df['win%'] = df.apply(
        lambda row: (float(row['wins']) / (float(row['wins']) + float(row['losses'])) * 100) 
        if pd.notna(row['wins']) and pd.notna(row['losses']) and pd.notna(row['wins']) and pd.notna(row['losses']) and (float(row['wins']) + float(row['losses'])) > 0 
        else pd.NA, 
        axis=1
    )
    
    # Convert to numeric and round win% to 3 decimal places
    df['win%'] = pd.to_numeric(df['win%'], errors='coerce')
    df['win%'] = df['win%'].round(3)
    
    # Check for championship indicators in all columns
    notes_cols = [c for c in df.columns if any(term in str(c).lower() for term in ['note', 'tourney', 'champ', 'ncaa', 'result', 'postseason'])]
    df['national_championship'] = 'no'
    for c in notes_cols:
        if c in df.columns:
            # Check if this row indicates a national championship
            champ_mask = df[c].astype(str).str.contains('champ|national.*champion|won.*ncaa|ncaa.*champion', case=False, na=False, regex=True)
            df.loc[champ_mask, 'national_championship'] = 'yes'
    
    # Filter out rows with no valid wins/losses data
    df = df[df['wins'].notna() & df['losses'].notna()].copy()
    
    # Only return if we have data
    if df.empty:
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'win%', 'national_championship'])
    
    # Sort by year in descending order (most recent first: 25-26, 24-25, etc.)
    def year_to_sort_key(yr_str):
        """Convert 25-26 to 2025 for sorting"""
        parts = str(yr_str).split('-')
        if len(parts) == 2:
            first = int(parts[0])
            # Handle century: 00-25 = 2000-2025, 26-99 = 1926-1999
            if first < 26:
                return 2000 + first
            else:
                return 1900 + first
        return 0
    
    df['_sort_key'] = df['year'].apply(year_to_sort_key)
    df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1).reset_index(drop=True)
    
    # Return columns in order: year, wins, losses, win%, national_championship
    return df[['year', 'wins', 'losses', 'win%', 'national_championship']]

def clean_softball_wikipedia(df):
    """Clean softball data from Wikipedia and return: year, wins, losses, ties, win%, national_championship"""
    if df.empty:
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'ties', 'win%', 'national_championship'])
    df = df.copy()
    
    # Remove header row if it exists (row where season column contains "Florida Gators")
    if 'season' in df.columns:
        initial_len = len(df)
        df = df[df['season'].astype(str) != 'Florida Gators'].copy()
        df = df[~df['season'].astype(str).str.contains('Florida Gators', case=False, na=False)].copy()
        # Also remove if wins column has "Florida Gators"
        if 'wins' in df.columns:
            df = df[df['wins'].astype(str) != 'Florida Gators'].copy()
            df = df[~df['wins'].astype(str).str.contains('Florida Gators', case=False, na=False)].copy()
        print(f"  Removed header rows: {initial_len} -> {len(df)}")
    
    # Flatten multi-level column names if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    
    # Handle duplicate column names by making them unique
    # This can happen when pandas reads HTML tables with duplicate headers
    if df.columns.duplicated().any():
        # Rename duplicates by appending .1, .2, etc.
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
    
    # Find wins, losses, ties columns - softball uses 'overall', 'overall.1', 'overall.2'
    wins_col = None
    losses_col = None
    ties_col = None
    
    # Look for overall columns (softball structure)
    if 'overall' in df.columns:
        wins_col = 'overall'
    if 'overall.1' in df.columns:
        losses_col = 'overall.1'
    if 'overall.2' in df.columns:
        ties_col = 'overall.2'
    
    # Fallback: look for standard wins/losses/ties columns
    if wins_col is None:
        for c in df.columns:
            if str(c).lower().strip() == 'wins' or (str(c).lower().strip() == 'w' and 'win' not in str(c).lower()):
                wins_col = c
                break
    if losses_col is None:
        for c in df.columns:
            if str(c).lower().strip() == 'losses' or str(c).lower().strip() == 'l':
                losses_col = c
                break
    if ties_col is None:
        for c in df.columns:
            if str(c).lower().strip() in ['ties', 'tie', 't']:
                ties_col = c
                break
    
    # Filter out rows with "No games played" BEFORE processing
    before_filter = len(df)
    if wins_col is not None and wins_col in df.columns:
        try:
            mask = ~df[wins_col].astype(str).str.contains('No games played', case=False, na=False)
            df = df[mask].copy()
        except Exception as e:
            print(f"  Warning: Error filtering wins column: {e}")
    if losses_col is not None and losses_col in df.columns:
        try:
            mask = ~df[losses_col].astype(str).str.contains('No games played', case=False, na=False)
            df = df[mask].copy()
        except Exception as e:
            print(f"  Warning: Error filtering losses column: {e}")
    print(f"  After filtering 'No games played': {before_filter} -> {len(df)}")
    
    # Set year column and filter valid years
    if 'season' in df.columns:
        df['year'] = df['season'].astype(str)
    elif 'year' in df.columns:
        df['year'] = df['year'].astype(str)
    else:
        df['year'] = df.index.astype(str)
    
    # Remove rows with invalid years (not 4-digit numbers)
    before_year_filter = len(df)
    df = df[df['year'].astype(str).str.match(r'^\d{4}$', na=False)].copy()
    print(f"  After filtering valid years: {before_year_filter} -> {len(df)}")
    
    # Convert to numeric using the found columns
    # Handle potential duplicate column names
    if wins_col is not None and wins_col in df.columns:
        try:
            col_data = df[wins_col]
            if isinstance(col_data, pd.DataFrame):
                # Multiple columns with same name - take first column
                df['wins'] = pd.to_numeric(col_data.iloc[:, 0], errors='coerce')
            else:
                # Single column
                df['wins'] = pd.to_numeric(col_data, errors='coerce')
        except Exception as e:
            print(f"  Warning: Error accessing wins column: {e}")
            df['wins'] = pd.NA
    else:
        df['wins'] = pd.NA
        print(f"  Warning: Could not find wins column. Available: {list(df.columns)[:5]}")
    
    if losses_col is not None and losses_col in df.columns:
        try:
            col_data = df[losses_col]
            if isinstance(col_data, pd.DataFrame):
                df['losses'] = pd.to_numeric(col_data.iloc[:, 0], errors='coerce')
            else:
                df['losses'] = pd.to_numeric(col_data, errors='coerce')
        except Exception as e:
            print(f"  Warning: Error accessing losses column: {e}")
            df['losses'] = pd.NA
    else:
        df['losses'] = pd.NA
        print(f"  Warning: Could not find losses column. Available: {list(df.columns)[:5]}")
    
    if ties_col is not None and ties_col in df.columns:
        try:
            col_data = df[ties_col]
            if isinstance(col_data, pd.DataFrame):
                df['ties'] = pd.to_numeric(col_data.iloc[:, 0], errors='coerce').fillna(0)
            else:
                df['ties'] = pd.to_numeric(col_data, errors='coerce').fillna(0)
        except Exception as e:
            print(f"  Warning: Error accessing ties column: {e}")
            df['ties'] = 0
    else:
        df['ties'] = 0
    
    # Remove duplicate years - keep the row with the most games played (wins + losses + ties)
    # But first, preserve the original season column for championship detection
    if 'season' in df.columns:
        df['_original_season'] = df['season'].astype(str)
    
    # Normalize year first
    df['year'] = df['year'].apply(normalize_season_to_yryr)
    
    # Calculate total games and remove duplicates
    df['_total_games'] = df['wins'].fillna(0) + df['losses'].fillna(0) + df['ties'].fillna(0)
    # Sort by total games descending, then remove duplicates keeping the first (most games)
    df = df.sort_values('_total_games', ascending=False, na_position='last')
    before_dup_removal = len(df)
    df = df.drop_duplicates(subset=['year'], keep='first').copy()
    df = df.drop('_total_games', axis=1)
    print(f"  After removing duplicates: {before_dup_removal} -> {len(df)} rows")
    
    # Calculate win percentage (wins / (wins + losses + ties) * 100)
    df['win%'] = df.apply(
        lambda row: (float(row['wins']) / (float(row['wins']) + float(row['losses']) + float(row['ties'])) * 100) 
        if pd.notna(row['wins']) and pd.notna(row['losses']) and (float(row['wins']) + float(row['losses']) + float(row['ties'])) > 0 
        else pd.NA, 
        axis=1
    )
    
    # Convert to numeric and round
    df['win%'] = pd.to_numeric(df['win%'], errors='coerce')
    df['win%'] = df['win%'].round(3)
    
    # Check for national championship - UF softball won in 2014 and 2015
    df['national_championship'] = 'no'
    
    # Mark ONLY 2014 and 2015 championships
    if '_original_season' in df.columns:
        df.loc[df['_original_season'].isin(['2014', '2015']), 'national_championship'] = 'yes'
        df = df.drop('_original_season', axis=1)
    elif 'season' in df.columns:
        df.loc[df['season'].astype(str).isin(['2014', '2015']), 'national_championship'] = 'yes'
    
    # Filter out rows with no valid wins/losses data (should already be done, but double-check)
    before_final_filter = len(df)
    df = df[df['wins'].notna() & df['losses'].notna()].copy()
    print(f"  After final wins/losses filter: {before_final_filter} -> {len(df)}")
    
    if df.empty:
        print("  Warning: Softball dataframe is empty after all filtering")
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'ties', 'win%', 'national_championship'])
    
    # Ensure all required columns exist
    if 'ties' not in df.columns:
        df['ties'] = 0
    if 'win%' not in df.columns:
        df['win%'] = pd.NA
    if 'national_championship' not in df.columns:
        df['national_championship'] = 'no'
    
    # Sort by year in descending order (most recent first: 25-26, 24-25, etc.)
    def year_to_sort_key(yr_str):
        """Convert 25-26 to 2025 for sorting"""
        parts = str(yr_str).split('-')
        if len(parts) == 2:
            first = int(parts[0])
            # Handle century: 00-25 = 2000-2025, 26-99 = 1926-1999
            if first < 26:
                return 2000 + first
            else:
                return 1900 + first
        return 0
    
    if not df.empty:
        df['_sort_key'] = df['year'].apply(year_to_sort_key)
        df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1).reset_index(drop=True)
    
    # Return columns in order: year, wins, losses, ties, win%, national_championship
    return df[['year', 'wins', 'losses', 'ties', 'win%', 'national_championship']]

def clean_baseball_wikipedia(df):
    """Clean baseball data from Wikipedia and return: year, wins, losses, ties, win%, national_championship"""
    if df.empty:
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'ties', 'win%', 'national_championship'])
    df = df.copy()
    
    # Remove header row if it exists (row where season column contains "Florida Gators")
    # Also check if wins column has "Florida Gators" as first value
    if 'season' in df.columns:
        initial_len = len(df)
        # Remove rows where season is "Florida Gators"
        df = df[df['season'].astype(str) != 'Florida Gators'].copy()
        df = df[~df['season'].astype(str).str.contains('Florida Gators', case=False, na=False)].copy()
        # Also remove if wins column has "Florida Gators"
        if 'wins' in df.columns:
            df = df[df['wins'].astype(str) != 'Florida Gators'].copy()
            df = df[~df['wins'].astype(str).str.contains('Florida Gators', case=False, na=False)].copy()
        print(f"  Removed header rows: {initial_len} -> {len(df)}")
    
    # Flatten multi-level column names if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    
    # Find wins column
    wins_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower == 'wins':
            wins_col = c
            break
    
    # Find losses column
    losses_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower == 'losses':
            losses_col = c
            break
    
    # Find ties column
    ties_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower in ['ties', 'tie', 't']:
            ties_col = c
            break
    
    # Find year/season column
    season_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower in ['season', 'year']:
            season_col = c
            break
    
    # Filter out rows with "No games played" BEFORE processing
    before_filter = len(df)
    if wins_col is not None and wins_col in df.columns:
        df = df[~df[wins_col].astype(str).str.contains('No games played', case=False, na=False)].copy()
    if losses_col is not None and losses_col in df.columns:
        df = df[~df[losses_col].astype(str).str.contains('No games played', case=False, na=False)].copy()
    print(f"  After filtering 'No games played': {before_filter} -> {len(df)}")
    
    # Note: We'll convert to numeric AFTER filtering by year to avoid index issues
    
    # Set year column and filter valid years BEFORE converting to numeric
    if season_col:
        df['year'] = df[season_col].astype(str)
    else:
        df['year'] = df.index.astype(str)
    
    # Remove rows with invalid years (not 4-digit numbers)
    before_year_filter = len(df)
    df = df[df['year'].astype(str).str.match(r'^\d{4}$', na=False)].copy()
    print(f"  After filtering valid years: {before_year_filter} -> {len(df)}")
    
    # Convert to numeric
    if 'wins' in df.columns:
        df['wins'] = pd.to_numeric(df['wins'], errors='coerce')
    if 'losses' in df.columns:
        df['losses'] = pd.to_numeric(df['losses'], errors='coerce')
    if 'ties' in df.columns:
        df['ties'] = pd.to_numeric(df['ties'], errors='coerce').fillna(0)
    else:
        df['ties'] = 0
    
    # Remove duplicate years - keep the row with the most games played (wins + losses + ties)
    # This should be the overall season record, not a subset
    # But first, preserve the original season column for championship detection
    if season_col and season_col in df.columns:
        df['_original_season'] = df[season_col].astype(str)
    
    # Normalize year first
    df['year'] = df['year'].apply(normalize_season_to_yryr)
    
    # Calculate total games and remove duplicates
    df['_total_games'] = df['wins'].fillna(0) + df['losses'].fillna(0) + df['ties'].fillna(0)
    # Sort by total games descending, then remove duplicates keeping the first (most games)
    df = df.sort_values('_total_games', ascending=False, na_position='last')
    before_dup_removal = len(df)
    df = df.drop_duplicates(subset=['year'], keep='first').copy()
    df = df.drop('_total_games', axis=1)
    print(f"  After removing duplicates: {before_dup_removal} -> {len(df)} rows")
    
    # Note: Conversion to numeric happens after duplicate removal now
    
    # Year normalization already done above before duplicate removal
    
    # Calculate win percentage (wins / (wins + losses + ties) * 100)
    df['win%'] = df.apply(
        lambda row: (float(row['wins']) / (float(row['wins']) + float(row['losses']) + float(row['ties'])) * 100) 
        if pd.notna(row['wins']) and pd.notna(row['losses']) and (float(row['wins']) + float(row['losses']) + float(row['ties'])) > 0 
        else pd.NA, 
        axis=1
    )
    
    # Convert to numeric and round
    df['win%'] = pd.to_numeric(df['win%'], errors='coerce')
    df['win%'] = df['win%'].round(3)
    
    # Check for national championship - UF baseball won ONLY in 2017
    # The 2017 season corresponds to year 16-17 (2016-2017 academic year, spring 2017)
    # Initialize all to 'no'
    df['national_championship'] = 'no'
    
    # Mark championships - use the preserved original season column
    if '_original_season' in df.columns:
        # For baseball: only 2017
        if 'baseball' in str(sport).lower() if 'sport' in locals() else True:
            df.loc[df['_original_season'] == '2017', 'national_championship'] = 'yes'
        df = df.drop('_original_season', axis=1)
    elif season_col and season_col in df.columns:
        # For baseball: only 2017
        if 'baseball' in str(sport).lower() if 'sport' in locals() else True:
            df.loc[df[season_col].astype(str) == '2017', 'national_championship'] = 'yes'
    else:
        # Fallback: mark based on wins/losses match for 2017 (52 wins, 19 losses)
        # This is the actual 2017 record for baseball
        df.loc[(df['wins'] == 52) & (df['losses'] == 19), 'national_championship'] = 'yes'
    
    # Filter out rows with no valid wins/losses data (should already be done, but double-check)
    before_final_filter = len(df)
    df = df[df['wins'].notna() & df['losses'].notna()].copy()
    print(f"  After final wins/losses filter: {before_final_filter} -> {len(df)}")
    
    if df.empty:
        print("  Warning: Baseball dataframe is empty after all filtering")
        return pd.DataFrame(columns=['year', 'wins', 'losses', 'ties', 'win%', 'national_championship'])
    
    # Ensure all required columns exist
    if 'ties' not in df.columns:
        df['ties'] = 0
    if 'win%' not in df.columns:
        df['win%'] = pd.NA
    if 'national_championship' not in df.columns:
        df['national_championship'] = 'no'
    
    # Sort by year in descending order (most recent first: 25-26, 24-25, etc.)
    # Convert year to sortable format for proper sorting
    def year_to_sort_key(yr_str):
        """Convert 25-26 to 2025 for sorting"""
        parts = str(yr_str).split('-')
        if len(parts) == 2:
            first = int(parts[0])
            # Handle century: 00-25 = 2000-2025, 26-99 = 1926-1999
            if first < 26:
                return 2000 + first
            else:
                return 1900 + first
        return 0
    
    if not df.empty:
        df['_sort_key'] = df['year'].apply(year_to_sort_key)
        df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1).reset_index(drop=True)
    
    # Return columns in order: year, wins, losses, ties, win%, national_championship
    return df[['year', 'wins', 'losses', 'ties', 'win%', 'national_championship']]

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
print('Cleaning and formatting data...')
print('='*60)

# Clean and save football data
# Re-read from CSV to ensure clean structure
cfb_raw_clean = pd.read_csv(f'{OUTPUT_DIR}/sportsref_cfb_raw.csv')
cleaned_cfb = clean_sportsref(cfb_raw_clean, 'football')
if not cleaned_cfb.empty:
    cleaned_cfb.to_csv(f'{OUTPUT_DIR}/football.csv', index=False)
    print(f'\n✓ Football: {len(cleaned_cfb)} seasons saved')
    print('\nFirst 10 rows:')
    print(cleaned_cfb.head(10))
    print(f'\nNational Championships: {(cleaned_cfb["national_championship"] == "yes").sum()}')
else:
    print('⚠ No football data extracted')

print('\n' + '='*60 + '\n')

# Clean and save basketball data
# Re-read from CSV to ensure clean structure
cbb_raw_clean = pd.read_csv(f'{OUTPUT_DIR}/sportsref_cbb_raw.csv')
cleaned_cbb = clean_sportsref(cbb_raw_clean, 'basketball')
if not cleaned_cbb.empty:
    cleaned_cbb.to_csv(f'{OUTPUT_DIR}/basketball.csv', index=False)
    print(f'✓ Basketball: {len(cleaned_cbb)} seasons saved')
    print('\nFirst 10 rows:')
    print(cleaned_cbb.head(10))
    print(f'\nNational Championships: {(cleaned_cbb["national_championship"] == "yes").sum()}')
else:
    print('⚠ No basketball data extracted')

# Quality check report
print('\n' + '='*60)
print('DATA QUALITY REPORT')
print('='*60)

if not cleaned_cfb.empty:
    print(f'\n📊 Football:')
    print(f'  Total seasons: {len(cleaned_cfb)}')
    print(f'  Date range: {cleaned_cfb["year"].min()} → {cleaned_cfb["year"].max()}')
    print(f'  Valid win% records: {cleaned_cfb["win%"].notna().sum()}')
    print(f'  National Championships: {(cleaned_cfb["national_championship"] == "yes").sum()}')

if not cleaned_cbb.empty:
    print(f'\n📊 Basketball:')
    print(f'  Total seasons: {len(cleaned_cbb)}')
    print(f'  Date range: {cleaned_cbb["year"].min()} → {cleaned_cbb["year"].max()}')
    print(f'  Valid win% records: {cleaned_cbb["win%"].notna().sum()}')
    print(f'  National Championships: {(cleaned_cbb["national_championship"] == "yes").sum()}')

print('\n' + '='*60 + '\n')

# Scrape and clean baseball data from Wikipedia
print('Scraping Baseball from Wikipedia...')
print('='*60)
baseball_dfs = read_html_tables(sources['wikipedia_baseball'])

if baseball_dfs:
    # Find the main table (usually the largest one)
    baseball_raw = extract_main_table(baseball_dfs)
    if not baseball_raw.empty:
        baseball_raw.to_csv(f'{OUTPUT_DIR}/wikipedia_baseball_raw.csv', index=False)
        print(f'✓ Baseball raw: {len(baseball_raw)} rows saved')
        
        # Clean baseball data directly from the extracted dataframe
        cleaned_baseball = clean_baseball_wikipedia(baseball_raw)
        if not cleaned_baseball.empty:
            cleaned_baseball.to_csv(f'{OUTPUT_DIR}/baseball.csv', index=False)
            print(f'\n✓ Baseball: {len(cleaned_baseball)} seasons saved')
            print('\nFirst 10 rows:')
            print(cleaned_baseball.head(10))
            print(f'\nNational Championships: {(cleaned_baseball["national_championship"] == "yes").sum()}')
            
            if not cleaned_baseball.empty:
                print(f'\n📊 Baseball:')
                print(f'  Total seasons: {len(cleaned_baseball)}')
                print(f'  Date range: {cleaned_baseball["year"].min()} → {cleaned_baseball["year"].max()}')
                print(f'  Valid win% records: {cleaned_baseball["win%"].notna().sum()}')
                print(f'  National Championships: {(cleaned_baseball["national_championship"] == "yes").sum()}')
        else:
            print('⚠ No baseball data extracted after cleaning')
    else:
        print('⚠ No baseball table found')
else:
    print('⚠ Could not scrape baseball data from Wikipedia')

print('\n' + '='*60 + '\n')

# Scrape and clean softball data from Wikipedia
print('Scraping Softball from Wikipedia...')
print('='*60)
softball_dfs = read_html_tables(sources['wikipedia_softball'])

if softball_dfs:
    # Find the main table (usually the largest one)
    softball_raw = extract_main_table(softball_dfs)
    if not softball_raw.empty:
        softball_raw.to_csv(f'{OUTPUT_DIR}/wikipedia_softball_raw.csv', index=False)
        print(f'✓ Softball raw: {len(softball_raw)} rows saved')
        
        # Clean softball data directly from the extracted dataframe
        cleaned_softball = clean_softball_wikipedia(softball_raw)
        if not cleaned_softball.empty:
            cleaned_softball.to_csv(f'{OUTPUT_DIR}/softball.csv', index=False)
            print(f'\n✓ Softball: {len(cleaned_softball)} seasons saved')
            print('\nFirst 10 rows:')
            print(cleaned_softball.head(10))
            print(f'\nNational Championships: {(cleaned_softball["national_championship"] == "yes").sum()}')
            
            if not cleaned_softball.empty:
                print(f'\n📊 Softball:')
                print(f'  Total seasons: {len(cleaned_softball)}')
                print(f'  Date range: {cleaned_softball["year"].min()} → {cleaned_softball["year"].max()}')
                print(f'  Valid win% records: {cleaned_softball["win%"].notna().sum()}')
                print(f'  National Championships: {(cleaned_softball["national_championship"] == "yes").sum()}')
        else:
            print('⚠ No softball data extracted after cleaning')
    else:
        print('⚠ No softball table found')
else:
    print('⚠ Could not scrape softball data from Wikipedia')

print('\n✅ Done! CSV files saved to cleaned_data/ folder')

