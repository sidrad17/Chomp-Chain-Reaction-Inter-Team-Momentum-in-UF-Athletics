#!/usr/bin/env python3
#run the data scraping and cleaning to generate CSV files for all 9 UF sports

import os, re
import requests
import pandas as pd
from io import StringIO

output_dir = "files_and_interface"
os.makedirs(output_dir, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0 (compatible; UFDataCollector/1.0)"}

#source dictionary
sources = {
    "sportsref_cfb": "https://www.sports-reference.com/cfb/schools/florida/",
    "sportsref_cbb": "https://www.sports-reference.com/cbb/schools/florida/",
    "sportsref_wbb": "https://www.sports-reference.com/cbb/schools/florida/women/",
    "wikipedia_softball": "https://en.wikipedia.org/wiki/List_of_Florida_Gators_softball_seasons",
    "wikipedia_wtennis": "https://en.wikipedia.org/wiki/Florida_Gators_women%27s_tennis",
    "wikipedia_soccer": "https://en.wikipedia.org/wiki/Florida_Gators_women%27s_soccer",
    "wikipedia_volleyball": "https://en.wikipedia.org/wiki/Florida_Gators_women%27s_volleyball",
    "wikipedia_tennis": "https://en.wikipedia.org/wiki/Florida_Gators_men%27s_tennis",
    "wikipedia_baseball": "https://en.wikipedia.org/wiki/List_of_Florida_Gators_baseball_seasons"
}

#read tables from url
def read_html_tables(url, match=None, header=0):
    try:
        print(f"Trying pandas.read_html on {url}")
        kwargs = {"header": header}
        if match is not None:
            kwargs["match"] = match
        dfs = pd.read_html(url, **kwargs)
        return dfs
    except Exception as e:
        print("Primary read_html failed:", e)
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            html = resp.text
            kwargs = {"header": header}
            if match is not None:
                kwargs["match"] = match
            dfs = pd.read_html(StringIO(html), **kwargs)
            return dfs
        except Exception as e2:
            print("Fallback failed:", e2)
            return []

#extract main tables from url
def extract_main_table(dfs, prefer_season_table=False):
    if not dfs:
        print("Warning: No tables found")
        return pd.DataFrame()

    if prefer_season_table:
        for df in sorted(dfs, key=lambda x: x.shape[0], reverse=True):
            cols = [str(c).lower() for c in df.columns]
            if "season" in cols or "year" in cols:
                if df.shape[0] > 10:
                    return df

    df = max(dfs, key=lambda x: x.shape[0])
    if df.empty:
        print("Warning: Largest table is empty")
        return pd.DataFrame()

    first_row = df.iloc[0].astype(str).str.lower()
    if any(term in ' '.join(first_row.values) for term in ["year", "season", "rk"]):
        df.columns = df.iloc[0].astype(str)
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[~df.iloc[:, 0].astype(str).str.contains("rk|year|season", case=False, na=False)]
    df = df.dropna(how="all").reset_index(drop=True)

    return df

#convert any season to YYYY-YYYY format
def normalize_to_yyyy_yyyy(s):
    s = str(s).strip()
    s = s.replace("–", "-").replace("—", "-")

    m = re.match(r"(\d{4})-(\d{4})$", s)
    if m:
        return s

    m = re.match(r"(\d{4})-(\d{2})$", s)
    if m:
        year1 = int(m.group(1))
        year2_suffix = int(m.group(2))
        century = (year1 // 100) * 100
        year2 = century + year2_suffix
        if year2 < year1:
            year2 += 100
        return f"{year1}-{year2}"

    m = re.match(r"(\d{4})$", s)
    if m:
        year = int(m.group(1))
        return f'{year}-{year + 1}'

    m = re.match(r"(\d{2})-(\d{2})$", s)
    if m:
        year1_suffix = int(m.group(1))
        year2_suffix = int(m.group(2))

        if year1_suffix <= 25:
            century = 2000
        else:
            century = 1900

        year1 = century + year1_suffix
        year2 = century + year2_suffix

        if year2 < year1:
            year2 += 100

        return f"{year1}-{year2}"

    return s

#adjust year format for spring sport
def adjust_spring_sport_year(year_str):
    year_str = str(year_str).strip()
    m = re.match(r"(\d{4})-(\d{4})$", year_str)
    if m:
        year1 = int(m.group(1))
        year2 = int(m.group(2))
        return f"{year1 - 1}-{year2 - 1}"
    return year_str

#filter to only include seasons from 1925 onwards
def filter_last_100_years(df):
    if df.empty or "year" not in df.columns:
        return df

    def extract_start_year(yr_str):
        parts = str(yr_str).split("-")
        if len(parts) >= 1:
            return int(parts[0])
        return 0

    df = df.copy()
    df["_start_year"] = df["year"].apply(extract_start_year)

    before = len(df)
    df = df[df["_start_year"] >= 1925].copy()
    after = len(df)

    if before != after:
        print(f"  Filtered to 1925+: {before} -> {after} rows")

    df = df.drop("_start_year", axis=1)
    return df

#clean Sports-Reference data
def clean_sportsref(df, sport):
    if df.empty:
        return pd.DataFrame(columns=["year", "wins", "losses", "win_loss_pct", "national_championship"])
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns.values]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]

    wins_col = None
    for c in df.columns:
        if str(c).lower().strip() in ["w", "wins"]:
            wins_col = c
            break

    losses_col = None
    for c in df.columns:
        if str(c).lower().strip() in ["l", "losses"]:
            losses_col = c
            break

    season_col = None
    for c in df.columns:
        if str(c).lower().strip() in ["season", "year", "yr"]:
            season_col = c
            break

    if season_col:
        df["year"] = df[season_col].astype(str)
    else:
        df["year"] = df.index.astype(str)

    if wins_col:
        df["wins"] = pd.to_numeric(df[wins_col], errors="coerce")
    else:
        df["wins"] = pd.NA

    if losses_col:
        df["losses"] = pd.to_numeric(df[losses_col], errors="coerce")
    else:
        df["losses"] = pd.NA

    df = df[df["wins"].notna() & df["losses"].notna()].copy()

    df["year"] = df["year"].apply(normalize_to_yyyy_yyyy)

    df["_total"] = df["wins"] + df["losses"]
    df = df.sort_values("_total", ascending=False)
    before = len(df)
    df = df.drop_duplicates(subset=["year"], keep="first")
    df = df.drop("_total", axis=1)
    if before != len(df):
        print(f"  Removed {before - len(df)} duplicates")

    df["win_loss_pct"] = ((df["wins"] / (df["wins"] + df["losses"])) * 100).round(3)

    notes_cols = [c for c in df.columns if any(term in str(c).lower()
                                               for term in ["note", "tourney", "champ", "ncaa", "bowl"])]
    df["national_championship"] = "no"
    for c in notes_cols:
        if c in df.columns:
            mask = df[c].astype(str).str.contains(
                "won.*ncaa.*tournament.*national.*final|national.*champion|bcs.*championship.*\\(w\\)",
                case=False, na=False, regex=True)
            df.loc[mask, "national_championship"] = "yes"

    if df.empty:
        return pd.DataFrame(columns=["year", "wins", "losses", "win_loss_pct", "national_championship"])

    def year_sort_key(yr):
        yr_clean = str(yr).replace("–", "-").replace("—", "-")
        parts = yr_clean.split("-")
        if parts and parts[0].isdigit():
            return int(parts[0])
        return 0

    df["_sort"] = df["year"].apply(year_sort_key)
    df = df.sort_values("_sort", ascending=False).drop("_sort", axis=1).reset_index(drop=True)

    return df[["year", "wins", "losses", "win_loss_pct", "national_championship"]]

#clean Wikipedia data
def clean_wikipedia(df, sport_name, championship_years=None):
    if df.empty:
        return pd.DataFrame(columns=["year", "wins", "losses", "win_loss_pct", "national_championship"])
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in col).strip() for col in df.columns.values]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]

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

    print(f"  Available columns: {list(df.columns)[:10]}")

    year_col = None
    for c in df.columns:
        col_lower = str(c).lower().strip()
        if col_lower in ["season", "year"]:
            year_col = c
            break

    if year_col:
        df["year"] = df[year_col].astype(str)
    else:
        df["year"] = df.index.astype(str)

    wins_col = None
    losses_col = None

    if "overall" in df.columns:
        wins_col = "overall"
    elif "record_overall" in df.columns:
        wins_col = "record_overall"

    if "overall.1" in df.columns:
        losses_col = "overall.1"
    elif "record_overall.1" in df.columns:
        losses_col = "record_overall.1"

    if not wins_col:
        for c in df.columns:
            col_lower = str(c).lower().strip()
            if "win" in col_lower and "loss" not in col_lower and "%" not in col_lower:
                wins_col = c
                break

    if not losses_col:
        for c in df.columns:
            col_lower = str(c).lower().strip()
            if "loss" in col_lower and "%" not in col_lower:
                losses_col = c
                break

    if not wins_col or not losses_col:
        record_col = None
        for c in df.columns:
            col_lower = str(c).lower().strip()
            if col_lower in ["record", "w–l", "w-l", "overall", "overall record", "regular season record"]:
                record_col = c
                break

        if record_col and record_col in df.columns:
            print(f"  Found record column: {record_col}")
            df["_record"] = df[record_col].astype(str)

            df["wins"] = df["_record"].str.replace("–", "-").str.extract(r"(\d+)-", expand=False)
            df["losses"] = df["_record"].str.replace("–", "-").str.extract(r"-(\d+)", expand=False)

            df["wins"] = pd.to_numeric(df["wins"], errors="coerce")
            df["losses"] = pd.to_numeric(df["losses"], errors="coerce")

            df = df.drop("_record", axis=1)
    else:
        if wins_col and wins_col in df.columns:
            col_data = df[wins_col]
            if isinstance(col_data, pd.DataFrame):
                df["wins"] = pd.to_numeric(col_data.iloc[:, 0], errors="coerce")
            else:
                df["wins"] = pd.to_numeric(col_data, errors="coerce")
        else:
            df["wins"] = pd.NA

        if losses_col and losses_col in df.columns:
            col_data = df[losses_col]
            if isinstance(col_data, pd.DataFrame):
                df["losses"] = pd.to_numeric(col_data.iloc[:, 0], errors="coerce")
            else:
                df["losses"] = pd.to_numeric(col_data, errors="coerce")
        else:
            df["losses"] = pd.NA

    if "season" in df.columns or year_col:
        before = len(df)
        df = df[~df['year'].astype(str).str.contains(
            "Florida Gators|Season|Total|Year|Record", case=False, na=False)].copy()
        if before != len(df):
            print(f"  Removed header rows: {before} -> {len(df)}")

    before = len(df)
    df = df[df["year"].astype(str).str.match(r"^\d{2,4}", na=False)].copy()
    print(f"  After year filter: {before} -> {len(df)}")

    df["_original_year"] = df["year"].copy()

    df["year"] = df["year"].apply(normalize_to_yyyy_yyyy)

    if "wins" not in df.columns:
        df["wins"] = pd.NA
    if "losses" not in df.columns:
        df["losses"] = pd.NA

    df["_total"] = df["wins"].fillna(0) + df["losses"].fillna(0)
    df = df.sort_values("_total", ascending=False, na_position="last")
    before = len(df)
    df = df.drop_duplicates(subset=["year"], keep="first")
    df = df.drop("_total", axis=1)
    if before != len(df):
        print(f"  Removed duplicates: {before} -> {len(df)}")

    df["win_loss_pct"] = df.apply(
        lambda row: ((float(row["wins"]) / (float(row["wins"]) + float(row["losses"]))) * 100)
        if pd.notna(row["wins"]) and pd.notna(row["losses"]) and (float(row["wins"]) + float(row["losses"])) > 0
        else pd.NA,
        axis=1
    )
    df["win_loss_pct"] = pd.to_numeric(df["win_loss_pct"], errors="coerce")
    df["win_loss_pct"] = df["win_loss_pct"].round(3)

    #check for national championships in columns
    df["national_championship"] = "no"

    #look for columns that might contain championship info
    national_cols = [c for c in df.columns if
                     "national" in str(c).lower() or "final" in str(c).lower() or "ranking" in str(c).lower()]
    print(f"  Checking for championships in columns: {national_cols}")

    for col in national_cols:
        if col in df.columns:
            #check for "1st", "National Champion", etc.
            mask = df[col].astype(str).str.contains(
                r"1st|national\s+champion|ncaa\s+champion",
                case=False, na=False, regex=True)
            df.loc[mask, "national_championship"] = "yes"
            if mask.any():
                print(f"  Found championships in column '{col}': {mask.sum()} matches")

    #also check if specific years were provided
    if championship_years and "_original_year" in df.columns:
        champ_year_strs = [str(y) for y in championship_years]
        print(f"  Also checking for specific championship years: {champ_year_strs}")

        for idx, row in df.iterrows():
            orig_year = str(row["_original_year"]).strip()
            for champ_year in champ_year_strs:
                if champ_year in orig_year or orig_year == champ_year:
                    df.at[idx, "national_championship"] = "yes"
                    print(f"  Matched championship year: {orig_year}")
                    break

    print(f"  Total championships marked: {(df['national_championship'] == 'yes').sum()}")

    df = df.drop("_original_year", axis=1, errors="ignore")

    before = len(df)
    df = df[df["wins"].notna() & df["losses"].notna()].copy()
    print(f"  Final filter: {before} -> {len(df)}")

    if df.empty:
        print(f"  WARNING: {sport_name} has no data after cleaning")
        return pd.DataFrame(columns=["year", "wins", "losses", "win_loss_pct", "national_championship"])

    def year_sort_key(yr):
        yr_clean = str(yr).replace("–", "-").replace("—", "-")
        parts = yr_clean.split("-")
        if parts and parts[0].isdigit():
            return int(parts[0])
        return 0

    df["_sort"] = df["year"].apply(year_sort_key)
    df = df.sort_values("_sort", ascending=False).drop("_sort", axis=1).reset_index(drop=True)

    return df[["year", "wins", "losses", "win_loss_pct", "national_championship"]]


#main execution
print("=" * 80)
print("SCRAPING UF ATHLETICS DATA - 9 SPORTS")
print("=" * 80)

#scrape football
print("\n1. Football (Sports-Reference)")
print("=" * 80)
cfb_dfs = read_html_tables(sources["sportsref_cfb"])
cfb_raw = extract_main_table(cfb_dfs)
if not cfb_raw.empty:
    cfb_raw.to_csv(f"{output_dir}/sportsref_cfb_raw.csv", index=False)
    cfb_raw_clean = pd.read_csv(f"{output_dir}/sportsref_cfb_raw.csv")
    cleaned_cfb = clean_sportsref(cfb_raw_clean, "football")
    cleaned_cfb = filter_last_100_years(cleaned_cfb)
    if not cleaned_cfb.empty:
        cleaned_cfb.to_csv(f"{output_dir}/football.csv", index=False)
        print(f"  Football: {len(cleaned_cfb)} seasons | {cleaned_cfb["year"].min()} → {cleaned_cfb["year"].max()}")
        print(f"  Championships: {(cleaned_cfb["national_championship"] == "yes").sum()}")

#scrape basketball
print("\n2. Basketball (Sports-Reference)")
print("=" * 80)
cbb_dfs = read_html_tables(sources['sportsref_cbb'])
cbb_raw = extract_main_table(cbb_dfs)
if not cbb_raw.empty:
    cbb_raw.to_csv(f"{output_dir}/sportsref_cbb_raw.csv", index=False)
    cbb_raw_clean = pd.read_csv(f"{output_dir}/sportsref_cbb_raw.csv")
    cleaned_cbb = clean_sportsref(cbb_raw_clean, "basketball")
    cleaned_cbb = filter_last_100_years(cleaned_cbb)
    if not cleaned_cbb.empty:
        cleaned_cbb.to_csv(f'{output_dir}/basketball.csv', index=False)
        print(f"  Basketball: {len(cleaned_cbb)} seasons | {cleaned_cbb["year"].min()} → {cleaned_cbb["year"].max()}")
        print(f"  Championships: {(cleaned_cbb["national_championship"] == "yes").sum()}")

#scrape women's basketball
print("\n3. Women\'s Basketball (Sports-Reference)")
print("=" * 80)
wbb_dfs = read_html_tables(sources['sportsref_wbb'])
wbb_raw = extract_main_table(wbb_dfs)
if not wbb_raw.empty:
    wbb_raw.to_csv(f"{output_dir}/sportsref_wbb_raw.csv", index=False)
    wbb_raw_clean = pd.read_csv(f"{output_dir}/sportsref_wbb_raw.csv")
    cleaned_wbb = clean_sportsref(wbb_raw_clean, "womens_basketball")
    cleaned_wbb = filter_last_100_years(cleaned_wbb)
    if not cleaned_wbb.empty:
        cleaned_wbb.to_csv(f'{output_dir}/womens_basketball.csv', index=False)
        print(
            f"  Women\'s Basketball: {len(cleaned_wbb)} seasons | {cleaned_wbb["year"].min()} → {cleaned_wbb["year"].max()}")
        print(f"  Championships: {(cleaned_wbb["national_championship"] == "yes").sum()}")

#scrape softball
print("\n4. Softball (Wikipedia)")
print("=" * 80)
softball_dfs = read_html_tables(sources["wikipedia_softball"])
if softball_dfs:
    softball_raw = extract_main_table(softball_dfs)
    if not softball_raw.empty:
        softball_raw.to_csv(f"{output_dir}/wikipedia_softball_raw.csv", index=False)
        cleaned_softball = clean_wikipedia(softball_raw, "softball", championship_years=[2014, 2015])
        cleaned_softball = filter_last_100_years(cleaned_softball)
        if not cleaned_softball.empty:
            #adjust years for spring sport (subtract 1 from both years)
            cleaned_softball["year"] = cleaned_softball["year"].apply(adjust_spring_sport_year)
            cleaned_softball.to_csv(f"{output_dir}/softball.csv", index=False)
            print(
                f"  Softball: {len(cleaned_softball)} seasons | {cleaned_softball["year"].min()} → {cleaned_softball["year"].max()}")
            print(f"  Championships: {(cleaned_softball["national_championship"] == "yes").sum()}")

#scrape women's tennis
print("\n5. Women\'s Tennis (Wikipedia)")
print("=" * 80)
wtennis_dfs = read_html_tables(sources["wikipedia_wtennis"])
if wtennis_dfs:
    wtennis_raw = extract_main_table(wtennis_dfs, prefer_season_table=True)
    if not wtennis_raw.empty:
        wtennis_raw.to_csv(f"{output_dir}/wikipedia_wtennis_raw.csv", index=False)
        #don't pass specific years - let it detect from "National Champion" in national column
        cleaned_wtennis = clean_wikipedia(wtennis_raw, "womens_tennis", championship_years=None)
        cleaned_wtennis = filter_last_100_years(cleaned_wtennis)
        if not cleaned_wtennis.empty:
            cleaned_wtennis.to_csv(f"{output_dir}/womens_tennis.csv", index=False)
            print(
                f"  Women\'s Tennis: {len(cleaned_wtennis)} seasons | {cleaned_wtennis["year"].min()} → {cleaned_wtennis["year"].max()}")
            print(f"  Championships: {(cleaned_wtennis["national_championship"] == "yes").sum()}")

#scrape soccer
print("\n6. Women\'s Soccer (Wikipedia)")
print("=" * 80)
soccer_dfs = read_html_tables(sources["wikipedia_soccer"])
if soccer_dfs:
    soccer_raw = extract_main_table(soccer_dfs, prefer_season_table=True)
    if not soccer_raw.empty:
        soccer_raw.to_csv(f"{output_dir}/wikipedia_soccer_raw.csv", index=False)
        #1998 national championship for soccer
        cleaned_soccer = clean_wikipedia(soccer_raw, "soccer", championship_years=[1998])
        cleaned_soccer = filter_last_100_years(cleaned_soccer)
        if not cleaned_soccer.empty:
            cleaned_soccer.to_csv(f"{output_dir}/soccer.csv", index=False)
            print(
                f"  Soccer: {len(cleaned_soccer)} seasons | {cleaned_soccer["year"].min()} → {cleaned_soccer["year"].max()}")
            print(f"  Championships: {(cleaned_soccer["national_championship"] == "yes").sum()}")

#scrape volleyball
print("\n7. Women\'s Volleyball (Wikipedia)")
print("=" * 80)
volleyball_dfs = read_html_tables(sources['wikipedia_volleyball'])
if volleyball_dfs:
    volleyball_raw = extract_main_table(volleyball_dfs, prefer_season_table=True)
    if not volleyball_raw.empty:
        volleyball_raw.to_csv(f"{output_dir}/wikipedia_volleyball_raw.csv", index=False)
        #no national championships for women's volleyball
        cleaned_volleyball = clean_wikipedia(volleyball_raw, "volleyball", championship_years=[])
        cleaned_volleyball = filter_last_100_years(cleaned_volleyball)
        if not cleaned_volleyball.empty:
            cleaned_volleyball.to_csv(f"{output_dir}/volleyball.csv", index=False)
            print(
                f"  Volleyball: {len(cleaned_volleyball)} seasons | {cleaned_volleyball["year"].min()} → {cleaned_volleyball["year"].max()}")
            print(f"  Championships: {(cleaned_volleyball["national_championship"] == "yes").sum()}")

#scrape tennis
print("\n8. Men\'s Tennis (Wikipedia)")
print("=" * 80)
tennis_dfs = read_html_tables(sources["wikipedia_tennis"])
if tennis_dfs:
    tennis_raw = extract_main_table(tennis_dfs, prefer_season_table=True)
    if not tennis_raw.empty:
        tennis_raw.to_csv(f"{output_dir}/wikipedia_tennis_raw.csv", index=False)
        #don't pass specific years - let it detect from "1st" in national ranking column
        cleaned_tennis = clean_wikipedia(tennis_raw, "tennis", championship_years=None)
        cleaned_tennis = filter_last_100_years(cleaned_tennis)
        if not cleaned_tennis.empty:
            cleaned_tennis.to_csv(f"{output_dir}/tennis.csv", index=False)
            print(
                f"  Tennis: {len(cleaned_tennis)} seasons | {cleaned_tennis["year"].min()} → {cleaned_tennis["year"].max()}")
            print(f"  Championships: {(cleaned_tennis["national_championship"] == "yes").sum()}")

#scrape baseball
print("\n9. Baseball (Wikipedia)")
print("=" * 80)
baseball_dfs = read_html_tables(sources["wikipedia_baseball"])
if baseball_dfs:
    baseball_raw = extract_main_table(baseball_dfs)
    if not baseball_raw.empty:
        baseball_raw.to_csv(f"{output_dir}/wikipedia_baseball_raw.csv", index=False)
        cleaned_baseball = clean_wikipedia(baseball_raw, "baseball", championship_years=[2017])
        cleaned_baseball = filter_last_100_years(cleaned_baseball)
        if not cleaned_baseball.empty:
            #adjust years for spring sport (subtract 1 from both years)
            cleaned_baseball["year"] = cleaned_baseball["year"].apply(adjust_spring_sport_year)
            cleaned_baseball.to_csv(f'{output_dir}/baseball.csv', index=False)
            print(
                f"  Baseball: {len(cleaned_baseball)} seasons | {cleaned_baseball["year"].min()} → {cleaned_baseball["year"].max()}")
            print(f"  Championships: {(cleaned_baseball["national_championship"] == "yes").sum()}")

#print summary of scraped data
print("\n" + "=" * 80)
print("FINAL SUMMARY - ALL SPORTS")
print("=" * 80)

sports = [
    ("football.csv", "Football"),
    ("basketball.csv", "Basketball"),
    ("womens_basketball.csv", "Women's Basketball"),
    ("softball.csv", "Softball"),
    ("womens_tennis.csv", "Women's Tennis"),
    ("soccer.csv", "Women's Soccer"),
    ("volleyball.csv", "Women's Volleyball"),
    ("tennis.csv", "Men's Tennis"),
    ("baseball.csv", "Baseball"),
]

for filename, name in sports:
    path = f"{output_dir}/{filename}"
    if os.path.exists(path):
        df = pd.read_csv(path)
        champs = (df["national_championship"] == "yes").sum() if not df.empty else 0
        print(f"{name:25} | {len(df):3} seasons | {champs} championships")
    else:
        print(f"{name:25} | FILE NOT FOUND")

#print format of outputted csv files
print("\nFormat: year (YYYY-YYYY), wins, losses, win_loss_pct, national_championship")