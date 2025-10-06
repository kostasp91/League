from playwright.sync_api import sync_playwright
from openpyxl import Workbook, load_workbook
import os

# === CONFIGURATION ===
MODE = "full"  # "full" → rebuild from scratch, "update" → append new weeks
OUTPUT_FILE = "players_data_dunkest.xlsx"
WEEKS = 2  # how many total weeks exist right now
ROUNDS = 2  # how many rounds per week


# === HELPER FUNCTIONS ===
def parse_float(value):
    """Convert string to float, handling commas and minus signs."""
    value = value.strip().replace(",", ".").replace("\u2212", "-")
    try:
        return float(value)
    except:
        return 0.0


def parse_row(cols, week, rnd):
    """Extract player data from table row cells."""
    return [
        cols[0].inner_text().strip(),  # Player
        cols[1].inner_text().strip(),  # Pos
        cols[2].inner_text().strip(),  # Team
        parse_float(cols[3].inner_text()),  # FPT
        parse_float(cols[4].inner_text()),  # CR
        parse_float(cols[7].inner_text()),  # Minutes
        week,
        rnd,
    ]


# === MAIN SCRAPER FUNCTION ===
def scrape_data():
    # === MODE HANDLING ===
    if MODE.lower() == "full" or not os.path.exists(OUTPUT_FILE):
        print("Running in FULL mode — creating new file.")
        wb = Workbook()
        # remove default empty sheet
        default_sheet = wb.active
        wb.remove(default_sheet)
        start_week = 1
    elif MODE.lower() == "update":
        print("Running in UPDATE mode — adding missing weeks.")
        wb = load_workbook(OUTPUT_FILE)
        existing_sheets = wb.sheetnames
        last_week_num = 0
        for s in existing_sheets:
            if s.startswith("Week "):
                try:
                    week_num = int(s.split(" ")[1])
                    last_week_num = max(last_week_num, week_num)
                except:
                    pass
        start_week = last_week_num + 1
    else:
        raise ValueError(f"Invalid MODE: {MODE}. Use 'full' or 'update'.")

    print(f"Starting from Week {start_week}")

    # === SCRAPE LOOP ===
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()

        for week in range(start_week, WEEKS + 1):
            # create new sheet for the week
            ws = wb.create_sheet(title=f"Week {week}")
            ws.append(["Player", "Pos", "Team", "FPT", "CR", "Minutes", "Week", "Round"])

            for rnd in range(1, ROUNDS + 1):
                print(f"Fetching Week {week}, Round {rnd}...")
                url = (
                    f"https://www.dunkest.com/en/euroleague/stats/players/table?"
                    f"season_id=23&mode=dunkest&stats_type=tot&weeks[]={week}&rounds[]={rnd}"
                    f"&teams[]=32&teams[]=33&teams[]=34&teams[]=35&teams[]=36&teams[]=37"
                    f"&teams[]=38&teams[]=39&teams[]=40&teams[]=41&teams[]=42&teams[]=43"
                    f"&teams[]=44&teams[]=45&teams[]=46&teams[]=47&teams[]=48&teams[]=56"
                    f"&teams[]=60&teams[]=75&positions[]=1&positions[]=2&positions[]=3"
                    f"&player_search=&min_cr=4&max_cr=35&sort_by=pdk&sort_order=desc"
                    f"&iframe=yes&noadv=yes"
                )

                page.goto(url)
                page.wait_for_selector("table tbody tr")

                # scrape table pages
                while True:
                    rows = page.query_selector_all("table tbody tr")
                    for row in rows:
                        cols = row.query_selector_all("td")
                        if cols:
                            ws.append(parse_row(cols, week, rnd))

                    # next page check
                    next_btn = page.query_selector("a[href='']:has-text('»')")
                    if not next_btn or "disabled" in (next_btn.get_attribute("class") or ""):
                        break
                    next_btn.click()
                    page.wait_for_timeout(1000)

        browser.close()

    # === SAVE RESULTS ===
    wb.save(OUTPUT_FILE)
    print(f"Done. Data saved to {OUTPUT_FILE}")


# === ENTRY POINT ===
if __name__ == "__main__":
    scrape_data()
