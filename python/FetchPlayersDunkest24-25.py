from playwright.sync_api import sync_playwright
from openpyxl import Workbook

WEEKS = 35
ROUNDS = 2

def parse_float(value):
    value = value.strip().replace(",", ".").replace("\u2212", "-")
    try:
        return float(value)
    except:
        return 0.0

def parse_row(cols, week, rnd):
    return [
        cols[0].inner_text().strip(),  # Player
        cols[1].inner_text().strip(),  # Pos
        cols[2].inner_text().strip(),  # Team
        parse_float(cols[3].inner_text()),  # FPT
        parse_float(cols[4].inner_text()),  # CR
        parse_float(cols[7].inner_text()),  # Minutes
        week,
        rnd
    ]

def scrape_data():
    wb = Workbook()

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()

        for week in range(1, WEEKS + 1):
            if week == 1:
                ws = wb.active
                ws.title = "Week 1"   # <-- rename first sheet properly
            else:
                ws = wb.create_sheet(title=f"Week {week}")

            ws.append(["Player", "Pos", "Team", "FPT", "CR", "Minutes", "Week", "Round"])

            for rnd in range(1, ROUNDS + 1):
                url = (
                    f"https://www.dunkest.com/en/euroleague/stats/players/table/season/2024-2025"
                    f"?season_id=17&mode=dunkest&stats_type=tot&weeks[]={week}&rounds[]={rnd}"
                    f"&teams[]=31&teams[]=32&teams[]=33&teams[]=34&teams[]=35&teams[]=36&teams[]=37"
                    f"&teams[]=38&teams[]=39&teams[]=40&teams[]=41&teams[]=42&teams[]=43&teams[]=44"
                    f"&teams[]=45&teams[]=47&teams[]=48&teams[]=60&positions[]=1&positions[]=2&positions[]=3"
                    f"&player_search=&min_cr=4&max_cr=35&sort_by=pdk&sort_order=desc&iframe=yes&noadv=yes"
                )

                page.goto(url)
                try:
                    page.wait_for_selector("table tbody tr", timeout=15000)
                except:
                    print(f"No data for week {week}, round {rnd}")
                    continue

                # Handle pagination
                while True:
                    rows = page.query_selector_all("table tbody tr")
                    for row in rows:
                        cols = row.query_selector_all("td")
                        if cols:
                            ws.append(parse_row(cols, week, rnd))

                    # Click "next page" if available
                    next_btn = page.query_selector("a.page-link:has-text('»')")
                    if not next_btn or "disabled" in (next_btn.get_attribute("class") or ""):
                        break
                    next_btn.click()
                    page.wait_for_timeout(1500)  # small delay to load new page

        browser.close()

    wb.save("players_data_dunkest_24-25.xlsx")
    print("Done. Data saved to players_data_dunkest_24-25.xlsx")

if __name__ == "__main__":
    scrape_data()


