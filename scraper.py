# ──────────────────────────────────────────────────────────────
#  scrape.py  —  ML-Optimized Selenium scraper for teacheron.com
#  pip install selenium beautifulsoup4
# ──────────────────────────────────────────────────────────────

import os, csv, time, random, re
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# ── CONFIG ─────────────────────────────────────────────────────

SUBJECTS = {
    "math":             "maths",
    "physics":          "physics",
    "chemistry":        "chemistry",
    "biology":          "biology",
    "english":          "english",
    "computer science": "computer_science",
}

PAGES_PER_SUBJECT = 10
OUTPUT_FILE       = "tutors_ml_dataset.csv"
BASE_URL          = "https://www.teacheron.com"
DEBUG_DIR         = "debug_html"

# Optimized CSV Fields for ML/EDA
CSV_FIELDS = [
    # Reference/Metadata (Drop these in your ML pipeline if not needed)
    "search_subject", "name", "url",
    
    # Geospatial Features
    "country",
    
    # Numerical Experience Features
    "total_experience", "online_experience",
    
    # Numerical Price Features (Standardized to USD)
    "price_min_usd", "price_max_usd", "price_avg_usd", "price_unit",
    
    # Categorical & NLP Features
    "num_subjects", "subjects_taught",
    "desc_length", "description"
]

# ── DRIVER ─────────────────────────────────────────────────────

def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": (
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            "window.chrome={runtime:{}};"
        )
    })
    return driver

# ── ML PARSING HELPERS ─────────────────────────────────────────

def parse_experience(text):
    """Extract numerical years from experience string."""
    if not text or text == "N/A": return 0.0
    match = re.search(r'([\d.]+)', text)
    return float(match.group(1)) if match else 0.0

def parse_price(text):
    """Extract USD min/max and billing period from price string."""
    if not text or text == "N/A": return 0.0, 0.0, "unknown"
    
    # Extract USD values
    usd_match = re.search(r'USD\s*([\d,.]+)\s*(?:-\s*([\d,.]+))?', text)
    min_usd, max_usd = 0.0, 0.0
    if usd_match:
        min_usd = float(usd_match.group(1).replace(',', ''))
        max_usd = float(usd_match.group(2).replace(',', '')) if usd_match.group(2) else min_usd
        
    # Extract period (hour, month, day)
    period_match = re.search(r'/(hour|month|day)', text, re.IGNORECASE)
    period = period_match.group(1).lower() if period_match else "unknown"
    
    return min_usd, max_usd, period

def parse_location(text):
    """Extract country from location string."""
    if not text or text == "N/A": return "unknown"
    parts = [p.strip() for p in text.split(',')]
    country = parts[-1] if parts else "unknown"
    return country

# ── EXTRACTION & SCRAPING ──────────────────────────────────────

def build_url(slug, page):
    return f"{BASE_URL}/tutors/{slug}-tutors?p={page}"

def save_debug_html(html, subject, page_num):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, f"{subject.replace(' ','_')}_p{page_num}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def extract_fields(card):
    """Extract raw strings from the HTML card."""
    name = card.find("span", {"itemprop": "name"}).get_text(strip=True) if card.find("span", {"itemprop": "name"}) else None
    
    url = None
    url_meta = card.find("meta", {"itemprop": "url"})
    if url_meta: url = url_meta.get("content")
    else:
        a_tag = card.find("a", target="_blank")
        if a_tag:
            href = a_tag.get("href")
            url = href if href and href.startswith("http") else BASE_URL + href if href else None

    subjects = []
    subject_tags = card.find("ul", class_=lambda c: c and "subjectTags" in c)
    if subject_tags:
        for a in subject_tags.find_all("a"): subjects.append(a.get_text(strip=True))
    subjects_str = ", ".join(subjects)

    desc_tag = card.find("p", class_="profile-description")
    description = desc_tag.get_text(strip=True) if desc_tag else None

    location = None
    loc_icon = card.find("i", class_=lambda c: c and "fa-map-marker" in c)
    if loc_icon:
        li = loc_icon.find_parent("li")
        if li: location = li.get("data-original-title") or li.get_text(strip=True)

    price = None
    for li in card.find_all("li", class_="tooltips"):
        title = li.get("data-original-title", "")
        if "USD" in title or "INR" in title or "month" in title.lower() or "hour" in title.lower():
            price = title or li.get_text(strip=True)
            break

    online_exp, total_exp = None, None
    for li in card.find_all("li", class_="tooltips"):
        title_attr = li.get("data-original-title", "").lower()
        if "online teaching experience" in title_attr: online_exp = li.get("data-original-title")
        if "total teaching experience" in title_attr: total_exp = li.get("data-original-title")

    return name, subjects_str, location, price, online_exp, total_exp, url, description

def find_job_cards(container):
    cards = container.find_all("div", class_=lambda c: c and "inner-results" in c)
    if cards: return cards, "div.inner-results"
    return container.find_all("div", recursive=False), "direct child divs"

def scrape_page(html, subject, page_num, is_first_page):
    rows = []
    soup = BeautifulSoup(html, "html.parser")

    container = soup.find(id="tutorOrJobSearchItemList")
    if not container:
        first_card = soup.find("div", class_=lambda c: c and "inner-results" in c)
        if first_card: container = first_card.parent
        else: return rows

    cards, _ = find_job_cards(container)
    if not cards: return rows

    sep = "─" * 76
    for card in cards:
        name, subjects_taught, location, price, online_exp, total_exp, link, description = extract_fields(card)
        if not name and not link: continue

        # ── FEATURE ENGINEERING FOR ML ──────────────────────────
        total_exp_num = parse_experience(total_exp)
        online_exp_num = parse_experience(online_exp)
        
        min_usd, max_usd, price_unit = parse_price(price)
        avg_usd = round((min_usd + max_usd) / 2, 2) if max_usd > 0 else 0.0
        
        country = parse_location(location)
        
        subjects_list = [s.strip() for s in subjects_taught.split(',')] if subjects_taught and subjects_taught != "N/A" else []
        num_subjects = len(subjects_list)
        
        desc_len = len(description) if description and description != "N/A" else 0

        row = {
            "search_subject": subject, "name": name or "N/A", "url": link or "N/A",
            "country": country,
            "total_experience": total_exp_num, "online_experience": online_exp_num,
            "price_min_usd": min_usd, "price_max_usd": max_usd, "price_avg_usd": avg_usd, "price_unit": price_unit,
            "num_subjects": num_subjects, "subjects_taught": subjects_taught or "N/A",
            "desc_length": desc_len, "description": description or "N/A"
        }
        rows.append(row)
        
        if is_first_page and len(rows) <= 3: # Print first 3 for verification
            print(f"  {sep}\n  NAME: {row['name']} | EXP: {row['total_experience']}yr | AVG ${row['price_avg_usd']}/{row['price_unit']} | {row['country']}")

    print(f"  ✓ Processed {len(rows)} records on page {page_num}.")
    return rows

def scrape_subject(driver, subject, slug, num_pages):
    rows = []
    print(f"\n{'='*80}\n  Subject: {subject.upper()}\n{'='*80}")
    for page_num in range(1, num_pages + 1):
        url = build_url(slug, page_num)
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.inner-results")))
            time.sleep(random.uniform(1.0, 2.0))
            html = driver.page_source
        except Exception:
            continue
        
        if page_num == 1: save_debug_html(html, subject, page_num)
        rows.extend(scrape_page(html, subject, page_num, is_first_page=(page_num == 1)))
        time.sleep(random.uniform(2.0, 4.0))
    return rows

def main():
    print("\n  Launching ML-Optimized Scraper …")
    driver = make_driver()
    all_rows = []
    try:
        for subject, slug in SUBJECTS.items():
            all_rows.extend(scrape_subject(driver, subject, slug, PAGES_PER_SUBJECT))
    finally:
        driver.quit()

    if all_rows:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  ✅ SUCCESS! Saved {len(all_rows)} ML-ready records → {OUTPUT_FILE}")
    else:
        print("  ❌ No records scraped.")

if __name__ == "__main__":
    main()