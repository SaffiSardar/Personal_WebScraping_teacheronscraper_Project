# 📚 TeacherOn Tutor Profile Scraper (ML-Optimized)

A robust, Selenium-based web scraping tool that extracts tutor profiles from [TeacherOn](https://www.teacheron.com). Unlike basic scrapers, this tool is specifically engineered to generate a **clean, structured, and Machine Learning-ready dataset** (CSV) for Exploratory Data Analysis (EDA) and predictive modeling.

This project is perfect for data scientists and learners who want to practice real-world web scraping, dynamic DOM parsing, and automated feature engineering.

---

## 🚀 Features

* **Multi-Subject Scraping:** Automatically scrapes profiles for Math, Physics, Chemistry, Biology, English, and Computer Science.
* **Selenium-Powered:** Uses Headless Chrome with stealth anti-detection flags to bypass basic bot protections and render dynamic JavaScript content.
* **ML-Ready Feature Engineering:**
  * **Numerical Parsing:** Extracts raw floats from experience strings (e.g., `"4.0 years..."` → `4.0`).
  * **Price Standardization:** Parses complex pricing strings into `price_min_usd`, `price_max_usd`, `price_avg_usd`, and `price_unit` (hour/month/day).
  * **Geospatial Features:** Extracts the `country` and creates a binary `is_india` flag.
  * **NLP Metrics:** Calculates `num_subjects` and `desc_length` for quick quantitative text analysis.
* **Debugging Tools:** Automatically saves raw HTML and card structures to a `debug_html/` folder if scraping breaks.
* **CSV Export:** Outputs a highly structured dataset ready for Pandas, Scikit-Learn, or XGBoost.

---

## 🛠️ Tech Stack

* **Python 3**
* **Selenium** (Headless Chrome automation)
* **BeautifulSoup4** (HTML parsing and DOM traversal)
* **Regex** (Data extraction and feature engineering)

---

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/teacheron-ml-tutor-scraper.git
cd teacheron-ml-tutor-scraper
```

2. Install Python dependencies:
```bash
pip install selenium beautifulsoup4
```
*(Note: Modern Selenium (v4.6+) automatically manages ChromeDriver via Selenium Manager, so you don't need to download it manually.)*

---

## ▶️ Usage

Run the script:
```bash
python scrape.py
```

The script will launch a headless browser, iterate through the configured subjects, and print a summary of the extracted ML features to the terminal. 

Once finished, it generates **`tutors_ml_dataset.csv`** with the following optimized columns:

| Feature Type | Columns |
| :--- | :--- |
| **Metadata** | `search_subject`, `name`, `url` |
| **Geospatial** | `country`, `is_india` (Binary: 1/0) |
| **Numerical** | `total_experience`, `online_experience` |
| **Pricing (USD)** | `price_min_usd`, `price_max_usd`, `price_avg_usd`, `price_unit` |
| **NLP / Categorical** | `num_subjects`, `subjects_taught`, `desc_length`, `description` |

---

## ⚙️ How It Works (Feature Engineering)

1. **Stealth Browsing:** Launches Chrome with arguments like `--disable-blink-features=AutomationControlled` and injects JS to hide the `navigator.webdriver` property.
2. **DOM Traversal:** Locates tutor cards using the `div.inner-results` class and extracts raw text using BeautifulSoup.
3. **Data Transformation:** 
   * Uses **Regex** to isolate numerical values from messy UI strings (e.g., extracting `15.84` from `"USD 15.84 - 42.23/month (INR 1,500 - 4,000/month)"`).
   * Splits location strings to isolate the country and apply binary encoding.
4. **Graceful Fallbacks:** If a specific tooltip or icon is missing, the script falls back to alternative parsing strategies to prevent data loss.

---

## 🧩 Customization

**Change the subjects being scraped:**
Update the `SUBJECTS` dictionary at the top of `scrape.py`. The key is your display name, and the value is the URL slug.
```python
SUBJECTS = {
    "math":             "maths",
    "physics":          "physics",
    "data science":     "data-science",  # Add new subjects here
}
```

**Change the number of pages scraped per subject:**
```python
PAGES_PER_SUBJECT = 10  # Default is 3
```

---

## ⚠️ Disclaimer

This script is for **educational and data science portfolio purposes only**. 
Please respect the website’s Terms of Service and `robots.txt` before scraping in production environments. Do not use aggressive request rates that could burden the target server.

---

## 🌟 Future Improvements

* [ ] Integrate NLP pipelines (TF-IDF, Word2Vec, HuggingFace) to generate embeddings for the `description` column.
* [ ] Build a predictive ML model (e.g., XGBoost) to predict `price_avg_usd` based on experience and subjects.
* [ ] Implement proxy rotation and randomized delay systems for large-scale scraping.
* [ ] Create a Streamlit dashboard to visualize the EDA of the generated CSV.

---

## 👨‍💻 Author

Built with Python, Selenium, and Data Science curiosity by **Saffi** 🧠✨  
Happy scraping & modeling!
