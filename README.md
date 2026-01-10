# 📚 TeacherOn Math Tutor Job Scraper

A Python web scraping script that extracts **online Math tutoring job listings** from [TeacherOn](https://www.teacheron.com), displaying job title, teaching mode, price range, and job link directly in the terminal.

This tool is designed for learners who want to practice real-world web scraping with `requests` and `BeautifulSoup`.

---

## 🚀 Features

* Scrapes **multiple pages automatically**.
* Filters only **Math-related jobs**.
* Extracts:

  * Job title
  * Teaching mode (Online / Offline)
  * Price range
  * Job URL
* Displays clean, readable output in the terminal.
* Uses browser-like headers to avoid simple bot blocking.

---

## 🛠️ Tech Stack

* Python 3
* requests
* BeautifulSoup (bs4)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/teacheron-math-job-scraper.git
cd teacheron-math-job-scraper
```

Install dependencies:

```bash
pip install requests beautifulsoup4
```

---

## ▶️ Usage

Run the script:

```bash
python scraper.py
```

It will scrape the first **3 pages** of online tutoring jobs and print results like:

```
--------------------------------------------------------
JOB       -->  Looking for Online Math Tutor
MODE      -->  Online
PRICE     -->  $10-15/hr
VISIT     -->  https://www.teacheron.com/...
```

---

## ⚙️ How It Works

* Iterates through multiple pages of TeacherOn job listings.
* Finds all listings containing the keyword **“math”**.
* Traverses HTML structure to extract:

  * Title
  * Mode of teaching
  * Price tooltip
  * Job link
* Gracefully handles missing data using exception handling.

---

## 🧩 Customization

Change number of pages scraped:

```python
for i in range(1, 4):
```

For 10 pages:

```python
for i in range(1, 11):
```

Change keyword from **math** to anything else:

```python
math_jobs = results.find_all("span", string=lambda text: "physics" in text.lower())
```

---

## ⚠️ Disclaimer

This script is for **educational purposes only**.
Respect the website’s terms of service and robots.txt before scraping in production environments.

---

## 🌟 Future Improvements

* Save data to CSV or Excel
* Add pagination auto-detection
* Add CLI filters for subject keywords
* Proxy rotation and delay system

---

## 👨‍💻 Author

Built with Python and curiosity by **Saffi_Mac** 🧠✨
Happy scraping!
