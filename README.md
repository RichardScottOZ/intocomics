# IntoComics
#### <em>Your guide to getting into comic books</em>
---

## Comic Recommender Engine

A content-based comic recommender engine that uses manifest data from [Comic-Analysis](https://github.com/RichardScottOZ/Comic-Analysis) to recommend new comics from platforms like [Neon Ichiban](https://neonichiban.com/) and [GlobalComix](https://globalcomix.com/).

### How It Works

1. **Load your library** from a Comic-Analysis manifest CSV (canonical_id + image paths) or provide a list of titles you like
2. **Scrape candidates** from comic platforms (Neon Ichiban, GlobalComix)
3. **Get recommendations** ranked by content similarity using text embeddings

### Quick Start

```python
from src.intocomics.manifest import load_manifest
from src.intocomics.recommender import ComicRecommender
from src.intocomics.scrapers import ScrapedComic
from src.intocomics.scrapers.neonichiban import NeonIchibanScraper
from src.intocomics.scrapers.globalcomix import GlobalComixScraper

# Option 1: Load from Comic-Analysis manifest
comics = load_manifest("path/to/master_manifest.csv")

# Option 2: Just provide titles you like
recommender = ComicRecommender()
recommender.load_liked_titles([
    "Batman Dark Knight Returns",
    "Saga",
    "Sandman",
])

# Scrape candidates from comic platforms
scraper = NeonIchibanScraper()
candidates = scraper.scrape_listings(max_items=50)

# Get recommendations
results = recommender.recommend(candidates, top_k=5)
for rec in results:
    print(f"{rec.title} (score: {rec.score:.3f}) - {rec.url}")
```

### Installation

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
python -m pytest tests/ -v
```

### Architecture

```
src/intocomics/
├── __init__.py          # Package init
├── manifest.py          # Load Comic-Analysis manifest CSVs
├── embeddings.py        # Text & image embedding generation
├── recommender.py       # Content-based recommendation engine
└── scrapers/
    ├── __init__.py      # Base scraper class & ScrapedComic dataclass
    ├── neonichiban.py   # Neon Ichiban scraper
    └── globalcomix.py   # GlobalComix scraper
```

### Manifest Format

The recommender reads the same manifest CSV format used by [Comic-Analysis](https://github.com/RichardScottOZ/Comic-Analysis):

```csv
canonical_id,absolute_image_path
NeonIchiban/MyComic/page_0000,/path/to/extracted/NeonIchiban/MyComic/page_0000.png
NeonIchiban/MyComic/page_0001,/path/to/extracted/NeonIchiban/MyComic/page_0001.png
```

Pages are grouped by comic (from the canonical_id path) and the first page is used as the cover image.

### Embedding Strategy

The recommender supports multiple embedding backends with graceful fallbacks:

- **sentence-transformers** (best quality) → TF-IDF → bag-of-words for text
- **CLIP** (best quality) → pixel features for images

The fallback chain ensures the recommender works without GPU or heavy ML dependencies.

---

## Legacy: Original IntoComics (ALS Recommender)

The original project used Amazon review data and ALS collaborative filtering to recommend comics from movie/TV preferences.

* [Amazon Review data exploratory data analysis](https://github.com/jnawjux/intocomics/blob/master/amazon_reviews_eda.ipynb)
* [Data preparation and modeling](https://github.com/jnawjux/intocomics/blob/master/data_prep_and_modeling.ipynb)
* [Web app development](https://github.com/jnawjux/intocomics/blob/master/web_app_development.ipynb) 
* [Presentation (Google Slides)](https://docs.google.com/presentation/d/17ZCj6XF-yz0qAAhKozr-Nzvy8R7UQNkzM-Hzz_ehBs0/edit?usp=sharing)

### Business Understanding
A passion of mine is comic books. As much as I love the medium, the industry at large is often looked over in favor of the next big movie or tv show using their stories or characters. Further, there are a number of great stories and worlds done by smaller artists/writers that are waiting for a chance to shine. My goal is to create a recommendation system where people will be able to match their movie and television preferences to comic books, helping open up a new world of entertainment for them. 

### Data Understanding
To build my model, I used a large repository of Amazon reviews previously collected in a research project at the University of California, San Diego (~24GB of book and movies/tv reviews from 1996-2014, [more information here](http://jmcauley.ucsd.edu/data/amazon/links.html)). My goal was to use ratings from users who reviewed both comic books and movies/tv.

### Data Preparation
In order to extract the correct users and ratings, I had to spend a good amount of time learning and exploring this dataset. All of the comic books/graphic novels are lumped within all other book reviews with no shortcuts to pull them out. I started with a smaller amount of Amazon IDs (ASIN) for comic books from scraping Amazons bestseller pages. Based on data exploration, I found a pattern in the ids to help shorcut get a few large chunks of ids for comic books. With this set of ids, I found any corresponding reviews. I then took the reviewers in that set and found any that also had reviewed movies/tv. After removing items with less than 5 reivews, and dropping any data missing relevant metadata, I was working with ~84,000 reviews, with ~8,500 distinct users and ~7,400 items (~1,300 comic books/graphic novels, ~6,100 movies/tv).

### Modeling
My approach was to build a Alternative Least Squares (ALS) model to have a collaborative filtering recommender system. I treated both movies/tv and comic books equally as items. Using a matrix for all users and items, I created the model, but filtered based on type when getting top recommendations.  

### Evaluation
For evaluation, I optimized the performance of my model based for Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Tuning the hyper-parameters of my model did not prove to offer too much in terms of performance, but I settled on my best model using a rank of 50, regularization parameter at .1, and max number of iterations at 20. My best performing model has an RMSE of 1.17.  In general, I would hope to get that number under 1, but think this is a fairly good performance under my current scope. I am working under a larger assumption that each user's taste preferences (regardless of media) are the same, which in general might seem like a big leap, but with this performance actually seems to speak well to that point. 

### Next Steps
##### Model improvements:
* Add reviews from Amazon Prime Video into the model as well - ✅
* Explore adjustments to data to account for user and item bias.
* Gather more recent review data to improve and deepend my models ability.

##### Web app improvements:
* Add additional user features such as filtering and content for recommended comic books (Examples: filter by independent or major company published comics, display short description or review of item)
* Add more movies and television options to choose from for user to rate on.
