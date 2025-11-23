
# 🎥 Semantic Movie Recommender

An intelligent movie recommendation system powered by LangChain, ChromaDB, HuggingFace embeddings, and a Gradio web interface. Users can describe a movie, specify a genre or emotional tone, or even directly search by title, actor, director, or year to get accurate suggestions.

👉 [Liveive app on HuggingFace Spaces](https://huggingface.co/spaces/shoot4dastars/semantic-movie-recommender)

## 🚀 Features

🔍 Semantic Search

- Enter any natural-language description such as:
    
    "shy girl falls in love", "dark thriller with a twist"...
The system embeds user text and finds thematically similar films using ChromaDB.

🎯 Metadata-Aware Matching

Your query is also matched against:
- movie names
- actors
- directors
- release year
If a match is found, the recommender prioritizes those movies for higher accuracy.

🎭 Emotion-Based Ranking

Movies can also be ranked by emotional tone:
- Happy
- Surprising
- Angry
- Suspenseful
- Sad
- Gross

🎬 Genre Filtering

Filter recommendations by movie genre.

🌐 Interactive Web UI

Built with Gradio — clean, fast, and responsive UI.

## Demo

[xyz](rec.gif)

## Run Locally

Clone the project

```bash
  git clone https://github.com/shoot4dastars/semantic-movie-recommender
```

Go to the project directory

```bash
  cd semantic-movie-recommender
```

Install dependencies

```bash
  pip Install -r requirements.txt
```

Run

```bash
  python app.py
```

Gradio will show a local URL

```bash
  Running on http://127.0.0.1:7860
```

Open the link on your browser.


## Acknowledgements

 - [Dataset](https://www.kaggle.com/datasets/gsimonx37/letterboxd)
 - [Embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
 - [Emotional Text Classifier](https://huggingface.co/michellejieli/emotion_text_classifier)


## Authors

- [Silon Pant](https://github.com/shoot4dastars)

