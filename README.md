# 📰 Fake News Detection System

A simple machine-learning mini project that predicts whether a news article is **REAL** or **FAKE**.

## Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Folder structure

```text
fake_news_detection_project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model.pkl
├── vectorizer.pkl
│
└── data/
    ├── fake.csv
    └── true.csv
```

## Setup

1. Put `fake.csv` and `true.csv` inside the `data` folder.
2. Install the packages:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Start the web interface:

```bash
streamlit run app.py
```

5. Open the local URL shown by Streamlit in your browser.

## Important

The accuracy shown by the training script is test-set accuracy on the supplied dataset. The model is not a live fact-checker and can make incorrect predictions on new articles.
