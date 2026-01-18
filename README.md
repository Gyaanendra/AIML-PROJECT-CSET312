# AIML_PROJECT 🧠📊

This project is an **AI/ML workflow project** for building and preparing datasets for tasks like **Fake News Detection / Text Classification** using multiple sources (FakeNewsCorpus + Kaggle dataset), and combining them into a final merged dataset.

## 📂 Folder Structure

```bash
AIML_PROJECT/
│
├── .git/
├── .venv/
│
├── Datasets/
│   ├── FakeNewsCorpus/
│   │   └── news.csv.7z
│   │
│   ├── kaggle/
│   │   └── isot_welfake_correct.csv
│   │
│   ├── merged/
│   │   └── final_combined_corpus.csv
│   │
│   └── datasets_links.txt
│
├── Notebooks/
│   ├── data_gathering.ipynb
│   └── Preprocessing.ipynb
│
├── .gitattributes
├── .gitignore
├── .python-version
├── LICENSE
├── project_structure.txt
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 📌 Project Overview

### ✅ Dataset Sources

This project uses datasets stored inside the `Datasets/` folder:

- https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0
- https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset
- https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

---

## 📓 Notebooks

All experiment and preprocessing work is stored inside:

### `Notebooks/`

- `data_gathering.ipynb` → dataset collection & loading
- `Preprocessing.ipynb` → cleaning, preprocessing, and merging

---

## ⚙️ Setup (Recommended)

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd AIML_PROJECT
```

### 2️⃣ Install dependencies (uv)

```bash
uv sync
```

### 3️⃣ Activate virtual environment

✅ Windows:

```bash
.venv\Scripts\activate
```

✅ Mac/Linux:

```bash
source .venv/bin/activate
```

---

## ▶️ Running the Project

Open the notebooks inside:

```bash
Notebooks/
```

Run using Jupyter:

```bash
jupyter notebook
```

Or use VS Code Notebook interface.

---

## 📄 Notes

- `.venv/` is only for local development and should not be uploaded.
- Dataset links are maintained in:
  - `Datasets/datasets_links.txt`

---

## 📜 License

This project is licensed under the **MIT License** (see `LICENSE`).
