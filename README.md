# PokerMind AI

PokerMind AI is a progressive machine learning project focused on poker win probability and decision quality analysis.

The first objective is to analyze poker hand histories and estimate which player has the highest probability of winning during a hand. Later, the project may evaluate whether player decisions were reasonable using equity, pot odds, stack size, position, and betting history.

This project is currently in the exploration stage. The goal is to understand available poker data, inspect file formats, and prepare a clean foundation before building models or applications.

## Data

Raw data is stored outside this repository at:

```text
/Users/maxime/Documents/Projet/POC ML/data
```

This external folder should not be moved, deleted, or modified automatically.

The project also contains a local raw data folder:

```text
data/raw/
```

Use this local folder only for small sample files or manually selected data. Large datasets should not be copied automatically into the project.

Processed data can later be stored in:

```text
data/processed/
```

## Candidate Datasets

Possible datasets to investigate:

- PHH Dataset / Poker Hand Histories
- Kaggle Poker Heads Up
- PokerBench
- UCI Poker Hand Dataset, only as a fallback

Poker hand history datasets are preferred because they can contain action sequences, betting history, stack sizes, positions, and outcomes. The UCI Poker Hand Dataset is less useful for decision analysis because it focuses mainly on final hand classification.

## First ML Objective

The first machine learning objective is win probability prediction:

- Input: available information during a poker hand
- Output: estimated probability that each player wins the hand

At the beginning, this may start with simplified features such as cards, board state, position, pot size, and number of active players.

## Later Objective

A later objective is decision quality analysis:

- Compare player actions against estimated equity
- Consider pot odds and stack sizes
- Include table position and betting history
- Identify decisions that appear reasonable or questionable

This later stage should only be attempted after the data format and win probability baseline are understood.

## Project Structure

```text
PokerMind AI/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── poker_utils.py
├── README.md
└── requirements.txt
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the first notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Disclaimer

This project is for educational and analytical purposes only. It is not gambling advice and should not be used to encourage or support real-money gambling decisions.
