# Data

The dataset is not stored in this repository.

**BBC News Summary** (Kaggle: [`pariza/bbc-news-summary`](https://www.kaggle.com/datasets/pariza/bbc-news-summary))
contains 2,225 BBC news articles from 2004 and 2005 in five categories
(business, entertainment, politics, sport, tech) and, for each article, a
summary made of sentences taken from it. The articles come from the BBC
dataset of Greene and Cunningham (ICML 2006).

Expected layout:

```
BBC News Summary/
  News Articles/<category>/<nnn>.txt   # first line: headline, then the body
  Summaries/<category>/<nnn>.txt       # extracted sentences, joined without spaces
```

On Kaggle, add the dataset as input and the notebook finds it under
`/kaggle/input`. Locally, download it and pass the folder with `--data`.

The pipeline writes `data/splits.csv` (document id, category, split) to the
output folder so that the exact split can be reproduced and shared.
