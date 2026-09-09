# ML Foundations — Coursework

Supervised and unsupervised learning exercises completed during the M.P.S. Analytics program at
Northeastern University. Each is a self-contained notebook on a public dataset.

**These are learning exercises, not portfolio work.** For applied analysis, see:

- [**seattle-vision-zero-analysis**](https://github.com/janemeimei/seattle-vision-zero-analysis)
  — 262K police-reported collisions; found the city's highest-risk corridor has 89% of its
  signals in poor condition
- [**customer-retention-causal-analysis**](https://github.com/janemeimei/customer-retention-causal-analysis)
  — causal inference and targeting policy; found 19.3% of a customer base was harmed by the same
  intervention that helped others
- [**us-traffic-disruption-analysis**](https://github.com/janemeimei/us-traffic-disruption-analysis)
  — 7.7M records in R; ANOVA and logistic regression on what makes a crash disrupt traffic
- [**movie-analytics-pyspark**](https://github.com/janemeimei/movie-analytics-pyspark)
  — 1.2M records at scale; clustering, recommendation, topic modeling

---

## Contents

| Notebook | Problem | Method | Outcome |
|---|---|---|---|
| `linear-regression/` | E-commerce company deciding whether to invest in its mobile app or its website | Linear regression on session length, app time, website time, membership length | Membership length was the dominant predictor of annual spend — the choice between app and website mattered less than retention |
| `logistic-regression/` | Predicting whether an internet user clicks an advertisement | Logistic regression on five features including time on site, age, area income, daily internet usage | Users who click tend to be those who spend *less* time online — engagement and click propensity move in opposite directions |
| `decision-tree-random-forest/` | Classifying whether a LendingClub borrower repaid a loan in full (2007–2010) | Decision tree vs random forest; categorical `purpose` one-hot encoded | Decision tree 73%, random forest 85% — the ensemble gain came from reducing variance on a noisy target |
| `kmeans-clustering/` | Grouping universities into private and public without labels | K-Means on 16 dimensions | **22% accuracy — the method failed.** Clusters did not align with the private/public distinction; class imbalance and the geometry of the feature space make K-Means a poor fit for this task |
| `nlp/` | Classifying Yelp reviews as 1-star or 5-star from review text | CountVectorizer → TF-IDF → Multinomial Naive Bayes, in a scikit-learn pipeline | **81% accuracy — but the model never predicts the minority class.** See below |

---

## Two results worth keeping

Both of these are included rather than removed, because what went wrong is more instructive than
what went right.

### K-Means: 22% accuracy

Unsupervised clustering does not know which distinction you want it to find — it finds the one
the feature geometry supports. A 22% match against the private/public label means the dominant
variation in the data is something else, most likely size and cost. That is a real result about
the dataset. The useful conclusion is that K-Means was the wrong tool for a question that already
had a known label.

### Naive Bayes: 81% accuracy that means nothing

```
              precision    recall   f1-score   support
   1 star          0.00      0.00       0.00       228
   5 star          0.81      1.00       0.90       998

    accuracy                            0.81      1226

confusion matrix
   [[  0  228]      ← every 1-star review misclassified
    [  0  998]]     ← every 5-star review correct
```

**The model labels every review 5-star.** It never predicts the minority class once. The 81%
accuracy is simply the class split — 998 of 1,226 reviews are 5-star, so predicting the majority
every time gets you 81% for free.

Accuracy on an imbalanced binary target is not a measure of anything. Precision, recall, and the
confusion matrix are the only readings that show what the model is doing. Fixes would be class
weighting, resampling, or a threshold chosen against the actual cost of each error type.

---

## Stack

Python · Jupyter · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

---

**Jane (Jingjie) Mei** · M.P.S. Analytics, Northeastern University
[LinkedIn](https://linkedin.com/in/janemei-analytics) · [Tableau](https://public.tableau.com/app/profile/jane.mei)
