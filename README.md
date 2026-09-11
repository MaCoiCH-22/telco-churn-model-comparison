# Customer Churn Prediction — Model Comparison

Four classifiers built, tuned, and compared on the IBM Telco customer churn dataset, with model selection driven by the metrics that actually matter on imbalanced data.

**My role:** modeling and evaluation. Graduate coursework (ADS-502, M.S. Applied Data Science, University of San Diego), completed as a three-person team — Samer Marroki did data cleaning and repository setup, Ryan Salas did exploratory analysis and feature preparation, and I built, tuned, and evaluated the models and made the final model selection. Original team repository: https://github.com/samermarroki/ADS502Group5FinalProject

---

## The problem

A telecom provider wants to know which customers are likely to leave, early enough to do something about it. That's binary classification, and the business asymmetry matters: a missed churner is a customer walking out the door, while a false alarm is one wasted retention offer. Those costs aren't equal, so accuracy is the wrong thing to optimize.

## The data

IBM Telco Customer Churn — 7,043 customers, 21 columns covering demographics, subscribed services, contract terms, and billing. The target is `Churn`, split 73.5% No / 26.5% Yes.

Preprocessing: dropped the ID column, coerced `TotalCharges` to numeric and median-filled the blanks, mapped `Churn` to 1/0, and one-hot encoded the categoricals with `drop_first=True`, giving 30 features.

## Method

**Stratified 70/30 split** (4,930 train / 2,113 test) with a fixed seed, so the 1-in-4 churn rate holds in both sets and the run reproduces.

**Scaling without leakage.** `StandardScaler` is fit on the training set only, then applied to the test set. KNN measures distance, so `TotalCharges` (up to ~8,700) would otherwise drown out `tenure` (0–72). Fitting the scaler on all the data first would let test-set information reach the model and inflate the scores.

**KNN tuned with GridSearchCV** across k = 1–31 and two weighting schemes, scored on AUC with 5-fold cross-validation. Best: k = 31, uniform weights.

**Three comparison models** — logistic regression, decision tree, and random forest — trained on the same scaled features and scored on the same held-out test set.

## Results

| Model | Accuracy | Precision | Sensitivity | F1 | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **.809** | **.665** | .563 | **.610** | **.845** |
| KNN (k=31) | .782 | .591 | **.586** | .589 | .821 |
| Random Forest | .788 | .630 | .494 | .553 | .819 |
| Decision Tree | .732 | .496 | .499 | .497 | .658 |

**Confusion matrix — logistic regression** (rows = predicted, columns = actual):

|  | Actual: No churn | Actual: Churn |
|---|---|---|
| **Predicted: No churn** | 1,393 | 245 |
| **Predicted: Churn** | 159 | 316 |

## Model selection

Logistic regression wins on AUC (.845) and F1 (.610), the two metrics that hold up when 74% of the data sits in one class. A model predicting "no churn" for every customer would score 73.5% accuracy while catching nobody, which is why accuracy alone isn't the deciding number here.

KNN catches slightly more churners (.586 sensitivity vs .563), but the gap is 23 basis points, and logistic regression is stronger everywhere else. It's also the model you can explain: coefficients map to features, so a retention team can see which contract terms and service combinations drive the prediction. That interpretability is worth more than a marginal sensitivity gain when the output has to survive a conversation with people who don't work in Python.

The decision tree is the instructive failure. Unpruned, it overfits, and its AUC (.658) lands far below the ensemble built from the same algorithm.

**The trade-off in business terms:** the model catches 316 of 561 churners and raises 159 false alarms. Those 245 missed churners are the real cost — customers leaving without an intervention — while the false alarms cost one retention offer each. If retention offers are cheap relative to losing a customer, the classification threshold should move down from 0.5 to catch more churners at the price of more false alarms. That threshold is a business decision, not a modeling one.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook Team5_Modeling_MaCoi.ipynb
```

The notebook pulls the dataset from IBM's public GitHub, so no local data file is needed. It also reproduces the cleaning step so it runs standalone.

**Tools:** Python, pandas, NumPy, scikit-learn, matplotlib.

## What I'd do differently

- **Threshold tuning.** Everything above uses the default 0.5 cutoff. Optimizing the threshold against a real cost ratio would likely beat all four models as configured.
- **Class weighting.** `class_weight='balanced'` on logistic regression and random forest is the obvious next experiment for the sensitivity problem.
- **Prune the tree.** The decision tree ran unpruned. Depth limits and minimum leaf sizes would give it a fairer showing.
- **Coefficient analysis.** Logistic regression won partly on interpretability, but the notebook stops short of reporting which features actually drive churn.

---

*Coursework project. AI assistance was used in drafting the modeling code; the interpretation, model selection, and conclusions are my own.*
