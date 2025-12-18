import React, { useState } from "react";

const CV_METHODS = [
  /* =========================
   BASIC
========================= */

  {
    name: "Hold-Out Validation",
    category: "Basic",
    why: "Fastest evaluation with minimal computation",
    useCase: "Very large datasets",
    when: "Speed > stability",
    internalAlgorithm: [
      "Shuffle dataset (optional)",
      "Split dataset into train and test subsets",
      "Train model once using training subset",
      "Freeze model parameters",
      "Evaluate model on test subset",
      "Report test metric",
    ],
    compatible: "All ML models",
    notCompatible: "Small datasets",
    python: `
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(Xtr, ytr)
print(model.score(Xte, yte))
`,
  },

  {
    name: "Train–Validation–Test Split",
    category: "Basic",
    why: "Separates tuning from final evaluation",
    useCase: "Hyperparameter tuning",
    when: "Need unbiased test set",
    internalAlgorithm: [
      "Shuffle dataset",
      "Split into training, validation, and test sets",
      "Train model on training set",
      "Tune hyperparameters using validation set",
      "Freeze best model",
      "Evaluate once on test set",
    ],
    compatible: "All ML models",
    notCompatible: "Tiny datasets",
    python: `
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3)
Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5)
`,
  },

  /* =========================
   K-FOLD FAMILY
========================= */

  {
    name: "K-Fold Cross Validation",
    category: "K-Fold",
    why: "Balances bias and variance",
    useCase: "General evaluation",
    when: "IID data",
    internalAlgorithm: [
      "Partition dataset into K equal folds",
      "For each fold i from 1 to K:",
      "  Use fold i as test set",
      "  Use remaining K−1 folds as training set",
      "  Train model and evaluate on test fold",
      "Aggregate metrics across all folds",
    ],
    compatible: "Most ML models",
    notCompatible: "Time series",
    python: `
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=kf)
print(scores.mean())
`,
  },

  {
    name: "Stratified K-Fold",
    category: "K-Fold",
    why: "Preserves class proportions",
    useCase: "Imbalanced classification",
    when: "Class ratios must be preserved",
    internalAlgorithm: [
      "Group samples by class labels",
      "Split each class into K folds",
      "Merge corresponding folds across classes",
      "Rotate test fold across K iterations",
      "Aggregate evaluation metrics",
    ],
    compatible: "Classification models",
    notCompatible: "Regression",
    python: `
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
skf = StratifiedKFold(n_splits=5)

model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=skf)
print(scores.mean())
`,
  },

  {
    name: "Group K-Fold",
    category: "K-Fold",
    why: "Prevents group leakage",
    useCase: "Patient / user data",
    when: "Grouped samples exist",
    internalAlgorithm: [
      "Assign each sample to a group",
      "Split groups into K disjoint folds",
      "Ensure no group appears in both train and test",
      "Train and evaluate across folds",
      "Aggregate results",
    ],
    compatible: "All ML models",
    notCompatible: "Ungrouped data",
    python: `
from sklearn.model_selection import GroupKFold
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)
groups = np.random.randint(0, 3, size=len(y))

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups):
    print(len(train), len(test))
`,
  },

  {
    name: "Repeated K-Fold",
    category: "K-Fold",
    why: "Reduces randomness",
    useCase: "Stable performance estimation",
    when: "Metrics vary across runs",
    internalAlgorithm: [
      "Perform standard K-Fold split",
      "Repeat K-Fold multiple times with reshuffling",
      "Collect metrics from all repetitions",
      "Average metrics across runs",
    ],
    compatible: "All ML models",
    notCompatible: "Time series",
    python: `
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
rkf = RepeatedKFold(n_splits=5, n_repeats=3)

model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=rkf)
print(scores.mean())
`,
  },

  {
    name: "Repeated Stratified K-Fold",
    category: "K-Fold",
    why: "Stable stratified evaluation",
    useCase: "Highly imbalanced data",
    when: "Need low-variance stratified metrics",
    internalAlgorithm: [
      "Apply stratified K-Fold splitting",
      "Repeat stratified splitting multiple times",
      "Train and evaluate model each time",
      "Aggregate metrics across repetitions",
    ],
    compatible: "Classification models",
    notCompatible: "Regression",
    python: `
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

model = LogisticRegression(max_iter=200)
`,
  },

  /* =========================
   LEAVE-OUT
========================= */

  {
    name: "Leave-One-Out (LOOCV)",
    category: "Leave-Out",
    why: "Maximum training data usage",
    useCase: "Very small datasets",
    when: "N is tiny",
    internalAlgorithm: [
      "For each sample i:",
      "  Use sample i as test set",
      "  Train on remaining N−1 samples",
      "  Evaluate prediction for i",
      "Aggregate metrics across all samples",
    ],
    compatible: "All ML models",
    notCompatible: "Large datasets",
    python: `
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
loo = LeaveOneOut()

model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=loo)
print(scores.mean())
`,
  },

  {
    name: "Leave-P-Out",
    category: "Leave-Out",
    why: "Exhaustive validation",
    useCase: "Research experiments",
    when: "Dataset extremely small",
    internalAlgorithm: [
      "Enumerate all combinations of P samples",
      "Use P samples as test set",
      "Train on remaining samples",
      "Evaluate model",
      "Aggregate all evaluations",
    ],
    compatible: "All ML models",
    notCompatible: "Large datasets",
    python: `
from sklearn.model_selection import LeavePOut
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
lpo = LeavePOut(p=2)

for train, test in lpo.split(X):
    print(len(train), len(test))
    break
`,
  },

  {
    name: "Leave-One-Group-Out",
    category: "Leave-Out",
    why: "Strict group isolation",
    useCase: "Medical studies",
    when: "One group must be fully unseen",
    internalAlgorithm: [
      "Identify unique groups",
      "Iteratively select one group as test set",
      "Train model on all remaining groups",
      "Evaluate on held-out group",
      "Aggregate group-wise metrics",
    ],
    compatible: "All ML models",
    notCompatible: "Ungrouped data",
    python: `
from sklearn.model_selection import LeaveOneGroupOut
`,
  },

  /* =========================
   TIME SERIES
========================= */

  {
    name: "TimeSeriesSplit (Forward Chaining)",
    category: "Time Series",
    why: "Preserves temporal order",
    useCase: "Forecasting",
    when: "Time dependency exists",
    internalAlgorithm: [
      "Sort data by time",
      "Start with minimal training window",
      "Train model on past data",
      "Test on immediate future data",
      "Expand training window forward",
      "Repeat until end of series",
    ],
    compatible: "Regression, LSTM",
    notCompatible: "IID assumptions",
    python: `
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

X = np.arange(50).reshape(-1,1)
y = X.flatten()

tscv = TimeSeriesSplit(n_splits=5)
model = LinearRegression()

for tr, te in tscv.split(X):
    model.fit(X[tr], y[tr])
    print(model.score(X[te], y[te]))
`,
  },

  {
    name: "Rolling Window CV",
    category: "Time Series",
    why: "Handles concept drift",
    useCase: "Financial time series",
    when: "Recent data more important",
    internalAlgorithm: [
      "Fix training window size",
      "Train on most recent window",
      "Test on next time step",
      "Slide window forward",
      "Repeat for entire series",
    ],
    compatible: "Time-series models",
    notCompatible: "Static datasets",
    python: `
window = 20
for i in range(window, len(X)):
    X_train = X[i-window:i]
    X_test = X[i:i+1]
`,
  },

  {
    name: "Expanding Window CV",
    category: "Time Series",
    why: "Uses increasing historical data",
    useCase: "Long-term forecasting",
    when: "Older data still relevant",
    internalAlgorithm: [
      "Start with small training window",
      "Train model",
      "Test on next time block",
      "Expand training window",
      "Repeat sequentially",
    ],
    compatible: "Time-series models",
    notCompatible: "IID datasets",
    python: `
# Similar to TimeSeriesSplit with expanding window
`,
  },

  {
    name: "Blocking Time Series CV",
    category: "Time Series",
    why: "Avoids leakage across time blocks",
    useCase: "Autocorrelated data",
    when: "Strong temporal correlation",
    internalAlgorithm: [
      "Divide time series into blocks",
      "Hold out one block as test",
      "Train on remaining blocks",
      "Rotate blocks",
      "Aggregate metrics",
    ],
    compatible: "Time-series models",
    notCompatible: "IID data",
    python: `
# Custom block splitting logic
`,
  },

  /* =========================
   BOOTSTRAP
========================= */

  {
    name: "Bootstrap Validation",
    category: "Bootstrap",
    why: "Effective for small datasets",
    useCase: "Statistical estimation",
    when: "Limited data",
    internalAlgorithm: [
      "Sample N points with replacement",
      "Train model on bootstrap sample",
      "Evaluate on out-of-bag samples",
      "Repeat many times",
      "Aggregate metrics",
    ],
    compatible: "Most ML models",
    notCompatible: "Highly biased estimators",
    python: `
from sklearn.utils import resample
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)

scores = []
for _ in range(20):
    Xb, yb = resample(X, y)
    model.fit(Xb, yb)
    scores.append(model.score(X, y))
print(np.mean(scores))
`,
  },

  {
    name: ".632 Bootstrap",
    category: "Bootstrap",
    why: "Bias correction",
    useCase: "Small datasets",
    when: "Bootstrap optimism exists",
    internalAlgorithm: [
      "Train model on bootstrap sample",
      "Evaluate on training and OOB samples",
      "Weight errors: 0.368 * train + 0.632 * OOB",
      "Repeat across samples",
      "Aggregate weighted error",
    ],
    compatible: "Most ML models",
    notCompatible: "Streaming data",
    python: `
# Conceptual implementation of .632 bootstrap
`,
  },

  {
    name: ".632+ Bootstrap",
    category: "Bootstrap",
    why: "Improved bias correction",
    useCase: "Highly overfitted models",
    when: "High overfitting risk",
    internalAlgorithm: [
      "Compute .632 bootstrap estimate",
      "Estimate no-information error",
      "Adjust weights based on overfitting",
      "Compute corrected error",
    ],
    compatible: "Most ML models",
    notCompatible: "Online learning",
    python: `
# Advanced statistical bootstrap correction
`,
  },

  /* =========================
   RANDOMIZED
========================= */

  {
    name: "Monte Carlo CV (Shuffle Split)",
    category: "Random",
    why: "Randomized robustness",
    useCase: "Quick repeated evaluation",
    when: "No strict fold structure",
    internalAlgorithm: [
      "Randomly split data into train and test",
      "Train model and evaluate",
      "Repeat random splitting many times",
      "Aggregate metrics",
    ],
    compatible: "All ML models",
    notCompatible: "Time series",
    python: `
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
ss = ShuffleSplit(n_splits=5, test_size=0.2)

model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=ss)
print(scores.mean())
`,
  },

  /* =========================
   MODEL SELECTION
========================= */

  {
    name: "Nested Cross Validation",
    category: "Model Selection",
    why: "Unbiased model comparison",
    useCase: "Hyperparameter tuning",
    when: "Selecting best model",
    internalAlgorithm: [
      "Split dataset using outer CV",
      "For each outer training set:",
      "  Run inner CV to tune hyperparameters",
      "  Select best hyperparameters",
      "Train final model on outer training data",
      "Evaluate on outer test set",
      "Aggregate outer scores",
    ],
    compatible: "All ML models",
    notCompatible: "Very large datasets",
    python: `
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
inner = KFold(n_splits=3)
outer = KFold(n_splits=5)

clf = GridSearchCV(SVC(), {"C":[0.1,1,10]}, cv=inner)
scores = cross_val_score(clf, X, y, cv=outer)
print(scores.mean())
`,
  },

  /* =========================
   ONLINE / STREAMING
========================= */

  {
    name: "Prequential Cross Validation",
    category: "Online",
    why: "Streaming evaluation",
    useCase: "Online learning",
    when: "Data arrives sequentially",
    internalAlgorithm: [
      "Receive next data point",
      "Predict output using current model",
      "Compare prediction with true label",
      "Update model with same data point",
      "Repeat for entire stream",
    ],
    compatible: "Online ML models",
    notCompatible: "Batch-only models",
    python: `
from river import linear_model, metrics

model = linear_model.LogisticRegression()
metric = metrics.Accuracy()

for x, y in stream:
    y_pred = model.predict_one(x)
    metric.update(y, y_pred)
    model.learn_one(x, y)

print(metric)
`,
  },

  /* =========================
   SPECIAL STRUCTURED
========================= */

  {
    name: "Spatial Cross Validation",
    category: "Special",
    why: "Prevents spatial leakage",
    useCase: "GIS, climate data",
    when: "Spatial autocorrelation exists",
    internalAlgorithm: [
      "Partition data based on spatial regions",
      "Hold out one region as test set",
      "Train on remaining regions",
      "Rotate held-out region",
      "Aggregate metrics",
    ],
    compatible: "Spatial ML models",
    notCompatible: "IID assumptions",
    python: `
# Custom spatial split logic
`,
  },

  {
    name: "Hierarchical Cross Validation",
    category: "Special",
    why: "Respects hierarchy",
    useCase: "School → class → student data",
    when: "Nested structure exists",
    internalAlgorithm: [
      "Identify hierarchy levels",
      "Split data at highest hierarchy level",
      "Train on remaining hierarchy branches",
      "Evaluate on held-out branch",
      "Aggregate results",
    ],
    compatible: "Hierarchical models",
    notCompatible: "Flat IID data",
    python: `
# Hierarchical splitting logic
`,
  },

  {
    name: "Subject / Patient-Wise CV",
    category: "Special",
    why: "Prevents identity leakage",
    useCase: "Healthcare datasets",
    when: "Multiple samples per subject",
    internalAlgorithm: [
      "Group samples by subject",
      "Ensure subject appears in only one fold",
      "Train on other subjects",
      "Evaluate on unseen subject",
    ],
    compatible: "All ML models",
    notCompatible: "Single-sample subjects",
    python: `
# Equivalent to GroupKFold
`,
  },

  {
    name: "Cluster-Based Cross Validation",
    category: "Special",
    why: "Cluster-aware evaluation",
    useCase: "Unsupervised + ML pipelines",
    when: "Clustered data",
    internalAlgorithm: [
      "Cluster dataset",
      "Treat each cluster as a group",
      "Apply group-based CV",
      "Aggregate cluster-wise metrics",
    ],
    compatible: "All ML models",
    notCompatible: "Unclustered data",
    python: `
# Cluster + GroupKFold
`,
  },

  /* =========================
   BAYESIAN
========================= */

  {
    name: "Bayesian Cross Validation",
    category: "Bayesian",
    why: "Posterior predictive evaluation",
    useCase: "Bayesian models",
    when: "Probabilistic modeling",
    internalAlgorithm: [
      "Fit Bayesian model on full dataset",
      "Compute posterior distribution",
      "Estimate predictive likelihood",
      "Adjust for model complexity",
      "Report expected log predictive density",
    ],
    compatible: "Bayesian models",
    notCompatible: "Frequentist-only models",
    python: `
# PyMC / Stan based evaluation
`,
  },

  {
    name: "PSIS-LOO",
    category: "Bayesian",
    why: "Efficient LOO approximation",
    useCase: "Bayesian model comparison",
    when: "Exact LOO is expensive",
    internalAlgorithm: [
      "Compute importance weights",
      "Apply Pareto smoothing",
      "Estimate leave-one-out likelihood",
      "Aggregate predictive scores",
    ],
    compatible: "Bayesian models",
    notCompatible: "Non-probabilistic models",
    python: `
# arviz.loo(model)
`,
  },

  {
    name: "WAIC",
    category: "Bayesian",
    why: "Information criterion",
    useCase: "Bayesian model comparison",
    when: "Posterior samples available",
    internalAlgorithm: [
      "Compute log-likelihood per sample",
      "Estimate effective number of parameters",
      "Penalize model complexity",
      "Compute WAIC score",
    ],
    compatible: "Bayesian models",
    notCompatible: "Non-Bayesian models",
    python: `
# arviz.waic(model)
`,
  },
];

/* =========================
        MAIN COMPONENT
========================= */
export default function CrossValidationSelector() {
  const [selected, setSelected] = useState(CV_METHODS[0]);

  return (
    <div className="h-screen overflow-hidden bg-gradient-to-br from-slate-900 to-indigo-900 text-white">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-full min-h-0 p-6">

        {/* ================= LEFT: METHODS ================= */}
        <Sidebar
          methods={CV_METHODS}
          selected={selected}
          onSelect={setSelected}
        />

        {/* ================= RIGHT: DETAILS ================= */}
        <DetailsPanel method={selected} />

      </div>
    </div>
  );
}

/* =========================
        SIDEBAR
========================= */
function Sidebar({ methods, selected, onSelect }) {
  return (
    <div className="bg-white/10 backdrop-blur rounded-2xl min-h-0 flex flex-col">
      
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-slate-900/80 backdrop-blur rounded-t-2xl p-4">
        <h2 className="text-xl font-semibold">Methods</h2>
      </div>

      {/* Scrollable List */}
      <div className="overflow-y-auto min-h-0 p-4 space-y-2 pr-2">
        {methods.map((m) => (
          <button
            key={m.name}
            onClick={() => onSelect(m)}
            className={`w-full text-left p-3 rounded-xl transition ${
              selected.name === m.name
                ? "bg-indigo-600 shadow-lg"
                : "bg-white/10 hover:bg-white/20"
            }`}
          >
            <div className="font-medium">{m.name}</div>
            <div className="text-xs text-indigo-200">
              {m.category}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

/* =========================
      DETAILS PANEL
========================= */
function DetailsPanel({ method }) {
  return (
    <div className="md:col-span-2 bg-white/10 backdrop-blur rounded-2xl min-h-0 flex flex-col">

      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-slate-900/80 backdrop-blur rounded-t-2xl p-5">
        <h1 className="text-2xl md:text-3xl font-bold">
          {method.name}
        </h1>
        <p className="text-indigo-300 text-sm">
          Category: {method.category}
        </p>
      </div>

      {/* Scrollable Content */}
      <div className="overflow-y-auto min-h-0 p-6 space-y-6 pr-3">

        <InfoGrid method={method} />

        <Section title="Internal Algorithm (How it Works)">
          <ol className="list-decimal list-inside space-y-1 text-slate-200">
            {method.internalAlgorithm.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        </Section>

        <Section title="Visual Fold Diagram">
          <pre className="bg-black/50 rounded-xl p-4 text-sm whitespace-pre overflow-x-auto">
            {method.visualDiagram || "Diagram coming soon"}
          </pre>
        </Section>

        <Section title="Python Example">
          <pre className="bg-black/50 rounded-xl p-4 text-sm overflow-x-auto">
            {method.python}
          </pre>

          <button
            onClick={() => navigator.clipboard.writeText(method.python)}
            className="mt-3 px-4 py-2 bg-indigo-600 rounded-lg hover:bg-indigo-700 transition"
          >
            Copy Code
          </button>
        </Section>

      </div>
    </div>
  );
}

/* =========================
      INFO GRID
========================= */
function InfoGrid({ method }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Info title="Why used?" value={method.why} />
      <Info title="Use case" value={method.useCase} />
      <Info title="When to use?" value={method.when} />
      <Info title="Compatible models" value={method.compatible} />
      <Info title="Not compatible with" value={method.notCompatible} />
    </div>
  );
}

/* =========================
      UI HELPERS
========================= */
function Section({ title, children }) {
  return (
    <div>
      <h3 className="text-lg font-semibold text-indigo-300 mb-2">
        {title}
      </h3>
      {children}
    </div>
  );
}

function Info({ title, value }) {
  return (
    <div className="bg-white/5 rounded-xl p-3">
      <div className="text-indigo-300 text-sm font-semibold">
        {title}
      </div>
      <div className="text-slate-200 text-sm">
        {value}
      </div>
    </div>
  );
}
