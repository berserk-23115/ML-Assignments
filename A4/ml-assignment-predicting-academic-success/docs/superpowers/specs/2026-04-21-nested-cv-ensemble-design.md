# Nested CV Ensemble with Fold-Specific Weights

**Date:** 2026-04-21  
**Goal:** Maximize accuracy without reducing generalizability by preventing weight optimization overfitting to training distribution.

## Problem Statement

The current approach (run.ipynb) uses Monte Carlo optimization (15,000 weight combinations) on out-of-fold training predictions to find the best ensemble weights. Even though OOF predictions are unbiased, optimizing so many weight combinations on a single training fold can cause the weights to fit quirks of the training distribution rather than learn generalizable combinations.

## Solution: Nested Cross-Validation with Fold-Specific Weights

### Architecture

Execute a 10-fold stratified CV outer loop. In each fold:
1. Train 5 candidate models on the training fold
2. Generate OOF predictions on validation fold
3. Evaluate each model's accuracy on validation fold
4. Select top 2-3 models based on validation accuracy (most robust approach)
5. Optimize weights on that fold's OOF predictions using limited Monte Carlo (1500 combinations with Dirichlet distribution)
6. Store fold-specific weights and model predictions

For the test set, average predictions from all 10 folds using their respective fold-specific optimized weights.

**Generalization Benefit:** Weights are validated on held-out data in each fold, preventing overfitting to any single training distribution. By averaging across 10 different fold-level solutions, the final ensemble is robust.

### Data Flow

**Inputs:**
- `train_assignment.csv` — training data with Target labels
- `test_assignment.csv` — test data without Target

**Processing:**
1. Load data and apply feature engineering (existing logic)
2. Initialize 10-fold stratified CV splitter
3. For fold in range(10):
   - Split data: training fold (90%) → train; validation fold (10%) → evaluate & optimize
   - Train 5 candidate models (XGBoost, LightGBM, CatBoost, HistGradientBoosting, ExtraTrees) on training fold
   - Generate OOF predictions on validation fold for all 5 models
   - Calculate accuracy for each model on validation fold
   - **Model Selection:** Keep top 2-3 models by validation accuracy
   - **Weight Optimization:** Run 1500 random weight combinations (using Dirichlet distribution for weights summing to 1) on only the top 2-3 models' OOF predictions
   - Store: fold-specific best weights, selected model indices, fold metrics
4. Test prediction generation:
   - For each fold: use fold's top 2-3 models + fold's optimized weights to generate test predictions
   - Average all 10 fold-level test predictions using fold-specific weights
5. Convert probability predictions to class labels using the label encoder
6. Output submission CSV

**Generalization Tracking:**
- Log validation accuracy for each model in each fold
- Track which 2-3 models were selected per fold
- Calculate and report fold-to-fold stability (min, max, mean accuracy across folds)
- Report generalization gap: difference between best and worst fold performance

### Output Specification

**File:** `submission.csv`

**Format:**
```
id,Target
76518,Graduate
76519,Dropout
76520,Enrolled
```

- Column 1 (`id`): Sample identifier from test_assignment.csv
- Column 2 (`Target`): Predicted class label (Graduate/Dropout/Enrolled)
- No additional columns or probability scores

### Implementation Structure

**Main script:** `train_nested_cv.py` (~150-200 lines)

**Components:**
- **Config section:** N_SPLITS=10, RANDOM_STATE=42, model hyperparameters (same as run.ipynb)
- **Helper functions:**
  - `add_features(df)` — Feature engineering (existing logic)
  - `train_fold(X_train, y_train, X_val, y_val, X_test)` — Train models, select top 2-3, optimize weights, return fold results
  - `predict_test(fold_results, X_test)` — Generate fold-averaged test predictions
  - `main()` — Orchestrate 10-fold CV loop, aggregation, output
- **Logging:** Print fold-level metrics, model selection summary, final accuracy estimate
- **Output:** `submission.csv`

### Success Criteria

1. ✅ Weights are optimized on validation folds, not training folds
2. ✅ Model selection is based on cross-validated performance (most robust)
3. ✅ Feature selection is implicit (tree models naturally weight important features)
4. ✅ Final ensemble is averaged across 10 fold-level solutions for stability
5. ✅ Generalization metrics are tracked and reported
6. ✅ Output is Kaggle-compliant CSV with id and Target columns only

### Trade-offs

| Aspect | Gain | Cost |
|--------|------|------|
| **Weight Generalization** | Weights validated on held-out data per fold | ~10× more model training than single-fold optimization |
| **Model Robustness** | Top 2-3 models per fold ensures consistency | May miss rare strong performers in some folds |
| **Simplicity** | Easier to interpret than meta-learners | More complex than single Monte Carlo search |
| **Stability** | Averaging 10 fold solutions reduces variance | Slight reduction in peak performance from any single fold |

---

**Design Status:** Ready for implementation planning
