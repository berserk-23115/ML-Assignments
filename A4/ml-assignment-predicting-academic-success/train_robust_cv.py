import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

def train_robust_model():
    print("Loading data...")
    train = pd.read_csv('train_assignment.csv')
    test = pd.read_csv('test_assignment.csv')
    
    # Feature engineering/preprocessing
    target_col = 'Target'
    id_col = 'id'
    
    X = train.drop([id_col, target_col], axis=1)
    y = train[target_col]
    X_test = test.drop([id_col], axis=1)
    
    # Identify categorical features
    # (Assuming columns ending in _code or _flag are categorical)
    cat_features = [col for col in X.columns if col.endswith('_code') or col.endswith('_flag')]
    
    # Model parameters aimed at preventing overfitting (high generalisability)
    catboost_params = {
        'iterations': 2000,          # High number of trees
        'learning_rate': 0.03,       # Low learning rate
        'depth': 6,                  # Shallow trees to prevent overfitting
        'l2_leaf_reg': 5,            # L2 regularization
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 100,              # Early stopping rounds
        'verbose': 200,
        'allow_writing_files': False
    }

    # Stratified K-Fold setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros((len(test), n_splits, len(np.unique(y)))) # Storing probabilities
    
    # Mapping target to integers if necessary
    target_mapping = {val: idx for idx, val in enumerate(np.unique(y))}
    reverse_mapping = {idx: val for idx, val in enumerate(np.unique(y))}
    y_mapped = y.map(target_mapping)
    
    catboost_params['classes_count'] = len(target_mapping)
    catboost_params['loss_function'] = 'MultiClass'
    
    print(f"Starting {n_splits}-fold Stratified CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_mapped)):
        print(f"\n--- Fold {fold + 1} ---")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y_mapped.iloc[train_idx], y_mapped.iloc[val_idx]
        
        train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_features)
        val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_features)
        
        model = CatBoostClassifier(**catboost_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True
        )
        
        # OOF Predictions
        val_preds = model.predict(val_pool).flatten()
        oof_predictions[val_idx] = val_preds
        
        fold_acc = accuracy_score(y_val_fold, val_preds)
        print(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")
        
        # Test predictions (Probability averaging)
        test_pool = Pool(X_test, cat_features=cat_features)
        test_predictions[:, fold, :] = model.predict_proba(test_pool)

    # Calculate overall OOF CV Score
    oof_acc = accuracy_score(y_mapped, oof_predictions)
    print(f"\nOverall Out-Of-Fold (OOF) Accuracy: {oof_acc:.4f}")
    
    # Final predictions on test set
    print("Averaging predictions over all folds...")
    avg_test_probs = np.mean(test_predictions, axis=1)
    final_test_preds = np.argmax(avg_test_probs, axis=1)
    final_test_labels = [reverse_mapping[pred] for pred in final_test_preds]
    
    # Create submission file
    submission = pd.DataFrame({
        id_col: test[id_col],
        target_col: final_test_labels
    })
    
    submission_path = 'robust_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
if __name__ == "__main__":
    train_robust_model()
