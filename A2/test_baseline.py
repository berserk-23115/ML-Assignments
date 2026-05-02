from PIL import Image
import os, numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

base = './data/TRAIN/IMAGE'
test_base = './data/TEST/IMAGE'

X_train, y_train = [], []
X_test, y_test = [], []

for cls_idx, cls in enumerate(sorted(os.listdir(base))):
    if cls.startswith('.'): continue
    cpath = os.path.join(base, cls)
    for f in sorted(os.listdir(cpath)):
        if f.startswith('.'): continue
        img = Image.open(os.path.join(cpath, f)).convert('RGB')
        arr = np.array(img, dtype=np.float32).ravel() / 255.0
        X_train.append(arr)
        y_train.append(cls_idx)

for cls_idx, cls in enumerate(sorted(os.listdir(test_base))):
    if cls.startswith('.'): continue
    cpath = os.path.join(test_base, cls)
    for f in sorted(os.listdir(cpath)):
        if f.startswith('.'): continue
        img = Image.open(os.path.join(cpath, f)).convert('RGB')
        arr = np.array(img, dtype=np.float32).ravel() / 255.0
        X_test.append(arr)
        y_test.append(cls_idx)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Test PCA + SVM variations
for n_comp in [100, 200, 300]:
    for C in [1, 10, 50]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp, whiten=True)),
            ('svm', SVC(kernel='rbf', C=C, gamma='scale'))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        print(f"PCA({n_comp},whiten) + SVM(C={C}): {acc:.4f}")

# Test without whitening
for n_comp in [200]:
    for C in [10]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp, whiten=False)),
            ('svm', SVC(kernel='rbf', C=C, gamma='scale'))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        print(f"PCA({n_comp},no-whiten) + SVM(C={C}): {acc:.4f}")

# Test ExtraTrees (no PCA needed)
print("\n--- ExtraTrees ---")
clf = ExtraTreesClassifier(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"ExtraTrees(500): {acc:.4f}")

# Test with 10% data
np.random.seed(42)
idx10 = np.random.choice(len(X_train), int(0.1*len(X_train)), replace=False)
X_10 = X_train[idx10]
y_10 = y_train[idx10]
print(f"\n--- 10% data ({len(X_10)} samples) ---")
for n_comp in [100, 150]:
    for C in [10, 50]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp, whiten=True)),
            ('svm', SVC(kernel='rbf', C=C, gamma='scale'))
        ])
        pipe.fit(X_10, y_10)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        print(f"10%-PCA({n_comp},whiten) + SVM(C={C}): {acc:.4f}")
