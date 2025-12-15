"""Train and test LSTM classifier (supports fast mode with 1-fold)"""
import numpy as np
import dga_classifier.data as data
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight


def build_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128))  # Removed input_length (deprecated)
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model


def run(max_epoch=20, nfolds=5, batch_size=128):
    indata = data.get_data()
    X_raw = [x[1] for x in indata]
    y_raw = np.array([0 if x[0] == 'benign' else 1 for x in indata])

    # Build character vocabulary
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X_raw)))}
    max_features = len(valid_chars) + 1
    maxlen = min(np.max([len(x) for x in X_raw]), 50)  # Cap at 50 to avoid memory issues

    X = [[valid_chars[c] for c in s] for s in X_raw]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    final_data = []

    # --- FAST MODE ---
    if nfolds <= 1:
        print("\nRunning in FAST MODE (single train/test split)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, stratify=y_raw, random_state=42)
        folds = [(X_train, X_test, y_train, y_test)]
    else:
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        folds = []
        for train_idx, test_idx in skf.split(X, y_raw):
            folds.append((X[train_idx], X[test_idx], y_raw[train_idx], y_raw[test_idx]))

    # --- TRAIN LOOP ---
    for fold, (X_train, X_test, y_train, y_test) in enumerate(folds, start=1):
        print(f"\nFold {fold}/{len(folds)}")
        
        # Calculate class weights for imbalanced data (Cost-sensitive learning)
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
        
        benign_count = np.sum(y_train == 0)
        malicious_count = np.sum(y_train == 1)
        ratio = malicious_count / benign_count if benign_count > 0 else 0
        print(f"  Dataset: {benign_count} benign, {malicious_count} malicious (ratio={ratio:.4f})")
        print(f"  Class weights: benign={class_weight_dict[0]:.4f}, malicious={class_weight_dict[1]:.4f}")
        
        model = build_model(max_features, maxlen)
        best_auc, best_iter = 0.0, -1
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0,
                     class_weight=class_weight_dict)
            preds = model.predict(X_test, verbose=0)
            t_auc = metrics.roc_auc_score(y_test, preds)
            print(f"Epoch {ep}: auc={t_auc:.6f} (best={best_auc:.6f})")

            if t_auc > best_auc:
                best_auc, best_iter = t_auc, ep
                out_data = {
                    "y": y_test,
                    "probs": preds,
                    "confusion_matrix": metrics.confusion_matrix(y_test, preds > 0.5)
                }
            elif ep - best_iter > 3:
                break

        print(out_data["confusion_matrix"])
        final_data.append(out_data)

    return final_data
