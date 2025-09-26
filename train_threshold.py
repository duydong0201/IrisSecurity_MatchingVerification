# train_threshold.py
import glob, os, random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

def hamming_distance(c1, c2):
    return np.sum(c1 != c2) / len(c1)

def collect_distances(codes_dir="codes_cropped", test_size=0.3, num_persons=100, random_state=None):
    persons = {}
    for f in glob.glob(os.path.join(codes_dir, "*.npy")):
        name = os.path.basename(f).split("_")[0]
        persons.setdefault(name, []).append(f)  # lưu path

    selected = dict(list(persons.items())[:num_persons])

    labels, distances = [], []
    test_files = []

    for person, file_list in selected.items():
        train_files, test_files_person = train_test_split(
            file_list, test_size=test_size, random_state=random_state
        )
        train_codes = [np.load(f) for f in train_files]
        test_codes  = [np.load(f) for f in test_files_person]

        # dùng median thay vì mean
        template = np.median(train_codes, axis=0) > 0.5

        # intra-class
        for f, code in zip(test_files_person, test_codes):
            dist = hamming_distance(template, code)
            labels.append(1)
            distances.append(dist)
            test_files.append(f)

        # inter-class (sample để cân bằng số lượng)
        inter_candidates = []
        for other_person, other_files in selected.items():
            if other_person == person:
                continue
            inter_candidates.extend(other_files)
        if inter_candidates:
            sampled = random.sample(inter_candidates, min(len(test_codes), len(inter_candidates)))
            for f in sampled:
                code = np.load(f)
                dist = hamming_distance(template, code)
                labels.append(0)
                distances.append(dist)

    return np.array(labels), np.array(distances), test_files

def find_best_threshold(y_true, distances):
    best_t, best_score = 0, 0
    best_metrics = {}
    for t in np.linspace(0.0, 1.0, 200):
        y_pred = (distances < t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)

        # dùng macro-F1 (trung bình giữa class 0 và 1)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        if f1_macro > best_score:
            best_score = f1_macro
            best_t = t
            best_metrics = dict(acc=acc, pre=pre, rec=rec, f1=f1, f1_macro=f1_macro)

    return best_t, best_metrics

if __name__ == "__main__":
    epochs = 50
    thresholds = []
    all_test_files = []

    for epoch in range(epochs):
        y_true, distances, test_files = collect_distances(
            "codes_cropped", test_size=0.3, num_persons=1000, random_state=epoch
        )
        best_t, metrics = find_best_threshold(y_true, distances)
        thresholds.append(best_t)
        all_test_files.extend(test_files)

        print(f"[Epoch {epoch+1}] Th={best_t:.3f} | "
              f"Acc={metrics['acc']*100:.2f}% | "
              f"Pre={metrics['pre']*100:.2f}% | "
              f"Rec={metrics['rec']*100:.2f}% | "
              f"F1={metrics['f1']*100:.2f}% | "
              f"Macro-F1={metrics['f1_macro']*100:.2f}%")

    avg_th = np.mean(thresholds)
    print("="*60)
    print(f"Average Threshold : {avg_th:.3f}")
    print("="*60)

    # lưu threshold trung bình
    with open("best_threshold.txt", "w") as f:
        f.write(str(avg_th))

    # lưu test files
    with open("test_list.txt", "w") as f:
        for path in all_test_files:
            f.write(path + "\n")
