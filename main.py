import glob, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def hamming_distance(c1, c2):
    return np.sum(c1 != c2) / len(c1)

def collect_distances(codes_dir="codes", test_size=0.3):
    persons = {}
    # gom tất cả code theo person
    for f in glob.glob(os.path.join(codes_dir, "*.npy")):
        name = os.path.basename(f).split("_")[0]  # person1, person2
        persons.setdefault(name, []).append(np.load(f))

    intra, inter = [], []
    labels, distances = [], []

    for person, codes in persons.items():
        # chia 7:3 train:test
        train_codes, test_codes = train_test_split(codes, test_size=test_size, random_state=42)

        # template = trung bình train
        template = np.mean(train_codes, axis=0) > 0.5

        # intra-class (cùng người)
        for code in test_codes:
            dist = hamming_distance(template, code)
            intra.append(dist)
            labels.append(1)
            distances.append(dist)

        # inter-class (khác người)
        for other_person, other_codes in persons.items():
            if other_person == person:
                continue
            for code in other_codes:
                dist = hamming_distance(template, code)
                inter.append(dist)
                labels.append(0)
                distances.append(dist)

    return np.array(labels), np.array(distances), intra, inter

def find_best_threshold(y_true, distances):
    best_t, best_f1 = 0, 0
    for t in np.linspace(0.1, 0.6, 50):  # quét từ 0.1 đến 0.6
        y_pred = (distances < t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

if __name__ == "__main__":
    y_true, distances, intra, inter = collect_distances("codes", test_size=0.3)

    # tìm threshold tốt nhất
    best_t, best_f1 = find_best_threshold(y_true, distances)

    # đánh giá với threshold đó
    y_pred = (distances < best_t).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    print("="*40)
    print(f"Best threshold : {best_t:.3f}")
    print(f"Accuracy       : {acc*100:.2f}%")
    print(f"Precision      : {pre*100:.2f}%")
    print(f"Recall         : {rec*100:.2f}%")
    print(f"F1-Score       : {f1*100:.2f}%")
    print("="*40)
