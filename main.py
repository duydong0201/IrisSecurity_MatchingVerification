import glob, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def hamming_distance(c1, c2):
    return np.sum(c1 != c2) / len(c1)

def collect_distances(codes_dir="codes_cropped", test_size=0.3, num_persons=5):
    persons = {}
    # gom tất cả code theo person
    for f in glob.glob(os.path.join(codes_dir, "*.npy")):
        name = os.path.basename(f).split("_")[0]  # person1, person2
        persons.setdefault(name, []).append(np.load(f))

    # chỉ lấy num_persons đầu tiên
    selected = dict(list(persons.items())[:num_persons])

    intra, inter = [], []
    labels, distances = [], []

    for person, codes in selected.items():
        # chia 7:3 train:test
        train_codes, test_codes = train_test_split(
            codes, test_size=test_size, random_state=None
        )

        # template = trung bình train
        template = np.mean(train_codes, axis=0) > 0.5

        # intra-class (cùng người)
        for code in test_codes:
            dist = hamming_distance(template, code)
            intra.append(dist)
            labels.append(1)
            distances.append(dist)

        # inter-class (khác người)
        for other_person, other_codes in selected.items():
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
    for t in np.linspace(0.1, 0.6, 50):
        y_pred = (distances < t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

if __name__ == "__main__":
    all_acc, all_pre, all_rec, all_f1, all_thres = [], [], [], [], []

    for epoch in range(10):  # chạy 10 lần
        y_true, distances, intra, inter = collect_distances(
            "codes_cropped", test_size=0.3, num_persons=50
        )

        best_t, best_f1 = find_best_threshold(y_true, distances)

        y_pred = (distances < best_t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred)

        all_acc.append(acc)
        all_pre.append(pre)
        all_rec.append(rec)
        all_f1.append(f1)
        all_thres.append(best_t)

        print(f"[Epoch {epoch+1}] "
              f"Acc={acc:.4f}, Pre={pre:.4f}, Rec={rec:.4f}, "
              f"F1={f1:.4f}, Th={best_t:.3f}")

    print("="*40)
    print(f"Mean Accuracy   : {np.mean(all_acc)*100:.2f}%")
    print(f"Mean Precision  : {np.mean(all_pre)*100:.2f}%")
    print(f"Mean Recall     : {np.mean(all_rec)*100:.2f}%")
    print(f"Mean F1-Score   : {np.mean(all_f1)*100:.2f}%")
    print(f"Mean Threshold  : {np.mean(all_thres):.3f}")
    print("="*40)
