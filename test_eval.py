# test_eval.py
import os, glob, random
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def hamming_distance(c1, c2):
    return np.sum(c1 != c2) / len(c1)

if __name__ == "__main__":
    # đọc threshold từ file (lấy dòng cuối cùng)
    with open("best_threshold.txt", "r") as f:
        lines = f.readlines()
        last_line = lines[-1].strip()
        best_t = float(last_line.split(":")[-1])

    # đọc danh sách test files
    with open("test_list.txt", "r") as f:
        test_files = [line.strip() for line in f.readlines()]

    labels, distances = [], []

    # tạo template theo từng người
    persons = {}
    for f in test_files:
        name = f.split(os.sep)[-1].split("_")[0]
        persons.setdefault(name, []).append(np.load(f))

    for person, codes in persons.items():
        template = np.mean(codes, axis=0) > 0.5
        for code in codes:
            dist = hamming_distance(template, code)
            labels.append(1)
            distances.append(dist)

    y_true = np.array(labels)
    y_pred = (np.array(distances) < best_t).astype(int)

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    print("="*40)
    print(f"Threshold used : {best_t:.3f}")
    print(f"Accuracy       : {acc*100:.2f}%")
    print(f"Precision      : {pre*100:.2f}%")
    print(f"Recall         : {rec*100:.2f}%")
    print(f"F1-Score       : {f1*100:.2f}%")
    print("="*40)



# # ==============================
# # Thử xác thực 1 mẫu bất kỳ
# # ==============================

# def verify_identity(template, new_code, threshold):
#     dist = hamming_distance(template, new_code)
#     return dist < threshold, dist

# # lấy 1 file test bất kỳ
# test_file = random.choice(test_files)
# person_id = os.path.basename(test_file).split("_")[0]
# new_code = np.load(test_file)

# # xây template từ tập train của đúng person
# person_files = [
#     f for f in glob.glob("codes_cropped/*.npy")
#     if os.path.basename(f).split("_")[0] == person_id
# ]
# train_files, _ = train_test_split(person_files, test_size=0.3, random_state=42)
# train_codes = [np.load(f) for f in train_files]
# template = np.median(train_codes, axis=0) > 0.5

# # xác thực
# is_match, dist = verify_identity(template, new_code, best_t)
# match_rate = (1 - dist) * 100  # tỉ lệ match %

# print("="*40)
# print(f"Person ID       : {person_id}")
# print(f"Test sample     : {os.path.basename(test_file)}")
# print(f"Hamming dist    : {dist:.4f}")
# print(f"Threshold used  : {best_t:.3f}")
# print(f"Match rate      : {match_rate:.2f}%")
# print("Result          : MATCH ✅" if is_match else "Result          : NOT MATCH ❌")
# print("="*40)

