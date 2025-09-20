import numpy as np
import glob, os

def hamming_distance(c1, c2):
    return np.sum(c1 != c2) / len(c1)

def evaluate(codes_dir="codes", threshold=0.35):
    persons = {}
    # gom tất cả code theo person
    for f in glob.glob(os.path.join(codes_dir, "*.npy")):
        name = os.path.basename(f).split("_")[0]  # person1, person2
        persons.setdefault(name, []).append(np.load(f))

    y_true, y_pred = [], []

    # với mỗi person, chọn 1 ảnh làm template (train), còn lại làm test
    for person, codes in persons.items():
        template = codes[0]
        for i in range(1, len(codes)):
            dist = hamming_distance(template, codes[i])
            match = (dist < threshold)
            y_true.append(1)   # vì đây là cùng person
            y_pred.append(1 if match else 0)

        # so với template của người khác (negative test)
        for other_person, other_codes in persons.items():
            if other_person == person: continue
            for code in other_codes:
                dist = hamming_distance(template, code)
                match = (dist < threshold)
                y_true.append(0)   # khác người
                y_pred.append(1 if match else 0)

    # tính metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    return acc, pre, rec, f1

if __name__ == "__main__":
    acc, pre, rec, f1 = evaluate("codes")
    print("="*40)
    print(f"{'Accuracy':<10}: {acc*100:.2f}%")
    print(f"{'Precision':<10}: {pre*100:.2f}%")
    print(f"{'Recall':<10}: {rec*100:.2f}%")
    print(f"{'F1-Score':<10}: {f1*100:.2f}%")
    print("="*40)
