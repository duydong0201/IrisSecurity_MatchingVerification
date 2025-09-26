import os, glob, random, numpy as np
from sklearn.model_selection import train_test_split
from iris_Function import hamming_distance  # import từ file utils

def verify_identity(template, new_code, threshold):
    dist = hamming_distance(template, new_code)
    return dist < threshold, dist

if __name__ == "__main__":
    with open("best_threshold.txt", "r") as f:
        lines = f.readlines()
        last_line = lines[-1].strip()
        best_t = float(last_line.split(":")[-1])

   
    with open("test_list.txt", "r") as f:
        test_files = [line.strip() for line in f.readlines()]


    test_file = random.choice(test_files)
    person_id = os.path.basename(test_file).split("_")[0]
    new_code = np.load(test_file)


    person_files = [
        f for f in glob.glob("codes_cropped/*.npy")
        if os.path.basename(f).split("_")[0] == person_id
    ]
    train_files, _ = train_test_split(person_files, test_size=0.3, random_state=42)
    train_codes = [np.load(f) for f in train_files]
    template = np.median(train_codes, axis=0) > 0.5


    is_match, dist = verify_identity(template, new_code, best_t)
    match_rate = (1 - dist) * 100  # tỉ lệ match %

    print("="*40)
    print(f"Person ID       : {person_id}")
    print(f"Test sample     : {os.path.basename(test_file)}")
    print(f"Hamming dist    : {dist:.4f}")
    print(f"Threshold used  : {best_t:.3f}")
    print(f"Match rate      : {match_rate:.2f}%")
    print("Result          : MATCH ✅" if is_match else "Result          : NOT MATCH ❌")
    print("="*40)
