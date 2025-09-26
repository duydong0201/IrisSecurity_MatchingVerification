# iris_seg_eval.py
import os, glob, argparse
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

# =========================
# 0) TIỆN ÍCH CHUNG
# =========================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def list_images(folder: str):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def ensure_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def clahe_gray(img, clip_limit=2.0, tile_grid=(8,8)):
    img = ensure_uint8(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)

# =========================
# 1) PHÂN ĐOẠN & CHUẨN HOÁ
# =========================
def detect_circles(gray, dp=1.2, minDist=40, param1=100, param2=30, minR=10, maxR=120):
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
        param1=param1, param2=param2, minRadius=minR, maxRadius=maxR
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles[0]))
    # Trả về theo bán kính tăng dần (giúp tách pupil nhỏ, iris lớn)
    circles = circles[np.argsort(circles[:,2])]
    return circles

def segment_pupil_iris(gray):
    """
    Trả về (cx, cy, r_pupil, r_iris) hoặc None nếu thất bại.
    """
    h, w = gray.shape
    blur = cv2.medianBlur(gray, 5)

    # Dò PUPIL (nhỏ)
    pupil_candidates = detect_circles(
        blur, dp=1.2, minDist=min(h,w)//8, param1=100, param2=20,
        minR=max(6, min(h,w)//40), maxR=min(h,w)//10
    )
    # Dò IRIS (to hơn)
    iris_candidates = detect_circles(
        blur, dp=1.2, minDist=min(h,w)//6, param1=120, param2=35,
        minR=max(20, min(h,w)//12), maxR=min(h,w)//3
    )

    if pupil_candidates is None or iris_candidates is None:
        return None

    # Lấy pupil lớn nhất trong nhóm "nhỏ" và iris lớn nhất trong nhóm "to"
    pupil = pupil_candidates[-1]    # (x, y, r)
    iris  = iris_candidates[-1]

    # Bắt buộc pupil nằm trong iris
    cx, cy, rp = int(pupil[0]), int(pupil[1]), int(pupil[2])
    Ri = int(iris[2])
    # Dùng tâm của iris (ổn định hơn nếu khác nhau)
    cxi, cyi = int(iris[0]), int(iris[1])

    # Nếu tâm chênh nhiều, nội suy tâm trung bình
    cx = int(0.5*cx + 0.5*cxi)
    cy = int(0.5*cy + 0.5*cyi)

    if rp >= Ri or Ri <= 0 or rp <= 0:
        return None
    return cx, cy, rp, Ri

def rubber_sheet_normalize(gray, cx, cy, r_in, r_out, H=64, W=256):
    """
    Chuẩn hoá iris thành ảnh HxW (r theo trục dọc, theta theo trục ngang).
    Dùng warpPolar để chuyển sang toạ độ cực rồi cắt dải [r_in, r_out].
    """
    # warpPolar cho ra (radius, angle). Đặt maxRadius = r_out + 5 để đủ vùng.
    max_radius = int(r_out + 5)
    polar = cv2.warpPolar(
        gray,
        (W, max_radius),                # (width=angles, height=radius)
        (cx, cy),
        max_radius,
        flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
    )  # shape: [radius, angle]
    # Lấy dải [r_in : r_out] theo trục radius → resize về H
    r0, r1 = max(0, r_in), min(max_radius, r_out)
    band = polar[r0:r1, :]  # shape ~ [(r_out-r_in), W]
    if band.size == 0:
        return cv2.resize(gray, (W, H))
    norm = cv2.resize(band, (W, H), interpolation=cv2.INTER_LINEAR)
    return norm

def segment_and_normalize(img_path, H=64, W=256):
    g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(img_path)
    g = clahe_gray(g)
    seg = segment_pupil_iris(g)
    if seg is None:
        # fallback: cắt giữa ảnh cho chắc
        g = cv2.resize(g, (W, H))
        return g
    cx, cy, r_in, r_out = seg
    norm = rubber_sheet_normalize(g, cx, cy, r_in, r_out, H=H, W=W)
    return norm

# =========================
# 2) LOG-GABOR & IRISCODES
# =========================
def log_gabor_kernel(shape, f0=0.2, sigma=0.55):
    """
    Tạo kernel Log-Gabor trong miền tần số (không định hướng) cho shape (H,W).
    """
    H, W = shape
    u = (np.arange(W) - W/2) / W
    v = (np.arange(H) - H/2) / H
    U, V = np.meshgrid(u, v)
    R = np.sqrt(U*U + V*V)
    # Tránh log(0)
    R[R == 0] = 1e-6
    lg = np.exp(-(np.log(R/f0)**2) / (2*(np.log(sigma)**2)))
    lg[R < 1e-6] = 0.0
    return np.fft.fftshift(lg)

def apply_log_gabor_stack(norm_img, freqs=(0.10, 0.20, 0.35), sigma=0.55):
    """
    Áp 3 tần số Log-Gabor → trả về list magnitude responses (float32).
    """
    img = norm_img.astype(np.float32)
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    outs = []
    for f0 in freqs:
        ker = log_gabor_kernel(img.shape, f0=f0, sigma=sigma)
        resp = np.fft.ifft2(np.fft.ifftshift(Fshift * ker))
        outs.append(np.abs(resp).astype(np.float32))
    return outs  # list of HxW

def responses_to_code_and_mask(responses, mask_quantile=0.15):
    """
    - Nhị phân hoá từng kênh theo median (ổn định hơn mean).
    - Mask nhiễu: pixel có độ lớn nhỏ hơn ngưỡng (quantile) ở mọi kênh → loại bỏ.
    """
    H, W = responses[0].shape
    # Code: ghép kênh → vector nhị phân
    bits = []
    strong = np.ones((H, W), dtype=bool)
    for r in responses:
        thr = np.median(r)
        b = (r > thr)
        bits.append(b.astype(np.uint8))
        # cập nhật mask "mạnh": yêu cầu ít nhất vượt quantile của kênh
        q = np.quantile(r, mask_quantile)
        strong &= (r >= q)

    code = np.concatenate([b.reshape(-1, 1) for b in bits], axis=1).reshape(-1).astype(np.uint8)

    # Lặp mask cho đủ số kênh
    mask_base = strong.reshape(-1)
    mask = np.repeat(mask_base, len(responses))
    return code, mask


def iris_to_code_and_mask(norm_img, freqs=(0.10,0.20,0.35), sigma=0.55, mask_q=0.15):
    reps = apply_log_gabor_stack(norm_img, freqs=freqs, sigma=sigma)
    return responses_to_code_and_mask(reps, mask_quantile=mask_q)

# =========================
# 3) DATA LOADER
# =========================
def _get_person_list(data_dir="data", num_persons: int | None = None):
    persons = sorted([p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))])
    if num_persons is not None and num_persons > 0:
        persons = persons[:num_persons]
    return persons

def load_dataset(data_dir="data", persons_limit: int | None = None, H=64, W=256,
                 freqs=(0.10,0.20,0.35), sigma=0.55, mask_q=0.15):
    X, M, y = [], [], []
    persons = _get_person_list(data_dir, persons_limit)
    for person in persons:
        for eye in ["L", "R"]:
            eye_dir = os.path.join(data_dir, person, eye)
            if not os.path.exists(eye_dir):
                continue
            images = list_images(eye_dir)
            for img_path in images:
                norm = segment_and_normalize(img_path, H=H, W=W)
                code, mask = iris_to_code_and_mask(norm, freqs=freqs, sigma=sigma, mask_q=mask_q)
                X.append(code)
                M.append(mask)
                y.append(person)
    if len(X) == 0:
        return None, None, None
    return np.array(X, dtype=np.uint8), np.array(M, dtype=bool), np.array(y)

# =========================
# 4) KHOẢNG CÁCH HAMMING (MASK)
# =========================
def hamming_distance_masked(c1, c2, m1, m2):
    valid = m1 & m2
    n = np.count_nonzero(valid)
    if n == 0:
        return 1.0
    return np.sum(c1[valid] != c2[valid]) / float(n)

def _infer_side_from_code_length(code_length: int, nbits_per_pixel: int):
    # code là ghép nhiều kênh → độ dài phải chia hết cho nbits_per_pixel
    if code_length % nbits_per_pixel != 0:
        raise ValueError("Code length không chia hết cho số kênh.")
    L = code_length // nbits_per_pixel
    side = int(np.sqrt(L))
    if side * side != L:
        raise ValueError("Không suy ra được kích thước ảnh vuông cho bù lệch.")
    return side

def circular_shift_min_hamming_masked(t_code, t_mask, code, mask, max_shift, nbits=3):
    """
    Bù lệch tròn theo trục cột, xét theo 'pixel' (gồm nbits liên tiếp).
    """
    if max_shift <= 0:
        return hamming_distance_masked(t_code, code, t_mask, mask), code, mask

    side = _infer_side_from_code_length(len(code), nbits_per_pixel=nbits)
    # Reshape về (H, W, nbits)
    tC = t_code.reshape(side*side, nbits).reshape(side, side, nbits)
    tM = t_mask.reshape(side*side).reshape(side, side)

    C  = code.reshape(side*side, nbits).reshape(side, side, nbits)
    M  = mask.reshape(side*side).reshape(side, side)

    best_d = 1.0
    bestC, bestM = code, mask

    for s in range(-max_shift, max_shift+1):
        Cshift = np.roll(C, shift=s, axis=1)
        Mshift = np.roll(M, shift=s, axis=1)

        flatC = Cshift.reshape(side*side, nbits).reshape(-1)
        flatM = Mshift.reshape(-1)

        d = hamming_distance_masked(t_code, flatC, t_mask, flatM)
        if d < best_d:
            best_d, bestC, bestM = d, flatC, flatM

    return best_d, bestC, bestM

# =========================
# 5) ĐÁNH GIÁ
# =========================
def evaluate_classify(data_dir="data", persons_limit=None, test_size=0.3, random_state=42):
    X, M, y = load_dataset(data_dir, persons_limit)
    if X is None:
        print("Không có dữ liệu.")
        return

    Xtr, Xte, Mtr, Mte, ytr, yte = train_test_split(
        X, M, y, test_size=test_size, stratify=y, random_state=random_state
    )

    y_pred = []
    for i, code in enumerate(Xte):
        dists = [hamming_distance_masked(code, tr, Mte[i], Mtr[j]) for j, tr in enumerate(Xtr)]
        idx = int(np.argmin(dists))
        y_pred.append(ytr[idx])

    acc = accuracy_score(yte, y_pred)
    pre = precision_score(yte, y_pred, average="macro", zero_division=0)
    rec = recall_score(yte, y_pred, average="macro", zero_division=0)
    f1  = f1_score(yte, y_pred, average="macro", zero_division=0)

    print("="*44)
    print("Mode           : Classification (1-NN, masked Hamming)")
    print(f"Accuracy       : {acc*100:.2f}%")
    print(f"Precision      : {pre*100:.2f}%")
    print(f"Recall         : {rec*100:.2f}%")
    print(f"F1-Score       : {f1*100:.2f}%")
    print("="*44)

def find_best_threshold(labels, distances, steps=100):
    best_t, best_f1 = 0.0, 0.0
    ts = np.linspace(0.05, 0.6, steps)
    y_true = np.array(labels).astype(int)
    for t in ts:
        y_pred = (distances < t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def compute_eer(labels, distances):
    # EER: điểm FAR = FRR
    y = np.array(labels).astype(int)
    d = np.array(distances)
    # sort theo ngưỡng tăng
    ths = np.unique(np.round(d, 6))
    fars, frrs = [], []
    for t in ths:
        pred = (d < t).astype(int)
        tp = np.sum((pred==1)&(y==1))
        fp = np.sum((pred==1)&(y==0))
        tn = np.sum((pred==0)&(y==0))
        fn = np.sum((pred==0)&(y==1))
        far = fp / (fp + tn + 1e-9)
        frr = fn / (tp + fn + 1e-9)
        fars.append(far); frrs.append(frr)
    fars = np.array(fars); frrs = np.array(frrs)
    idx = np.argmin(np.abs(fars - frrs))
    return 0.5*(fars[idx] + frrs[idx])

def refine_template_with_alignment(train_codes, train_masks, epochs=1, max_shift=0, nbits=3):
    """
    Căn chỉnh theo circular shift + lấy majority vote (bit-wise).
    """
    if len(train_codes) == 0:
        raise ValueError("Empty train set")

    # Khởi tạo template bằng median bit
    T = (np.mean(train_codes, axis=0) >= 0.5).astype(np.uint8)

    # Mask gốc theo majority
    TM_base = (np.mean(train_masks, axis=0) > 0.2)  # shape = H*W
    TM = np.repeat(TM_base, nbits)                  # lặp cho đủ số kênh

    if epochs <= 1 and max_shift <= 0:
        return T, TM

    for _ in range(max(1, epochs)):
        alignedC, alignedM = [], []
        for c, m in zip(train_codes, train_masks):
            _, c2, m2 = circular_shift_min_hamming_masked(
                T, TM, c, m, max_shift=max_shift, nbits=nbits
            )
            alignedC.append(c2)
            alignedM.append(m2)

        A = np.stack(alignedC, axis=0)
        AM = np.stack(alignedM, axis=0)

        # majority bits + majority mask
        T = (np.mean(A, axis=0) >= 0.5).astype(np.uint8)
        TM_base = (np.mean(AM, axis=0) > 0.5)
        TM = np.repeat(TM_base, nbits)

    return T, TM


def evaluate_verification(
    data_dir="data",
    persons_limit=None,
    test_size=0.3,
    random_state=42,
    epochs=1,
    max_shift=0,
    H=64, W=256, freqs=(0.10,0.20,0.35), sigma=0.55, mask_q=0.15
):
    # Gom theo (person, eye)
    grouped = {}
    persons = _get_person_list(data_dir, persons_limit)
    for person in persons:
        for eye in ["L", "R"]:
            eye_dir = os.path.join(data_dir, person, eye)
            if not os.path.exists(eye_dir):
                continue
            codes, masks = [], []
            for img_path in list_images(eye_dir):
                norm = segment_and_normalize(img_path, H=H, W=W)
                c, m = iris_to_code_and_mask(norm, freqs=freqs, sigma=sigma, mask_q=mask_q)
                codes.append(c); masks.append(m)
            if codes:
                grouped.setdefault(person, {})[eye] = (np.array(codes, dtype=np.uint8),
                                                       np.array(masks, dtype=bool))

    labels, distances = [], []
    templates, testsets = {}, {}

    # Chia train/test từng (person,eye) → tạo template
    for person, eyes in grouped.items():
        for eye, (codes, masks) in eyes.items():
            if len(codes) < 2:
                continue
            c_tr, c_te, m_tr, m_te = train_test_split(
                codes, masks, test_size=test_size, random_state=random_state
            )
            T, TM = refine_template_with_alignment(c_tr, m_tr, epochs=epochs, max_shift=max_shift, nbits=len(freqs))
            templates[(person, eye)] = (T, TM)
            testsets[(person, eye)]  = (c_te, m_te)

    if len(testsets) == 0:
        print("No sufficient data to evaluate verification.")
        return

    # Genuine
    for key, (c_te, m_te) in testsets.items():
        T, TM = templates[key]
        for c, m in zip(c_te, m_te):
            d, _, _ = circular_shift_min_hamming_masked(T, TM, c, m, max_shift=max_shift, nbits=len(freqs))
            distances.append(d); labels.append(1)

    # Impostor
    keys = list(testsets.keys())
    for i, keyA in enumerate(keys):
        TA, TMA = templates[keyA]
        for j, keyB in enumerate(keys):
            if keyA == keyB: 
                continue
            c_teB, m_teB = testsets[keyB]
            for c, m in zip(c_teB, m_teB):
                d, _, _ = circular_shift_min_hamming_masked(TA, TMA, c, m, max_shift=max_shift, nbits=len(freqs))
                distances.append(d); labels.append(0)

    distances = np.array(distances); labels = np.array(labels).astype(int)

    # Tối ưu theo F1
    best_t, best_f1 = find_best_threshold(labels, distances, steps=150)
    y_pred = (distances < best_t).astype(int)

    # Chỉ số
    acc = accuracy_score(labels, y_pred)
    pre = precision_score(labels, y_pred, zero_division=0)
    rec = recall_score(labels, y_pred, zero_division=0)
    f1  = f1_score(labels, y_pred, zero_division=0)
    # ROC-AUC (cần cả 0 và 1)
    try:
        auc = roc_auc_score(labels, -distances)  # khoảng cách nhỏ là 'positive'
    except Exception:
        auc = float("nan")
    eer = compute_eer(labels, distances)

    print("="*44)
    print("Mode           : Verification (per-eye)")
    print(f"Best threshold : {best_t:.3f}")
    print(f"Accuracy       : {acc*100:.2f}%")
    print(f"Precision      : {pre*100:.2f}%")
    print(f"Recall         : {rec*100:.2f}%")
    print(f"F1-Score       : {f1*100:.2f}%")
    print(f"ROC-AUC        : {auc:.4f}")
    print(f"EER            : {eer*100:.2f}%")
    print(f"Epochs         : {epochs}")
    print(f"Max shift      : {max_shift}")
    print("="*44)

# =========================
# 6) MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Iris feature extraction and evaluation (improved)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--mode", type=str, choices=["classify", "verify"], default="verify")
    parser.add_argument("--num_persons", type=int, default=0, help="Limit first N persons (0=all)")
    parser.add_argument("--epochs", type=int, default=1, help="Template refinement epochs (verification)")
    parser.add_argument("--max_shift", type=int, default=0, help="Circular shift compensation (columns)")
    parser.add_argument("--norm_h", type=int, default=64, help="Normalized iris height")
    parser.add_argument("--norm_w", type=int, default=256, help="Normalized iris width")
    parser.add_argument("--freqs", type=str, default="0.10,0.20,0.35", help="Log-Gabor frequencies CSV")
    parser.add_argument("--sigma", type=float, default=0.55, help="Log-Gabor sigma (in log domain)")
    parser.add_argument("--mask_q", type=float, default=0.15, help="Noise mask quantile per channel (0..1)")
    args = parser.parse_args()

    persons_limit = args.num_persons if args.num_persons and args.num_persons > 0 else None
    freqs = tuple(float(x) for x in args.freqs.split(","))

    if args.mode == "classify":
        evaluate_classify(args.data_dir, persons_limit)
    else:
        evaluate_verification(
            data_dir=args.data_dir,
            persons_limit=persons_limit,
            epochs=max(1, args.epochs),
            max_shift=max(0, args.max_shift),
            H=args.norm_h, W=args.norm_w, freqs=freqs, sigma=args.sigma, mask_q=args.mask_q
        )

if __name__ == "__main__":
    main()
