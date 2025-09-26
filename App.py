import cv2
import numpy as np
import glob, os
from sklearn.model_selection import train_test_split

# ===============================
# Các hàm xử lý ảnh -> IrisCode
# ===============================
def segment_iris(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {img_path}")
    img_blur = cv2.medianBlur(img, 5)

    # Dò hình tròn (iris)
    circles = cv2.HoughCircles(
        img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=100, param2=30, minRadius=20, maxRadius=80
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        iris = img[max(0,y-r):y+r, max(0,x-r):x+r]
        if iris.size > 0:
            iris = cv2.resize(iris, (128, 128))
            return iris
    # fallback: resize toàn ảnh
    return cv2.resize(img, (128,128))

def log_gabor_filter(img, frequency=0.2, sigma=3.0):
    rows, cols = img.shape
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    log_gabor = np.exp((-(np.log(radius/frequency))**2) / (2 * (np.log(sigma))**2))
    log_gabor[radius == 0] = 0
    img_fft = np.fft.fft2(img)
    img_fftshift = np.fft.fftshift(img_fft)
    filtered = np.fft.ifft2(np.fft.ifftshift(img_fftshift * log_gabor))
    return np.abs(filtered)

def iris_to_code(img):
    mean_val = np.mean(img)
    binary = (img > mean_val).astype(np.uint8)
    return binary.flatten()

def hamming_distance(c1, c2):
    return np.sum(c1 != c2) / len(c1)

# ===============================
# Xác thực từ ảnh PNG với npy
# ===============================
def verify_from_image(img_path, person_id, threshold=0.2):
    # tạo code từ ảnh PNG
    iris = segment_iris(img_path)
    filtered = log_gabor_filter(iris)
    new_code = iris_to_code(filtered)

    # load npy template của person
    person_files = [
        f for f in glob.glob("codes_cropped/*.npy")
        if os.path.basename(f).split("_")[0] == person_id
    ]
    if not person_files:
        raise ValueError(f"Không tìm thấy dữ liệu .npy cho {person_id}")

    train_files, _ = train_test_split(person_files, test_size=0.3, random_state=42)
    train_codes = [np.load(f) for f in train_files]
    template = np.median(train_codes, axis=0) > 0.5

    # so khớp
    dist = hamming_distance(template, new_code)
    return dist < threshold, dist
