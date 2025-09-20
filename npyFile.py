import cv2
import numpy as np
import os
import glob

def preprocess(img_path):
    """Đọc và chuẩn hóa ảnh"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    return img

def log_gabor_filter(img, frequency=0.2, sigma=3.0):
    """Lọc Log-Gabor đơn giản"""
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
    """Chuyển ảnh → mã nhị phân"""
    mean_val = np.mean(img)
    binary = (img > mean_val).astype(np.uint8)
    return binary.flatten()  # vector 0/1

def process_dataset(data_dir="data", out_dir="codes"):
    os.makedirs(out_dir, exist_ok=True)

    persons = os.listdir(data_dir)
    for person in persons:
        for eye in ["L", "R"]:
            eye_dir = os.path.join(data_dir, person, eye)
            if not os.path.exists(eye_dir):
                continue
            images = glob.glob(os.path.join(eye_dir, "*.jpg"))
            for img_path in images:
                img = preprocess(img_path)
                filtered = log_gabor_filter(img)
                iris_code = iris_to_code(filtered)

                # tên file lưu
                fname = os.path.basename(img_path).replace(".jpg", ".npy")
                save_path = os.path.join(out_dir, f"{person}_{eye}_{fname}")
                np.save(save_path, iris_code)

                print(f"Saved {save_path} | IrisCode length: {len(iris_code)}")

if __name__ == "__main__":
    process_dataset("data", "codes")