import os
import glob
import argparse
import numpy as np

# Tái sử dụng pipeline có sẵn
from iris_seg_eval import segment_iris, log_gabor_filter, iris_to_code


def process_dataset(data_dir: str = "data", out_dir: str = "codes_cropped") -> None:
    os.makedirs(out_dir, exist_ok=True)

    persons = os.listdir(data_dir)
    for person in persons:
        for eye in ["L", "R"]:
            eye_dir = os.path.join(data_dir, person, eye)
            if not os.path.exists(eye_dir):
                continue

            images = glob.glob(os.path.join(eye_dir, "*.jpg"))
            for img_path in images:
                # 1) Cắt mống mắt bằng HoughCircles
                iris = segment_iris(img_path)

                # 2) Lọc Log-Gabor
                filtered = log_gabor_filter(iris)

                # 3) Sinh mã nhị phân
                iris_code = iris_to_code(filtered)

                # 4) Lưu thành .npy
                fname = os.path.basename(img_path).replace(".jpg", ".npy")
                save_path = os.path.join(out_dir, f"{person}_{eye}_{fname}")
                np.save(save_path, iris_code)

                print(f"Saved {save_path} | IrisCode length: {len(iris_code)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate iris binary codes after segmentation and save as .npy")
    parser.add_argument("--data_dir", type=str, default="data", help="Input images root directory: data/<person>/<L|R>/*.jpg")
    parser.add_argument("--out_dir", type=str, default="codes_cropped", help="Output directory to save .npy codes")
    args = parser.parse_args()

    process_dataset(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()


