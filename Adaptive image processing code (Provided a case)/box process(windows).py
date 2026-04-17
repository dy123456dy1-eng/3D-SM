import os
import cv2
import numpy as np
from datetime import datetime
import re
from collections import defaultdict


def process_reactors(batch_folder):
    """Process reactor images in the specified batch folder"""

    # ================== Path settings ==================
    input_folder = "input"
    output_colored_root = "output"

    os.makedirs(output_colored_root, exist_ok=True)

    reactor_names = ["R1-Left", "R2-Mid", "R3-Right"]
    images_per_row = 90  # Number of images per row in mosaic

    # ================== Fixed ROI coordinates ==================
    rois = [
        (890, 702, 15, 215),  # R1-Left
        (797, 702, 15, 215),  # R2-Mid
        (704, 702, 15, 215)   # R3-Right
    ]

    # ================== Helper functions ==================
    def format_date_for_folder(filename):
        parts = filename.split('_')
        for part in parts:
            if part.isdigit() and len(part) == 8:
                date_obj = datetime.strptime(part, "%Y%m%d")
                return f"{date_obj.month}.{date_obj.day}"
        return datetime.now().strftime("%m.%d").lstrip('0').replace('.0', '.')

    def extract_timestamp(fname):
        m = re.search(r'(\d{8})_(\d{6})', fname)
        return m.group(1) + m.group(2) if m else "0"

    # ================== Batch processing ==================
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    grouped_cropped = defaultdict(list)

    print(f"📁 Processing batch folder: {input_folder}")
    print(f"Found {len(image_files)} images")

    # ========== Cropping ==========
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Cannot read image: {filename}")
            continue

        date_folder = format_date_for_folder(filename)

        for i in range(3):
            x, y, w_r, h_r = rois[i]
            if y + h_r > img.shape[0] or x + w_r > img.shape[1]:
                print(f"[WARNING] {filename}'s {reactor_names[i]} out of bounds, skipping")
                continue

            cropped = img[y:y + h_r, x:x + w_r]
            grouped_cropped[(date_folder, reactor_names[i])].append((cropped, filename))

    # ============================ Mosaic & Pseudocolor =============================
    for (date_folder, reactor), file_list in grouped_cropped.items():

        file_list.sort(key=lambda f: extract_timestamp(f[1]))
        images = []

        # Read images and unify height
        for f in file_list:
            img = f[0]
            if img is not None:
                h = 200
                scale = h / img.shape[0]
                w = int(img.shape[1] * scale)
                images.append(cv2.resize(img, (w, h)))

        if not images:
            continue

        # ========== Mosaic ==========
        rows = []
        for i in range(0, len(images), images_per_row):
            row_imgs = images[i:i + images_per_row]

            max_h = max(im.shape[0] for im in row_imgs)
            max_w = max(im.shape[1] for im in row_imgs)

            padded = []
            for im in row_imgs:
                canvas = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
                canvas[:im.shape[0], :im.shape[1]] = im
                padded.append(canvas)

            rows.append(np.hstack(padded))

        final_img = np.vstack(rows)

        # Output path
        output_folder = os.path.join(output_colored_root, "merged", date_folder)
        os.makedirs(output_folder, exist_ok=True)
        merged_path = os.path.join(output_folder,
                                   f"{date_folder}_{reactor.split('-')[0][-1]}.png")
        cv2.imwrite(merged_path, final_img)
        print(f"✅ Merged image saved: {merged_path}")

        # ================== Automatically extract representative colors (right region) ==================
        lab_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab_img)

        h_img, w_img = lab_img.shape[:2]
        right_x = int(w_img * 0.97)

        # Supernatant takes the top
        top_region = lab_img[:int(h_img * 0.03), right_x:, :]

        # Sludge takes the bottom
        bottom_region = lab_img[int(h_img * 0.97):, right_x:, :]

        # Clustering function
        def extract_representative_color(region, k=2):
            Z = region.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        20, 1.0)
            _, labels, centers = cv2.kmeans(
                Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
            return centers[np.argmin(centers[:, 0])], centers[np.argmax(centers[:, 0])]

        sludge_lab, _ = extract_representative_color(bottom_region)
        _, water_lab = extract_representative_color(top_region)

        print(f"🎯 Sludge(LAB): {sludge_lab}, Water(LAB): {water_lab}")

        # Debug display selection area
        debug_img = final_img.copy()
        cv2.rectangle(debug_img, (right_x, int(h_img * 0.97)),
                      (w_img, h_img), (0, 255, 0), 3)
        cv2.rectangle(debug_img, (right_x, 0),
                      (w_img, int(h_img * 0.03)), (255, 0, 0), 3)

        debug_path = os.path.join(output_folder,
                                  f"{date_folder}_{reactor.split('-')[0][-1]}_auto_sample_region.png")
        cv2.imwrite(debug_path, debug_img)

        # ================== Color distance pseudocolor mapping ==================
        wL, wA, wB = 0.5, 1.2, 1.2
        weighted_lab = np.dstack([L * wL, A * wA, B * wB])

        dist_sludge = np.linalg.norm(weighted_lab -
                                     sludge_lab * [wL, wA, wB], axis=2)
        dist_water = np.linalg.norm(weighted_lab -
                                    water_lab * [wL, wA, wB], axis=2)

        norm_img = dist_water / (dist_water + dist_sludge + 1e-6)
        norm_img = np.power(norm_img, 2.0)
        norm_img = cv2.normalize(norm_img, None, 0, 1, cv2.NORM_MINMAX)

        grad_img = (norm_img * 255).astype(np.uint8)
        grad_img = cv2.GaussianBlur(grad_img, (3, 3), 0)

        colored_jet = cv2.applyColorMap(grad_img, cv2.COLORMAP_JET)
        colored_smooth = cv2.GaussianBlur(colored_jet, (5, 5), 0)
        colored_enhanced = cv2.addWeighted(
            colored_jet, 0.7, colored_smooth, 0.3, 0)

        lab_colored = cv2.cvtColor(colored_enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_colored)
        l = cv2.equalizeHist(l)
        colored_final = cv2.cvtColor(cv2.merge([l, a, b]),
                                     cv2.COLOR_LAB2BGR)

        # (Keep colored_final without bottom replacement)

        colored_path = os.path.join(output_folder,
                                    f"{date_folder}_{reactor.split('-')[0][-1]}_colored_jet.png")
        cv2.imwrite(colored_path, colored_final)
        print(f"🌈 Improved pseudocolor saved: {colored_path}")

    print("🎉 All processing completed!")


# ================== Entry point ==================
if __name__ == "__main__":
    batch_folder = r""
    process_reactors(batch_folder)
