import os
import numpy as np
from skimage import io, measure, color

# J (IoU) 계산 함수
def calculate_J(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()  # 교집합
    union = np.logical_or(pred_mask, gt_mask).sum()          # 합집합
    return intersection / union if union > 0 else 0

# F (Boundary Similarity) 계산 함수
def calculate_F(pred_mask, gt_mask):
    pred_contours = measure.find_contours(pred_mask, 0.5)
    gt_contours = measure.find_contours(gt_mask, 0.5)

    pred_contour_len = sum(len(c) for c in pred_contours)
    gt_contour_len = sum(len(c) for c in gt_contours)

    return min(pred_contour_len, gt_contour_len) / max(pred_contour_len, gt_contour_len) if max(pred_contour_len, gt_contour_len) > 0 else 0

# 객체별 IoU 계산 함수
def calculate_objectwise_J(pred_mask, gt_mask):
    pred_labels = label_colored_objects(pred_mask)
    gt_labels = label_colored_objects(gt_mask)

    object_J_scores = []
    for pred_region in measure.regionprops(pred_labels):
        pred_region_mask = pred_labels == pred_region.label
        best_J = 0

        for gt_region in measure.regionprops(gt_labels):
            gt_region_mask = gt_labels == gt_region.label
            J = calculate_J(pred_region_mask, gt_region_mask)
            best_J = max(best_J, J)

        object_J_scores.append(best_J)

    return np.mean(object_J_scores) if object_J_scores else 0

# 객체별 Boundary Similarity 계산 함수
def calculate_objectwise_F(pred_mask, gt_mask):
    pred_labels = label_colored_objects(pred_mask)
    gt_labels = label_colored_objects(gt_mask)

    object_F_scores = []
    for pred_region in measure.regionprops(pred_labels):
        pred_region_mask = pred_labels == pred_region.label
        best_F = 0

        for gt_region in measure.regionprops(gt_labels):
            gt_region_mask = gt_labels == gt_region.label
            F = calculate_F(pred_region_mask, gt_region_mask)
            best_F = max(best_F, F)

        object_F_scores.append(best_F)

    return np.mean(object_F_scores) if object_F_scores else 0

# 색상 기반 객체 라벨링 함수
def label_colored_objects(mask):
    # 색상별 객체를 라벨링
    unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    labeled_mask = np.zeros(mask.shape[:2], dtype=int)

    for i, color in enumerate(unique_colors[1:], start=1):  # 첫 색상은 배경(0)
        color_mask = np.all(mask == color, axis=-1)
        labeled_mask[color_mask] = i

    return measure.label(labeled_mask)

# 시퀀스 설정
sequence_indices = [
    "00001", "00009", "00015", "00017", "00019", "00021", "00025", "00029",
    "00031", "00036", "00039", "00049", "00050", "00051", "00058", "00059",
    "00060", "00061", "00064", "00067"
]

# 경로 설정
pred_path = "/home/bjh/LLE_VOS/output/default/lle-vos/Annotations"
gt_path = "/mnt/hdddate/bjh/LLE_VOS/LLE_VOS/Annotations"

def evaluate_sequences():
    J_scores = []
    F_scores = []

    for seq in sequence_indices:
        print(f"Evaluating Sequence {seq}...")

        pred_seq_path = os.path.join(pred_path, seq)
        gt_seq_path = os.path.join(gt_path, seq)

        # 프레임별 점수 계산
        for frame in sorted(os.listdir(gt_seq_path)):
            pred_mask_path = os.path.join(pred_seq_path, frame)
            gt_mask_path = os.path.join(gt_seq_path, frame)

            if os.path.exists(pred_mask_path) and os.path.exists(gt_mask_path):
                # 마스크 불러오기
                pred_mask = io.imread(pred_mask_path)
                gt_mask = io.imread(gt_mask_path)

                # 객체별 J와 F 계산
                J = calculate_objectwise_J(pred_mask, gt_mask)
                F = calculate_objectwise_F(pred_mask, gt_mask)

                J_scores.append(J)
                F_scores.append(F)

    # 평균 계산
    mean_J = np.mean(J_scores)
    mean_F = np.mean(F_scores)
    print("\n=== Final Results ===")
    print(f"Mean Object-wise J (IoU) Score: {mean_J:.4f}")
    print(f"Mean Object-wise F (Boundary) Score: {mean_F:.4f}")

    return mean_J, mean_F

# 실행
evaluate_sequences()
