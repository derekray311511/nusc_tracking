import numpy as np
import cv2

def calculate_distance(gt_translation, pred_translation):
    return np.linalg.norm(np.array(gt_translation) - np.array(pred_translation))

def evaluate_nuscenes(predictions, ground_truths, distance_threshold=2.0):
    TP = 0
    FP = 0
    FN = 0
    matched_predictions = []

    for gt in ground_truths:
        matched = False
        for pred in predictions:
            distance = calculate_distance(gt['translation'], pred['translation'])
            if distance < distance_threshold:
                matched = True
                if pred not in matched_predictions:
                    TP += 1
                    matched_predictions.append(pred)
                break
        if not matched:
            FN += 1

    FP = len(predictions) - len(matched_predictions)
    return TP, FP, FN, matched_predictions

# Assume `predictions` and `ground_truths` are lists of dictionaries with 'translation', 'size', 'velocity'


def draw_boxes(image, predictions, ground_truths, matched_predictions):
    if image is None:
        return
    # Draw FN boxes in blue
    for gt in ground_truths:
        if gt not in matched_predictions:
            top_left = (int(gt['translation'][0] - gt['size'][0]/2), int(gt['translation'][1] - gt['size'][1]/2))
            bottom_right = (int(gt['translation'][0] + gt['size'][0]/2), int(gt['translation'][1] + gt['size'][1]/2))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    
    # Draw TP and FP boxes
    for pred in predictions:
        top_left = (int(pred['translation'][0] - pred['size'][0]/2), int(pred['translation'][1] - pred['size'][1]/2))
        bottom_right = (int(pred['translation'][0] + pred['size'][0]/2), int(pred['translation'][1] + pred['size'][1]/2))
        color = (0, 255, 0) if pred in matched_predictions else (0, 0, 255)
        cv2.rectangle(image, top_left, bottom_right, color, 2)

# `image` is your image on which you want to draw
# Make sure to adjust the bounding box drawing logic if your 'translation' represents the center.

def test():
    predictions = [
        {'translation': [5, 5], 'size': [2, 2], 'velocity': [0, 0]},
        {'translation': [10, 10], 'size': [2, 2], 'velocity': [0, 0]},
        {'translation': [3, 3], 'size': [2, 2], 'velocity': [0, 0]},
    ]
    ground_truths = [
        {'translation': [5, 5], 'size': [2, 2], 'velocity': [0, 0]},
        {'translation': [10, 10], 'size': [2, 2], 'velocity': [0, 0]},
        {'translation': [4, 4], 'size': [2, 2], 'velocity': [0, 0]},
        {'translation': [15, 15], 'size': [2, 2], 'velocity': [0, 0]},
    ]
    TP, FP, FN, matched_predictions = evaluate_nuscenes(predictions, ground_truths)
    print(TP, FP, FN)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    draw_boxes(image, predictions, ground_truths, matched_predictions)
    cv2.imshow('image', image)
    cv2.waitKey(0)

if __name__ == "__main__":
    test()
