import cv2
import numpy as np
import imutils
import json

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 15)
    return thresh

def find_document_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def extract_omr_answers(warped, config):
    # Placeholder: expects config['omr_layout'] with grid info
    # This function should be adapted for your OMR layout
    num_questions = config['omr_layout']['num_questions']
    choices_per_question = config['omr_layout']['choices_per_question']
    bubble_grid = config['omr_layout']['bubble_grid']
    bubble_size = config['omr_layout']['bubble_size']
    answers = {}
    for q in range(num_questions):
        max_fill = 0
        marked = None
        for c in range(choices_per_question):
            # Example: calculate bubble position (customize for your layout)
            x = bubble_grid[0][0] + c * (bubble_size[0] + 10)
            y = bubble_grid[0][1] + q * (bubble_size[1] + 10)
            bubble = warped[y:y + bubble_size[1], x:x + bubble_size[0]]
            total = cv2.countNonZero(bubble)
            if total > max_fill:
                max_fill = total
                marked = chr(65 + c)  # 'A', 'B', ...
        answers[q + 1] = marked
    return answers
