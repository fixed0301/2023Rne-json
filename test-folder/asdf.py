import cv2
import numpy as np

def apply_outfocus_effect(image):
    height, width = image.shape[:2]
    blur_radius = min(height, width) // 2
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    # 마스크 생성
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), blur_radius, 255, -1)
    # 원형 마스크의 경계를 흐림
    soft_mask = cv2.GaussianBlur(mask, (301, 301), 0)

    result = cv2.copyTo(image, soft_mask, blurred)
    return result

image = cv2.imread('multiperson.jpg')

outfocus_result = apply_outfocus_effect(image)

cv2.imshow('outfocus_result', outfocus_result)
#cv2.imwrite('./outfocused.jpg', outfocus_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
