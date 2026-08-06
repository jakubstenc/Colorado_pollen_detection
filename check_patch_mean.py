import cv2
import numpy as np
img1 = cv2.imread('patch_raw.jpg')
img2 = cv2.imread('patch_swapped.jpg')
print("Raw mean BGR:", img1.mean(axis=(0,1)))
print("Swapped mean BGR:", img2.mean(axis=(0,1)))
