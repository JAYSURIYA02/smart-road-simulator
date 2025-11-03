import cv2

# read the two images (must be same size and type)
img1 = cv2.imread(r'C:\Users\jaysu\OneDrive\Desktop\Nexsync project\lanenet-lane-detection\output\image\lane_result.jpg')
img2 = cv2.imread(r'C:\Users\jaysu\OneDrive\Desktop\Nexsync project\lanenet-lane-detection\data\tusimple_test_image\2.jpg')

img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
# blend them using transparency (alpha and beta are weights)
alpha = 0.5 # transparency for img1
beta = 1   # transparency for img2

blended = cv2.addWeighted(img1, alpha, img2, beta, 0)

cv2.imshow('Blended Image', blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
