# motion blur
import cv2
import numpy as np

KERNEL_SIZE = 10

def motion_blur(img, t):
    # 生成运动模糊卷积核
    kernel_size = int(KERNEL_SIZE * t)  # 根据模糊程度 t 确定卷积核大小
    kernel = np.eye(kernel_size) / kernel_size
    
    # 对图像进行卷积，得到运动模糊的效果
    blurred = cv2.filter2D(img, -1, kernel)
    
    return blurred

import cv2
import numpy as np

# 定义图片大小和噪声均值、方差
img_size = (360, 640, 3)
mean = 0
var = 20

# 生成高斯噪声
sigma = var ** 0.5
# gaussian分布np.random.normal
gaussian = np.random.normal(mean, sigma, img_size)

def main():
    # 生成灰度图像，并加入高斯噪声
    img = np.zeros(img_size, dtype=np.uint8)
    cv2.randu(img, 0, 255)
    img_noisy = cv2.add(img, gaussian.astype(np.uint8))
    
    # img_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_GRAY2BGR)

    # 显示并保存图片
    cv2.imshow('Noisy Image', img_noisy)
    cv2.imwrite('noisy_image.png', img_noisy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
