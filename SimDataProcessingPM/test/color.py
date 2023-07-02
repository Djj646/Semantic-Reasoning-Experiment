import numpy as np
import cv2

def generate_color_block(rgb, filename):
    # 创建一个空白图像
    color_block = np.zeros((240, 300, 3), dtype=np.uint8)
    color_block[:, :] = rgb

    # 保存图像
    cv2.imwrite(filename, color_block)

    # 显示图像
    cv2.imshow("Color Block", color_block)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rgb = (0, 0, 0)
generate_color_block(rgb, "Color Block.png")