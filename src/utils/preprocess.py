import cv2
import numpy as np

def read_img(img_path):
    return cv2.imread(img_path, 0)
def show(img_mat):
    cv2.imshow("gear_img" , img_mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_img(path , img_mat):
    return cv2.imwrite(path , img_mat)

def main():
    img_mat = read_img("/home/sgarg10/gear_defect_det/GDIM/Simulink/Defect_Gear_3_frames/frame_00001.jpg")
    img_mat_arr = np.array(img_mat)
    print(np.max(img_mat_arr, 1))
    imge_id = 0
    write_img(f"/home/sgarg10/gear_defect_det/playground/test_{imge_id}.png" , img_mat)
    
if __name__ == '__main__':
    main()
    