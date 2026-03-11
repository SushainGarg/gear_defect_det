import numpy as np
import cv2 as cv
from tqdm import tqdm

def border_padding(arr: np.ndarray):
    return None

def get_neighbors(x,y):
    return[
        (x , y-1) , (x+1 , y-1) , (x+1 , y) , (x+1 , y+1),
        (x, y+1) , (x-1 , y+1) , (x-1 , y) , (x-1 , y-1)
    ]

def find_contours(bin: np.ndarray):
    h,w = bin.shape
    checked = np.zeros_like(bin , dtype=bool)
    contours = []
    
    for y in tqdm(range(1 , h-1) , desc="iterating y-axis"):
        for x in range(1 , w-1):
            if bin[y,x] == 1 and not checked[y,x]:
                contour = []
                curr_p = (x,y)
                start_p = curr_p
                
                last_diff = (x-1,y)
                
                while True:
                    contour.append(curr_p)
                    checked[curr_p[1] , curr_p[0]] = True
                    neigh = get_neighbors(curr_p[0] , curr_p[1])
                    
                    st_idx = 0
                    for i in range(8):
                        if neigh[i] == last_diff:
                            st_idx = (i+1) % 8
                            break
                    next = False
                    for i in range(8):
                        idx = (st_idx + i) % 8
                        nx , ny = neigh[idx]
                        
                        if 0<= ny < bin.shape[0] and 0<= nx <bin.shape[1]:
                            if bin[ny,nx]==1:  
                                curr_p = (nx,ny)
                                next = True
                                break
                    
                    if not next or curr_p == start_p:
                        break
                    if len(contour) > 5000:
                        contours.append(np.array(contour))
                        break
    return contours


if __name__ == "__main__":
    img = cv.imread("/home/sgarg10/gear_defect_det/solid_mask.png")
    img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    img[img>0] = 1
    print(np.max(img))
    contours = find_contours(img)
    print(contours)