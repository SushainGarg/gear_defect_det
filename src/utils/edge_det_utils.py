import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import binary_propagation

def read_img(img_path):
    return cv2.imread(img_path, 0)

def write_img(path , img_mat):
    return cv2.imwrite(path , img_mat)

def gaussian_kernel_2d(size: int , sigma: float):
    n = -(size-1)/2
    coord_arr = np.linspace(n , -n , size)
    x,y = np.meshgrid(coord_arr , coord_arr)
    denom = np.square(sigma)
    kernel = np.exp(-0.5*((np.square(x) + np.square(y))/denom))
    return kernel / np.sum(kernel)

def convolve_gauss_kernel_2d(kernel, img_mat):
    result = np.zeros_like(img_mat , dtype=np.float64)
    kernel_dim = kernel.shape[0]
    offset = kernel_dim // 2 # For a 5x5, offset is 2
    
    # Range should ensure the kernel doesn't go out of bounds
    # For a 5x5 kernel and 1000px image, i goes 0 to 995
    for i in tqdm(range(img_mat.shape[0] - kernel_dim + 1), desc="x axis"):
        for j in range(img_mat.shape[1] - kernel_dim + 1):
            # Calculate the weighted sum
            neighborhood = img_mat[i:i+kernel_dim, j:j+kernel_dim]
            # Store it in the center-aligned pixel of the result
            result[i + offset, j + offset] = np.sum(kernel * neighborhood)
            
    return result.astype(np.uint8)

def intensity_gradient(img_mat):
    img = img_mat.astype(np.float32)
    
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    
    gx[1:-1 , 1:-1] = (
        (img[0:-2 , 2:] - img[0:-2 , 0:-2]) + 
        2*(img[1:-1 , 2:] - img[1:-1 , 0:-2]) + 
        (img[2:, 2:] - img[2: , 0:-2])
    )
    
    gy[1:-1 , 1:-1] = (
        (img[2:, 0:-2] - img[0:-2 , 0:-2]) + 
        2*(img[2: , 1:-1] - img[0:-2 , 1:-1]) +
        (img[2: , 2:] - img[0:-2 , 2:])
    )
    
    mag = np.hypot(gx , gy)
    mag *= 255.0/(np.max(mag) if np.max(mag) > 0 else 1)
    # gx *= 255.0/(np.max(gx) if np.max(gx) > 0 else 1)
    # gy *= 255.0/(np.max(gy) if np.max(gy) > 0 else 1)
    dirn = np.arctan2(gy , gx)
    
    return {'mag': mag,
            'gx': gx,
            'gy':gy,
            'dirn': dirn}

def nm_sup(mag : np.array , dirn: np.array):
    print(mag[mag>255].shape)
    print(mag[mag>255][:20])
    angle = (dirn * 180.0) / np.pi
    angle[angle<0] += 180.0
    bins = np.int32((angle + 22.5) /45) % 4
    # print(bins[0:10 , 0:10])
    M = np.pad(mag , 1 , "constant")
    
    mask0 = (bins==0)
    mask1 = (bins==1)
    mask2 = (bins==2)
    mask3 = (bins==3)

    nms = np.zeros_like(mag)
    
    compare_xl    = mag>= M[1:-1,:-2]
    c_xr          = mag>= M [1:-1,2:]
    c_yup         = mag>= M[:-2,1:-1]
    c_ydown       = mag>= M [2:,1:-1]
    c_diag45up    = mag>= M  [:-2,2:]
    c_diag45down  = mag>= M  [2:,:-2]
    c_diag135up   = mag>= M [:-2,:-2]
    c_diag135down = mag>= M   [2:,2:]
    
    print(f"mask0: {mask0.shape} , compare_xl: {compare_xl.shape} , c_xr: {c_xr.shape} , mag : {mag.shape} , bins: {bins.shape} , nms: {nms.shape}")
    nms = np.where(mask0 & compare_xl & c_xr , mag , nms)
    nms = np.where(mask1 & c_diag45up & c_diag45down , mag , nms)
    nms = np.where(mask2& c_yup & c_ydown , mag, nms)
    nms = np.where(mask3 & c_diag135up & c_diag135down , mag , nms)
    
    print(nms[nms>255][:20])
    # print(nms[:10 , :10])
    return nms

def hysterisis(nms , low , high):
    strong_mask = nms>=high
    weak_mask = nms>=low
    fin_mask = binary_propagation(strong_mask , mask=weak_mask)
    return (fin_mask*255).astype(np.uint8)

def thresh_comp(nms: np.ndarray , sigma):
    print(nms[nms>0][:20])
    v = np.median(nms[nms>0])
    print(f"Degub Median: {v}")
    low = int(max(0 , (1.0 - sigma) * v))
    high = int(min(255 , (1.0 + sigma) * v))
    print(f"Hysteresis Thresholds: {high} - {low}")
    return low , high

def orchecterate_canny_arr(img: np.ndarray , gauss_kernel_size: int, gauss_std: int, sigma = 0.33):
    kernel = gaussian_kernel_2d(gauss_kernel_size,gauss_std)
    res = convolve_gauss_kernel_2d(kernel , img)
    int_grad = intensity_gradient(res)
    nms = nm_sup(int_grad['mag'] , int_grad['dirn'])
    low , high = thresh_comp(nms , sigma)
    canny_edge = hysterisis(nms , low , high)
    write_img(f'edge_det010.png' , canny_edge)
    return canny_edge , high

def orchecterate_canny_path(img_path: str):
    img = read_img(img_path)
    kernel = gaussian_kernel_2d(11,3)
    res = convolve_gauss_kernel_2d(kernel , img)
    int_grad = intensity_gradient(res)
    nms = nm_sup(int_grad['mag'] , int_grad['dirn'])
    low , high = thresh_comp(nms , sigma=0.33)
    canny_edge = hysterisis(nms , low , high)
    write_img('edge_det010.png' , canny_edge)

# theta = rad * 360/pi
# binning rads into 0 , 90 , 45 and 135
# equivalent rads = 0 , pi/2 , pi/4 , 3pi/4
# create a n*n array, for every entry, i,j starting from -(n-1)/2 to (n-1)/2 , calc l2 norm with x=i , y=j
# upper triangular matrix of distance calculations
if __name__ == '__main__':
    orchecterate_canny_path("/home/sgarg10/gear_defect_det/GDIM/Simulink/Defect_Gear_1_frames/frame_00488.jpg")
#     print(img_mat.shape)
#     kernel = gaussian_kernel_2d(5, 1)
#     print(kernel.shape)
#     res = convolve_gauss_kernel_2d(kernel , img_mat)
#     print(res.shape)
#     int_grad = intensity_gradient(res)
#     print(int_grad[-1][:10 , :10])
#     nms = nm_sup(int_grad[0] , int_grad[-1])
#     v = np.median(nms[nms>0])
#     sigma = 0.33
#     low = int(max(0 , (1.0 - sigma) * v))
#     high = int(min(255 , (1.0 + sigma) * v))
#     fin_img = hysterisis(nms , low , high)
#     write_img('fin_canny.png', fin_img)
#     # for i , img in enumerate(int_grad):
    #     write_img(f"/home/sgarg10/gear_defect_det/playground/test_{i+100}.png" , img.astype(np.uint8))
        # img_mat_2 = read_img("/home/sgarg10/gear_defect_det/playground/test_1.png")
        
        # print("Original patch:\n", img_mat[10:13, 10:13])
        # print("Blurred patch:\n", res[10:13, 10:13])
        
        # diff = np.abs(img_mat.astype(float) - res.astype(float))
        # print("Max_ Differnece:\n", np.max(diff))
    # imge_id = 1
    # write_img(f"/home/sgarg10/gear_defect_det/playground/test_{imge_id}.png" , res)
    # # img_mat_2 = read_img("/home/sgarg10/gear_defect_det/playground/test_1.png")
    
    # print("Original patch:\n", img_mat[10:13, 10:13])
    # print("Blurred patch:\n", res[10:13, 10:13])
    
    # diff = np.abs(img_mat.astype(float) - res.astype(float))
    # print("Max_ Differnece:\n", np.max(diff))