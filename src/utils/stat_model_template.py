import numpy as np
import cv2 as cv
from collections import deque
from tqdm import tqdm
from edge_det_utils import orchecterate_canny_arr
# 207 - 628
# width: 2160 , height: 3840
# (772, 3840, 2160)
# 400 , 700 ,, 2160 , 700 ,, 2160 , 2600 ,, 400 , 2600
# 1790 - 2980
# 2290 - 1370
def read_frames(dir_path = "/home/sgarg10/gear_defect_det/GDIM/Simulink/Defect_Gear_1.mp4"):
    teeth_collection = []
    cap = cv.VideoCapture(dir_path)
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    tot_f = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total = tot_f, desc="Processings Frames")
    print(f"width: {width} , height: {height}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
        teeth_collection.append(gray)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    return teeth_collection , width , height


def gt_frames(teeth_all: np.array , start , end):
    gtf = teeth_all[start:end , ...]
    print(f"good frames: {gtf.shape}")
    return gtf

def otsu_bin(frame: np.array , thresh):
    _ , binary = cv.threshold(frame , thresh , 255 , cv.THRESH_BINARY_INV)
    cv.imwrite('otsu_check_02.png' , binary)
    return binary

def otsu_thresh(hist: np.array):
    total_pix = hist.sum()
    probs = hist.ravel() / total_pix
    bins = np.arange(256)
    mu_total = np.dot(bins , probs)
    thresh = 0
    var = 0
    for t in tqdm(range(1 , 256) , desc="Calculating best Otsu Threshold"):
        w0 = np.sum(probs[:t])
        if w0 == 0: continue
        
        w1 = 1.0 - w0
        if w1 == 0: break
        
        mu0 = np.dot(bins[:t], probs[:t]) / w0
        mu1 = (mu_total - (mu0*w0)) / w1
        var_loc = w0*w1*((mu0 - mu1)**2)
        if var_loc > var:
            var = var_loc
            thresh = t
    
    print(f"Otsu Max Var: {var}")
    print(f"Otsu Best intensity: {thresh}")
    return var , thresh
 
def master_hist(gtf: np.array):
    master_hist = np.zeros((256 , 1))
    for f in tqdm(range(gtf.shape[0]) , desc="Creating Master Histogram for Pixel Intensity ranging from [0 , 255]"):
        hist = cv.calcHist([gtf[f]] , [0] , None , [256] , [0 , 256])
        master_hist += hist
    return master_hist
    
def bg_sub(gtf: np.array , check_int: int):
    backsub = cv.createBackgroundSubtractorMOG2(history=420)
    backsub2 = cv.createBackgroundSubtractorKNN(history=420 , detectShadows=1)
    lr = 0.001
    deque(tqdm(((backsub.apply(gtf[i] , learningRate=lr) , backsub2.apply(gtf[i] , learningRate=lr)) for i in range(gtf.shape[0] - 1)) , total=int(gtf.shape[0]-1) , desc="training MOG and KNN") , maxlen=0)
    # fgm = backsub.apply(gtf[-1 , ...])
    fgm2 = backsub2.apply(gtf[check_int , ...])
    # print(f"fgm: {fgm.shape}")
    # cv.imwrite('fg_MOG1.png' , fgm)
    
    # print(f"fgm2: {fgm2.shape}")
    # cv.imwrite('fg_KNN1.png' , fgm2)
    return fgm2

def roi_def(binary: np.array , img: np.array):
    mask = np.zeros_like(binary)
    cv.rectangle(mask , (700 , 700) , (2160, 2600) , 255 , -1)
    fin_mask = cv.bitwise_and(binary , mask)
    orig_mask = cv.bitwise_and(img , mask)
    return fin_mask[700:2600 , 700:2160] , orig_mask

def s_moments(mask: np.array):
    mask = (mask > 0).astype(np.uint8)
    M = cv.moments(mask)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    area = M['m00']
    print(f"area: {area} , COM x: {cx} , COM y: {cy}")
    return cx , cy , area

def otsu_knn(binary_otsu: np.array , fg_knn: np.array):
    ref_mask = cv.bitwise_and(binary_otsu , cv.dilate(fg_knn , (20,20), iterations=4))
    kernel = np.ones((5,2) , np.uint8)
    solid_mask = cv.morphologyEx(ref_mask , cv.MORPH_OPEN, kernel)
    solid_mask = (solid_mask > 0).astype(np.uint8) * 255
    return solid_mask

def write_video(fname: str , f_obj: np.array):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(f'{fname}.mp4' , fourcc , 20.0 , (2160 , 3840) , isColor=0)
    deque(tqdm((out.write(f_obj[i]) for i in range(f_obj.shape[0])) , total = f_obj.shape[0] , desc = "Writing Video") , maxlen=0)
    
def radius_boundaries(area):
    r = np.sqrt(area / np.pi)
    delta = int(r * 0.2)
    return r , delta

def cv_canny_v(binary: np.ndarray):
    return cv.Canny(binary , 100 , 200)

def extract_triple_wave(img: np.ndarray , y_com , strip_ht=10 , inter_strip=0.3):
    offset = img.shape[0] * inter_strip
    y_anchors = [y_com - offset, y_com, y_com + offset]
    waves = [extract_wave(img, y, strip_ht) for y in y_anchors]
    return waves

def extract_wave(img: np.ndarray , y_com , height=10):
    y_start = max(0 , int(y_com - height))
    y_end = min(img.shape[0], int(y_com + height))
    strip = img[y_start:y_end , :]
    return np.mean(strip, axis=0)

# def calculate_radial_profile(ellipse: cv.typing.RotatedRect, n=1000):
#     (xc,yc) , (MA , ma) , angle = ellipse
#     a = MA/2.0
#     b = ma/2.0
    
#     phi = np.radians(angle)
    
#     theta = np.linspace(0 , 2*np.pi, n,endpoint=False)
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
    
#     underside = np.sqrt((b*cos_theta)**2 + (a*sin_theta)**2)
#     r_theta = (a*b) / underside
    
#     x_loc = r_theta*cos_theta
#     y_loc = r_theta*sin_theta
    
#     x = xc + (x_loc * np.cos(phi) - y_loc*np.sin(phi))
#     y = yc + (x_loc * np.sin(phi) + y_loc*np.cos(phi))
    
#     return x,y
def horizontal_shift(wave_cur , wave_arr):
    
    N= len(wave_cur)
    w1 = (wave_cur - np.mean(wave_cur)) / (np.std(wave_cur) + 1e-5)
    w2 = (wave_arr - np.mean(wave_arr)) / (np.std(wave_arr) + 1e-5)
    
    f1 = np.fft.fft(w1)
    f2 = np.fft.fft(w2)
    
    cps = (f1*np.conj(f2)) / (np.abs(f1*np.conj(f2)) + 1e-12)
    shift_vector = np.fft.ifft(cps).real
    shift = np.argmax(shift_vector)
    
    y0 = shift_vector[shift]
    y_left=shift_vector[(shift-1) % N]
    y_right=shift_vector[(shift-1) % N]
    delta = (y_right - y_left) / (2*(2*y0 - y_right - y_left) + 1e-12)
    fin_shift = shift + delta
    if fin_shift > N //2:
        fin_shift -= N
        
    return fin_shift

if __name__ == '__main__':
    teeths , w , h = read_frames()
    fame_coll = gt_frames(np.array(teeths) , 207 , 628)
    check_frame_int = 23
    fg_mask = bg_sub(fame_coll , check_frame_int)
    hist = master_hist(fame_coll)
    var , thresh = otsu_thresh(hist)
    binary = otsu_bin(fame_coll[check_frame_int] , thresh)
    solid_mask = otsu_knn(binary , fg_mask)
    fin_mask , orig_mask = roi_def(solid_mask , fame_coll[check_frame_int])
    # print(f"final mask Shape , width: {fin_mask.shape[1]} , height: {fin_mask.shape[0]}")
    cx , cy , area = s_moments(fin_mask)
    r_com , delta = radius_boundaries(area)
    print(f"r_com: {r_com} , delta: {delta}")
    cv.imwrite("solid_mask.png" , solid_mask)
    cv.imwrite("otsu_roi.png" , fin_mask)
    cv.imwrite("img_roi.png" , orig_mask)
    waves = extract_triple_wave(fin_mask , cy)
    print(np.array(waves).shape)
    # edge_img , hysteresis_high = orchecterate_canny_arr(fin_mask,5,1)
    # contours , _ = cv.findContours(fin_mask , cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    # ellipse = cv.fitEllipse(max(contours , key=cv.contourArea))
    
    
    # col_img = cv.cvtColor(np.zeros_like(fin_mask) , cv.COLOR_GRAY2BGR)
    # cv.drawContours(col_img , contours , -1 , (0 , 255 , 0) , 2)
    # cv.ellipse(col_img,ellipse,(255 , 0 , 0), 2)
    # cv.imwrite('contaour_mask.png', col_img)
    
    # circles = cv.HoughCircles(edge_img[700:2600, 700:2160], cv.HOUGH_GRADIENT, dp=1, param1=hysteresis_high ,param2=500, minDist=50 , minRadius=int(r_com-delta) , maxRadius=int(r_com+delta))
    # print(circles)
    
    
 # Macro Pipeline -> parallel
 # Phase correlation for sub-pixel horizontal shift
 # fuzzy matching the 1D waves
 # teeth angle angle and Discontinuity
 
 
 # Golden Template Extraction
 # Convolve over the 1D wave using the GABOR locator kernel
 # Isolate ROI using Hanning window Extraction
 # calculate mean and variance map per pixel
 # z_score computation per pixel, high z-score pitting/stains -> parallel
 # Sobel + LBP -> LBP_unformity low and Sobel(grad mag) High -> surface defect
 # 1D fourier Projection from ROI -> parallel
 # Entropy computation -> parallel
 # scoring function -> not yet decided
