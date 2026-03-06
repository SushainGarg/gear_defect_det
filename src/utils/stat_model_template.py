import numpy as np
import cv2 as cv
from collections import deque
from tqdm import tqdm
from edge_det_utils import orchecterate_canny_arr
from concurrent.futures import ThreadPoolExecutor
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

def otsu_bin(frames: np.array , thresh):
    binary = (frames < thresh).astype(np.uint8) * 255
    cv.imwrite('otsu_check_02.png' , binary[123])
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
    
def bg_sub(gtf: np.array , check_int: int = 0):
    backsub = cv.createBackgroundSubtractorMOG2(history=420)
    backsub2 = cv.createBackgroundSubtractorKNN(history=420 , detectShadows=1)
    lr = 0.001
    deque(tqdm(((backsub.apply(gtf[i] , learningRate=lr) , backsub2.apply(gtf[i] , learningRate=lr)) for i in range(gtf.shape[0] - 1)) , total=int(gtf.shape[0]-1) , desc="training MOG and KNN") , maxlen=0)
    # fgm = backsub.apply(gtf[-1 , ...]) 
    fgm2 = np.array([backsub2.apply(frame, learningRate=0) for frame in gtf])
    print(f"fgm2 shape: {fgm2.shape}")
    # print(f"fgm: {fgm.shape}")
    # cv.imwrite('fg_MOG1.png' , fgm)
    
    # print(f"fgm2: {fgm2.shape}")
    # cv.imwrite('fg_KNN1.png' , fgm2)
    return fgm2

def roi_def(binary: np.array , img: np.array):
    y1, y2, x1, x2 = 700, 2600, 700, 2160
    # mask = np.zeros_like(binary)
    # cv.rectangle(mask , (700 , 700) , (2160, 2600) , 255 , -1)
    fin_mask = binary[: , y1:y2 , x1:x2]
    orig_mask = img[: , y1:y2 , x1:x2]
    return fin_mask , orig_mask

def s_moments(mask: np.array):
    # mask = (mask > 0).astype(np.uint8)
    # M = cv.moments(mask)
    areas = np.sum(mask > 0, axis=(1,2))
    H,W = mask.shape[1], mask.shape[2]
    x_coords = np.arange(W)
    y_coords = np.arange(H)
    
    m10 = np.sum(mask , axis=1) @ x_coords
    m01 = np.sum(mask , axis=2) @ y_coords
    cx_all = (m10 / 255) / (areas + 1e-6)
    cy_all = (m01 / 255) / (areas + 1e-6)
    # cx = M['m10'] / M['m00']
    # cy = M['m01'] / M['m00']
    # area = M['m00']
    print(f"area: {areas[123]} , COM x: {cx_all[123]} , COM y: {cy_all[123]}")
    return cx_all , cy_all , areas

def batch_ref_mask(fg_knn, iterations=4):
    N= fg_knn.shape[0]
    dil_stack = np.empty_like(fg_knn)
    def dilate_worker(i):
        dil_stack[i] = cv.dilate(fg_knn[i], np.ones((20,20), np.uint8), iterations=iterations)
    with ThreadPoolExecutor() as executor:
        executor.map(dilate_worker, range(N))
    return dil_stack
def otsu_knn(binary_otsu: np.array , fg_knn: np.array):
    # dil_kern = cv.dilate(fg_knn , np.ones((20,20), np.uint8), iterations=4)
    # ref_mask = (binary_otsu & dil_kern).astype(np.uint8)
    dil_stack = batch_ref_mask(fg_knn)
    ref_mask = (binary_otsu & dil_stack)
    kernel = np.ones((5,2) , np.uint8)
    # solid_mask = cv.morphologyEx(ref_mask , cv.MORPH_OPEN, kernel)
    # solid_mask = (solid_mask > 0).astype(np.uint8) * 255
    solid_mask = np.array([cv.morphologyEx(f , cv.MORPH_OPEN , kernel) for f in ref_mask])
    solid_mask[solid_mask > 0] = 255
    return solid_mask

def write_video(fname: str , f_obj: np.array):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(f'{fname}.mp4' , fourcc , 20.0 , (2160 , 3840) , isColor=0)
    deque(tqdm((out.write(f_obj[i]) for i in range(f_obj.shape[0])) , total = f_obj.shape[0] , desc = "Writing Video") , maxlen=0)
    
def radius_boundaries(area):
    r = np.sqrt(area / np.pi)
    delta = (r * 0.2).astype(np.uint8)
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

def batch_canny(stack , high , low):
    out = np.empty_like(stack)
    def worker(i):
        out[i] = cv.Canny(stack[i] , low , high)
    
    with ThreadPoolExecutor() as executor:
        executor.map(worker , range(stack.shape[0]))
    
    return out

def batch_hough(edge_arr, theta, threshold , min_theta , max_theta):
    N = edge_arr.shape[0]
    out = [None] * N
    threshold = 400
    def worker(i):
        out[i] = cv.HoughLines(edge_arr[i] , rho=1 ,theta = theta, threshold=threshold , min_theta=min_theta , max_theta = max_theta)
    
    with ThreadPoolExecutor() as executor:
        executor.map(worker , range(N))
    
    val_res = [res for res in out if res is not None]
    # print(np.array(val_res).shape)
    return val_res

def find_best_line(lines , cx , cy):
    center_line = np.zeros((2))
    l_arr = lines.reshape(-1 , 2)
    rhos = l_arr[: , 0]
    theta = l_arr[: , 1]
    
    dists = np.abs(cx*np.cos(theta) + cy*np.sin(theta) - rhos)
    min_id = np.argmin(dists)
    center_line = [theta[min_id] , rhos[min_id]]
    return center_line
    
def batch_center_line(lines , cx , cy):
    N = len(lines)
    out = np.empty((N , 2))
    
    def workers(i):
        out[i] = find_best_line(lines[i] , cx , cy)
    
    with ThreadPoolExecutor() as executor:
        executor.map(workers , range(N))
    
    return out
def batch_center_line_vec(lines, cx, cy):
    N = len(lines)
    out = np.empty((N, 2))

    for i in tqdm(range(N) , desc="center_line per frame"):
        if lines[i] is None:
            out[i] = [np.nan, np.nan]
            continue
            
        l_arr = lines[i][:, 0, :]
        r, t = l_arr[:, 0], l_arr[:, 1]
        dists = np.abs(cx * np.cos(t) + cy * np.sin(t) - r)
        
        idx = np.argmin(dists)
        out[i, 0] = t[idx]
        out[i, 1] = r[idx] 
    return out

def best_line(lines , mu , std):
    lines = lines.reshape(-1,2)
    t , r = lines[: , 1], lines[: , 0]
    mu_t , mu_r = mu[0], mu[1]
    std_t, std_r=std[0], std[1]
    z_theta = (t - mu_t) / (std_t + 1e-6)
    z_rho = (r - mu_r) / (std_r + 1e-6)
    prob = np.sqrt(z_theta ** 2 + z_rho**2)
    best_idx = np.argmin(prob)
    return lines[best_idx]

def rot_trlate_img(img , line , patch_size):
    t , r = line[1] , line[0]
    theta_deg = np.rad2deg(t)
    # rot_angle = -theta_deg
    
    pivot_x = r*np.cos(t)
    pivot_y = r*np.sin(t)
    
    M = cv.getRotationMatrix2D((pivot_x , pivot_y) , theta_deg , 1.0)
    
    M[0, 2] += (patch_size / 2 - pivot_x)
    
    return cv.warpAffine(img , M , (patch_size , img.shape[0]), flags=cv.INTER_LINEAR)

def draw_line(rho , theta , line_img):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0+3000 * (-b))
    y1 = int(y0+3000* (a))
    x2 = int(x0-3000 * (-b))
    y2 = int(y0-3000 * (a))
    cv.line(line_img , (x1, y1), (x2, y2), (0, 0, 255), 5)
    return line_img

def draw_lines(lines, edge_img):
    line_img = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)
    min_dist = float('inf')
    for j in tqdm(range(0, len(lines)), desc="Checking line Fidelity"):
        rho = lines[j][0][0]
        theta = lines[j][0][1]
        dist = abs(cx*np.cos(theta) + cy*np.sin(theta) - rho)
        if dist < min_dist:
            min_dist = dist
            line_img = draw_line(rho , theta , line_img)
    cv.imwrite(f"edege_img__run_1.png" , line_img)

def crop_roi(edge_img , orig_img , threshold , minA , maxA):
    line_ls = cv.HoughLines(edge_img , rho=1 ,theta = np.pi/180, threshold=threshold , min_theta=minA , max_theta = maxA)
    line = best_line(line_ls , means , stds)
    line_img = draw_line(line[0] , line[1] , cv.cvtColor(orig_img , cv.COLOR_GRAY2BGR))
    crop_roi = rot_trlate_img(orig_img, line , 300)
    return crop_roi , line_img

def batch_crop_roi(edge_arr , orig_mask, threshold , minA , maxA):
    def worker(i):
        crop_roi_img , line_img = crop_roi(edge_arr[i] , orig_mask[i], threshold , minA , maxA)
        cv.imwrite(f"cropped_rois/cropped_{i}.png", crop_roi_img)
        cv.imwrite(f"best_hough_line/best_line_img{i}.png", line_img)
    
    with ThreadPoolExecutor() as executor:
        executor.map(worker , range(edge_arr.shape[0]))

if __name__ == '__main__':
    teeths , w , h = read_frames()
    fame_coll = gt_frames(np.array(teeths) , 207 , 628)
    fg_mask = bg_sub(fame_coll)
    hist = master_hist(fame_coll)
    var , thresh = otsu_thresh(hist)
    binary = otsu_bin(fame_coll , thresh)
    solid_mask = otsu_knn(binary , fg_mask)
    fin_mask , orig_mask = roi_def(solid_mask , fame_coll)
    # print(f"final mask Shape , width: {fin_mask.shape[1]} , height: {fin_mask.shape[0]}")
    cx , cy , area = s_moments(fin_mask)
    cx , cy = np.mean(cx) , np.mean(cy)
    r_com , delta = radius_boundaries(area)
    print(f"cx: {cx} , cy: {cy} , r_com: {np.mean(r_com)} , delta: {np.mean(delta)}")
    cv.imwrite("solid_mask.png" , solid_mask[123])
    cv.imwrite("otsu_roi.png" , fin_mask[123])
    cv.imwrite("img_roi.png" , orig_mask[123])
    # waves = extract_triple_wave(fin_mask , cy)
    # edge_img , hysteresis_high = orchecterate_canny_arr(orig_mask[123],5,1)
    edge_arr = batch_canny(orig_mask , 3 ,1)
    # contours , _ = cv.findContours(fin_mask , cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    # ellipse = cv.fitEllipse(max(contours , key=cv.contourArea))
    
    
    # col_img = cv.cvtColor(np.zeros_like(fin_mask) , cv.COLOR_GRAY2BGR)
    # cv.drawContours(col_img , contours , -1 , (0 , 255 , 0) , 2)
    # cv.ellipse(col_img,ellipse,(255 , 0 , 0), 2)
    # cv.imwrite('contaour_mask.png', col_img)
    minA = np.deg2rad(5)
    maxA = np.deg2rad(20)
    threshold = 400
    lines = batch_hough(edge_arr, theta=np.pi/180 , threshold=threshold , min_theta=minA , max_theta=maxA)
    # line_img = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)
    # center_line = np.zeros((len(lines) , 2))
    # for i in tqdm(range(0 , len(lines)) , desc="Processing img Lines"):
        # print(np.array(lines[i].shape))
        # min_dist = float('inf')
        # tmp_ln = np.zeros((2))
        # for j in range(0, len(lines[i])):
        #     rho = lines[i][j][0][0]
        #     theta = lines[i][j][0][1]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a*rho
        #     y0 = b*rho
        #     dist = abs(cx*np.cos(theta) + cy*np.sin(theta) - rho)
        #     if dist < min_dist: 
        #         min_dist = dist
        #         tmp_ln[0] = theta
        #         tmp_ln[1] = rho
        # center_line[i][0] = tmp_ln[0]
        # center_line[i][1] = tmp_ln[1]
    
            # x1 = int(x0+3000 * (-b))
            # y1 = int(y0+3000* (a))
            # x2 = int(x0-3000 * (-b))
            # y2 = int(y0-3000 * (a))
            # cv.line(line_img , (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv.imwrite(f"hough_lines/edege_img__run1_{i}.png" , line_img)
    # print(circles)
    center_line = batch_center_line_vec(lines , cx , cy)
    print(f"Center_line Shape: {center_line.shape}")
    print(f"Center_Line Rand Entery : {center_line[10]}")
    # print(f"distance: {min_dist} , theta: {center_line[0]} , rho: {center_line[1]}")

    means = np.mean(center_line , axis=0)
    stds = np.std(center_line, axis=0)
    print(f"Gaussian Char mean: {means} , std {stds}")
    batch_crop_roi(edge_arr , orig_mask , threshold , minA , maxA)
    # line_ls = cv.HoughLines(edge_arr[123] , rho=1 ,theta = np.pi/180, threshold=threshold , min_theta=minA , max_theta = maxA)
    # print(f"line_ls: {type(line_ls)}")
    # draw_lines(line_ls , edge_arr[123])
    # line = best_line(line_ls , means , stds)
    # print(f"best line: {line}")
    # cv.imwrite("best_line_img.png" , draw_line(line[0] , line[1] , cv.cvtColor(orig_mask[123] , cv.COLOR_GRAY2BGR)))
    # crop_roi = rot_trlate_img(orig_mask[123], line , 500)
    # print(crop_roi)
    # cv.imwrite("crop_roi.png" , crop_roi)
    # cv.imwrite("crop_roi_canny.png" , edge_arr[123])
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
 
 # Hough Lines -> Output is a set of angle(theta) and shortest distance from origin to line(rho)
 # TO caluluate the best hough lines, we need to find the center hough line for each frame, and take the mean ans std for all center lines over rho and theta separately.
 # Construct a joint gaussian distribution using computed mean and std
 # figure out the Pixel range to crop from center hough line
 # Implement this on matlab, in realtime, retrive frame by frame, do canny, and get the teeth using the above method and thresholds
 # Feed it to the Matlab resnet-18 model 
