from skimage import measure
from skimage import morphology
import scipy.ndimage as ndimage
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Iterable, List, Set, Tuple
import time

def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))




def calc_back_skelen_torch(x):
    B, H, W = x.size()
    x_b_skelen = torch.zeros((B, H, W)).to(x.device)
    x_f_skelen = torch.zeros((B, H, W)).to(x.device)
    x_np = x.cpu().detach().numpy()
    for i in range(B):
        a = x_np[i, :, :]
        skelen = np.zeros_like(a)
        mask_label, label_num = measure.label(a, connectivity=1, background=1, return_num=True) # label it
        image_props = measure.regionprops(mask_label, cache=False) # get regions props

        fore_sk = a 
        #fore_sk = morphology.skeletonize(a, method="lee") #/255 # Lee Skeleton method   # fore skelen
        x_f_skelen[i, :, :] = torch.from_numpy(fore_sk).to(x.device)

        for li in range(label_num):
            image_prop = image_props[li]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            #bool_sub_sk = morphology.skeletonize(bool_sub, method="lee") #/255 # Lee Skeleton method
            bool_sub_sk = bool_sub
            skelen[min_row:max_row, min_col:max_col] += bool_sub_sk.astype(np.float32)
            #for mid check
            # plt.figure(figsize=(20, 10))
            # plt.subplot(1,2,1), plt.imshow(bool_sub, cmap='gray'), plt.axis("off"), plt.title('bool_sub', fontsize=24)
            # plt.subplot(1,2,2), plt.imshow(bool_sub_sk, cmap='gray'), plt.axis("off"), plt.title('bool_sub_sk', fontsize=24)
            # plt.show()
        x_b_skelen[i, :, :] = torch.from_numpy(skelen).to(x.device)  # back skelen
    return x_b_skelen, soft_dilate(x_b_skelen.unsqueeze(1)).squeeze(1), x_f_skelen


# get critical pixels

def minus_torch(a, b):
    c = F.relu(a-b)
    return c

# get connected regions
def get_tfn_conn_torch(fn, tfn):
    B, H, W = tfn.size()
    tfn_conn = torch.zeros((B, H, W)).to(tfn.device)
    fn_np = fn.cpu().detach().numpy() 
    tfn_np = tfn.cpu().detach().numpy()
    for i in range(B):
        fn = fn_np[i, :, :]
        tfn = tfn_np[i, :, :]
        ttfn = np.zeros_like(fn)
        mask_label, label_num = measure.label(fn, connectivity=1, background=0, return_num=True) 
        image_props = measure.regionprops(mask_label, cache=False) 
        for li in range(label_num):
            image_prop = image_props[li]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            tfn_sub = tfn[min_row:max_row, min_col:max_col].copy()

            tfn_sub[bool_sub < 1.0] = 0
            if np.sum(tfn_sub) > 0:
                ttfn[min_row:max_row, min_col:max_col] += bool_sub
            # for mid check
            #plt.figure(figsize=(10, 10))
            #plt.subplot(2,2,1), plt.imshow(fn, cmap='gray'), plt.axis("off"), plt.title('bool_sub', fontsize=24)
            #plt.subplot(2,2,2), plt.imshow(tfn, cmap='gray'), plt.axis("off"), plt.title('bool_sub', fontsize=24)
            #plt.subplot(2,2,3), plt.imshow(bool_sub, cmap='gray'), plt.axis("off"), plt.title('bool_sub_sk', fontsize=24)
            #plt.subplot(2,2,4), plt.imshow(ttfn, cmap='gray'), plt.axis("off"), plt.title('bool_sub_sk', fontsize=24)
            #plt.show()
        tfn_conn[i, :, :] = torch.from_numpy(ttfn).to(tfn_conn.device)
        tfn_conn[tfn_conn > 1] = 1
    return tfn_conn



def get_tfn(p, l, p_b_sk, l_f_sk, get_conn=False):
    fn = minus_torch(l, p)  # fn  label = 1 & pred = 0
    tfn = minus_torch(fn, 1-p_b_sk)  # fn - pred_back_skelen
    if get_conn:
        tfn = get_tfn_conn_torch(fn, tfn)
    tfn = tfn * fn  # critical pixels 2

    dp = soft_dilate(p.unsqueeze(1)).squeeze(1)  # tolerate some offset
    tfn2 = minus_torch(l_f_sk, dp)  # label_fore_sklen - dilate_pred
    tfn2 = get_tfn_conn_torch(fn, tfn2)
    tfn2 = tfn2 * fn  # critical pixels 1

    tfn += tfn2  # critical pixels
    tfn[tfn > 1] = 1

    return tfn


# calculate weight map

class OverlapTile:
    """
        This stategy is implementated from: Ma B, Ban X, Huang H, et al. Deep learning-based image segmentation
        for al-la alloy microscopic images[J]. Symmetry, 2018, 10(4): 107.
        For figure 4
        crop_size is the size of blue rectangle, which is equal to input_size
        roi_size is the size of yellow rectangle
    """

    def __init__(self, crop_size: int = 256, overlap_size: int = 32):
        self._crop_size = crop_size
        self._overlap_size = overlap_size
        self._roi_size = crop_size - 2 * overlap_size
        self._in_img_shape = None

    def crop(self, in_img: np.ndarray) -> List:
        """
            Crop in_img to sub-img List,
            hint: self._roi_size is used in crop and stitch stage, please check the paper careful when read this code
        """
        self._in_img_shape = in_img.shape
        # Pad image before cropping
        in_pad_img = np.pad(in_img, self._overlap_size, mode='symmetric')
        # calculate the number of cropping, which consider the influence of remainder
        h_pad_num = math.ceil(in_pad_img.shape[0] / self._roi_size)
        w_pad_num = math.ceil(in_pad_img.shape[1] / self._roi_size)
        # if in_pad_img.shape[0] % (self._roi_size) == 0:h_pad_num = h_pad_num - 1
        # if in_pad_img.shape[1] % (self._roi_size) == 0:w_pad_num = w_pad_num - 1
        in_crop_imgs = []
        for i in range(h_pad_num):
            # row analysis
            # overlap_crop, it is need to calculate the start of cropping
            start_h = i * self._roi_size
            end_h = start_h + self._crop_size

            # if there is some remainder result for cropping, change start_h and start_w
            if end_h > in_pad_img.shape[0]:
                start_h = in_pad_img.shape[0] - self._crop_size
                end_h = in_pad_img.shape[0]

            for j in range(w_pad_num):
                # column analysis
                start_w = j * self._roi_size
                end_w = start_w + self._crop_size

                if end_w > in_pad_img.shape[1]:
                    start_w = in_pad_img.shape[1] - self._crop_size
                    end_w = in_pad_img.shape[1]

                crop_img = in_pad_img[start_h: end_h, start_w: end_w]
                # print("cropping: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}, crop_shape={}".\
                #      format(i, start_h, start_w, j, end_h, end_w, crop_img.shape))
                in_crop_imgs.append(crop_img)
                if end_w == in_pad_img.shape[1]:
                    break
            if end_h == in_pad_img.shape[0]:
                break
        return in_crop_imgs

    def stitch(self, out_crop_imgs: List) -> np.ndarray:
        """
            Stitch sub-img List to whole out img
        """
        out_img = np.zeros(self._in_img_shape)

        # calculate the number of cropping, which consider the influence of remainder
        # h_num = math.ceil(self._in_img_shape[0] / self._roi_size) # it is no need to use that
        w_num = math.ceil(self._in_img_shape[1] / self._roi_size)

        for img_idx, out_crop_img in enumerate(out_crop_imgs):
            roi_img = out_crop_img[self._overlap_size: self._overlap_size + self._roi_size,
                      self._overlap_size: self._overlap_size + self._roi_size]
            i = int(img_idx / w_num)
            j = int(img_idx - (i * w_num))
            start_h = int(i * self._roi_size);
            end_h = start_h + self._roi_size
            start_w = int(j * self._roi_size);
            end_w = start_w + self._roi_size

            if end_h > self._in_img_shape[0]:
                start_h = self._in_img_shape[0] - self._roi_size
                end_h = self._in_img_shape[0]
            if end_w > self._in_img_shape[1]:
                start_w = self._in_img_shape[1] - self._roi_size
                end_w = self._in_img_shape[1]
                # print("stitching: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}"\
            # .format(i, start_h, start_w, j, end_h, end_w))
            out_img[start_h: end_h, start_w: end_w] = roi_img
        return out_img
    
    
class SkeletonAwareWeight():
    """
    Skeleton Aware Weight
    
    """
    def __init__(self, eps = 1e-20):
        """
        At this time, the weight is only suited to binary segmentation, so class_num = 2
        """
        self._class_num = 2
        self._eps = eps
        self.method=None
       # self.overlap_tile = OverlapTile()  # speed up

    
    def _get_weight(self, mask: np.ndarray, method=None, single_border=False) -> np.ndarray:
        """
        Get skeleton aware weight map
        :param mask: binary gt mask with shape (H, W)
        :return weight:  weight map with shape (H, W, 2)
        """
        self.method = method
        # Get class weight for two channels
        weight = np.zeros((mask.shape[0], mask.shape[1], 2))
        class_weight = np.zeros((self._class_num, 1))
        for class_idx in range(self._class_num):
            idx_num = np.count_nonzero(mask == class_idx)
            class_weight[class_idx, 0] = idx_num
        min_num = np.amin(class_weight)
        class_weight = class_weight * 1.0 / min_num
        class_weight = np.sum(class_weight) - class_weight
        
        
        # Get weight for each channel
        for class_idx in range(self._class_num):
            temp_mask = np.zeros_like(mask)
            temp_mask[mask == class_idx] = 1.0
            dis_trf = ndimage.distance_transform_edt(temp_mask)
            
           # st = time.time()
            if class_idx == 1: 
                # Get weight for border
                if single_border:
                    temp_weight = 1.0
                else:
                    #label_map, label_num = measure.label(temp_mask, connectivity=1, background=1, return_num=True)
                    #temp_weight = self._get_border_weight(class_weight[class_idx, 0], temp_mask, dis_trf, label_map, label_num)
                    ''' # speed up
                    # crop_images = self.overlap_tile.crop(temp_mask)
                    # temp = []
                    # for crop_mask in crop_images:
                    #     label_map, label_num = measure.label(crop_mask, connectivity=1, background=1, return_num=True)
                    #     temp_weight = self._get_border_weight(class_weight[class_idx, 0], crop_mask, dis_trf, label_map, label_num)
                    #     temp.append(temp_weight)
                    # temp_weight = self.overlap_tile.stitch(temp)
                    '''
            else:
                # Get weight for objects
                label_map, label_num = measure.label(temp_mask, connectivity=1, background=0, return_num=True)
                temp_weight = self._get_object_weight(class_weight[class_idx, 0], dis_trf, label_map, label_num)
           # ed = time.time()
            # print("class:{}, time {:.4f}".format(class_idx, ed - st))
            weight[:, :, class_idx] = temp_weight * temp_mask
        return weight
    
    def _get_border_weight(self, wc: float, mask: np.ndarray, dis_trf: np.ndarray, label_map: np.ndarray, label_num: int) -> np.ndarray:
        """
        Get border weight
        :param wc: class weight of border channel
        :param mask: real mask with shape (H,W), the border pixels equal 1 and the object pixels equal 0
        :param dis_trf: distance transform of border (shape (H,W)), it means the distance of each border pixel to the nearest object
        :param label_map: label map of connected components with shape (H, W)
        :param label_num: the number of connected components
        :return weight:  weight map of border with shape (H, W)        
        """
        weight = np.zeros(label_map.shape)
        image_props = measure.regionprops(label_map, cache=False)
        h, w = mask.shape[:2]
        min_dis = np.ones((h, w, label_num))
        
        # For each connected region, calculate the closest distance from other pixels to this region.
        for label_idx in range(label_num): 
            image_prop = image_props[label_idx]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0
            # mask this connected region by set its value to 0 
            mask_label = np.ones_like(mask)
            mask_label[min_row: max_row, min_col: max_col] -= bool_sub
            # Get the distance transform to masked connect component
            min_dis[:,:,label_idx] = ndimage.distance_transform_edt(mask_label) * mask

        # Calculate for each pixel the distances to the first nearest(d1) and second nearest(d2) connected regions
        min_dis.sort(axis=2)
        min_dis = min_dis[:, :, 0:2]
        # Calculate weight map by d1 and d2 for each pixel
        max_dis_trf = np.amax(dis_trf)
        weight =  1.0 + ((2 * max_dis_trf - min_dis[:, :, 0] - min_dis[:, :, 1] + self._eps) / (2 * max_dis_trf + self._eps))

        weight[weight < 0] = 0.0
        return weight

    def _get_object_weight(self, wc: float, dis_trf: np.ndarray, label_map: np.ndarray, label_num: int) -> np.ndarray:
        """
        Get object weight
        :param wc: class weight of border channel
        :param dis_trf: distance transform of object (shape (H,W)), it means the distance of each pixel to the nearest border
        :param label_map: label map of connected components with shape (H, W)
        :param label_num: the number of connected components
        :return weight:  weight map of object with shape (H, W)        
        """
        weight = np.zeros(label_map.shape)
        image_props = measure.regionprops(label_map, cache=False)
        
        # For each connect component, calculate its weight by its skeleton
        for label_idx in range(label_num):
            image_prop = image_props[label_idx]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            #bool_sub_sk = morphology.skeletonize(bool_sub, method="lee") #/255 # Lee Skeleton method
            bool_sub_sk = bool_sub
            if np.count_nonzero(bool_sub_sk == 1.0) == 0:
                # If there is no skelenton pixel, continue
                continue
            # Get the distance transform of skeleton pixel
            dis_trf_sk_sub = dis_trf[min_row: max_row, min_col: max_col] * bool_sub_sk

            # Get the distance transform to skeleton pixel 
            indices = np.zeros(((np.ndim(bool_sub_sk),) + bool_sub_sk.shape), dtype=np.int32)
            dis_trf_to_sk = ndimage.distance_transform_edt(1-bool_sub_sk, return_indices=True, indices=indices)

            h, w = bool_sub.shape[:2]
            dis_sk_map = np.ones((h, w, 2))
            dis_sk_map[:, :, 0] = dis_trf_to_sk  # d0 
            dis_sk_map[:, :, 1] = dis_trf_sk_sub[indices[0, :, :], indices[1, :, :]]   # d1 

            # Rectify, enusre d0 <= d1, d0: the distance of pixel to nearest skeleton pixel, d1: the distance d1 of nearest skeleton pixel to border
            dis_sk_map[:, :, 0][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]] = dis_sk_map[:, :, 1][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]]    
 
            weight_sub = 1-(dis_sk_map[:, :, 0] / (dis_sk_map[:, :, 1] + self._eps))
            
            weight[min_row: max_row, min_col: max_col] += weight_sub * bool_sub

        
        return weight
    
    
def calculate_class_weight(gt_images):
    class_weight = torch.zeros((gt_images.shape[0], 2, 1), device=gt_images.device)
    for class_idx in range(2):
        idx_num = torch.count_nonzero(gt_images == class_idx)
        class_weight[:, class_idx, 0] = idx_num
    min_num = torch.min(class_weight)
    class_weight = class_weight * 1.0 / min_num
    class_weight = torch.sum(class_weight) - class_weight
    
    return class_weight 

