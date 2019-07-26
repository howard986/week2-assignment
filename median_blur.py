# In[1]:
import cv2
import numpy as np
import datetime
import heapq

# 查找中位数的类
class MedianFinder(object):
    def __init__(self, ):
        self.len = 0
        # 大顶堆,因为heapq默认是小顶堆，所以存储的是负数，就变成了大顶堆
        self.lo = []
        # 小顶堆
        self.hi = [] 

    # 添加一个元素
    def addArray(self, list):
        for item in list:
            self.len += 1
            # 先插入大顶堆，再弹出顶部数据
            val = -heapq.heappushpop(self.lo, -item)
            # 将弹出的数据放入小顶堆
            heapq.heappush(self.hi, val)
            # 调整堆的平衡，如果小顶堆的元素更多，就弹出一个给大顶堆
            if(len(self.lo) < len(self.hi)):
                heapq.heappush(self.lo, -heapq.heappop(self.hi))

    # 查找中位数
    def findMedian(self):
        if self.len & 1 == 0:
            return (self.hi[0] - self.lo[0]) / 2.0 # 因为lo里存的负数，所以是减号
        else:
            return -self.lo[0]
            
# In[2]: 生成padding后的图像
def paddingImage(img, kernel, padding_way):
    h_img, w_img = img.shape
    h_kernel, w_kernel = kernel.shape
    w_padding = int((w_kernel - 1) / 2)
    h_padding = int((h_kernel - 1) / 2)
    # 生成一幅新的图片用于平滑
    h_new_img = img.shape[0] + h_kernel - 1
    w_new_img = img.shape[1] + w_kernel - 1
    new_img = np.zeros((h_new_img, w_new_img), dtype=np.uint8)
    new_img[h_padding:-h_padding, w_padding:-w_padding] = img
    print(new_img)
    if padding_way == 'REPLICA':
        new_img[:h_padding, :w_padding] = img[0, 0]
        new_img[-h_padding:, :w_padding] = img[-1, 0]
        new_img[-h_padding:, -w_padding:] = img[-1, -1]
        new_img[:h_padding, -w_padding:] = img[0, -1]

        new_img[:h_padding, w_padding:-w_padding] = img[0,:]
        new_img[h_padding:-h_padding,:w_padding] = img[:,0].reshape(h_img,-1)
        new_img[-h_padding:, w_padding:-w_padding] = img[-1, :]
        new_img[h_padding:-h_padding, -w_padding:] = img[:, -1].reshape(h_img,-1)
        print(new_img)
    if padding_way == 'ZERO':
        pass

    return new_img

def medianBlur(img, kernel, padding_way):
    #img & kernel is List of List; padding_way a string
    print('aa')

    # p_img = paddingImage(img, kernel, padding_way)

    # 对图像进行median blur
    h_img, w_img = img.shape
    h_kernel, w_kernel = kernel.shape
    w_padding = int((w_kernel - 1) / 2)
    h_padding = int((h_kernel - 1) / 2)
    if padding_way == 'REPLICA':
        p_img = cv2.copyMakeBorder(img, h_padding, h_padding, w_padding, w_padding, cv2.BORDER_REPLICATE)
    else:
        p_img = cv2.copyMakeBorder(img, h_padding, h_padding, w_padding, w_padding, cv2.BORDER_CONSTANT, value=0)
    cv2.imshow('cat', p_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_img = np.empty(img.shape, dtype=np.uint8)
    for w in range(w_padding, p_img.shape[1] - w_padding):
        for h in range(h_padding, p_img.shape[0] - h_padding):
            box = p_img[h - h_padding:h + h_padding+1, w - w_padding:w+w_padding+1]
            # 使用自定义类查找中位数
            m_finder = MedianFinder()
            m_finder.addArray(box.reshape(-1).tolist())
            med_val = m_finder.findMedian()
            # med_val = int(np.median(box))
            out_img[h-h_padding, w-w_padding] = med_val
    return out_img

# In[3]:
path = "mycat.jpg"
img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)# 0 grey

#cv2.imshow('cat', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
kernel = np.zeros((3, 3))
t1 = datetime.datetime.now()
blur_img = medianBlur(img, kernel, 'REPLICA')
t2 = datetime.datetime.now()
print(t2-t1)
cv2.imshow('blur_img', blur_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

