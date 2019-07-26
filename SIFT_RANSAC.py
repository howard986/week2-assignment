#############################################################################
#
# 在两幅图像查找匹配的点对
# 参考内容：
#      1. 周四老师的答疑内容中的图像拼接的代码
#      2. opencv中关于 findHomography 的源码内容
#      3. 重写了通过RANSAC算法删除匹配错误点的内容
# 1. 用opencv的sift算法查找

# In[1]:
# coding: utf-8
import numpy as np
import cv2
# In[2]:
# runRANSAC
# 通过RANSAC算法，删除匹配点对中匹配错误的内容
def runRANSAC(src_pts, dst_pts, n_minSample, k_maxIteration, t_errorThreshold, d):
    #   输入:
    #    data - 样本点
    #    model - 假设模型:事先自己确定
    #    n_minSample - 生成模型所需的最少样本点
    #    k_maxIteration - 最大迭代次数
    #    t_errorThreshold - 阈值:作为判断点满足模型的条件
    #    d - 拟合较好时,需要的样本点最少的个数,当做阈值看待 
    #   输出:
    #    mask - 最优拟合解（返回null,如果未找到）
    iter = 0
    bestfit = 0
    besterror = np.inf  # 误差默认值
    best_inlier_idxs = None

    max_goodCount = 0   # 最大内点的数量
    model_points = 4
    ## 建立一个mask，用于标记映射关系
    mask_final = np.zeros(len(src_pts), dtype=np.uint8)
    
    while iter < k_maxIteration:
        src_s, dst_s = getSubset(src_pts, dst_pts, model_points)
        print(iter)
        # 计算单应矩阵
        h_matrix = cv2.getPerspectiveTransform(np.float32(src_s), np.float32(dst_s))
        # 如果单应矩阵找到了，则根据单应矩阵计算内点 
        if h_matrix is None:
            iter += 1
            continue

        mask = findInnerPts(h_matrix, src_pts, dst_pts, t_errorThreshold)
        # 如果新的内点大于最少内点数目要求时，就计算新的模型
        if sum(mask) > max_goodCount: 
            mask_final = mask
            max_goodCount = sum(mask)
            h_matrix_final = h_matrix
        iter += 1

    if max_goodCount < d:
        return None

    return mask_final

# 选取一个子集，尚未考虑数据点共线的异常处理
def getSubset(src, dst, model_points):
    rand_idxs = np.random.randint(0, len(src), model_points)
    src_s = src[rand_idxs]
    dst_s = dst[rand_idxs]
    return src_s, dst_s

# 查找内点
def findInnerPts(h_m, src, dst, t_errorThreshold):
    # 将笛卡尔坐系转换为齐次坐标
    src_homo_cord = np.hstack((src, np.ones(len(src)).reshape(len(src),-1))).T
    dst_homo_cord = np.hstack((dst, np.ones(len(dst)).reshape(len(dst),-1))).T
    # 将原图视角转换到目标图视角
    trans_homo_cord = np.dot(h_m, src_homo_cord)
    trans_homo_cord[0] = trans_homo_cord[0] / trans_homo_cord[2]
    trans_homo_cord[1] = trans_homo_cord[1] / trans_homo_cord[2]
    trans_homo_cord[2] = 1
    # 计算误差
    delt = trans_homo_cord - dst_homo_cord
    err = delt[0] ** 2 + delt[1]** 2
    t_errorThreshold *= t_errorThreshold
    # 将误差与阈值做比较，提取出mask
    mask = err < t_errorThreshold
    return mask

# In[3]: 
# 读取图片
left_img = cv2.imread("captain.jpg")
left_img = cv2.resize(left_img, (500, 500))
right_img = cv2.imread("captain_dst.jpg")
right_img = cv2.resize(right_img, (500, 500))
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
# 用SIFT提取特征点
hessian = 300
surf = cv2.xfeatures2d.SIFT_create(hessian) # 将Hessian Threshold设置为400,阈值越大能检测的特征就越少
kp1, des1 = surf.detectAndCompute(left_gray, None)  # 查找关键点和描述符
kp2, des2 = surf.detectAndCompute(right_gray, None)
# 显示特征点
img_with_drawKeyPoint_left = cv2.drawKeypoints(left_gray, kp1, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img_with_drawKeyPoint_left", img_with_drawKeyPoint_left)
img_with_drawKeyPoint_right = cv2.drawKeypoints(right_gray, kp2, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img_with_drawKeyPoint_right", img_with_drawKeyPoint_right)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[4]:
# FlannBasedMatcher 进行特征点的匹配操作
FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
searchParams = dict(checks=50)  # 指定递归次数
flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器
matches = flann.knnMatch(des1, des2, k=2)  # 得出匹配的关键点

# In[5]:  
# 绘制FlannBasedMatcher的匹配结果，
# 尚未使用RANSAC删除匹配不准确的特征点
good = []
# 提取优秀的特征点
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        good.append(m)

src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # 查询图像的特征描述子索引
dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # 训练(模板)图像的特征描述子索引
src_pts[0], dst_pts[0]
img_matches = np.zeros((left_img.shape[0], left_img.shape[1]*2, 3), dtype=np.uint8)
img_matches.fill(255)
cv2.drawMatches(left_gray, 
                kp1, 
                right_gray, 
                kp2, 
                good, 
                img_matches);

# In[6]: 
# 使用RANSAC删除匹配不准确的特征点，并绘制处理后的匹配结果
mask  = runRANSAC(src_pts, dst_pts, 4, 1000, 3, len(src_pts)*0.6)

ransac_good = []
# 提取优秀的特征点
for i in range(len(mask)):
    if mask[i]:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        ransac_good.append(good[i])

img_ransac = np.zeros((left_img.shape[0], left_img.shape[1]*2, 3), dtype=np.uint8)
img_ransac.fill(255)
cv2.drawMatches(left_gray, 
                kp1, 
                right_gray, 
                kp2, 
                ransac_good, 
                img_ransac);
# 匹配不准确的结果
wrong_match = [i for i in good if i not in ransac_good]
img_wrong_match = np.zeros((left_img.shape[0], left_img.shape[1]*2, 3), dtype=np.uint8)
img_wrong_match.fill(255)
cv2.drawMatches(left_gray, 
                kp1, 
                right_gray, 
                kp2, 
                wrong_match, 
                img_wrong_match);

# 绘制FlannBasedMatcher的匹配结果，
cv2.imshow("Matches Before RANSAC", img_matches)
# 绘制RANSAC处理后的匹配结果，
cv2.imshow("Matches After RANSAC", img_ransac)
# 绘制匹配不准确的结果
cv2.imshow("Matches Wrong Matches", img_wrong_match)
cv2.waitKey(0)
cv2.destroyAllWindows()
