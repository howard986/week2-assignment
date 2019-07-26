# week2-assignment-  
第二周的作业 
20190726  
1. 此次上传完成了两个内容：  
   a. median_blur.py 中值滤波  
   b. SIFT_RANSAC.py 通过RANSAC算法对图像特征点匹配的结果进行过滤，剔除外点  
2. 程序说明：  
   a. median_blur.py   
      在计算中位数时使用了两个堆，大顶堆存储小的一半的数，小顶堆存储大的一半的数据。中位数通过堆顶的两个数产生。  
   b. SIFT_RANSAC.py   
      1）先用opencv内置的SIFT算法提取特征点  
      2）然后使用FlannBasedMatcher对特征点进行匹配  
      3）最后通过手写的runRANSAC算法，对上一步匹配错误的点进行了剔除  
      4）用三幅图形显示了RANSAC效果：  
         * 使用RANSAC前.PNG  
         * 使用RANSAC后.PNG  
         * 匹配不准确的结果.PNG  (把被RANSAC剔除的结果也单独显示出来)  
