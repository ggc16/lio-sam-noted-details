/************************************************* 
功能简介:
    对经过运动畸变校正之后的当前帧激光点云，计算每个点的曲率，进而提取角点、平面点（用曲率的大小进行判定）。

订阅：
    1、订阅当前激光帧运动畸变校正后的点云信息，来自ImageProjection。

发布：
    1、发布当前激光帧提取特征之后的点云信息，包括的历史数据有：运动畸变校正，点云数据，初始位姿，姿态角，有效点云数据，角点点云，平面点点云等，发布给MapOptimization；
    2、发布当前激光帧提取的角点点云，用于rviz展示；
    3、发布当前激光帧提取的平面点点云，用于rviz展示。
**************************************************/ 

#include "utility.h"
#include "lio_sam/cloud_info.h"

/**
 * 激光点曲率
*/
struct smoothness_t{ 
    float value;  // 曲率值
    size_t ind;   // 激光点一维索引
};

/**
 * 曲率比较函数，从小到大排序
*/
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:
    // 订阅当前激光帧运动畸变校正后的点云信息，来自ImageProjection。
    ros::Subscriber subLaserCloudInfo;
    
    // 发布当前激光帧提取特征之后的点云信息
    ros::Publisher pubLaserCloudInfo;
    // 发布当前激光帧提取的角点点云
    ros::Publisher pubCornerPoints;
    // 发布当前激光帧提取的平面点点云
    ros::Publisher pubSurfacePoints;
    
    // 当前激光帧运动畸变校正后的有效点云
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    // 当前激光帧角点点云集合
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    // 当前激光帧平面点点云集合
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;  //体素滤波降采样
    
    // 当前激光帧点云信息，包括的历史数据有：运动畸变校正，点云数据，初始位姿，姿态角，有效点云数据，角点点云，平面点点云等
    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;
    
    // 当前激光帧点云的曲率
    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    // 特征提取标记，1表示遮挡、平行，或者已经进行特征提取的点，0表示还未进行特征提取处理
    int *cloudNeighborPicked;
    // 1表示角点，-1表示平面点
    int *cloudLabel;

    /**
     * 构造函数
    */
    FeatureExtraction()
    {
        // 订阅当前激光帧运动畸变校正后的点云信息
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        
        // 发布当前激光帧提取特征之后的点云信息
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/feature/cloud_info", 1);
        // 发布当前激光帧的角点点云
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        // 发布当前激光帧的面点点云
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
        // 初始化
        initializationValue();
    }
    
    // 初始化
    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    /**
     * 订阅当前激光帧运动畸变校正后的点云信息
     * 1、计算当前激光帧点云中每个点的曲率
     * 2、标记属于遮挡、平行两种情况的点，不做特征提取
     * 3、点云角点、平面点特征提取
     *   1) 遍历扫描线，每根扫描线扫描一周的点云划分为6段，针对每段提取20个角点、不限数量的平面点，加入角点集合、平面点集合
     *   2) 认为非角点的点都是平面点，加入平面点云集合，最后降采样
     * 4、发布角点、面点点云，发布带特征点云数据的当前激光帧点云信息
    */
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        // 把提取出来的有效的点转成pcl的格式，当前激光帧运动畸变校正后的有效点云
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction
        
        // 计算当前激光帧点云中每个点的曲率
        calculateSmoothness();
        
        // 标记属于遮挡、平行两种情况的点，不做特征提取
        markOccludedPoints();
        
        // 点云角点、平面点特征提取
        // 1、遍历扫描线，每根扫描线扫描一周的点云划分为6段，针对每段提取20个角点、不限数量的平面点，加入角点集合、平面点集合
        // 2、认为非角点的点都是平面点，加入平面点云集合，最后降采样
        extractFeatures();
        
        // 发布角点、面点点云，发布带特征点云数据的当前激光帧点云信息
        publishFeatureCloud();
    }
    
    // 计算当前激光帧点云中每个点的曲率
    void calculateSmoothness()
    {
        // 遍历当前激光帧运动畸变校正后的有效点云
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            // 用当前激光点前后5个点计算当前点的曲率，平坦位置处曲率较小，角点处曲率较大；这个方法很简单但有效
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            
            // 距离差值平方作为曲率
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;
            // 下面两个值赋成初始值
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // 用来进行曲率排序，存储该点曲率值、激光点一维索引
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }
    // 标记属于遮挡、平行两种情况的点，不做特征提取
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // 取出相邻两个点距离信息
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            // 两个激光点之间的一维索引差值，如果在一条扫描线上，那么值为1；如果两个点之间有一些无效点被剔除了，可能会比1大，但不会特别大
            // 如果恰好前一个点在扫描一周的结束时刻，下一个点是另一条扫描线的起始时刻，那么值会很大
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));
            
            // 两个点在同一扫描线上，且距离相差大于0.3，认为存在遮挡关系（也就是这两个点不在同一平面上，如果在同一平面上，距离相差不会太大）
            // 远处的点会被遮挡，标记一下该点以及相邻的5个点，后面不再进行特征提取
            if (columnDiff < 10){
                // 这样depth1容易被遮挡，因此其之前的5个点走设置为无效点
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){   // 这里同理
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            
            // 用前后相邻点判断当前点所在平面是否与激光束方向平行
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));
            // 如果两点距离比较大 就很可能是平行的点，也很可能失去观测，那么就标记一下
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    /**
     * 点云角点、平面点特征提取
     * 1、遍历扫描线，每根扫描线扫描一周的点云划分为6段，针对每段提取20个角点、不限数量的平面点，加入角点集合、平面点集合
     * 2、认为非角点的点都是平面点，加入平面点云集合，最后降采样
    */
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());
        // 遍历扫描线
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();
            // 把每一根scan等分成6份，每份分别提取特征点
            for (int j = 0; j < 6; j++)
            {
                // 根据之前得到的每个scan的起始和结束id来均分
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;
                // 这种情况就不正常
                if (sp >= ep)
                    continue;
                // 按照曲率从小到大排序点云
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());
                // 按照曲率从大到小遍历
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    // 找到这个点对应的原先的idx
                    int ind = cloudSmoothness[k].ind;
                    // 如果没有被认为是遮挡点 且曲率大于边缘点门限，则认为是角点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        // 每段只取20个角点，如果单条扫描线扫描一周是1800个点，则划分6段，每段300个点，从中提取20个角点
                        largestPickedNum++;
                        // 每段最多找20个角点
                        if (largestPickedNum <= 20){
                            // 标签置1表示是角点
                            cloudLabel[ind] = 1;
                            // 加入角点点云
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }
                        // 将这个点周围的几个点设置成遮挡点，避免选取太集中，标记已被处理
                        cloudNeighborPicked[ind] = 1;
                        // 同一条扫描线上后5个点标记一下，不再处理，避免特征聚集
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            // 列idx距离太远就算了，空间上也不会太集中
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        // 同一条扫描线上前5个点标记一下，不再处理，避免特征聚集
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 按照曲率从小到大遍历，开始收集面点
                for (int k = sp; k <= ep; k++)
                {
                    // 激光点的索引
                    int ind = cloudSmoothness[k].ind;
                    // 同样要求不是遮挡点且曲率小于给定阈值，则认为是平面点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        // -1表示是面点
                        cloudLabel[ind] = -1;
                        // 同理 把周围的点都设置为遮挡点，标记已被处理
                        cloudNeighborPicked[ind] = 1;
                        // 同一条扫描线上后5个点标记一下，不再处理，避免特征聚集
                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        // 同一条扫描线上前5个点标记一下，不再处理，避免特征聚集
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 平面点和未被处理的点，都认为是平面点，加入平面点云集合
                for (int k = sp; k <= ep; k++)
                {
                    // 注意这里是小于等于0,也就是说不是角点的都认为是面点了
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }
            
            // 平面点云降采样
            surfaceCloudScanDS->clear();
            // 因为面点太多了，所以做一个下采样
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);
            
            // 加入平面点云集合
            *surfaceCloud += *surfaceCloudScanDS;
        }
    }
    // 将一些不会用到的存储空间释放掉
    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    /**
     * 发布角点、面点点云，发布带特征点云数据的当前激光帧点云信息
    */
    void publishFeatureCloud()
    {
        // 清理
        freeCloudInfoMemory();
        // 把角点和面点发送给后端，发布角点、面点点云，用于rviz展示
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // 发布当前激光帧点云信息，加入了角点、面点点云数据，发布给mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}