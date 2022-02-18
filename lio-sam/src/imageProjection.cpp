// 功能简介:
//     1、利用当前激光帧起止时刻间的imu数据计算旋转增量，IMU里程计数据（来自ImuPreintegration）计算平移增量，进而对该帧激光每一时刻的激光点进行运动畸变校正;
//      （利用相对于激光帧起始时刻的位姿增量，变换当前激光点到起始时刻激光点的坐标系下，实现校正）；
//     2、同时用IMU数据的姿态角（RPY，roll、pitch、yaw）、IMU里程计数据的的位姿，对当前帧激光位姿进行粗略初始化。

// 订阅：
//     1、订阅原始IMU数据;
//     2、订阅IMU里程计数据，来自ImuPreintegration，表示每一时刻对应的位姿；
//     3、订阅原始激光点云数据。

// 发布：
//     1、发布当前帧激光运动畸变校正之后的有效点云，用于rviz展示；
//     2、发布当前帧激光运动畸变校正之后的点云信息，包括点云数据、初始位姿、姿态角、有效点云数据等，发布给FeatureExtraction进行特征提取。

#include "utility.h"
#include "lio_sam/cloud_info.h"

/**
 * Velodyne点云结构，变量名XYZIRT是每个变量的首字母
*/
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D;    // 位置
    PCL_ADD_INTENSITY;  // 激光点反射强度，也可以存点的索引
    uint16_t ring;      // 扫描线
    float time;         // 时间戳，记录相对于当前帧第一个激光点的时差，第一个点time=0
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;        // 内存16字节对齐，EIGEN SSE优化要求
// 注册为PCL点云格式
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

/**
 * Ouster点云结构
*/
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// 本程序使用Velodyne点云结构
using PointXYZIRT = VelodynePointXYZIRT;

// imu数据队列长度
const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:
    // imu队列、odom队列互斥锁
    std::mutex imuLock;
    std::mutex odoLock;
    
    // 订阅原始激光点云
    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    // 发布当前帧校正后点云，有效点
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;
    
    // imu数据队列（原始数据，转lidar系下）
    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;
    
    // imu里程计队列
    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;
    
    // 激光点云数据队列
    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    // 队列front帧，作为当前处理帧点云
    sensor_msgs::PointCloud2 currentCloudMsg;
    
    // 当前激光帧从起始时刻到当前时刻所对应的每一个时刻的imu数据，计算相对于起始时刻的旋转增量，以及时间戳；
    // 用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;
    
    // 当前帧原始激光点云
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    // 当前帧运动畸变校正之后的激光点云
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    // 从fullCloud中提取有效点
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;
   
    bool odomDeskewFlag;
    // 当前激光帧起止时刻对应imu里程计位姿变换，该变换对应的平移增量；用于插值计算当前激光帧起止时间范围内，每一时刻的位置  
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    // 当前帧激光点云运动畸变校正之后的数据，包括点云数据、初始位姿、姿态角等，发布给featureExtraction进行特征提取
    lio_sam::cloud_info cloudInfo;
    // 当前帧起始时刻
    double timeScanCur;
    // 当前帧结束时刻
    double timeScanEnd;
    // 当前帧header，包含时间戳信息
    std_msgs::Header cloudHeader;


public:
    ImageProjection():
    deskewFlag(0)
    {
        // 订阅原始imu数据
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅原始lidar数据
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
       
        // 发布当前激光帧运动畸变校正后的点云，有效点   
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        // 发布当前激光帧运动畸变校正后的点云信息
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);
        // 初始化
        allocateMemory();
        // 重置参数
        resetParameters();
        // pcl日志级别，只打ERROR日志
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    /**
     * 初始化，构造指针，分配内存
    */
    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

   /**
     * 重置参数，接收每帧lidar数据都要重置这些参数
    */
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    /**
     * 订阅原始imu数据
     * 1、imu原始测量数据转换到lidar系，加速度、角速度、RPY
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        // imu原始测量数据转换到lidar系，加速度、角速度、RPY，相对于是lidar的测量数据
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);   // 对imu做一个坐标转换
        // 加一个线程锁，把imu数据保存进队列，添加数据的时候队列不可用
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    /**
     * 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }
    /**
     * 订阅原始lidar数据
     * 1、添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
     * 2、当前帧起止时刻对应的imu数据、imu里程计数据处理
     *   imu数据：
     *   1) 遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     *   2) 用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
     *   imu里程计数据：
     *   1) 遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
     *   2) 用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
     * 3、当前帧激光点云运动畸变校正
     *   1) 检查激光点距离、扫描线是否合规
     *   2) 激光运动畸变校正，保存激光点
     * 4、提取有效激光点，存extractedCloud
     * 5、发布当前帧校正后点云，有效点
     * 6、重置参数，接收每帧lidar数据都要重置这些参数
    */
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    { 
        // 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 当前帧起止时刻对应的imu数据、imu里程计数据处理
        if (!deskewInfo())
            return;
        
        // 当前帧激光点云运动畸变校正
        // 1、检查激光点距离、扫描线是否合规
        // 2、激光运动畸变校正，保存激光点
        projectPointCloud();
        
        // 提取有效激光点，存extractedCloud
        cloudExtraction();
        // 发布当前帧校正后点云，有效点
        publishClouds();
        // 重置参数，接收每帧lidar数据都要重置这些参数
        resetParameters();
    }

    /**
     * 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
    */
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 点云数据保存进队列
        cloudQueue.push_back(*laserCloudMsg);
        // 确保队列里大于两帧点云数据
        if (cloudQueue.size() <= 2)
            return false;

        // 缓存了足够多的点云之后，取出激光点云队列中最早的一帧
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);    // 转成pcl的点云格式
        }
        else if (sensor == SensorType::OUSTER)
        {
            // 转换成Velodyne格式
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // 当前帧头部
        cloudHeader = currentCloudMsg.header;
        // 当前帧起始时刻
        timeScanCur = cloudHeader.stamp.toSec();
        // 当前帧结束时刻，注：点云中激光点的time记录相对于当前帧第一个激光点的时差，第一个点time=0
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // is_dense是点云是否有序排列的标志，存在无效点，Nan或者Inf
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // 检查是否存在ring通道，注意static只检查一次
        // 查看驱动里是否把每个点属于哪一根扫描scan这个信息
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            // 如果没有这个信息就需要像loam或者lego loam那样手动计算scan id，现在velodyne的驱动里都会携带这些信息的
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // 检查是否存在time通道
        // 同样，检查是否有时间戳信息
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    // 获取运动补偿所需的信息，当前帧起止时刻对应的imu数据、imu里程计数据处理
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // 确保imu的数据覆盖这一帧的点云
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }
        // 当前帧对应imu数据处理
        // 1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
        // 2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
        // 注：imu数据都已经转换到lidar系下了
        imuDeskewInfo();

        // 当前帧对应imu里程计处理
        // 1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
        // 2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
        // 注：imu数据都已经转换到lidar系下了
        odomDeskewInfo();

        return true;
    }

    /**
     * 当前帧对应imu数据处理
     * 1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     * 2、用角速度、时间积分，计算 每一时刻 相对于 初始时刻 的旋转量，初始时刻旋转设为0
     * 注：imu数据都已经转换到lidar系下了
    */
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;
        
        // 从imu队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01) // 扔掉把过早的imu
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;
        
        // 遍历当前激光帧起止时刻（前后扩展0.01s）之间的imu数据
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // 提取imu姿态角RPY，作为当前lidar帧初始姿态角
            if (currentImuTime <= timeScanCur)
                // 把imu的姿态转成欧拉角
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
            // 超过当前激光帧结束时刻0.01s，结束
            if (currentImuTime > timeScanEnd + 0.01)    // 这一帧遍历完了就break
                break;
            
            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0){    // 起始帧
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            double angular_x, angular_y, angular_z;
            // 取出当前帧imu的角速度
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // imu帧间时差
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            // 计算每一个时刻的姿态角，方便后续查找对应每个点云时间的值
            // 当前时刻旋转角 = 前一时刻旋转角 + 角速度 * 时差
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;
        // 没有合规的imu数据
        if (imuPointerCur <= 0)
            return;
        // 可以使用imu数据进行运动补偿
        cloudInfo.imuAvailable = true;
    }

    /**
     * 当前帧对应imu里程计处理
     * 1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
     * 2、用起始、终止时刻对应imu里程计，计算 相对位姿 变换，保存平移增量
     * 注：imu数据都已经转换到lidar系下了
    */
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;
        
        // 从imu里程计队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!odomQueue.empty())
        {
            // 扔掉过早的数据
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;
        // 点云时间   ×××××××
        // odom时间     ×××××
        // 显然不能覆盖整个点云的时间
        // 要求必须有当前激光帧时刻之前的imu里程计数据
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // 提取当前激光帧起始时刻的imu里程计
        nav_msgs::Odometry startOdomMsg;
        // 找到对应的最早的点云时间的odom数据
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }
        // 将ros消息格式中的姿态转成tf的格式
        // 提取imu里程计姿态角
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);
        // 然后将四元数转成欧拉角
        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        // 记录点云起始时刻的对应的odom姿态
        // 用当前激光帧起始时刻的imu里程计，初始化lidar位姿，后面用于mapOptmization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true; // odom提供了这一帧点云的初始位姿

        // get end odometry at the end of the scan
        odomDeskewFlag = false;
        // 这里发现没有覆盖到最后的点云，那就不能用odom数据来做运动补偿
        // 如果当前激光帧结束时刻之后没有imu里程计数据，返回
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;
        
        // 提取当前激光帧结束时刻的imu里程计
        nav_msgs::Odometry endOdomMsg;
        // 找到点云最晚时间对应的odom数据
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        // 如果起止时刻对应imu里程计的方差不等，返回。这个代表odom退化了，就置信度不高了。
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;
        // 起始位姿和结束位姿都转成Affine3f这个数据结构
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 计算起始位姿和结束位姿之间的delta pose， 起止时刻之间imu里程计的相对变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
        // 将这个增量转成xyz和欧拉角的形式
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;  // 表示可以用odom来做运动补偿
    }

    /**
     * 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
    */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        // 查找当前时刻在imuTime下的索引
        int imuPointerFront = 0;
        // imuPointerCur是imu计算的旋转buffer的总共大小，这里用的就是一种朴素的确保不越界的方法
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // imuPointerBack     imuPointerFront
        //    ×                      ×
        //               ×
        //           imuPointerCur
        
        // 如果时间戳不在两个imu的旋转之间，就直接赋值了
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 否则 做一个线性插值，得到相对旋转
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    /**
     * 在当前激光帧起止时间范围内，计算某一时刻的平移（相对于起始时刻的平移增量）
    */
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }
    /**
     * 激光运动畸变校正
     * 利用当前帧起止时刻之间的imu数据计算旋转增量，imu里程计数据计算平移增量
     * 进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿
    */
    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;
        // relTime是相对时间，加上起始时间就是绝对时间
        // relTime是当前激光点相对于激光帧起始时刻的时间，pointTime则是当前激光点的时间戳
        double pointTime = timeScanCur + relTime;
        
        // 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        
        // 在当前激光帧起止时间范围内，计算某一时刻的平移（相对于起始时刻的平移增量），这里没有计算平移补偿
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);
        
        // 第一个点的位姿增量（0），求逆
        if (firstPointFlag == true)
        {
            // 计算第一个点的相对位姿
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // 计算当前点和第一个点的相对位姿
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        
        // 当前激光点在第一个激光点坐标系下的坐标
        PointType newPoint;
        // 就是R × p + t，把激光点补偿到第一个点对应时刻的位姿坐标系下
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    /* 将点云投影到一个矩阵上。并且保存每个点的信息
     * 当前帧激光点云运动畸变校正
     * 1、检查激光点距离、扫描线是否合规
     * 2、激光运动畸变校正，保存激光点
    */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // 遍历当前帧激光点云
        for (int i = 0; i < cloudSize; ++i)
        {
            // pcl格式
            PointType thisPoint;
            // 取出对应的某个点
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // 计算这个点距离lidar中心的距离
            float range = pointDistance(thisPoint);
            // 距离太小或者太远都认为是异常点
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;
            // 取出对应的在第几根scan上
            int rowIdn = laserCloudIn->points[i].ring;
            // scan id必须合理
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 如果需要降采样，就根据scan id适当跳过
            if (rowIdn % downsampleRate != 0)
                continue;
            // 计算水平角
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            // 水平扫描角度步长，例如一周扫描1800次，则两次扫描间隔角度0.2°
            static float ang_res_x = 360.0/float(Horizon_SCAN);
            // 计算水平线束id，转换到x负方向e为起始，顺时针为正方向，范围[0,H]
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;
            // 对水平id进行检查
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            // 如果这个位置已经有填充了就跳过
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            // 对点做运动补偿
            // 利用当前帧起止时刻之间的imu数据计算旋转增量，imu里程计数据计算平移增量，进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            // 将这个点的距离数据保存进这个range矩阵中
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // 算出这个点的索引
            int index = columnIdn + rowIdn * Horizon_SCAN;
            // 转换成一维索引，存校正之后的激光点
            fullCloud->points[index] = thisPoint;
        }
    }

    // 提取有效激光点，存extractedCloud
    void cloudExtraction()
    {   
        // 有效激光点数量
        int count = 0;
        // 遍历每一根scan
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 这个scan可以计算曲率的起始点（计算曲率需要左右各五个点）
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                // 有效激光点
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // 记录激光点对应的Horizon_SCAN方向上的索引
                    cloudInfo.pointColInd[count] = j;
                    // 激光点距离
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // 加入有效激光点，他的3d坐标信息
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // count只在有效点才会累加
                    ++count;
                }
            }
            // 记录每根扫描线倒数第5个激光点在一维数组中的索引，也就是这个scan可以计算曲率的终点
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }

    /**
     * 发布当前帧校正后点云，有效点
    */    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        // 发布提取出来的有效的点
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");
    // 3个线程
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
