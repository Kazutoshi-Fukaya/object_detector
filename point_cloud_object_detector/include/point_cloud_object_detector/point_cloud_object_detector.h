#ifndef POINT_CLOUD_OBJECT_DETECTOR_H_
#define POINT_CLOUD_OBJECT_DETECTOR_H_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <pcl_ros/transforms.h>
#include <cv_bridge/cv_bridge.h>

// pcl
#include <pcl_ros/point_cloud.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types_conversion.h>
#include <pcl/filters/random_sample.h>
#include <pcl/segmentation/min_cut_segmentation.h>

// Eigen
#include <Eigen/Dense>

// opencv
#include <opencv2/opencv.hpp>

// Custom msg
#include "darknet_ros_msgs/BoundingBoxes.h"
#include "object_detector_msgs/ObjectPositions.h"
#include "object_detector_msgs/ObjectPositionsWithImage.h"

namespace object_detector
{
class PointCloudObjectDetector {
public:
    PointCloudObjectDetector();
    void process();

private:
    void pc_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void bbox_callback(const darknet_ros_msgs::BoundingBoxesConstPtr& msg);
    void img_callback(const sensor_msgs::ImageConstPtr& msg);

    void load_object_classes();
    
    bool is_valid_class(const std::string& class_name);

    void convert_from_vec_to_pc(std::vector<pcl::PointXYZRGB>& vec,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc);
    void clustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& output_cloud);
    void mincut_clustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& output_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& center_cloud);
    void calc_position(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,double& x,double& y,double& z);

    // node handle
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // subscriber
    ros::Subscriber pc_sub_;
    ros::Subscriber bbox_sub_;
    ros::Subscriber img_sub_;

    // publisher
    ros::Publisher obj_pub_;
    ros::Publisher pc_pub_;
    ros::Publisher cls_pc_pub_;
    ros::Publisher obj_img_pub_;
    ros::Publisher obj_img_debug_pub_;

    // point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;

    // camera image
    // sensor_msgs::Image img_;

    // image pointer
    cv_bridge::CvImagePtr cv_ptr_;

    // tf
    boost::shared_ptr<tf2_ros::Buffer> buffer_;
    boost::shared_ptr<tf2_ros::TransformListener> listener_;
    boost::shared_ptr<tf2_ros::TransformBroadcaster> broadcaster_;

    // buffer
    std::string pc_frame_id_;
    bool has_received_pc_;
    bool has_received_img_;

    // object classes
    std::vector<std::string> object_classes_;

    // parameter
    std::string CAMERA_FRAME_ID_;
    bool IS_SAMPLING_;
    bool IS_CLUSTERING_;
    bool IS_PCL_TF_;
    bool IS_DEBUG_;
    bool USE_MINCUT_;
    bool USE_IMG_MSG_;
    int HZ_;
    int SAMPLING_NUM_;
    int MIN_CUT_NEIGHBORS_;
    static const int CLUSTER_NUM_ = 3;
    double SAMPLING_RATIO_;
    double CLUSTER_TOLERANCE_;
    double MIN_CLUSTER_SIZE_;
    double MIN_CUT_SIGMA_;
    double MIN_CUT_RADIUS_;
    double MIN_CUT_SOURCE_WEIGHT_;
    double ERROR_PER_DISTANCE_;
};
} // object_detector

#endif  // POINT_CLOUD_OBJECT_DETECTOR_H_