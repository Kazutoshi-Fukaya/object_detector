#include "point_cloud_object_detector/point_cloud_object_detector.h"

PointCloudObjectDetector::PointCloudObjectDetector(ros::NodeHandle nh,ros::NodeHandle private_nh) : nh_(nh), private_nh_(private_nh)
{
    pc_sub_ = nh_.subscribe("/camera/depth_registered/points",1,&PointCloudObjectDetector::sensor_callback,this);
    bbox_sub_ = nh_.subscribe("/darknet_ros/bounding_boxes",1,&PointCloudObjectDetector::bbox_callback,this);
}

void PointCloudObjectDetector::sensor_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    has_received_pcl2 = true;
    cloud->points.clear();
    pcl::fromROSMsg(*msg,*cloud);
    //ROS_INFO("Subscribed pcl");
}


void PointCloudObjectDetector::bbox_callback(const darknet_ros_msgs::BoundingBoxesConstPtr& msg)
{
    has_received_bbox = true;
    bboxes = *msg;
    //ROS_INFO("Subscribed bbox");
}

void PointCloudObjectDetector::check_bbox(darknet_ros_msgs::BoundingBox bbox)
{
    std::cout << "Class: " << bbox.Class << std::endl;
    std::cout << "Id: " << bbox.id << std::endl;
    std::cout << "Probability: " << bbox.probability << std::endl;
    std::cout << "X_range: [" << bbox.xmin << "," << bbox.xmax << "]" << std::endl;
    std::cout << "Y_range: [" << bbox.ymin << "," << bbox.ymax << "]" << std::endl;
    std::cout << std::endl;
}

void PointCloudObjectDetector::bbox_process()
{
    for(const auto &b : bboxes.bounding_boxes){
        std::cout << b.Class << std::endl;
        std::vector<float> points;
        std::vector<std::vector<float>> z_points(cloud->height,std::vector<float>());
        std::vector<float> z_value;
        
        for(const auto &p : cloud->points) points.push_back(p.z);

        if(points.size() == cloud->width*cloud->height){
            for(int i = 0; i < cloud->height; i++){
                for(int j = 0; j < cloud->width; j++){
                    z_points.at(i).push_back(points.at(i*cloud->width+j));
                }
            }

            if(!(b.xmin == 0 && b.xmax == 0)){
                for(int x = b.xmin; x <= b.xmax; x++){
                    for(int y = b.ymin; y <= b.ymax; y++){
                        z_value.push_back(z_points.at(y).at(x));
                    }
                }

                double d = 0.0;
                int finite_count = 0;
                for(int i = 0; i < z_value.size(); i++){
                    if(isfinite(z_value.at(i))){
                        d += z_value.at(i);
                        finite_count ++;
                    }
                }
                d /= (double)finite_count;
                std::cout << d << std::endl;

                //double dist = *min_element(z_value.begin(),z_value.end());
                //std::cout << dist << std::endl;
            }
        }
    }
}

void PointCloudObjectDetector::bbox_process_2()
{
    for(const auto &b : bboxes.bounding_boxes){
        std::cout << b.Class << std::endl;
        std::vector<pcl::PointXYZRGB> points;
        std::vector<std::vector<pcl::PointXYZRGB>> rearranged_points(cloud->height,std::vector<pcl::PointXYZRGB>());
        std::vector<pcl::PointXYZRGB> values;
        

        for(const auto &p : cloud->points) points.push_back(p);
        
        if(points.size() == cloud->width*cloud->height){
            for(int i = 0; i < cloud->height; i++){
                for(int j = 0; j < cloud->width; j++){
                    rearranged_points.at(i).push_back(points.at(i*cloud->width+j));
                }
            }

            
            if(!(b.xmin == 0 && b.xmax == 0)){
                for(int x = b.xmin; x <= b.xmax; x++){
                    for(int y = b.ymin; y <= b.ymax; y++){
                        values.push_back(rearranged_points.at(y).at(x));
                    }
                }

                double sum_x = 0.0;
                double sum_y = 0.0;
                double sum_z = 0.0;
                int finite_count = 0;
                for(const auto &value : values){
                    if(isfinite(value.x) && isfinite(value.y) && isfinite(value.z)){
                        sum_x += value.x;
                        sum_y += value.y;
                        sum_z += value.z;
                        finite_count ++;
                    }
                }
                
                sum_x /= (double)finite_count;
                sum_y /= (double)finite_count;
                sum_z /= (double)finite_count;

                std::cout << "x: " << sum_x << std::endl;
                std::cout << "y: " << sum_y << std::endl;
                std::cout << "z: " << sum_z << std::endl;

                double d = sqrt(pow(sum_x,2)+pow(sum_z,2));
                double theta = atan2(sum_z,sum_x) - M_PI/2;
                std::cout << "d: " << d << std::endl;
                std::cout << "theta: " << theta << std::endl;

            }
        }
    }
}

void PointCloudObjectDetector::process()
{
    ros::Rate rate(1);
    while(ros::ok()){
        if(has_received_bbox && has_received_pcl2) bbox_process_2();
        ros::spinOnce();
        rate.sleep();
    }   
}