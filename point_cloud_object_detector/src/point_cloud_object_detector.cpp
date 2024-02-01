#include "point_cloud_object_detector/point_cloud_object_detector.h"

using namespace object_detector;

PointCloudObjectDetector::PointCloudObjectDetector() :
    private_nh_("~"),
    cloud_(new pcl::PointCloud<pcl::PointXYZRGB>),
    pc_frame_id_(std::string("")), has_received_pc_(false)
{
    private_nh_.param("CAMERA_FRAME_ID",CAMERA_FRAME_ID_,{std::string("base_link")});
    private_nh_.param("HZ",HZ_,{10});

    // sampling param
    private_nh_.param("IS_SAMPLING",IS_SAMPLING_,{true});
    private_nh_.param("SAMPLING_NUM",SAMPLING_NUM_,{100000});
    private_nh_.param("SAMPLING_RATIO",SAMPLING_RATIO_,{0.1});

    // clustring param
    private_nh_.param("IS_CLUSTERING",IS_CLUSTERING_,{true});
    private_nh_.param("USE_MINCUT",USE_MINCUT_,{false});
    private_nh_.param("CLUSTER_TOLERANCE",CLUSTER_TOLERANCE_,{0.02});
    private_nh_.param("MIN_CLUSTER_SIZE",MIN_CLUSTER_SIZE_,{100});

    // mincut param
    private_nh_.param("MIN_CUT_NEIGHBORS",MIN_CUT_NEIGHBORS_,{20});
    private_nh_.param("MIN_CUT_SIGMA",MIN_CUT_SIGMA_,{0.05});
    private_nh_.param("MIN_CUT_RADIUS",MIN_CUT_RADIUS_,{0.20});
    private_nh_.param("MIN_CUT_SOURCE_WEIGHT",MIN_CUT_SOURCE_WEIGHT_,{0.6});

    private_nh_.param("PUB_STATIC_PC",PUB_STATIC_PC_,{true});

    // image param
    private_nh_.param("USE_IMG_MSG",USE_IMG_MSG_,{false});

    // error param
    private_nh_.param("ERROR_PER_DISTANCE",ERROR_PER_DISTANCE_,{0.05});
    private_nh_.param("SD_FACTOR",SD_FACTOR_,{0.025});

    // pos correction param
    private_nh_.param("CORRECT_POS",CORRECT_POS_,{true});
    private_nh_.param("POS_CORRECTION_FACTOR",POS_CORRECTION_FACTOR_,{0.3});

    pc_sub_ = nh_.subscribe("pc_in",1,&PointCloudObjectDetector::pc_callback,this);
    bbox_sub_ = nh_.subscribe("bbox_in",1,&PointCloudObjectDetector::bbox_callback,this);
    img_sub_ = nh_.subscribe("img_in",1,&PointCloudObjectDetector::img_callback,this);
    
    obj_pub_ = nh_.advertise<object_detector_msgs::ObjectPositions>("obj_out",1);
    
    // private_nh_.param("USE_IMG_MSG",USE_IMG_MSG_,{false});
    if(USE_IMG_MSG_){
        obj_img_pub_ = nh_.advertise<object_detector_msgs::ObjectPositionsWithImage>("obj_img_out",1);
        obj_img_debug_pub_ = nh_.advertise<sensor_msgs::Image>("obj_img_debug_out",1);
    }

    private_nh_.param("IS_DEBUG",IS_DEBUG_,{false});
    if(IS_DEBUG_){
        pc_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("pc_out",1);
        if(IS_CLUSTERING_){
            cls_pc_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cls_pc_out",1);
        }
        if(PUB_STATIC_PC_){
            static_pc_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("static_pc_out",1);
        }
    }

    private_nh_.param("IS_PCL_TF",IS_PCL_TF_,{false});
    if(IS_PCL_TF_){
        buffer_.reset(new tf2_ros::Buffer);
        listener_.reset(new tf2_ros::TransformListener(*buffer_));
        broadcaster_.reset(new tf2_ros::TransformBroadcaster);
    }

    get_pc_ = false;

    // initialize point cloud
    cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    static_cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

    load_object_classes();
}

void PointCloudObjectDetector::pc_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    cloud_->clear();
    pcl::fromROSMsg(*msg,*cloud_);
    pc_frame_id_ = msg->header.frame_id;
    pc_stamp_ = msg->header.stamp;
    // if(IS_SAMPLING_){
    //     pcl::shared_ptr<pcl::RandomSample<pcl::PointXYZRGB>> sampler(new pcl::RandomSample<pcl::PointXYZRGB>);
    //     sampler->setInputCloud(cloud_);
    //     sampler->setSample(SAMPLING_NUM_);
    //     sampler->filter(*cloud_);
    //     // ROS_INFO("cloud size: %ld",cloud_->size());
    // }
    if(IS_PCL_TF_){
        geometry_msgs::TransformStamped transform_stamped;
        try{
            // transform_stamped = buffer_->lookupTransform(CAMERA_FRAME_ID_,msg->header.frame_id,ros::Time(0));
            transform_stamped = buffer_->lookupTransform(CAMERA_FRAME_ID_,msg->header.frame_id,msg->header.stamp);
        }
        catch(tf2::TransformException& ex){
            ROS_WARN("%s", ex.what());
            return;
        }
        Eigen::Matrix4f transform = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();
        pcl::transformPointCloud(*cloud_,*cloud_,transform);
    }
    has_received_pc_ = true;
    get_pc_ = true;
}

void PointCloudObjectDetector::bbox_callback(const darknet_ros_msgs::BoundingBoxesConstPtr& msg)
{
    // ROS_INFO("received bbox at %f",ros::Time::now().toSec());
    if(has_received_pc_){
        ros::Time bbox_time = msg->image_header.stamp;

        // object positions
        object_detector_msgs::ObjectPositions positions;
        positions.header.frame_id = CAMERA_FRAME_ID_;
        positions.header.stamp = bbox_time;

        // object positions with image
        object_detector_msgs::ObjectPositionsWithImage positions_with_image;
        positions_with_image.header.frame_id = CAMERA_FRAME_ID_;
        positions_with_image.header.stamp = bbox_time;

        // merged cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cls_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

        // other cloud
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr other_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

        // std::cout << "cloud_ points type: " << typeid(cloud_->points).name() << std::endl;
        // std::cout << "cloud_ point type: " << typeid(cloud_->points.at(0)).name() << std::endl;
        // std::vector<pcl::PointXYZRGB> points;
        // for(const auto &p : cloud_->points) points.emplace_back(p);

        // std::vector<std::vector<pcl::PointXYZRGB>> rearranged_points(cloud_->height,std::vector<pcl::PointXYZRGB>());
        // if(points.size() == cloud_->width*cloud_->height){
        //     for(int i = 0; i < cloud_->height; i++){
        //         for(int j = 0; j < cloud_->width; j++){
        //             rearranged_points.at(i).emplace_back(points.at(i*cloud_->width+j));
        //         }
        //     }
        // }else{
        //     ROS_WARN("points size is not cloud size");
        //     return;
        // }

        std::vector<int> obj_points_indices;    // overlap indices
        obj_points_indices.reserve(cloud_->points.size());

        for(const auto &bbox : msg->bounding_boxes){
            if(!is_valid_class(bbox.Class)) continue;

            // std::vector<pcl::PointXYZRGB> values;
            std::vector<int> bbox_indices;

            // for mincut
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr center_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

            if(!(bbox.xmin == 0 && bbox.xmax == 0)){
                for(int x = bbox.xmin; x < bbox.xmax; x++){
                    for(int y = bbox.ymin; y < bbox.ymax; y++){
                        // values.emplace_back(rearranged_points.at(y).at(x));
                        bbox_indices.emplace_back(y*cloud_->width+x);
                    }
                }

                if(IS_CLUSTERING_ && USE_MINCUT_){
                    int cx = (bbox.xmin + bbox.xmax)/2;
                    int cy = (bbox.ymin + bbox.ymax)/2;
                    // center_cloud->push_back(rearranged_points.at(cy).at(cx));
                    center_cloud->push_back(cloud_->points.at(cy*cloud_->width+cx));
                }
                
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
                obj_cloud->width = bbox.xmax - bbox.xmin;
                obj_cloud->height = bbox.ymax - bbox.ymin;
                obj_cloud->points.resize(obj_cloud->width*obj_cloud->height);
                // convert_from_vec_to_pc(values,obj_cloud);
                convert_from_indices_to_pc(bbox_indices,cloud_,obj_cloud);
                // trim_pc(cloud_,obj_cloud,bbox.xmin,bbox.xmax,bbox.ymin,bbox.ymax);
                *merged_cloud += *obj_cloud;
                for(const auto &i : bbox_indices) obj_points_indices.emplace_back(i);

                double x, y, z;              
                if(IS_CLUSTERING_){
                    if(IS_SAMPLING_){
                        pcl::shared_ptr<pcl::RandomSample<pcl::PointXYZRGB>> sampler(new pcl::RandomSample<pcl::PointXYZRGB>);
                        sampler->setInputCloud(obj_cloud);
                        sampler->setSample((int)(obj_cloud->size()*SAMPLING_RATIO_));
                        sampler->filter(*obj_cloud);
                    }
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cls_obj_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
                    if(USE_MINCUT_) mincut_clustering(obj_cloud,cls_obj_cloud,center_cloud);
                    else clustering(obj_cloud,cls_obj_cloud);
                    if(cls_obj_cloud->points.empty()) return;
                    // std::cout << "clustered cloud width: " << cls_obj_cloud->width << " object cloud width: " << obj_cloud->width << std::endl;
                    calc_position(cls_obj_cloud,x,y,z);
                    *merged_cls_cloud += *cls_obj_cloud;
                }
                else calc_position(obj_cloud,x,y,z);
                
                // object_position
                object_detector_msgs::ObjectPosition position;
                position.Class = bbox.Class;
                position.probability = bbox.probability;
                position.x = x;
                position.y = y;
                position.z = z;
                positions.object_position.emplace_back(position);

                // front_cam_color_optical_frame
                double d = std::sqrt(std::pow(position.x,2) + std::pow(position.z,2));
                // double d = std::sqrt(std::pow(position.x,2) + std::pow(position.y,2) + std::pow(position.z,2));
                // double d = std::sqrt(position.x*position.x + position.y*position.y + position.z*position.z);
                double theta = std::atan2(position.z,position.x) - M_PI/2;
                // std::cout << "distance: " << d << std::endl;
                // std::cout << "x: " << position.x << " y: " << position.y << " z: " << position.z << std::endl;

                // object_position_with_image
                if(USE_IMG_MSG_ && has_received_img_){
                    object_detector_msgs::ObjectPositionWithImage position_with_image;
                    cv::Mat obj_img = cv_ptr_->image(cv::Rect(bbox.xmin,bbox.ymin,bbox.xmax-bbox.xmin,bbox.ymax-bbox.ymin));
                    position_with_image.img = *cv_bridge::CvImage(std_msgs::Header(),"bgr8",obj_img).toImageMsg();
                    position_with_image.img.header.frame_id = CAMERA_FRAME_ID_;
                    // position_with_image.img.header.stamp = now_time;
                    position_with_image.img.header.stamp = bbox_time;
                    position_with_image.Class = bbox.Class;
                    position_with_image.probability = bbox.probability;
                    position_with_image.error = ERROR_PER_DISTANCE_*d;
                    position_with_image.sd = SD_FACTOR_*d;
                    position_with_image.x = x;
                    position_with_image.y = y;
                    position_with_image.z = z;
                    positions_with_image.object_positions_with_img.emplace_back(position_with_image);
                }

                // double d = std::sqrt(std::pow(position.x,2) + std::pow(position.z,2));
                // double theta = std::atan2(position.z,position.x) - M_PI/2;

                // std::cout << "(NAME,X,Y,Z): (" << bbox.Class << "," 
                //                                << position.x << "," 
                //                                << position.y << "," 
                //                                << position.z << ")" << std::endl;
                // std::cout << "Distance[m]: : " << d << std::endl;
                // std::cout << "Angle[rad] : " << theta << std::endl << std::endl;
            }
            else{
                ROS_WARN("No bbox range");
                return;
            }
        }
        if(positions.object_position.empty()) return;
        obj_pub_.publish(positions);

        if(USE_IMG_MSG_ && has_received_img_){
            // publish object positions with image
            if(positions_with_image.object_positions_with_img.empty()) return;
            obj_img_pub_.publish(positions_with_image);
            obj_img_debug_pub_.publish(positions_with_image.object_positions_with_img.at(0).img);
        }

        if(IS_DEBUG_){
            // convert_from_indices_to_pc(obj_points_indices,cloud_,merged_cloud);
            // separate_pc_by_indices(obj_points_indices,cloud_,merged_cloud,other_cloud);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*merged_cloud,cloud_msg);
            cloud_msg.header.frame_id = pc_frame_id_;
            // cloud_msg.header.stamp = bbox_time;
            cloud_msg.header.stamp = pc_stamp_;
            pc_pub_.publish(cloud_msg);

            if(IS_CLUSTERING_){
                sensor_msgs::PointCloud2 cls_cloud_msg;
                pcl::toROSMsg(*merged_cls_cloud,cls_cloud_msg);
                cls_cloud_msg.header.frame_id = pc_frame_id_;
                // cls_cloud_msg.header.stamp = bbox_time;
                cls_cloud_msg.header.stamp = pc_stamp_;
                cls_pc_pub_.publish(cls_cloud_msg);
                // ROS_INFO("published clustered pointcloud at %f",now_time.toSec());
            }

            if(PUB_STATIC_PC_){
                separate_pc_by_indices(obj_points_indices,cloud_,merged_cloud,static_cloud_);
                static_pc_stamp_ = pc_stamp_;
            }
        }
    }
    has_received_pc_ = false;
    has_received_img_ = false;
    // cloud_->clear();
}

void PointCloudObjectDetector::img_callback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_ptr_ = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& ex){
        ROS_ERROR("cv_bridge exception: %s",ex.what());
        return;
    }
    has_received_img_ = true;
}

void PointCloudObjectDetector::publish_static_pc()
{
    std::cout << "publish static pc at " << ros::Time::now() << std::endl;
    sensor_msgs::PointCloud2 static_cloud_msg;
    if(static_cloud_->points.empty())
    {
        std::cout << "There is no static cloud" << std::endl;
        if(cloud_->points.empty()){
            ROS_WARN("cloud is empty");
            return;
        }
        else{
            pcl::toROSMsg(*cloud_,static_cloud_msg);
            static_cloud_msg.header.frame_id = pc_frame_id_;
            static_cloud_msg.header.stamp = pc_stamp_;
        }
    }
    else{
        std::cout << "There is static cloud" << std::endl;
        pcl::toROSMsg(*static_cloud_,static_cloud_msg);
        static_cloud_msg.header.frame_id = pc_frame_id_;
        static_cloud_msg.header.stamp = static_pc_stamp_;
    }

    static_pc_pub_.publish(static_cloud_msg);
    static_cloud_->clear();
}

void PointCloudObjectDetector::load_object_classes()
{
    std::string yaml_file_name;
    private_nh_.param("YAML_FILE_NAME",yaml_file_name,{std::string("object_classes")});
    XmlRpc::XmlRpcValue object_classes;
    if(!private_nh_.getParam(yaml_file_name.c_str(),object_classes)){
        ROS_WARN("Could not load %s",yaml_file_name.c_str());
        return;
    }

    ROS_ASSERT(object_classes.getType() == XmlRpc::XmlRpcValue::TypeArray);
    for(int i = 0; i < object_classes.size(); i++){
        ROS_ASSERT(object_classes[i].getType() == XmlRpc::XmlRpcValue::TypeString);
        object_classes_.emplace_back(static_cast<std::string>(object_classes[i]));
        std::cout << object_classes_.at(i) << std::endl;
    }

    ROS_INFO("Loaded %s",yaml_file_name.c_str());
}

bool PointCloudObjectDetector::is_valid_class(const std::string& class_name)
{
    for(const auto &c : object_classes_){
        if(c == class_name) return true;
    }
    return false;
}

void PointCloudObjectDetector::convert_from_vec_to_pc(std::vector<pcl::PointXYZRGB>& vec,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc)
{
    int count = 0;
    for(const auto &v : vec){
        if(!std::isnan(v.x) && !std::isnan(v.y) && !std::isnan(v.z)){
            pc->points.at(count).x = v.x;
            pc->points.at(count).y = v.y;
            pc->points.at(count).z = v.z;
            pc->points.at(count).r = v.r;
            pc->points.at(count).g = v.g;
            pc->points.at(count).b = v.b;
            count++;
        }
    }
}

void PointCloudObjectDetector::convert_from_indices_to_pc(std::vector<int>& indices,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& output_cloud)
{
    int count = 0;
    for(const auto &i : indices){
        if(!std::isnan(input_cloud->points.at(i).x) && !std::isnan(input_cloud->points.at(i).y) && !std::isnan(input_cloud->points.at(i).z)){
            output_cloud->points.at(count).x = input_cloud->points.at(i).x;
            output_cloud->points.at(count).y = input_cloud->points.at(i).y;
            output_cloud->points.at(count).z = input_cloud->points.at(i).z;
            output_cloud->points.at(count).r = input_cloud->points.at(i).r;
            output_cloud->points.at(count).g = input_cloud->points.at(i).g;
            output_cloud->points.at(count).b = input_cloud->points.at(i).b;
            count++;
        }
    }
}

void PointCloudObjectDetector::trim_pc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& output_cloud,int x_min,int x_max,int y_min,int y_max)
{
    pcl::shared_ptr<pcl::ExtractIndices<pcl::PointXYZRGB>> t_ex(new pcl::ExtractIndices<pcl::PointXYZRGB>);
    t_ex->setInputCloud(input_cloud);
    t_ex->setNegative(false);
    
    size_t row_start = y_min;
    size_t col_start = x_min;
    size_t nb_rows = y_max - y_min + 1;
    size_t nb_cols = x_max - x_min + 1;

    t_ex->setIndices(row_start,col_start,nb_rows,nb_cols);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_trim_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    t_ex->filter(*tmp_trim_cloud);
    // std::cout << "tmp cloud size: " << tmp_cloud->size() << " tmp cloud width: " << tmp_cloud->width << " tmp cloud height: " << tmp_cloud->height << std::endl;
    // tmp_cloud->width = nb_cols;
    // tmp_cloud->height = nb_rows;
    output_cloud = tmp_trim_cloud;
    // std::cout << "output cloud size: " << output_cloud->size() << std::endl;
}

void PointCloudObjectDetector::clustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& output_cloud)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(input_cloud);

    std::vector<pcl::PointIndices> indices;
    pcl::shared_ptr<pcl::EuclideanClusterExtraction<pcl::PointXYZRGB>> ec(new pcl::EuclideanClusterExtraction<pcl::PointXYZRGB>);
    ec->setClusterTolerance(CLUSTER_TOLERANCE_);
    ec->setMinClusterSize(MIN_CLUSTER_SIZE_);
    ec->setMaxClusterSize(input_cloud->points.size());
    ec->setSearchMethod(tree);
    ec->setInputCloud(input_cloud);
    ec->extract(indices);

    pcl::shared_ptr<pcl::ExtractIndices<pcl::PointXYZRGB>> ex(new pcl::ExtractIndices<pcl::PointXYZRGB>);
    ex->setInputCloud(input_cloud);
    ex->setNegative(false);

    pcl::PointIndices::Ptr tmp_clustered_indices (new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    if(indices.size() <= 1) return;
    *tmp_clustered_indices = indices[0];
    ex->setIndices(tmp_clustered_indices);
    ex->filter(*tmp_cloud);
    output_cloud = tmp_cloud;
}

void PointCloudObjectDetector::mincut_clustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& output_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& center_cloud)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(input_cloud);

    std::vector<pcl::PointIndices> indices;
    pcl::shared_ptr<pcl::MinCutSegmentation<pcl::PointXYZRGB>> seg(new pcl::MinCutSegmentation<pcl::PointXYZRGB>);
    seg->setSigma(MIN_CUT_SIGMA_);
    seg->setRadius(MIN_CUT_RADIUS_);
    seg->setNumberOfNeighbours(MIN_CUT_NEIGHBORS_);
    seg->setSourceWeight(MIN_CUT_SOURCE_WEIGHT_);
    seg->setSearchMethod(tree);
    seg->setInputCloud(input_cloud);
    seg->setForegroundPoints(center_cloud);
    seg->extract(indices);
    
    pcl::shared_ptr<pcl::ExtractIndices<pcl::PointXYZRGB>> ex(new pcl::ExtractIndices<pcl::PointXYZRGB>);
    ex->setInputCloud(input_cloud);
    ex->setNegative(false);

    pcl::PointIndices::Ptr tmp_clustered_indices (new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    *tmp_clustered_indices = indices[1];
    ex->setIndices(tmp_clustered_indices);
    ex->filter(*tmp_cloud);
    output_cloud = tmp_cloud;
    ROS_INFO("mincut clustering at %f",ros::Time::now().toSec());
}

// void PointCloudObjectDetector::separate_pc_by_indices(std::vector<int>& indices,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& other_cloud)
// {
//     // too heavy!!
//     //
//     std::vector<int> other_indices;
//     other_indices.reserve(input_cloud->points.size());
//     std::vector<int> target_indices;
//     target_indices.reserve(indices.size());
//     for(int i = 0; i < input_cloud->points.size(); i++){
//         bool is_target = false;
//         for(const auto &j : indices){
//             if(i == j){
//                 is_target = true;
//                 break;
//             }
//         }
//         if(is_target) target_indices.emplace_back(i);
//         else other_indices.emplace_back(i);
//     }
//     //
//     // resize
//     if(target_cloud->points.size() != target_indices.size()){
//         target_cloud->width = target_indices.size();
//         target_cloud->height = 1;
//         target_cloud->points.resize(target_cloud->width*target_cloud->height);
//     }
//     if(other_cloud->points.size() != other_indices.size()){
//         other_cloud->width = other_indices.size();
//         other_cloud->height = 1;
//         other_cloud->points.resize(other_cloud->width*other_cloud->height);
//     }
//     //
//     // set target cloud
//     int count = 0;
//     for(const auto &i : target_indices){
//         target_cloud->points.at(count).x = input_cloud->points.at(i).x;
//         target_cloud->points.at(count).y = input_cloud->points.at(i).y;
//         target_cloud->points.at(count).z = input_cloud->points.at(i).z;
//         target_cloud->points.at(count).r = input_cloud->points.at(i).r;
//         target_cloud->points.at(count).g = input_cloud->points.at(i).g;
//         target_cloud->points.at(count).b = input_cloud->points.at(i).b;
//         count++;
//     }
//     //
//     // set other cloud
//     // count = 0;
//     // for(const auto &i : other_indices){
//     //     other_cloud->points.at(count).x = input_cloud->points.at(i).x;
//     //     other_cloud->points.at(count).y = input_cloud->points.at(i).y;
//     //     other_cloud->points.at(count).z = input_cloud->points.at(i).z;
//     //     other_cloud->points.at(count).r = input_cloud->points.at(i).r;
//     //     other_cloud->points.at(count).g = input_cloud->points.at(i).g;
//     //     other_cloud->points.at(count).b = input_cloud->points.at(i).b;
//     //     count++;
//     // }
// }

void PointCloudObjectDetector::separate_pc_by_indices(std::vector<int>& indices,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target_cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& other_cloud)
{
    pcl::shared_ptr<pcl::ExtractIndices<pcl::PointXYZRGB>> t_ex(new pcl::ExtractIndices<pcl::PointXYZRGB>);
    t_ex->setInputCloud(input_cloud);
    t_ex->setNegative(false);
    
    pcl::PointIndices::Ptr tmp_target_indices (new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_target_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    // *tmp_target_indices->indices = indices;
    for(const auto &i : indices) tmp_target_indices->indices.push_back(i);
    t_ex->setIndices(tmp_target_indices);
    t_ex->filter(*tmp_target_cloud);
    target_cloud = tmp_target_cloud;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_other_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    t_ex->setIndices(tmp_target_indices);
    t_ex->setNegative(true);
    t_ex->filter(*tmp_other_cloud);
    other_cloud = tmp_other_cloud;
}

void PointCloudObjectDetector::calc_position(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,double& x,double& y,double& z)
{
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    int count = 0;
    for(const auto &p : cloud->points){
        sum_x += p.x;
        sum_y += p.y;
        sum_z += p.z;
        count++;
    }
    x = sum_x/(double)count;
    y = sum_y/(double)count;
    z = sum_z/(double)count;
}

void PointCloudObjectDetector::process()
{
    ros::Rate rate(HZ_);
    while(ros::ok()){
        if(get_pc_ && IS_DEBUG_ && PUB_STATIC_PC_){
            publish_static_pc();
        }
        ros::spinOnce();
        rate.sleep();
    }
}