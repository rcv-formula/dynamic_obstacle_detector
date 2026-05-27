#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/int32.hpp"

#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

#include "obstacle_context_msgs/msg/obstacle_cluster.hpp"
#include "obstacle_context_msgs/msg/obstacle_cluster_array.hpp"

class ObstacleClusterContextNode : public rclcpp::Node
{
public:
  ObstacleClusterContextNode()
  : Node("obstacle_cluster_context_node")
  {
    declareParameters();
    loadParameters();

    if (input_type_ == "pointcloud2") {
      pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&ObstacleClusterContextNode::pointCloudCallback, this, std::placeholders::_1));
    } else {
      scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&ObstacleClusterContextNode::scanCallback, this, std::placeholders::_1));
    }

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_,
      rclcpp::QoS(100),
      std::bind(&ObstacleClusterContextNode::odomCallback, this, std::placeholders::_1));

    if (use_imu_motion_gate_) {
      imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&ObstacleClusterContextNode::imuCallback, this, std::placeholders::_1));
    }

    obstacle_mode_sub_ = this->create_subscription<std_msgs::msg::Int32>(
      obstacle_mode_topic_,
      rclcpp::QoS(10),
      std::bind(&ObstacleClusterContextNode::obstacleModeCallback, this, std::placeholders::_1));

    processed_scan_pub_ =
      this->create_publisher<sensor_msgs::msg::LaserScan>(
        processed_scan_topic_,
        rclcpp::SensorDataQoS());

    cluster_pub_ =
      this->create_publisher<obstacle_context_msgs::msg::ObstacleClusterArray>(
        output_topic_,
        rclcpp::SensorDataQoS());

    marker_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
        marker_topic_,
        rclcpp::QoS(1));

    dynamic_pointcloud_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        dynamic_pointcloud_topic_,
        rclcpp::SensorDataQoS());

    static_pointcloud_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        static_pointcloud_topic_,
        rclcpp::SensorDataQoS());

    RCLCPP_INFO(this->get_logger(), "obstacle_cluster_context_node started");
    RCLCPP_INFO(this->get_logger(), "use_sim_time: %s", use_sim_time_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "input_type: %s", input_type_.c_str());
    RCLCPP_INFO(this->get_logger(), "scan_topic: %s", scan_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "pointcloud_topic: %s", pointcloud_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "odom_topic: %s", odom_topic_.c_str());
    if (use_imu_motion_gate_) {
      RCLCPP_INFO(this->get_logger(), "imu_topic: %s", imu_topic_.c_str());
    } else {
      RCLCPP_INFO(this->get_logger(), "imu motion gate disabled");
    }
    RCLCPP_INFO(this->get_logger(), "processed_scan_topic: %s", processed_scan_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "output_topic: %s", output_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "marker_topic: %s", marker_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "dynamic_pointcloud_topic: %s", dynamic_pointcloud_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "static_pointcloud_topic: %s", static_pointcloud_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "obstacle_mode_topic: %s", obstacle_mode_topic_.c_str());
    RCLCPP_INFO(
      this->get_logger(),
      "pointcloud_use_latest_tf: %s",
      pointcloud_use_latest_tf_ ? "true" : "false");
  }

private:
  struct OdomPose
  {
    rclcpp::Time stamp;
    double x;
    double y;
    double yaw;
  };

  struct ScanPoint
  {
    size_t index;
    size_t source_index;
    double range;
    double angle;
    double x;
    double y;
    double z;
  };

  struct TrackState
  {
    double x;
    double y;
    rclcpp::Time stamp;
    OdomPose pose;
  };

  struct Track
  {
    uint32_t id;

    double x;
    double y;
    rclcpp::Time stamp;
    OdomPose pose;

    int observed_count;

    double vx;
    double vy;
    double speed;
    double compensated_displacement;

    uint8_t motion_label;
    int static_hit_count;
    int dynamic_hit_count;

    std::deque<TrackState> history;
  };

  struct ScanFrame
  {
    rclcpp::Time stamp;
    OdomPose pose;
    std::vector<ScanPoint> points;
    double angle_increment;
  };

  struct DensityClusterStats
  {
    double max_weight;
    int dense_point_count;
  };

  struct DynamicPointMetadata
  {
    uint32_t track_id;
    float relative_speed;
    float relative_yaw;
  };

  struct DynamicPoint
  {
    ScanPoint point;
    DynamicPointMetadata metadata;
  };

  using DensityCell = std::pair<int, int>;
  using DensityGrid = std::map<DensityCell, double>;

  static constexpr int OBSTACLE_MODE_STATIC_ONLY = 1;
  static constexpr int OBSTACLE_MODE_DYNAMIC_ONLY = 2;
  static constexpr int OBSTACLE_MODE_STATIC_AND_DYNAMIC = 3;

  void declareParameters()
  {
    this->declare_parameter<std::string>("input_type", "laser_scan");
    this->declare_parameter<std::string>("scan_topic", "/scan");
    this->declare_parameter<std::string>("pointcloud_topic", "/cloud");
    this->declare_parameter<std::string>("odom_topic", "/odom");
    this->declare_parameter<std::string>("imu_topic", "/imu/data");

    this->declare_parameter<std::string>("processed_scan_topic", "/processed_scan");
    this->declare_parameter<std::string>("output_topic", "/obstacle_clusters");
    this->declare_parameter<std::string>("marker_topic", "/obstacle_cluster_markers");
    this->declare_parameter<std::string>("dynamic_pointcloud_topic", "/dynamic_pointcloud");
    this->declare_parameter<std::string>("static_pointcloud_topic", "/static_pointcloud");
    this->declare_parameter<std::string>("obstacle_mode_topic", "/obstacle_mode");
    this->declare_parameter<bool>("pointcloud_use_latest_tf", true);

    this->declare_parameter<double>("min_valid_range", 0.15);
    this->declare_parameter<double>("max_valid_range", 10.0);

    this->declare_parameter<bool>("use_roi_filter", true);
    this->declare_parameter<double>("roi_angle_min_deg", -120.0);
    this->declare_parameter<double>("roi_angle_max_deg", 120.0);

    this->declare_parameter<double>("cluster_range_jump", 0.20);
    this->declare_parameter<int>("min_cluster_points", 2);
    this->declare_parameter<double>("min_cluster_width", 0.04);
    this->declare_parameter<double>("max_cluster_width", 0.80);

    this->declare_parameter<double>("association_distance", 0.60);
    this->declare_parameter<double>("track_timeout_sec", 0.50);
    this->declare_parameter<int>("min_observations", 5);
    this->declare_parameter<int>("track_history_size", 7);

    this->declare_parameter<double>("static_speed_threshold", 0.50);
    this->declare_parameter<double>("dynamic_speed_threshold", 1.5);
    this->declare_parameter<int>("static_confirm_count", 5);
    this->declare_parameter<int>("dynamic_confirm_count", 3);

    this->declare_parameter<bool>("use_wall_static_label", true);
    this->declare_parameter<double>("wall_static_min_width", 1.0);
    this->declare_parameter<int>("wall_static_min_points", 10);

    this->declare_parameter<bool>("use_scan_density_static_filter", true);
    this->declare_parameter<int>("scan_density_history_size", 8);
    this->declare_parameter<double>("scan_density_cell_size", 0.10);
    this->declare_parameter<double>("scan_density_min_weight", 6.0);
    this->declare_parameter<int>("scan_density_neighbor_cells", 1);
    this->declare_parameter<int>("scan_density_cluster_min_dense_points", 2);

    this->declare_parameter<bool>("use_imu_motion_gate", true);
    this->declare_parameter<double>("imu_accel_gate_threshold", 1.0);
    this->declare_parameter<double>("imu_yaw_rate_gate_threshold", 0.7);
    this->declare_parameter<double>("imu_motion_gate_hold_sec", 0.25);

    this->declare_parameter<double>("max_odom_age_sec", 0.35);
    this->declare_parameter<double>("max_pose_jump_dist", 1.0);
    this->declare_parameter<double>("max_pose_jump_yaw_deg", 20.0);
    this->declare_parameter<double>("odom_history_sec", 2.0);

    this->declare_parameter<double>("marker_scale", 0.15);
    this->declare_parameter<double>("marker_text_scale", 0.18);
    this->declare_parameter<double>("marker_lifetime_sec", 0.2);
  }

  void loadParameters()
  {
    rclcpp::Parameter use_sim_time_param;
    if (this->get_parameter("use_sim_time", use_sim_time_param)) {
      use_sim_time_ = use_sim_time_param.as_bool();
    }

    input_type_ = this->get_parameter("input_type").as_string();
    scan_topic_ = this->get_parameter("scan_topic").as_string();
    pointcloud_topic_ = this->get_parameter("pointcloud_topic").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    imu_topic_ = this->get_parameter("imu_topic").as_string();

    processed_scan_topic_ = this->get_parameter("processed_scan_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();
    marker_topic_ = this->get_parameter("marker_topic").as_string();
    dynamic_pointcloud_topic_ =
      this->get_parameter("dynamic_pointcloud_topic").as_string();
    static_pointcloud_topic_ =
      this->get_parameter("static_pointcloud_topic").as_string();
    obstacle_mode_topic_ = this->get_parameter("obstacle_mode_topic").as_string();
    pointcloud_use_latest_tf_ =
      this->get_parameter("pointcloud_use_latest_tf").as_bool();

    if (input_type_ != "laser_scan" && input_type_ != "pointcloud2") {
      RCLCPP_WARN(
        this->get_logger(),
        "invalid input_type '%s'. falling back to 'laser_scan'",
        input_type_.c_str());
      input_type_ = "laser_scan";
    }

    min_valid_range_ = this->get_parameter("min_valid_range").as_double();
    max_valid_range_ = this->get_parameter("max_valid_range").as_double();

    use_roi_filter_ = this->get_parameter("use_roi_filter").as_bool();
    roi_angle_min_rad_ = deg2rad(this->get_parameter("roi_angle_min_deg").as_double());
    roi_angle_max_rad_ = deg2rad(this->get_parameter("roi_angle_max_deg").as_double());

    cluster_range_jump_ = this->get_parameter("cluster_range_jump").as_double();
    min_cluster_points_ = this->get_parameter("min_cluster_points").as_int();
    min_cluster_width_ = this->get_parameter("min_cluster_width").as_double();
    max_cluster_width_ = this->get_parameter("max_cluster_width").as_double();

    association_distance_ = this->get_parameter("association_distance").as_double();
    track_timeout_sec_ = this->get_parameter("track_timeout_sec").as_double();
    min_observations_ = this->get_parameter("min_observations").as_int();
    track_history_size_ = this->get_parameter("track_history_size").as_int();

    static_speed_threshold_ = this->get_parameter("static_speed_threshold").as_double();
    dynamic_speed_threshold_ = this->get_parameter("dynamic_speed_threshold").as_double();
    static_confirm_count_ = this->get_parameter("static_confirm_count").as_int();
    dynamic_confirm_count_ = this->get_parameter("dynamic_confirm_count").as_int();

    use_wall_static_label_ = this->get_parameter("use_wall_static_label").as_bool();
    wall_static_min_width_ = this->get_parameter("wall_static_min_width").as_double();
    wall_static_min_points_ = this->get_parameter("wall_static_min_points").as_int();

    use_scan_density_static_filter_ =
      this->get_parameter("use_scan_density_static_filter").as_bool();
    scan_density_history_size_ =
      this->get_parameter("scan_density_history_size").as_int();
    scan_density_cell_size_ =
      this->get_parameter("scan_density_cell_size").as_double();
    scan_density_min_weight_ =
      this->get_parameter("scan_density_min_weight").as_double();
    scan_density_neighbor_cells_ =
      this->get_parameter("scan_density_neighbor_cells").as_int();
    scan_density_cluster_min_dense_points_ =
      this->get_parameter("scan_density_cluster_min_dense_points").as_int();

    use_imu_motion_gate_ = this->get_parameter("use_imu_motion_gate").as_bool();
    imu_accel_gate_threshold_ = this->get_parameter("imu_accel_gate_threshold").as_double();
    imu_yaw_rate_gate_threshold_ = this->get_parameter("imu_yaw_rate_gate_threshold").as_double();
    imu_motion_gate_hold_sec_ = this->get_parameter("imu_motion_gate_hold_sec").as_double();

    if (static_speed_threshold_ > dynamic_speed_threshold_) {
      RCLCPP_WARN(
        this->get_logger(),
        "static_speed_threshold is greater than dynamic_speed_threshold. swapping thresholds");
      std::swap(static_speed_threshold_, dynamic_speed_threshold_);
    }

    max_odom_age_sec_ = this->get_parameter("max_odom_age_sec").as_double();
    max_pose_jump_dist_ = this->get_parameter("max_pose_jump_dist").as_double();
    max_pose_jump_yaw_rad_ = deg2rad(this->get_parameter("max_pose_jump_yaw_deg").as_double());
    odom_history_sec_ = this->get_parameter("odom_history_sec").as_double();

    marker_scale_ = this->get_parameter("marker_scale").as_double();
    marker_text_scale_ = this->get_parameter("marker_text_scale").as_double();
    marker_lifetime_sec_ = this->get_parameter("marker_lifetime_sec").as_double();

    if (track_history_size_ < 1) {
      track_history_size_ = 1;
    }

    if (min_observations_ < 1) {
      min_observations_ = 1;
    }

    if (static_confirm_count_ < 1) {
      static_confirm_count_ = 1;
    }

    if (dynamic_confirm_count_ < 1) {
      dynamic_confirm_count_ = 1;
    }

    if (wall_static_min_points_ < 1) {
      wall_static_min_points_ = 1;
    }

    if (scan_density_history_size_ < 1) {
      scan_density_history_size_ = 1;
    }

    if (scan_density_cell_size_ < 1e-3) {
      scan_density_cell_size_ = 1e-3;
    }

    if (scan_density_min_weight_ < 1.0) {
      scan_density_min_weight_ = 1.0;
    }

    if (scan_density_neighbor_cells_ < 0) {
      scan_density_neighbor_cells_ = 0;
    }

    if (scan_density_cluster_min_dense_points_ < 1) {
      scan_density_cluster_min_dense_points_ = 1;
    }
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    OdomPose pose;
    if (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0) {
      pose.stamp = this->get_clock()->now();
      RCLCPP_INFO_ONCE(
        this->get_logger(),
        "odom header stamp is zero. using receive time for odom tracking");
    } else {
      pose.stamp = msg->header.stamp;
    }

    pose.x = msg->pose.pose.position.x;
    pose.y = msg->pose.pose.position.y;
    pose.yaw = yawFromQuaternion(msg->pose.pose.orientation);

    if (!odom_history_.empty()) {
      const OdomPose & prev = odom_history_.back();
      const double dist = std::hypot(pose.x - prev.x, pose.y - prev.y);
      const double dyaw = std::fabs(normalizeAngle(pose.yaw - prev.yaw));
      const double dt = (pose.stamp - prev.stamp).seconds();

      if (dt > 1e-3 && dist <= max_pose_jump_dist_) {
        current_ego_speed_ = dist / dt;
      }

      if (dist > max_pose_jump_dist_ || dyaw > max_pose_jump_yaw_rad_) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          1000,
          "large odom jump detected. dist=%.3f, dyaw=%.3f deg",
          dist,
          dyaw * 180.0 / M_PI);
      }
    }

    odom_history_.push_back(pose);

    while (!odom_history_.empty()) {
      const double age = (pose.stamp - odom_history_.front().stamp).seconds();

      if (age > odom_history_sec_) {
        odom_history_.pop_front();
      } else {
        break;
      }
    }
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
  {
    sensor_msgs::msg::LaserScan processed_scan = makeProcessedScan(*scan);
    processed_scan_pub_->publish(processed_scan);

    std::vector<ScanPoint> points = scanToPoints(processed_scan);
    processPoints(
      processed_scan.header,
      points,
      processed_scan.angle_increment,
      processed_scan.ranges.size());
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud)
  {
    const std::vector<ScanPoint> points = pointCloudToPoints(*cloud);
    processPoints(
      cloud->header,
      points,
      estimateAngleIncrement(points),
      pointCloudPointCount(*cloud));
  }

  void processPoints(
    const std_msgs::msg::Header & header,
    const std::vector<ScanPoint> & points,
    double angle_increment,
    size_t source_point_count)
  {
    OdomPose curr_pose;
    const bool has_odom = getInterpolatedOdom(header.stamp, curr_pose);

    if (!has_odom) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "no valid odom for scan time. clusters will be UNKNOWN");
    }

    std::vector<std::vector<ScanPoint>> clusters = makeClusters(points);
    const DensityGrid density_grid = has_odom ? makeScanDensityGrid(curr_pose) : DensityGrid{};
    std::vector<std::vector<ScanPoint>> valid_clusters;

    obstacle_context_msgs::msg::ObstacleClusterArray out;
    out.header = header;

    for (const auto & cluster : clusters) {
      if (!isValidCluster(cluster)) {
        continue;
      }

      auto msg = makeClusterMsg(header, cluster);
      if (has_odom) {
        const DensityClusterStats density_stats =
          computeDensityClusterStats(cluster, density_grid);
        msg.is_density_static = isDensityStaticCluster(density_stats);
        msg.max_dense_weight =
          static_cast<float>(density_stats.max_weight);
      }
      out.clusters.push_back(msg);
      valid_clusters.push_back(cluster);
    }

    if (has_odom) {
      updateTracks(out, curr_pose);
      pushScanDensityHistory(
        points,
        header.stamp,
        curr_pose,
        angle_increment);
    } else {
      setAllClustersUnknown(out);
      removeOldTracks(header.stamp);
    }

    applyObstacleMode(out);

    cluster_pub_->publish(out);

    auto marker_array = makeMarkerArray(out);
    marker_pub_->publish(marker_array);

    publishSegmentedPointClouds(header, points, valid_clusters, out, source_point_count);
  }

  void obstacleModeCallback(const std_msgs::msg::Int32::SharedPtr msg)
  {
    if (msg->data < OBSTACLE_MODE_STATIC_ONLY ||
      msg->data > OBSTACLE_MODE_STATIC_AND_DYNAMIC)
    {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "invalid obstacle mode %d. use 1=static only, 2=dynamic only, 3=both",
        msg->data);
      return;
    }

    if (obstacle_mode_ != msg->data) {
      obstacle_mode_ = msg->data;
      RCLCPP_INFO(
        this->get_logger(),
        "obstacle mode changed to %d (%s)",
        obstacle_mode_,
        obstacleModeText().c_str());
    }
  }

  sensor_msgs::msg::LaserScan makeProcessedScan(
    const sensor_msgs::msg::LaserScan & scan) const
  {
    sensor_msgs::msg::LaserScan out = scan;

    for (size_t i = 0; i < scan.ranges.size(); ++i) {
      const double range = scan.ranges[i];
      const double angle = scan.angle_min + static_cast<double>(i) * scan.angle_increment;

      bool keep = true;

      if (!std::isfinite(range)) {
        keep = false;
      }

      if (range < min_valid_range_ || range > max_valid_range_) {
        keep = false;
      }

      if (range < scan.range_min || range > scan.range_max) {
        keep = false;
      }

      if (use_roi_filter_) {
        if (angle < roi_angle_min_rad_ || angle > roi_angle_max_rad_) {
          keep = false;
        }
      }

      if (!keep) {
        out.ranges[i] = std::numeric_limits<float>::infinity();
      } else {
        out.ranges[i] = static_cast<float>(range);
      }
    }

    return out;
  }

  std::vector<ScanPoint> scanToPoints(
    const sensor_msgs::msg::LaserScan & scan) const
  {
    std::vector<ScanPoint> points;
    points.reserve(scan.ranges.size());

    for (size_t i = 0; i < scan.ranges.size(); ++i) {
      const double range = scan.ranges[i];

      if (!isValidRange(range, scan)) {
        continue;
      }

      const double angle = scan.angle_min + static_cast<double>(i) * scan.angle_increment;

      ScanPoint p;
      p.index = i;
      p.source_index = i;
      p.range = range;
      p.angle = angle;
      p.x = range * std::cos(angle);
      p.y = range * std::sin(angle);
      p.z = 0.0;

      points.push_back(p);
    }

    return points;
  }

  std::vector<ScanPoint> pointCloudToPoints(
    const sensor_msgs::msg::PointCloud2 & cloud)
  {
    std::vector<ScanPoint> points;
    points.reserve(pointCloudPointCount(cloud));

    if (!hasPointCloudField(cloud, "x") || !hasPointCloudField(cloud, "y")) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "PointCloud2 input must contain x and y fields");
      return points;
    }

    const bool has_z = hasPointCloudField(cloud, "z");
    if (!has_z) {
      RCLCPP_WARN_ONCE(
        this->get_logger(),
        "PointCloud2 input has no z field. using z=0.0 for output clouds");
    }

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");

    if (has_z) {
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

      for (size_t source_index = 0; iter_x != iter_x.end();
        ++iter_x, ++iter_y, ++iter_z, ++source_index)
      {
        ScanPoint point;
        if (makePointCloudScanPoint(*iter_x, *iter_y, *iter_z, source_index, point)) {
          points.push_back(point);
        }
      }
    } else {
      for (size_t source_index = 0; iter_x != iter_x.end();
        ++iter_x, ++iter_y, ++source_index)
      {
        ScanPoint point;
        if (makePointCloudScanPoint(*iter_x, *iter_y, 0.0, source_index, point)) {
          points.push_back(point);
        }
      }
    }

    std::sort(
      points.begin(),
      points.end(),
      [](const ScanPoint & lhs, const ScanPoint & rhs) {
        if (lhs.angle == rhs.angle) {
          return lhs.range < rhs.range;
        }
        return lhs.angle < rhs.angle;
      });

    for (size_t i = 0; i < points.size(); ++i) {
      points[i].index = i;
    }

    return points;
  }

  bool makePointCloudScanPoint(
    double x,
    double y,
    double z,
    size_t source_index,
    ScanPoint & point) const
  {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      return false;
    }

    const double range = std::hypot(x, y);
    if (range < min_valid_range_ || range > max_valid_range_) {
      return false;
    }

    const double angle = std::atan2(y, x);
    if (use_roi_filter_) {
      if (angle < roi_angle_min_rad_ || angle > roi_angle_max_rad_) {
        return false;
      }
    }

    point.index = source_index;
    point.source_index = source_index;
    point.range = range;
    point.angle = angle;
    point.x = x;
    point.y = y;
    point.z = z;

    return true;
  }

  double estimateAngleIncrement(const std::vector<ScanPoint> & points) const
  {
    if (points.size() < 2) {
      return 0.0;
    }

    double sum_increment = 0.0;
    int count = 0;

    for (size_t i = 1; i < points.size(); ++i) {
      const double increment = points[i].angle - points[i - 1].angle;
      if (std::isfinite(increment) && increment > 1e-9) {
        sum_increment += increment;
        count++;
      }
    }

    if (count == 0) {
      return 0.0;
    }

    return sum_increment / static_cast<double>(count);
  }

  bool hasPointCloudField(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std::string & field_name) const
  {
    for (const auto & field : cloud.fields) {
      if (field.name == field_name) {
        return true;
      }
    }

    return false;
  }

  size_t pointCloudPointCount(const sensor_msgs::msg::PointCloud2 & cloud) const
  {
    return static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
  }

  bool isValidRange(
    double range,
    const sensor_msgs::msg::LaserScan & scan) const
  {
    if (!std::isfinite(range)) {
      return false;
    }

    if (range < min_valid_range_ || range > max_valid_range_) {
      return false;
    }

    if (range < scan.range_min || range > scan.range_max) {
      return false;
    }

    return true;
  }

  std::vector<std::vector<ScanPoint>> makeClusters(
    const std::vector<ScanPoint> & points) const
  {
    std::vector<std::vector<ScanPoint>> clusters;

    if (points.empty()) {
      return clusters;
    }

    std::vector<ScanPoint> current;
    current.push_back(points.front());

    for (size_t i = 1; i < points.size(); ++i) {
      const ScanPoint & prev = points[i - 1];
      const ScanPoint & now = points[i];

      const bool adjacent_beam = (now.index == prev.index + 1);
      const double dist = std::hypot(now.x - prev.x, now.y - prev.y);

      if (adjacent_beam && dist < cluster_range_jump_) {
        current.push_back(now);
      } else {
        if (!current.empty()) {
          clusters.push_back(current);
        }

        current.clear();
        current.push_back(now);
      }
    }

    if (!current.empty()) {
      clusters.push_back(current);
    }

    return clusters;
  }

  bool isValidCluster(const std::vector<ScanPoint> & cluster) const
  {
    if (static_cast<int>(cluster.size()) < min_cluster_points_) {
      return false;
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto & p : cluster) {
      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);
    }

    const double width = std::hypot(max_x - min_x, max_y - min_y);

    if (width < min_cluster_width_) {
      return false;
    }

    if (width > max_cluster_width_) {
      return false;
    }

    return true;
  }

  obstacle_context_msgs::msg::ObstacleCluster makeClusterMsg(
    const std_msgs::msg::Header & header,
    const std::vector<ScanPoint> & cluster) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    ObstacleCluster msg;
    msg.header = header;

    double sum_x = 0.0;
    double sum_y = 0.0;

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    double min_range = std::numeric_limits<double>::infinity();
    double max_range = -std::numeric_limits<double>::infinity();

    for (const auto & p : cluster) {
      sum_x += p.x;
      sum_y += p.y;

      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);

      min_range = std::min(min_range, p.range);
      max_range = std::max(max_range, p.range);
    }

    const double n = static_cast<double>(cluster.size());

    msg.center_x = static_cast<float>(sum_x / n);
    msg.center_y = static_cast<float>(sum_y / n);

    msg.min_x = static_cast<float>(min_x);
    msg.min_y = static_cast<float>(min_y);
    msg.max_x = static_cast<float>(max_x);
    msg.max_y = static_cast<float>(max_y);

    msg.min_range = static_cast<float>(min_range);
    msg.max_range = static_cast<float>(max_range);

    msg.width = static_cast<float>(std::hypot(max_x - min_x, max_y - min_y));
    msg.point_count = static_cast<uint32_t>(cluster.size());
    msg.scan_start_index = static_cast<uint32_t>(cluster.front().index);
    msg.scan_end_index = static_cast<uint32_t>(cluster.back().index);
    msg.scan_start_angle = static_cast<float>(cluster.front().angle);
    msg.scan_end_angle = static_cast<float>(cluster.back().angle);

    msg.track_id = 0;
    msg.motion_label = ObstacleCluster::MOTION_UNKNOWN;
    msg.is_wall_static = false;
    msg.is_density_static = false;
    msg.max_dense_weight = 0.0f;
    msg.observed_count = 0;
    msg.static_hit_count = 0;
    msg.dynamic_hit_count = 0;
    msg.ego_motion_gate_active = false;

    msg.velocity_x = 0.0f;
    msg.velocity_y = 0.0f;
    msg.speed = 0.0f;
    msg.compensated_displacement = 0.0f;

    msg.risk_weight = 1.2f;

    return msg;
  }

  void publishSegmentedPointClouds(
    const std_msgs::msg::Header & header,
    const std::vector<ScanPoint> & points,
    const std::vector<std::vector<ScanPoint>> & valid_clusters,
    const obstacle_context_msgs::msg::ObstacleClusterArray & clusters_msg,
    size_t source_point_count) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    size_t mask_size = source_point_count;
    for (const auto & point : points) {
      mask_size = std::max(mask_size, point.source_index + 1);
    }

    std::vector<bool> dynamic_source_mask(mask_size, false);
    std::vector<bool> static_source_mask(mask_size, false);
    std::vector<DynamicPointMetadata> dynamic_metadata(mask_size);
    const size_t cluster_count = std::min(valid_clusters.size(), clusters_msg.clusters.size());

    for (size_t i = 0; i < cluster_count; ++i) {
      const auto & cluster_msg = clusters_msg.clusters[i];

      DynamicPointMetadata metadata;
      metadata.track_id = cluster_msg.track_id;
      metadata.relative_speed = cluster_msg.speed;
      metadata.relative_yaw = std::atan2(cluster_msg.center_y, cluster_msg.center_x);

      for (const auto & point : valid_clusters[i]) {
        if (point.source_index >= dynamic_source_mask.size()) {
          continue;
        }

        if (isDynamicOutputLabel(cluster_msg.motion_label)) {
          dynamic_source_mask[point.source_index] = true;
          dynamic_metadata[point.source_index] = metadata;
        }

        if (isStaticOutputLabel(cluster_msg.motion_label)) {
          static_source_mask[point.source_index] = true;
        }
      }
    }

    std::vector<DynamicPoint> dynamic_points;
    std::vector<ScanPoint> static_points;
    dynamic_points.reserve(points.size());
    static_points.reserve(points.size());

    for (const auto & point : points) {
      const bool is_dynamic =
        point.source_index < dynamic_source_mask.size() &&
        dynamic_source_mask[point.source_index];
      const bool is_static =
        point.source_index < static_source_mask.size() &&
        static_source_mask[point.source_index];

      if (is_dynamic) {
        DynamicPoint dynamic_point;
        dynamic_point.point = point;
        dynamic_point.metadata = dynamic_metadata[point.source_index];
        dynamic_points.push_back(dynamic_point);
      } else if (is_static) {
        static_points.push_back(point);
      }
    }

    dynamic_pointcloud_pub_->publish(makeDynamicPointCloudMsg(header, dynamic_points));
    static_pointcloud_pub_->publish(makePointCloudMsg(header, static_points));
  }

  sensor_msgs::msg::PointCloud2 makeDynamicPointCloudMsg(
    const std_msgs::msg::Header & header,
    const std::vector<DynamicPoint> & points) const
  {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = header;
    if (pointcloud_use_latest_tf_) {
      cloud.header.stamp.sec = 0;
      cloud.header.stamp.nanosec = 0;
    }
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = true;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
      6,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "track_id", 1, sensor_msgs::msg::PointField::UINT32,
      "relative_speed", 1, sensor_msgs::msg::PointField::FLOAT32,
      "relative_yaw", 1, sensor_msgs::msg::PointField::FLOAT32);
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_track_id(cloud, "track_id");
    sensor_msgs::PointCloud2Iterator<float> iter_relative_speed(cloud, "relative_speed");
    sensor_msgs::PointCloud2Iterator<float> iter_relative_yaw(cloud, "relative_yaw");

    for (const auto & dynamic_point : points) {
      *iter_x = static_cast<float>(dynamic_point.point.x);
      *iter_y = static_cast<float>(dynamic_point.point.y);
      *iter_z = static_cast<float>(dynamic_point.point.z);
      *iter_track_id = dynamic_point.metadata.track_id;
      *iter_relative_speed = dynamic_point.metadata.relative_speed;
      *iter_relative_yaw = dynamic_point.metadata.relative_yaw;
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_track_id;
      ++iter_relative_speed;
      ++iter_relative_yaw;
    }

    return cloud;
  }

  sensor_msgs::msg::PointCloud2 makePointCloudMsg(
    const std_msgs::msg::Header & header,
    const std::vector<ScanPoint> & points) const
  {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = header;
    if (pointcloud_use_latest_tf_) {
      cloud.header.stamp.sec = 0;
      cloud.header.stamp.nanosec = 0;
    }
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = true;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

    for (const auto & point : points) {
      *iter_x = static_cast<float>(point.x);
      *iter_y = static_cast<float>(point.y);
      *iter_z = static_cast<float>(point.z);
      ++iter_x;
      ++iter_y;
      ++iter_z;
    }

    return cloud;
  }

  void updateTracks(
    obstacle_context_msgs::msg::ObstacleClusterArray & clusters_msg,
    const OdomPose & curr_pose)
  {
    const rclcpp::Time now(clusters_msg.header.stamp);

    std::vector<bool> track_matched(tracks_.size(), false);

    for (auto & cluster : clusters_msg.clusters) {
      double best_dist = std::numeric_limits<double>::infinity();
      int best_idx = -1;
      double best_pred_x = 0.0;
      double best_pred_y = 0.0;

      for (size_t i = 0; i < tracks_.size(); ++i) {
        if (track_matched[i]) {
          continue;
        }

        auto pred = transformPrevPointToCurrentFrame(
          tracks_[i].x,
          tracks_[i].y,
          tracks_[i].pose,
          curr_pose);

        const double dist = std::hypot(
          static_cast<double>(cluster.center_x) - pred.first,
          static_cast<double>(cluster.center_y) - pred.second);

        if (dist < best_dist) {
          best_dist = dist;
          best_idx = static_cast<int>(i);
          best_pred_x = pred.first;
          best_pred_y = pred.second;
        }
      }

      if (best_idx >= 0 && best_dist < association_distance_) {
        Track & tr = tracks_[best_idx];

        const double mean_disp = computeMeanCompensatedDisplacement(
          tr,
          static_cast<double>(cluster.center_x),
          static_cast<double>(cluster.center_y),
          curr_pose);

        const double dt = (now - tr.stamp).seconds();

        if (dt > 1e-3) {
          tr.vx = (static_cast<double>(cluster.center_x) - best_pred_x) / dt;
          tr.vy = (static_cast<double>(cluster.center_y) - best_pred_y) / dt;
        } else {
          tr.vx = 0.0;
          tr.vy = 0.0;
        }

        const double history_speed = computeCompensatedSpeedOverHistory(
          tr,
          static_cast<double>(cluster.center_x),
          static_cast<double>(cluster.center_y),
          now,
          curr_pose);
        tr.speed = std::isfinite(history_speed) ? history_speed : std::hypot(tr.vx, tr.vy);
        tr.compensated_displacement = mean_disp;
        if (cluster.is_density_static) {
          tr.static_hit_count = std::max(tr.static_hit_count, static_confirm_count_);
          tr.dynamic_hit_count = 0;
          tr.motion_label = refineWallStaticLabel(
            obstacle_context_msgs::msg::ObstacleCluster::STATIC,
            cluster);
        } else {
          updateMotionEvidence(tr);
          tr.motion_label = classifyMotionWithHysteresis(tr, now);
          tr.motion_label = refineWallStaticLabel(tr.motion_label, cluster);
        }

        cluster.track_id = tr.id;
        cluster.motion_label = tr.motion_label;
        cluster.is_wall_static = tr.motion_label ==
          obstacle_context_msgs::msg::ObstacleCluster::WALL_STATIC;
        cluster.observed_count = static_cast<uint32_t>(tr.observed_count);
        cluster.static_hit_count = static_cast<uint32_t>(tr.static_hit_count);
        cluster.dynamic_hit_count = static_cast<uint32_t>(tr.dynamic_hit_count);
        cluster.ego_motion_gate_active = isEgoMotionUnstable(now);
        cluster.velocity_x = static_cast<float>(tr.vx);
        cluster.velocity_y = static_cast<float>(tr.vy);
        cluster.speed = static_cast<float>(tr.speed);
        cluster.compensated_displacement = static_cast<float>(mean_disp);
        cluster.risk_weight = decideRiskWeight(cluster.motion_label);

        tr.x = cluster.center_x;
        tr.y = cluster.center_y;
        tr.stamp = now;
        tr.pose = curr_pose;
        tr.observed_count++;

        pushTrackHistory(
          tr,
          static_cast<double>(cluster.center_x),
          static_cast<double>(cluster.center_y),
          now,
          curr_pose);

        track_matched[best_idx] = true;
      } else {
        Track tr;
        tr.id = next_track_id_++;

        tr.x = cluster.center_x;
        tr.y = cluster.center_y;
        tr.stamp = now;
        tr.pose = curr_pose;

        tr.observed_count = 1;

        tr.vx = 0.0;
        tr.vy = 0.0;
        tr.speed = 0.0;
        tr.compensated_displacement = 0.0;
        tr.motion_label = cluster.is_density_static ?
          refineWallStaticLabel(obstacle_context_msgs::msg::ObstacleCluster::STATIC, cluster) :
          obstacle_context_msgs::msg::ObstacleCluster::MOTION_UNKNOWN;
        tr.static_hit_count = cluster.is_density_static ? static_confirm_count_ : 0;
        tr.dynamic_hit_count = 0;

        pushTrackHistory(
          tr,
          static_cast<double>(cluster.center_x),
          static_cast<double>(cluster.center_y),
          now,
          curr_pose);

        tracks_.push_back(tr);
        track_matched.push_back(true);

        cluster.track_id = tr.id;
        cluster.motion_label = tr.motion_label;
        cluster.is_wall_static = tr.motion_label ==
          obstacle_context_msgs::msg::ObstacleCluster::WALL_STATIC;
        cluster.observed_count = static_cast<uint32_t>(tr.observed_count);
        cluster.static_hit_count = static_cast<uint32_t>(tr.static_hit_count);
        cluster.dynamic_hit_count = 0;
        cluster.ego_motion_gate_active = isEgoMotionUnstable(now);
        cluster.velocity_x = 0.0f;
        cluster.velocity_y = 0.0f;
        cluster.speed = 0.0f;
        cluster.compensated_displacement = 0.0f;
        cluster.risk_weight = decideRiskWeight(cluster.motion_label);
      }
    }

    removeOldTracks(now);
  }

  void pushScanDensityHistory(
    const std::vector<ScanPoint> & points,
    const rclcpp::Time & stamp,
    const OdomPose & pose,
    double angle_increment)
  {
    if (!use_scan_density_static_filter_) {
      return;
    }

    ScanFrame frame;
    frame.stamp = stamp;
    frame.pose = pose;
    frame.points = points;
    frame.angle_increment = angle_increment;
    scan_density_history_.push_back(frame);

    while (static_cast<int>(scan_density_history_.size()) > scan_density_history_size_) {
      scan_density_history_.pop_front();
    }
  }

  DensityGrid makeScanDensityGrid(const OdomPose & curr_pose) const
  {
    DensityGrid grid;

    if (!use_scan_density_static_filter_) {
      return grid;
    }

    for (const auto & frame : scan_density_history_) {
      for (const auto & point : frame.points) {
        const auto point_in_curr =
          transformPrevPointToCurrentFrame(point.x, point.y, frame.pose, curr_pose);
        grid[densityCellForPoint(point_in_curr.first, point_in_curr.second)] +=
          scanPointDensityWeight(point, frame.angle_increment);
      }
    }

    return grid;
  }

  DensityClusterStats computeDensityClusterStats(
    const std::vector<ScanPoint> & cluster,
    const DensityGrid & grid) const
  {
    DensityClusterStats stats;
    stats.max_weight = 0.0;
    stats.dense_point_count = 0;

    if (!use_scan_density_static_filter_ || grid.empty()) {
      return stats;
    }

    for (const auto & point : cluster) {
      const DensityCell cell = densityCellForPoint(point.x, point.y);
      const double point_density = cellDensityAt(grid, cell);

      stats.max_weight = std::max(stats.max_weight, point_density);

      if (point_density >= scan_density_min_weight_) {
        stats.dense_point_count++;
      }
    }

    return stats;
  }

  bool isDensityStaticCluster(const DensityClusterStats & stats) const
  {
    if (!use_scan_density_static_filter_) {
      return false;
    }

    return stats.dense_point_count >= scan_density_cluster_min_dense_points_;
  }

  double cellDensityAt(const DensityGrid & grid, const DensityCell & cell) const
  {
    double density = 0.0;

    for (int dx = -scan_density_neighbor_cells_; dx <= scan_density_neighbor_cells_; ++dx) {
      for (int dy = -scan_density_neighbor_cells_; dy <= scan_density_neighbor_cells_; ++dy) {
        const auto it = grid.find({cell.first + dx, cell.second + dy});

        if (it != grid.end()) {
          density += it->second;
        }
      }
    }

    return density;
  }

  double scanPointDensityWeight(
    const ScanPoint & point,
    double angle_increment) const
  {
    const double abs_angle_increment = std::fabs(angle_increment);

    if (!std::isfinite(point.range) || point.range <= 0.0 ||
      !std::isfinite(abs_angle_increment) || abs_angle_increment <= 1e-9)
    {
      return 1.0;
    }

    const double beam_spacing_at_point = point.range * abs_angle_increment;
    const double beam_spacing_at_one_meter = abs_angle_increment;

    return std::max(1.0, beam_spacing_at_point / beam_spacing_at_one_meter);
  }

  DensityCell densityCellForPoint(double x, double y) const
  {
    return {
      static_cast<int>(std::floor(x / scan_density_cell_size_)),
      static_cast<int>(std::floor(y / scan_density_cell_size_))};
  }

  void pushTrackHistory(
    Track & tr,
    double x,
    double y,
    const rclcpp::Time & stamp,
    const OdomPose & pose)
  {
    TrackState state;
    state.x = x;
    state.y = y;
    state.stamp = stamp;
    state.pose = pose;

    tr.history.push_back(state);

    while (static_cast<int>(tr.history.size()) > track_history_size_) {
      tr.history.pop_front();
    }
  }

  double computeMeanCompensatedDisplacement(
    const Track & tr,
    double curr_x,
    double curr_y,
    const OdomPose & curr_pose) const
  {
    if (tr.history.empty()) {
      return std::numeric_limits<double>::infinity();
    }

    double sum_dist = 0.0;
    int count = 0;

    for (const auto & h : tr.history) {
      auto prev_in_curr =
        transformPrevPointToCurrentFrame(
          h.x,
          h.y,
          h.pose,
          curr_pose);

      const double dist = std::hypot(
        curr_x - prev_in_curr.first,
        curr_y - prev_in_curr.second);

      sum_dist += dist;
      count++;
    }

    if (count == 0) {
      return std::numeric_limits<double>::infinity();
    }

    return sum_dist / static_cast<double>(count);
  }

  double computeCompensatedSpeedOverHistory(
    const Track & tr,
    double curr_x,
    double curr_y,
    const rclcpp::Time & curr_stamp,
    const OdomPose & curr_pose) const
  {
    if (tr.history.empty()) {
      return std::numeric_limits<double>::infinity();
    }

    const TrackState & oldest = tr.history.front();
    const double dt = (curr_stamp - oldest.stamp).seconds();

    if (dt <= 1e-3) {
      return std::numeric_limits<double>::infinity();
    }

    auto oldest_in_curr =
      transformPrevPointToCurrentFrame(
        oldest.x,
        oldest.y,
        oldest.pose,
        curr_pose);

    const double dist = std::hypot(
      curr_x - oldest_in_curr.first,
      curr_y - oldest_in_curr.second);

    return dist / dt;
  }

  void updateMotionEvidence(Track & tr) const
  {
    if (!std::isfinite(tr.speed)) {
      tr.static_hit_count = 0;
      tr.dynamic_hit_count = 0;
      return;
    }

    if (tr.speed >= dynamic_speed_threshold_) {
      tr.dynamic_hit_count++;
      tr.static_hit_count = 0;
      return;
    }

    if (tr.speed <= static_speed_threshold_) {
      tr.static_hit_count++;
      tr.dynamic_hit_count = 0;
      return;
    }

    tr.static_hit_count = 0;
    tr.dynamic_hit_count = 0;
  }

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (!use_imu_motion_gate_) {
      return;
    }

    const rclcpp::Time stamp(msg->header.stamp);
    const double horizontal_accel =
      std::hypot(msg->linear_acceleration.x, msg->linear_acceleration.y);
    const double yaw_rate = std::fabs(msg->angular_velocity.z);
    latest_imu_horizontal_accel_ = horizontal_accel;
    latest_imu_yaw_rate_ = yaw_rate;
    latest_imu_accel_gate_active_ = horizontal_accel >= imu_accel_gate_threshold_;
    latest_imu_yaw_gate_active_ = yaw_rate >= imu_yaw_rate_gate_threshold_;

    if (latest_imu_accel_gate_active_ || latest_imu_yaw_gate_active_) {
      last_imu_motion_spike_stamp_ = stamp;
      has_imu_motion_spike_ = true;
      imu_motion_gate_active_ = true;
      return;
    }

    if (imu_motion_gate_active_ && has_imu_motion_spike_) {
      const double age = (stamp - last_imu_motion_spike_stamp_).seconds();

      if (age > imu_motion_gate_hold_sec_) {
        imu_motion_gate_active_ = false;
      }
    }
  }

  uint8_t classifyMotionWithHysteresis(
    const Track & tr,
    const rclcpp::Time & now) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    if (tr.observed_count < min_observations_) {
      return ObstacleCluster::MOTION_UNKNOWN;
    }

    if (static_cast<int>(tr.history.size()) < min_observations_) {
      return ObstacleCluster::MOTION_UNKNOWN;
    }

    if (!std::isfinite(tr.speed)) {
      return ObstacleCluster::MOTION_UNKNOWN;
    }

    const bool ego_motion_unstable = isEgoMotionUnstable(now);

    if (isStaticLikeMotionLabel(tr.motion_label)) {
      if (ego_motion_unstable) {
        return ObstacleCluster::STATIC;
      }

      if (tr.dynamic_hit_count >= dynamic_confirm_count_) {
        return ObstacleCluster::DYNAMIC;
      }

      return ObstacleCluster::STATIC;
    }

    if (tr.motion_label == ObstacleCluster::DYNAMIC) {
      if (tr.static_hit_count >= static_confirm_count_) {
        return ObstacleCluster::STATIC;
      }

      return ObstacleCluster::DYNAMIC;
    }

    if (!ego_motion_unstable && tr.dynamic_hit_count >= dynamic_confirm_count_) {
      return ObstacleCluster::DYNAMIC;
    }

    if (tr.static_hit_count >= static_confirm_count_) {
      return ObstacleCluster::STATIC;
    }

    return ObstacleCluster::MOTION_UNKNOWN;
  }

  uint8_t refineWallStaticLabel(
    uint8_t motion_label,
    const obstacle_context_msgs::msg::ObstacleCluster & cluster) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    if (!isStaticLikeMotionLabel(motion_label)) {
      return motion_label;
    }

    if (!use_wall_static_label_) {
      return ObstacleCluster::STATIC;
    }

    if (isWallStaticCandidate(cluster)) {
      return ObstacleCluster::WALL_STATIC;
    }

    return ObstacleCluster::STATIC;
  }

  bool isStaticLikeMotionLabel(uint8_t motion_label) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    return motion_label == ObstacleCluster::STATIC ||
      motion_label == ObstacleCluster::WALL_STATIC;
  }

  bool isWallStaticCandidate(
    const obstacle_context_msgs::msg::ObstacleCluster & cluster) const
  {
    return cluster.width >= static_cast<float>(wall_static_min_width_) &&
      cluster.point_count >= static_cast<uint32_t>(wall_static_min_points_);
  }

  bool isEgoMotionUnstable(const rclcpp::Time & now) const
  {
    (void)now;

    if (!use_imu_motion_gate_) {
      return false;
    }

    return imu_motion_gate_active_;
  }

  float decideRiskWeight(uint8_t motion_label) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    if (motion_label == ObstacleCluster::DYNAMIC) {
      return 1.5f;
    }

    if (motion_label == ObstacleCluster::MOTION_UNKNOWN) {
      return 1.3f;
    }

    return 1.0f;
  }

  void applyObstacleMode(
    obstacle_context_msgs::msg::ObstacleClusterArray & clusters_msg) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    if (obstacle_mode_ == OBSTACLE_MODE_STATIC_AND_DYNAMIC) {
      return;
    }

    for (auto & cluster : clusters_msg.clusters) {
      if (obstacle_mode_ == OBSTACLE_MODE_STATIC_ONLY) {
        if (cluster.motion_label == ObstacleCluster::DYNAMIC) {
          cluster.motion_label = ObstacleCluster::STATIC;
        }
      } else if (obstacle_mode_ == OBSTACLE_MODE_DYNAMIC_ONLY) {
        if (cluster.motion_label == ObstacleCluster::STATIC) {
          cluster.motion_label = ObstacleCluster::MOTION_UNKNOWN;
        }
      }

      cluster.is_wall_static = cluster.motion_label == ObstacleCluster::WALL_STATIC;
      cluster.risk_weight = decideRiskWeight(cluster.motion_label);
    }
  }

  bool isDynamicOutputLabel(uint8_t motion_label) const
  {
    return motion_label == obstacle_context_msgs::msg::ObstacleCluster::DYNAMIC;
  }

  bool isStaticOutputLabel(uint8_t motion_label) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    if (obstacle_mode_ == OBSTACLE_MODE_DYNAMIC_ONLY) {
      return motion_label == ObstacleCluster::WALL_STATIC;
    }

    if (obstacle_mode_ == OBSTACLE_MODE_STATIC_ONLY) {
      return motion_label != ObstacleCluster::DYNAMIC;
    }

    return motion_label != ObstacleCluster::DYNAMIC;
  }

  void setAllClustersUnknown(
    obstacle_context_msgs::msg::ObstacleClusterArray & clusters_msg) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    for (auto & cluster : clusters_msg.clusters) {
      cluster.track_id = 0;
      cluster.motion_label = ObstacleCluster::MOTION_UNKNOWN;
      cluster.is_wall_static = false;
      cluster.is_density_static = false;
      cluster.max_dense_weight = 0.0f;
      cluster.observed_count = 0;
      cluster.static_hit_count = 0;
      cluster.dynamic_hit_count = 0;
      cluster.ego_motion_gate_active = false;
      cluster.velocity_x = 0.0f;
      cluster.velocity_y = 0.0f;
      cluster.speed = 0.0f;
      cluster.compensated_displacement = 0.0f;
      cluster.risk_weight = 1.3f;
    }
  }

  void removeOldTracks(const rclcpp::Time & now)
  {
    std::vector<Track> alive_tracks;
    alive_tracks.reserve(tracks_.size());

    for (const auto & tr : tracks_) {
      const double age = (now - tr.stamp).seconds();

      if (age < track_timeout_sec_) {
        alive_tracks.push_back(tr);
      }
    }

    tracks_ = alive_tracks;
  }

  std::pair<double, double> transformPrevPointToCurrentFrame(
    double px,
    double py,
    const OdomPose & prev_pose,
    const OdomPose & curr_pose) const
  {
    const double cos_prev = std::cos(prev_pose.yaw);
    const double sin_prev = std::sin(prev_pose.yaw);

    const double wx = prev_pose.x + cos_prev * px - sin_prev * py;
    const double wy = prev_pose.y + sin_prev * px + cos_prev * py;

    const double dx = wx - curr_pose.x;
    const double dy = wy - curr_pose.y;

    const double cos_curr = std::cos(curr_pose.yaw);
    const double sin_curr = std::sin(curr_pose.yaw);

    const double cx = cos_curr * dx + sin_curr * dy;
    const double cy = -sin_curr * dx + cos_curr * dy;

    return {cx, cy};
  }

  bool getInterpolatedOdom(
    const rclcpp::Time & target_time,
    OdomPose & out_pose) const
  {
    if (odom_history_.empty()) {
      return false;
    }

    if (odom_history_.size() == 1) {
      const double age = std::fabs((target_time - odom_history_.front().stamp).seconds());

      if (age <= max_odom_age_sec_) {
        out_pose = odom_history_.front();
        return true;
      }

      return false;
    }

    if (target_time < odom_history_.front().stamp) {
      return false;
    }

    if (target_time > odom_history_.back().stamp) {
      const double age = (target_time - odom_history_.back().stamp).seconds();

      if (age <= max_odom_age_sec_) {
        out_pose = odom_history_.back();
        out_pose.stamp = target_time;
        return true;
      }

      return false;
    }

    for (size_t i = 1; i < odom_history_.size(); ++i) {
      const auto & p0 = odom_history_[i - 1];
      const auto & p1 = odom_history_[i];

      if (p0.stamp <= target_time && target_time <= p1.stamp) {
        const double total_dt = (p1.stamp - p0.stamp).seconds();

        if (total_dt < 1e-6) {
          out_pose = p1;
          out_pose.stamp = target_time;
          return true;
        }

        const double ratio = (target_time - p0.stamp).seconds() / total_dt;

        out_pose.stamp = target_time;
        out_pose.x = p0.x + ratio * (p1.x - p0.x);
        out_pose.y = p0.y + ratio * (p1.y - p0.y);
        out_pose.yaw = interpolateYaw(p0.yaw, p1.yaw, ratio);

        return true;
      }
    }

    return false;
  }

  visualization_msgs::msg::MarkerArray makeMarkerArray(
    const obstacle_context_msgs::msg::ObstacleClusterArray & clusters_msg)
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    visualization_msgs::msg::MarkerArray marker_array;
    const rclcpp::Time marker_time(clusters_msg.header.stamp);
    const bool ego_motion_unstable = isEgoMotionUnstable(marker_time);

    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header = clusters_msg.header;
    delete_marker.header.stamp.sec = 0;
    delete_marker.header.stamp.nanosec = 0;
    delete_marker.ns = "obstacle_clusters";
    delete_marker.id = 0;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    int id = 1;

    {
      visualization_msgs::msg::Marker gate_marker;
      gate_marker.header = clusters_msg.header;
      gate_marker.header.stamp.sec = 0;
      gate_marker.header.stamp.nanosec = 0;
      gate_marker.ns = "obstacle_cluster_status";
      gate_marker.id = id++;
      gate_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      gate_marker.action = visualization_msgs::msg::Marker::ADD;

      gate_marker.pose.position.x = 2.0;
      gate_marker.pose.position.y = -1.8;
      gate_marker.pose.position.z = 1.2;
      gate_marker.pose.orientation.x = 0.0;
      gate_marker.pose.orientation.y = 0.0;
      gate_marker.pose.orientation.z = 0.0;
      gate_marker.pose.orientation.w = 1.0;

      gate_marker.scale.z = 0.25;
      gate_marker.color.r = 1.0f;
      gate_marker.color.g = ego_motion_unstable ? 0.85f : 1.0f;
      gate_marker.color.b = ego_motion_unstable ? 0.0f : 1.0f;
      gate_marker.color.a = 1.0f;

      std::ostringstream gate_text;
      gate_text << "ODOM TIME MODE\n"
                << "obstacle mode: " << obstacle_mode_
                << " " << obstacleModeText() << "\n"
                << "ego_v: " << std::fixed << std::setprecision(2)
                << current_ego_speed_ << " m/s\n"
                << "thr S/D: " << static_speed_threshold_ << " / "
                << dynamic_speed_threshold_ << " m/s";

      if (use_imu_motion_gate_) {
        gate_text << "\n"
                  << "IMU GATE: " << (ego_motion_unstable ? "ON" : "off") << "\n"
                  << "accel_xy: " << std::fixed << std::setprecision(2)
                  << latest_imu_horizontal_accel_ << " / "
                  << imu_accel_gate_threshold_ << " m/s^2"
                  << (latest_imu_accel_gate_active_ ? " *" : "") << "\n"
                  << "yaw_rate: " << latest_imu_yaw_rate_ << " / "
                  << imu_yaw_rate_gate_threshold_ << " rad/s"
                  << (latest_imu_yaw_gate_active_ ? " *" : "");
      }

      gate_marker.text = gate_text.str();
      gate_marker.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);

      marker_array.markers.push_back(gate_marker);
    }

    for (const auto & cluster : clusters_msg.clusters) {
      visualization_msgs::msg::Marker marker;

      marker.header = clusters_msg.header;
      marker.header.stamp.sec = 0;
      marker.header.stamp.nanosec = 0;
      marker.ns = "obstacle_clusters";
      marker.id = id++;

      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.pose.position.x = cluster.center_x;
      marker.pose.position.y = cluster.center_y;
      marker.pose.position.z = 0.15;

      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = marker_scale_;
      marker.scale.y = marker_scale_;
      marker.scale.z = marker_scale_;

      if (cluster.motion_label == ObstacleCluster::WALL_STATIC) {
        // wall static: cyan
        marker.color.r = 0.0f;
        marker.color.g = 0.8f;
        marker.color.b = 1.0f;
        marker.color.a = 1.0f;
      } else if (cluster.motion_label == ObstacleCluster::STATIC) {
        // static: blue
        marker.color.r = 0.1f;
        marker.color.g = 0.3f;
        marker.color.b = 1.0f;
        marker.color.a = 1.0f;
      } else if (cluster.motion_label == ObstacleCluster::DYNAMIC) {
        // dynamic: red
        marker.color.r = 1.0f;
        marker.color.g = 0.1f;
        marker.color.b = 0.1f;
        marker.color.a = 1.0f;
      } else {
        // unknown: green
        marker.color.r = 0.1f;
        marker.color.g = 1.0f;
        marker.color.b = 0.1f;
        marker.color.a = 1.0f;
      }

      marker.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);

      marker_array.markers.push_back(marker);

      visualization_msgs::msg::Marker text_marker;
      text_marker.header = marker.header;
      text_marker.ns = "obstacle_cluster_labels";
      text_marker.id = id++;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;

      const double label_offset = std::max(0.20, marker_scale_ * 1.5);
      text_marker.pose.position.x = cluster.center_x;
      text_marker.pose.position.y = cluster.center_y + label_offset;
      text_marker.pose.position.z = 0.35;

      text_marker.pose.orientation.x = 0.0;
      text_marker.pose.orientation.y = 0.0;
      text_marker.pose.orientation.z = 0.0;
      text_marker.pose.orientation.w = 1.0;

      text_marker.scale.z = marker_text_scale_;

      text_marker.color.r = 1.0f;
      text_marker.color.g = 1.0f;
      text_marker.color.b = 1.0f;
      text_marker.color.a = 1.0f;

      std::ostringstream label;
      label << "id: " << cluster.track_id << "\n"
            << "v: " << std::fixed << std::setprecision(2)
            << cluster.speed << " m/s\n"
            << "obs: " << cluster.observed_count << "\n"
            << "label: " << motionLabelText(cluster.motion_label) << "\n"
            << "S/D: " << cluster.static_hit_count << "/"
            << cluster.dynamic_hit_count << "\n"
            << "max_w: " << std::fixed << std::setprecision(1)
            << cluster.max_dense_weight << "/"
            << scan_density_min_weight_;

      if (cluster.is_density_static) {
        label << " dense";
      }

      if (cluster.ego_motion_gate_active) {
        label << "\ngate";
      }

      text_marker.text = label.str();

      text_marker.lifetime = marker.lifetime;

      marker_array.markers.push_back(text_marker);
    }

    return marker_array;
  }

  double yawFromQuaternion(
    const geometry_msgs::msg::Quaternion & q_msg) const
  {
    tf2::Quaternion q(
      q_msg.x,
      q_msg.y,
      q_msg.z,
      q_msg.w);

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;

    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    return yaw;
  }

  std::string motionLabelText(uint8_t motion_label) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    if (motion_label == ObstacleCluster::STATIC) {
      return "STATIC";
    }

    if (motion_label == ObstacleCluster::DYNAMIC) {
      return "DYNAMIC";
    }

    if (motion_label == ObstacleCluster::WALL_STATIC) {
      return "WALL";
    }

    return "UNKNOWN";
  }

  std::string obstacleModeText() const
  {
    if (obstacle_mode_ == OBSTACLE_MODE_STATIC_ONLY) {
      return "STATIC_ONLY";
    }

    if (obstacle_mode_ == OBSTACLE_MODE_DYNAMIC_ONLY) {
      return "DYNAMIC_ONLY";
    }

    return "STATIC_DYNAMIC";
  }

  double normalizeAngle(double a) const
  {
    while (a > M_PI) {
      a -= 2.0 * M_PI;
    }

    while (a < -M_PI) {
      a += 2.0 * M_PI;
    }

    return a;
  }

  double interpolateYaw(double yaw0, double yaw1, double ratio) const
  {
    const double diff = normalizeAngle(yaw1 - yaw0);
    return normalizeAngle(yaw0 + ratio * diff);
  }

  double deg2rad(double deg) const
  {
    return deg * M_PI / 180.0;
  }

private:
  bool use_sim_time_ = false;

  std::string input_type_;
  std::string scan_topic_;
  std::string pointcloud_topic_;
  std::string odom_topic_;
  std::string imu_topic_;

  std::string processed_scan_topic_;
  std::string output_topic_;
  std::string marker_topic_;
  std::string dynamic_pointcloud_topic_;
  std::string static_pointcloud_topic_;
  std::string obstacle_mode_topic_;
  bool pointcloud_use_latest_tf_;
  int obstacle_mode_ = OBSTACLE_MODE_STATIC_AND_DYNAMIC;

  double min_valid_range_;
  double max_valid_range_;

  bool use_roi_filter_;
  double roi_angle_min_rad_;
  double roi_angle_max_rad_;

  double cluster_range_jump_;
  int min_cluster_points_;
  double min_cluster_width_;
  double max_cluster_width_;

  double association_distance_;
  double track_timeout_sec_;
  int min_observations_;
  int track_history_size_;

  double static_speed_threshold_;
  double dynamic_speed_threshold_;
  int static_confirm_count_;
  int dynamic_confirm_count_;

  bool use_wall_static_label_;
  double wall_static_min_width_;
  int wall_static_min_points_;

  bool use_scan_density_static_filter_;
  int scan_density_history_size_;
  double scan_density_cell_size_;
  double scan_density_min_weight_;
  int scan_density_neighbor_cells_;
  int scan_density_cluster_min_dense_points_;

  bool use_imu_motion_gate_;
  double imu_accel_gate_threshold_;
  double imu_yaw_rate_gate_threshold_;
  double imu_motion_gate_hold_sec_;

  double max_odom_age_sec_;
  double max_pose_jump_dist_;
  double max_pose_jump_yaw_rad_;
  double odom_history_sec_;

  double marker_scale_;
  double marker_text_scale_;
  double marker_lifetime_sec_;

  std::deque<OdomPose> odom_history_;
  std::deque<ScanFrame> scan_density_history_;
  double current_ego_speed_ = 0.0;

  bool has_imu_motion_spike_ = false;
  bool imu_motion_gate_active_ = false;
  bool latest_imu_accel_gate_active_ = false;
  bool latest_imu_yaw_gate_active_ = false;
  double latest_imu_horizontal_accel_ = 0.0;
  double latest_imu_yaw_rate_ = 0.0;
  rclcpp::Time last_imu_motion_spike_stamp_{0, 0, RCL_ROS_TIME};

  std::vector<Track> tracks_;
  uint32_t next_track_id_ = 1;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr obstacle_mode_sub_;

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr processed_scan_pub_;
  rclcpp::Publisher<obstacle_context_msgs::msg::ObstacleClusterArray>::SharedPtr cluster_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pointcloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr static_pointcloud_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleClusterContextNode>());
  rclcpp::shutdown();
  return 0;
}
