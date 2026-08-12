/*
 * slam_toolbox
 * Copyright Work Modifications (c) 2019, Steve Macenski
 *
 * THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
 * COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
 * COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS
 * AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
 *
 * BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
 * BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS
 * CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
 * CONDITIONS.
 *
 */

/* Author: Steven Macenski */

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "slam_toolbox/slam_toolbox_localization.hpp"

namespace slam_toolbox
{

/*****************************************************************************/
LocalizationSlamToolbox::LocalizationSlamToolbox(rclcpp::NodeOptions options)
: SlamToolbox(options)
/*****************************************************************************/
{
  processor_type_ = PROCESS_LOCALIZATION;
  localization_pose_sub_ =
    this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "initialpose", 1,
    std::bind(&LocalizationSlamToolbox::localizePoseCallback,
    this, std::placeholders::_1));

  // Source-agnostic absolute pose correction input. Any node that produces an
  // estimate of base_link in the map frame with covariance (e.g. an AprilTag
  // relocalizer) can publish here to correct SLAM's internal pose graph.
  have_pose_correction_ = false;
  pose_correction_timeout_ = 0.5;
  pose_correction_timeout_ = this->declare_parameter("pose_correction_timeout",
      pose_correction_timeout_);
  pose_correction_verbose_ = false;
  pose_correction_verbose_ = this->declare_parameter("pose_correction_verbose",
      pose_correction_verbose_);
  pose_correction_sub_ =
    this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/pose_correction", 1,
    std::bind(&LocalizationSlamToolbox::poseCorrectionCallback,
    this, std::placeholders::_1));

  clear_localization_ = this->create_service<std_srvs::srv::Empty>(
    "slam_toolbox/clear_localization_buffer",
    std::bind(&LocalizationSlamToolbox::clearLocalizationBuffer, this,
    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  // in localization mode, we cannot allow for interactive mode
  enable_interactive_mode_ = false;

  // in localization mode, disable map saver
  map_saver_.reset();
}

/*****************************************************************************/
void LocalizationSlamToolbox::loadPoseGraphByParams()
/*****************************************************************************/
{
  std::string filename;
  geometry_msgs::msg::Pose2D pose;
  bool dock = false;
  if (shouldStartWithPoseGraph(filename, pose, dock)) {
    std::shared_ptr<slam_toolbox::srv::DeserializePoseGraph::Request> req =
      std::make_shared<slam_toolbox::srv::DeserializePoseGraph::Request>();
    std::shared_ptr<slam_toolbox::srv::DeserializePoseGraph::Response> resp =
      std::make_shared<slam_toolbox::srv::DeserializePoseGraph::Response>();
    req->initial_pose = pose;
    req->filename = filename;
    req->match_type =
      slam_toolbox::srv::DeserializePoseGraph::Request::LOCALIZE_AT_POSE;
    if (dock) {
      RCLCPP_WARN(get_logger(),
        "LocalizationSlamToolbox: Starting localization "
        "at first node (dock) is correctly not supported.");
    }

    deserializePoseGraphCallback(nullptr, req, resp);
  }
}

/*****************************************************************************/
bool LocalizationSlamToolbox::clearLocalizationBuffer(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<std_srvs::srv::Empty::Request> req,
  std::shared_ptr<std_srvs::srv::Empty::Response> resp)
/*****************************************************************************/
{
  boost::mutex::scoped_lock lock(smapper_mutex_);
  RCLCPP_INFO(get_logger(),
    "LocalizationSlamToolbox: Clearing localization buffer.");
  smapper_->clearLocalizationBuffer();
  return true;
}

/*****************************************************************************/
bool LocalizationSlamToolbox::serializePoseGraphCallback(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<slam_toolbox::srv::SerializePoseGraph::Request> req,
  std::shared_ptr<slam_toolbox::srv::SerializePoseGraph::Response> resp)
/*****************************************************************************/
{
  RCLCPP_ERROR(get_logger(), "LocalizationSlamToolbox: Cannot call serialize map "
    "in localization mode!");
  return false;
}

/*****************************************************************************/
bool LocalizationSlamToolbox::deserializePoseGraphCallback(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<slam_toolbox::srv::DeserializePoseGraph::Request> req,
  std::shared_ptr<slam_toolbox::srv::DeserializePoseGraph::Response> resp)
/*****************************************************************************/
{
  if (req->match_type != procType::LOCALIZE_AT_POSE) {
    RCLCPP_ERROR(get_logger(), "Requested a non-localization deserialization "
      "in localization mode.");
    return false;
  }
  return SlamToolbox::deserializePoseGraphCallback(request_header, req, resp);
}

/*****************************************************************************/
void LocalizationSlamToolbox::laserCallback(
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
/*****************************************************************************/
{
  // store scan header
  scan_header = scan->header;
  // no odom info
  Pose2 pose;
  if (!pose_helper_->getOdomPose(pose, scan->header.stamp)) {
    RCLCPP_WARN(get_logger(), "Failed to compute odom pose");
    return;
  }

  // ensure the laser can be used
  LaserRangeFinder * laser = getLaser(scan);

  if (!laser) {
    RCLCPP_WARN(get_logger(), "SynchronousSlamToolbox: Failed to create laser"
      " device for %s; discarding scan", scan->header.frame_id.c_str());
    return;
  }

  if (shouldProcessScan(scan, pose)) {
    addScan(laser, scan, pose);
  }
}

/*****************************************************************************/
LocalizedRangeScan * LocalizationSlamToolbox::addScan(
  LaserRangeFinder * laser,
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
  Pose2 & odom_pose)
/*****************************************************************************/
{
  boost::mutex::scoped_lock l(pose_mutex_);

  if (processor_type_ == PROCESS_LOCALIZATION && process_near_pose_) {
    processor_type_ = PROCESS_NEAR_REGION;
  }

  LocalizedRangeScan * range_scan = getLocalizedRangeScan(
    laser, scan, odom_pose);

  // Add the localized range scan to the smapper
  boost::mutex::scoped_lock lock(smapper_mutex_);
  bool processed = false, update_reprocessing_transform = false;

  Matrix3 covariance;
  covariance.SetToIdentity();

  if (processor_type_ == PROCESS_NEAR_REGION) {
    if (!process_near_pose_) {
      RCLCPP_ERROR(get_logger(),
        "Process near region called without a "
        "valid region request. Ignoring scan.");
      return nullptr;
    }

    // set our position to the requested pose and process
    range_scan->SetOdometricPose(*process_near_pose_);
    range_scan->SetCorrectedPose(range_scan->GetOdometricPose());
    process_near_pose_.reset(nullptr);
    processed = smapper_->getMapper()->ProcessAgainstNodesNearBy(range_scan, true, &covariance);

    // reset to localization mode
    update_reprocessing_transform = true;
    processor_type_ = PROCESS_LOCALIZATION;
  } else if (processor_type_ == PROCESS_LOCALIZATION) {
    processed = smapper_->getMapper()->ProcessLocalization(range_scan, &covariance);
    update_reprocessing_transform = false;
  } else {
    RCLCPP_FATAL(get_logger(), "LocalizationSlamToolbox: "
      "No valid processor type set! Exiting.");
    exit(-1);
  }

  // if successfully processed, create odom to map transformation
  if (!processed) {
    delete range_scan;
    range_scan = nullptr;
  } else {
    // fuse any pending absolute pose correction as a unary prior on this node
    // and re-optimize so the corrected pose reflects it before we publish
    applyPendingPoseCorrection(range_scan);

    // compute our new transform
    setTransformFromPoses(range_scan->GetCorrectedPose(), odom_pose,
      scan->header.stamp, update_reprocessing_transform);

    publishPose(range_scan->GetCorrectedPose(), covariance, scan->header.stamp);
  }

  return range_scan;
}

/*****************************************************************************/
void LocalizationSlamToolbox::poseCorrectionCallback(
  const
  geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
/*****************************************************************************/
{
  // Buffer the most recent correction; it will be applied as a unary prior on
  // the next scan node that gets processed into the graph.
  boost::mutex::scoped_lock lock(pose_correction_mutex_);
  const bool overwrote_pending = have_pose_correction_;
  pending_pose_correction_ = *msg;
  have_pose_correction_ = true;

  if (pose_correction_verbose_) {
    const rclcpp::Time stamp(msg->header.stamp);
    RCLCPP_INFO(get_logger(),
      "PoseCorrection: Received correction on /pose_correction in frame '%s' "
      "stamped %.3fs at (%.3f, %.3f, %.3f).%s",
      msg->header.frame_id.c_str(), stamp.seconds(),
      msg->pose.pose.position.x, msg->pose.pose.position.y,
      tf2::getYaw(msg->pose.pose.orientation),
      overwrote_pending ?
      " Overwrote a previous unprocessed correction." : "");
  }
}

/*****************************************************************************/
void LocalizationSlamToolbox::applyPendingPoseCorrection(
  LocalizedRangeScan * range_scan)
/*****************************************************************************/
{
  if (!range_scan || !solver_) {
    if (pose_correction_verbose_) {
      RCLCPP_INFO(get_logger(),
        "PoseCorrection: Skipping correction; %s is not available.",
        !range_scan ? "no scan node" : "no graph solver");
    }
    return;
  }

  geometry_msgs::msg::PoseWithCovarianceStamped correction;
  {
    boost::mutex::scoped_lock lock(pose_correction_mutex_);
    if (!have_pose_correction_) {
      if (pose_correction_verbose_) {
        RCLCPP_INFO(get_logger(),
          "PoseCorrection: No pending correction to apply to node %i.",
          range_scan->GetUniqueId());
      }
      return;
    }
    correction = pending_pose_correction_;
    have_pose_correction_ = false;
  }

  if (pose_correction_verbose_) {
    RCLCPP_INFO(get_logger(),
      "PoseCorrection: Processing pending correction for scan node %i.",
      range_scan->GetUniqueId());
  }

  // reject stale corrections relative to the scan being processed
  const rclcpp::Time correction_stamp(correction.header.stamp);
  const rclcpp::Time scan_stamp(scan_header.stamp);
  if (pose_correction_timeout_ > 0.0 &&
    correction_stamp.nanoseconds() > 0 && scan_stamp.nanoseconds() > 0)
  {
    const double dt = std::fabs((scan_stamp - correction_stamp).seconds());
    if (dt > pose_correction_timeout_) {
      RCLCPP_WARN(get_logger(),
        "PoseCorrection: Discarding correction stale by %.3fs (timeout %.3fs).",
        dt, pose_correction_timeout_);
      return;
    }
    if (pose_correction_verbose_) {
      RCLCPP_INFO(get_logger(),
        "PoseCorrection: Correction age %.3fs within timeout %.3fs; accepting.",
        dt, pose_correction_timeout_);
    }
  }

  // absolute measured pose of base_link in the map frame
  Eigen::Vector3d measured_pose(
    correction.pose.pose.position.x,
    correction.pose.pose.position.y,
    tf2::getYaw(correction.pose.pose.orientation));

  // extract the (x, y, yaw) sub-covariance from the 6x6 pose covariance
  const std::array<double, 36> & c = correction.pose.covariance;
  Eigen::Matrix3d covariance;
  covariance << c[0], c[1], c[5],
    c[6], c[7], c[11],
    c[30], c[31], c[35];

  // guard against a degenerate/uninitialized covariance
  if (!covariance.allFinite() || std::fabs(covariance.determinant()) < 1e-12) {
    RCLCPP_WARN(get_logger(),
      "PoseCorrection: Ignoring correction with singular covariance.");
    return;
  }

  if (pose_correction_verbose_) {
    RCLCPP_INFO(get_logger(),
      "PoseCorrection: Measured pose (%.3f, %.3f, %.3f) with (x, y, yaw) "
      "covariance diagonal (%.4e, %.4e, %.4e), determinant %.4e.",
      measured_pose(0), measured_pose(1), measured_pose(2),
      covariance(0, 0), covariance(1, 1), covariance(2, 2),
      covariance.determinant());
  }

  const int node_id = range_scan->GetUniqueId();

  // snapshot SLAM's current estimate for this node so we can report how far the
  // correction moves it once the prior is fused and the graph is re-optimized
  const Pose2 pose_before = range_scan->GetCorrectedPose();

  solver_->AddPrior(node_id, measured_pose, covariance);

  if (pose_correction_verbose_) {
    RCLCPP_INFO(get_logger(),
      "PoseCorrection: Added unary prior to node %i; re-optimizing pose graph.",
      node_id);
  }

  // re-optimize so the prior propagates into the graph and the scans'
  // corrected poses (and thus map->odom) reflect the correction
  smapper_->getMapper()->CorrectPoses();

  if (pose_correction_verbose_) {
    const Pose2 pose_after = range_scan->GetCorrectedPose();
    const double dx = pose_after.GetX() - pose_before.GetX();
    const double dy = pose_after.GetY() - pose_before.GetY();
    double dyaw = pose_after.GetHeading() - pose_before.GetHeading();
    dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));
    RCLCPP_INFO(get_logger(),
      "PoseCorrection: Node %i estimate shifted by (%.3f, %.3f, %.3f) "
      "[translation %.3fm, rotation %.3frad]: (%.3f, %.3f, %.3f) -> "
      "(%.3f, %.3f, %.3f).",
      node_id, dx, dy, dyaw, std::hypot(dx, dy), std::fabs(dyaw),
      pose_before.GetX(), pose_before.GetY(), pose_before.GetHeading(),
      pose_after.GetX(), pose_after.GetY(), pose_after.GetHeading());
  }

  RCLCPP_INFO(get_logger(),
    "PoseCorrection: Applied absolute prior to node %i at (%.2f, %.2f, %.2f).",
    node_id, measured_pose(0), measured_pose(1), measured_pose(2));
}

/*****************************************************************************/
void LocalizationSlamToolbox::localizePoseCallback(
  const
  geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
/*****************************************************************************/
{
  if (processor_type_ != PROCESS_LOCALIZATION) {
    RCLCPP_ERROR(get_logger(),
      "LocalizePoseCallback: Cannot process localization command "
      "if not in localization mode.");
    return;
  }

  boost::mutex::scoped_lock l(pose_mutex_);
  if (process_near_pose_) {
    process_near_pose_.reset(new Pose2(msg->pose.pose.position.x,
      msg->pose.pose.position.y, tf2::getYaw(msg->pose.pose.orientation)));
  } else {
    process_near_pose_ = std::make_unique<Pose2>(msg->pose.pose.position.x,
        msg->pose.pose.position.y, tf2::getYaw(msg->pose.pose.orientation));
  }

  first_measurement_ = true;

  boost::mutex::scoped_lock lock(smapper_mutex_);
  smapper_->clearLocalizationBuffer();

  RCLCPP_INFO(get_logger(),
    "LocalizePoseCallback: Localizing to: (%0.2f %0.2f), theta=%0.2f",
    msg->pose.pose.position.x, msg->pose.pose.position.y,
    tf2::getYaw(msg->pose.pose.orientation));
}

}  // namespace slam_toolbox

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(slam_toolbox::LocalizationSlamToolbox)
