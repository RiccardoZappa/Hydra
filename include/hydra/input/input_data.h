#pragma once

#include <Eigen/Geometry>
#include <limits>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <utility>
#include <vector>

#include "hydra/common/common_types.h"
#include "hydra/input/sensor.h"

namespace hydra {

/**
 * @class MaskData
 * @brief Struct to hold instance information, in this case the binary mask and the
 * class id
 *
 */
struct MaskData {
  using Ptr = std::shared_ptr<MaskData>;
  
  uint64 map_view_id;
  uint64 mask_id;
  uint64 class_id;
  cv::Mat mask;

  virtual ~MaskData() = default;
};

struct InputData {
  using Ptr = std::shared_ptr<InputData>;

  // Types of the stored image data.
  using ColorType = cv::Vec3b;
  using RangeType = float;
  using VertexType = cv::Vec3f;
  using LabelType = int;

  explicit InputData(Sensor::ConstPtr sensor) : sensor_(std::move(sensor)) {}
  virtual ~InputData() = default;

  // Time stamp this input data was captured.
  TimeStamp timestamp_ns;

  // Pose of the robot body in the world frame.
  Eigen::Isometry3d world_T_body;

  // Color image as RGB.
  cv::Mat color_image;

  // Depth image as planar depth in metres.
  cv::Mat depth_image;

  // Ray lengths in meters.
  cv::Mat range_image;

  // Label image for semantic input data.
  cv::Mat label_image;

  cv::Mat panoptic_image;

  // Instance masks for each instance in the label image
  uint64 map_view_id;
  std::vector<MaskData> instance_masks;

  // 3D points of the range image in sensor or world frame.
  cv::Mat vertex_map;
  bool points_in_world_frame = false;

  // Min and max range observed in the range image.
  float min_range = 0.0f;
  float max_range = std::numeric_limits<float>::infinity();

  /**
   * @brief Get the sensor that captured this data.
   */
  const Sensor& getSensor() const { return *sensor_; }

  /**
   * @brief Get the pose of the sensor in world frame when this data was captured.
   */
  Eigen::Isometry3d getSensorPose() const {
    return world_T_body * sensor_->body_T_sensor();
  }

 private:
  Sensor::ConstPtr sensor_;
};

};  // namespace hydra
