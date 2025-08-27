/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <optional>

#include <memory>

#include "hydra/common/dsg_types.h"
#include "hydra/common/output_sink.h"
#include "hydra/input/input_data.h"
#include "hydra/reconstruction/reconstruction_output.h"
#include "hydra/reconstruction/mesh_integrator_config.h"

namespace kimera_pgmo {
class MeshDelta;
}  // namespace kimera_pgmo

namespace hydra {

struct Cluster {
  Eigen::Vector3d centroid;
  std::vector<size_t> indices;
  pcl::PointCloud<pcl::PointXYZRGBA> mesh;
  MaskData mask;
};

class MeshIntegrator;

using LabelIndices = std::map<uint32_t, std::vector<size_t>>;

class MeshSegmenter {
 public:
  using Clusters = std::vector<Cluster>;
  using LabelClusters = std::map<uint32_t, Clusters>;
  using Sink = OutputSink<uint64_t,
                          const kimera_pgmo::MeshDelta&,
                          const std::vector<size_t>&,
                          const LabelIndices&>;

  struct ObjectDetectionParams {
    size_t min_cluster_size = 10; // default value
    float cloud_downsampling = 0.1f;
    float matching_threshold = 0.01f;
  };

  struct Config {
    char prefix = 'O';
    LayerId layer_id = DsgLayers::OBJECTS;
    double active_index_horizon_m = 7.0;
    double cluster_tolerance = 0.25;
    std::map<std::string, ObjectDetectionParams> per_object_params;
    ObjectDetectionParams default_object_params;
    size_t max_cluster_size = 100000;
    float angle_step = 10.0f;
    BoundingBox::Type bounding_box_type = BoundingBox::Type::AABB;
    std::set<uint32_t> labels;
    std::string timer_namespace = "frontend/objects";
    std::vector<Sink::Factory> sinks;
    float min_mesh_z = 0.0;
    float processing_grid_size = 0.1f;
    bool skip_clustering = false;
    bool use_kdtree_distance_check = true;
    float nodes_match_iou_threshold = 0.5;
    bool merge_active_nodes = false;
    float close_to_cloud_threshold = 0.025;
    MeshIntegratorConfig mesh_integrator_config;
  } const config;

  explicit MeshSegmenter(const Config& config);
  ~MeshSegmenter();

  LabelClusters detect(const ReconstructionOutput& input,
                       uint64_t timestamp_ns,
                       const kimera_pgmo::MeshDelta& active,
                       const std::optional<Eigen::Vector3d>& pos);

  void updateGraph(uint64_t timestamp,
                   const LabelClusters& clusters,
                   size_t num_archived_vertices,
                   DynamicSceneGraph& graph,
                   const ReconstructionOutput& input);

  std::unordered_set<NodeId> getActiveNodes() const;

 private:
  void archiveOldNodes(const DynamicSceneGraph& graph, size_t num_archived_vertices);

  void addNodeToGraph(DynamicSceneGraph& graph,
                      const Cluster& cluster,
                      uint32_t label,
                      uint64_t timestamp,
                      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr high_res_cloud);

  void updateNodeInGraph(DynamicSceneGraph& graph,
                         const Cluster& cluster,
                         const SceneGraphNode& node,
                         uint64_t timestamp,
                         pcl::PointCloud<pcl::PointXYZRGBA>::Ptr high_res_cloud);

  void mergeActiveNodes(DynamicSceneGraph& graph, uint32_t label);

 private:
  NodeSymbol next_node_id_;
  std::map<uint32_t, std::set<NodeId>> active_nodes_;
  Sink::List sinks_;
  std::unordered_map<NodeId, std::unique_ptr<VolumetricMap>> object_tsdf_map_;
  std::unique_ptr<hydra::MeshIntegrator> mesh_integrator_;
};

using Clusters = MeshSegmenter::Clusters;
using CloudPoint = pcl::PointXYZRGBA;
using KdTreeT = pcl::search::KdTree<CloudPoint>;
using MeshCloud = pcl::PointCloud<CloudPoint>;
using InstanceData = std::pair<MaskData, MeshCloud::Ptr>;
using ClassToInstance = std::unordered_map<int64, std::vector<InstanceData>>;

void declare_config(MeshSegmenter::Config& config);
void declare_config(MeshSegmenter::ObjectDetectionParams& config);

const MeshSegmenter::ObjectDetectionParams& getParamsForLabel(const MeshSegmenter::Config& config, uint32_t label);

/**
 * @brief Check if two nodes are close together -> consider as the same object, use when
 * mergeActiveNodes is used
 *
 * @param lhs_node the left hand side node
 * @param rhs_node the right hand side node
 * @param config MeshSegmenter's config
 * @return true if the two nodes overlap (one's bounding box contains the centroid of
 * the other's)
 */
bool nodesMatch(const SceneGraphNode& lhs_node,
                const SceneGraphNode& rhs_node,
                const MeshSegmenter::Config& config);

/**
 * @brief Check if a node and a cluster are close to each other -> consider as the same
 * object
 *
 * @param cluster the cluster to compare with
 * @param node the node to compare the cluster against
 * @param config MeshSegmenter config
 * @return true if the cluster and the node are close to each other (the node's bounding
 * box contains the centroid of the cluster)
 */
bool nodesMatch(const Cluster& cluster,
                const SceneGraphNode& node,
                const MeshSegmenter::Config& config);

/**
 * @brief Remove the floor of the input point cloud
 *
 * @param cloud the input cloud
 * @param z the minimum height above which the points are retained
 */
MeshCloud::Ptr removeFloor(MeshCloud::Ptr cloud, float z);
/**
 * @brief Downsamples a pointcloud using voxel downsampling
 *
 * @param cloud the input cloud
 * @param vox_size the voxel grid size for downsampling
 */
MeshCloud::Ptr downsampleCloud(MeshCloud::Ptr cloud, float vox_size);
/**
 * @brief remove outliers from a pointcloud using statistical outlier removal
 *
 * @param cloud the input cloud
 * @param mean_k the mean k value
 * @param stddev_mul_thresh the standard deviation
 */
MeshCloud::Ptr cloudStatisticalOutlierRemoval(MeshCloud::Ptr cloud,
                                              int mean_k,
                                              float stddev_mul_thresh);
/**
 * @brief Cluster a point cloud using Euclidean clustering
 *
 * @param cloud_ptr the input cloud
 * @param cluster_indices the indices of the input cloud
 * @param config MeshSegmenter's config
 */
void euclideanClustering(MeshCloud::Ptr cloud_ptr,
                         std::vector<pcl::PointIndices>& cluster_indices,
                         const MeshSegmenter::Config& config);
/**
 * @brief compute the median of a list of floats
 *
 * @param input a vector of floats
 * @return the median of the inputs
 */
float computeMedian(std::vector<float>& input);
/**
 * @brief compute the median point of a mesh
 *
 * @param mesh_ptr a pointer to the mesh
 * @return the median point of the mesh
 */
CloudPoint computeMeshMedian(MeshCloud::Ptr mesh_ptr);
/**
 * @brief A function to return if a point is close to the mesh by iteratively going
 * through all the possible pairs
 *
 * @param point_D the point to consider
 * @param instance_mesh the mesh to consider
 * @param threshold the threshold below which the point is considered close to the mesh
 * @return true if the point is close to the instance mesh
 */
bool isPointCloseToCloudNaive(const CloudPoint& point_D,
                              const MeshCloud::Ptr instance_mesh,
                              float threshold);
/**
 * @brief A function to return if a point is close to the mesh by considering the k
 * closest points using kd-tree search
 *
 * @param point_D the point to consider
 * @param instance_mesh the mesh to consider
 * @param kdtree the kdtree
 * @param k the k parameter for the kdtree
 * @param threshold the threshold below which the poiint is considered close to the
 * cloud
 * @return true if the point is considered close to the clouds
 */
bool isPointCloseToCloudKDTree(const CloudPoint& point_D,
                               const MeshCloud::Ptr instance_mesh,
                               pcl::KdTreeFLANN<CloudPoint> kdtree,
                               int k,
                               float threshold);
/**
 * @brief compute a hash map of instance information (cloud, instance view data) from
 * the reconstructed vertex map.
 *
 * @param input the reconstruction output
 * @param config MeshSegmenter config
 * @return unordered_map {class_id -> vector{<MaskData, MeshCloud>}}
 */
ClassToInstance computeInstancesClouds(const ReconstructionOutput& input,
                                       MeshSegmenter::Config config);

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr generateHighResPointCloud(const hydra::InputData& sensor_data,
                                                                  const hydra::MaskData& mask_data,
                                                                  MeshSegmenter::Config config);
/**
 * @brief extract pcl clusters for each instance from the monolithic map. From the
 * reconstructed clouds from compute InstancesClouds, go over the full map's point
 * cloud, if a point is close enough to a mesh, it is considered part of that mesh
 *
 * @param config MeshSegmenter config
 * @param delta kimera_pgmo delta (volumetric changes between t and t+1)
 * @param indices indices of delta that contains the class semantic
 * @param class_id the class semantic
 * @param class_to_instance map {class_id -> instance_info}. (see
 * computeInstancesClouds)
 * @param registered_indices set of indices that have already been registered
 * @return clusters of points from the monolithic map
 */
Clusters findInstanceClusters(const MeshSegmenter::Config& config,
                              const kimera_pgmo::MeshDelta& delta,
                              const std::vector<size_t>& indices,
                              const int64& class_id,
                              const ClassToInstance& class_to_instance,
                              std::unordered_set<size_t>& registered_indices,
                              size_t min_cluster_size);

spark_dsg::Mesh::Ptr generateMeshFromCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud);

}  // namespace hydra
