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
#include "hydra/utils/mesh_utilities.h"

#include <glog/logging.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> 
#include <pcl/common/centroid.h> 
#include <spark_dsg/bounding_box_extraction.h>

#include <algorithm>
#include <memory>

namespace hydra {

bool updateNodeCentroid(const spark_dsg::Mesh& mesh,
                        const std::vector<size_t>& indices,
                        NodeAttributes& attrs) {
  size_t num_valid = 0;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto idx : indices) {
    const auto pos = mesh.pos(idx).cast<double>();
    if (!pos.array().isFinite().all()) {
      continue;
    }

    centroid += pos;
    ++num_valid;
  }

  if (!num_valid) {
    return false;
  }

  attrs.position = centroid / num_valid;
  return true;
}

// TODO: Clean up mesh by removing outliers
void removeMeshOutliers(BoundingBox::MeshAdaptor mesh_adaptor,
                        pcl::PointIndices& removed_indices) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto& index : *mesh_adaptor.indices) {
    // LOG(INFO) << "Point: " << point.x() << " " << point.y() << " " << point.z();
    const auto& point = mesh_adaptor.mesh.points[index];
    cloud->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
  }
  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor(true);
  sor.setInputCloud(cloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(0.5);
  sor.filter(*cloud_filtered);
  sor.getRemovedIndices(removed_indices);
  VLOG(5) << "Cloud b4 stat removal: " << cloud->size();
  VLOG(5) << "Cloud af stat removal: " << cloud_filtered->size();
}

bool updateObjectGeometry(const spark_dsg::Mesh& mesh,
                          ObjectNodeAttributes& attrs,
                          const std::vector<size_t>* indices,
                          std::optional<BoundingBox::Type> type) {
  std::vector<size_t> mesh_connections;
  if (!indices) {
    mesh_connections.assign(attrs.mesh_connections.begin(),
                            attrs.mesh_connections.end());
  }

  const BoundingBox::MeshAdaptor adaptor(mesh, indices ? indices : &mesh_connections);

  pcl::PointIndices removed_indices;
  removeMeshOutliers(adaptor, removed_indices);
  VLOG(5) << "Removed indices size: " << removed_indices.indices.size();
  std::sort(removed_indices.indices.rbegin(), removed_indices.indices.rend());
  if (!indices) {
    for (const auto& i : removed_indices.indices) {
      mesh_connections.erase(mesh_connections.begin() + i);
    }
  }
  VLOG(5) << "Mesh connections after outlier rm: " << mesh_connections.size();
  attrs.bounding_box = BoundingBox(adaptor, type.value_or(attrs.bounding_box.type));
  if (indices) {
    return updateNodeCentroid(mesh, *indices, attrs);
  } else {
    return updateNodeCentroid(mesh, mesh_connections, attrs);
  }
  // return updateNodeCentroid(mesh, mesh_connections, attrs);
}

void updateBoundingBoxFromPointCloud(spark_dsg::ObjectNodeAttributes& attrs,
                                     spark_dsg::BoundingBox::Type type) {
    if (!attrs.point_cloud || attrs.point_cloud->empty()) {
        attrs.bounding_box.type = spark_dsg::BoundingBox::Type::INVALID;
        return;
    }

    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*attrs.point_cloud, min_pt, max_pt);

    const Eigen::Vector3f dimensions = max_pt.head<3>() - min_pt.head<3>();
    const Eigen::Vector3f world_P_center = min_pt.head<3>() + (dimensions / 2.0f);

    attrs.bounding_box = spark_dsg::BoundingBox(dimensions,
                                                world_P_center);
    attrs.bounding_box.type = type;
}

MeshLayer::Ptr getActiveMesh(const MeshLayer& mesh_layer,
                             const BlockIndices& archived_blocks) {
  auto active_mesh = std::make_shared<MeshLayer>(mesh_layer.blockSize());
  const BlockIndexSet archived_set(archived_blocks.begin(), archived_blocks.end());
  for (const auto& block : mesh_layer.updatedBlockIndices()) {
    if (archived_set.count(block)) {
      continue;
    }
    active_mesh->allocateBlock(block) = mesh_layer.getBlock(block);
  }
  return active_mesh;
}

}  // namespace hydra
