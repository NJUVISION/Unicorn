#pragma once
#include <PointSet.h>

#include <nanoflann.hpp>
#include <utilities.h>
#include <utility>
#include <vector>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
//#include <point_cloud_dataset.h>
#include <Eigen/Dense>

#include "nanoflann.hpp"
#include "utilities.h"
using namespace nanoflann;
using namespace Eigen;

class point_cloud_dataset {
protected:
public:

  //global data
  std::vector<Point> m_projected_point_position;

  //color
  std::vector<std::array<double, 3>> m_point_color_rgb;
  std::vector<std::array<double, 3>> m_point_color_lab;

  // projection and curvature
  std::vector<double> m_computed_curvature;
  PointSet m_projected_points_set;
  KdTree m_point_cloud;
  int m_nb_vertices;

  // MSDM
  std::vector<double> m_mean_curvature;
  std::vector<double> m_local_msdm;
  std::unordered_map<std::string, std::vector<double>> m_map_debug;

  //others
  std::string m_instance_name;
  static int objet_count;

  point_cloud_dataset(int nb_vertices = 0, std::string name = std::string("Default_dataset"))
      : m_point_cloud(3, m_projected_points_set, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    m_nb_vertices = nb_vertices;
    m_projected_point_position.assign(m_nb_vertices, Point());
    m_computed_curvature.assign(m_nb_vertices, 0.0);
    m_mean_curvature.assign(m_nb_vertices, 0.0);
    m_local_msdm.assign(m_nb_vertices, 0.0);
   
    m_instance_name = name;



    // m_computed_curvature.assign(m_nb_projected_vertices, Point());


    // m_projected_points.loadPointCloud(m_projected_point_position);
    // m_point_cloud(3, m_projected_points, KDTreeSingleIndexAdaptorParams(10));
  }

  // Set the inner-structure with the right points values and build the kdtree index.
  void build_kdtree() {

    m_projected_points_set.loadPointCloud(m_projected_point_position);
    m_point_cloud.buildIndex();
  }
  // Check if the key already exist then initialize value with the size of the structure
  void initialize_map_value(std::string key) {
    if (m_map_debug.find(key) == m_map_debug.end()) {
      // not found
      m_map_debug[key] = std::vector<double>(m_nb_vertices, 0.0);
      std::cout << "[" << m_instance_name <<"] => Key : " << key << " inserted in m_map_debug and structure initialized"
                << std::endl;
    } else {
      // found
      std::cout << "[" << m_instance_name << "] => Key : " << key << " allready inserted..." << std::endl;
    }
  }

   void initialize_map_debug() {
   
	initialize_map_value("Luminance");
	initialize_map_value("Contrast");
	initialize_map_value("Structure");

   
   }

};
