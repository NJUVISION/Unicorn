#ifndef POINT_SET
#define POINT_SET


#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>

#include <Eigen/Dense>
#include "tinyply.h"

class Point {
public:
  double x, y, z, nx, ny, nz, r, g, b, cie_L, cie_A, cie_B;

  friend std::ostream& operator<<(std::ostream& os, const Point& pts);
};

Point operator+(const Point& p1, const Point& p2) {
  Point p;
  p.x = p1.x + p2.x;
  p.y = p1.y + p2.y;
  p.z = p1.z + p2.z;
  return p;
}

Point operator+(const Point& p1, const Eigen::Vector3d& n) {
  Point p;
  p.x = p1.x + n(0);
  p.y = p1.y + n(1);
  p.z = p1.z + n(2);
  return p;
}

Point operator/(const Point& p1, const double a) {
  Point p;
  p.x = p1.x / a;
  p.y = p1.y / a;
  p.z = p1.z / a;
  return p;
}

Point operator*(const Point& p1, const double a) {
  Point p;
  p.x = p1.x * a;
  p.y = p1.y * a;
  p.z = p1.z * a;
  return p;
}

std::ostream& operator<<(std::ostream& os, const Point& pts) {
  os << "{" << pts.x << " , " << pts.y << " , " << pts.z << " }";
  return os;
}


Eigen::Matrix3d operator*(const Point& p1, const Point& p2) {
  Eigen::Matrix3d M(3, 3);
  M(0, 0) = p1.x * p2.x;
  M(0, 1) = p1.x * p2.y;
  M(0, 2) = p1.x * p2.z;
  M(1, 1) = p1.y * p2.y;
  M(1, 2) = p1.y * p2.z;
  M(2, 2) = p1.z * p2.z;
  M(1, 0) = M(0, 1);
  M(2, 0) = M(0, 2);
  M(2, 1) = M(1, 2);
  return M;
}

class PointSet {
public:
  std::vector<Point> pts;

  PointSet() { xmin = xmax = 0.0; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the distance between the vector "p1[0:size-1]" and the data point
  // with index "idx_p2" stored in the class:
  inline double kdtree_distance(const double* p1, const size_t idx_p2, size_t size) const {
    const double d0 = p1[0] - pts[idx_p2].x;
    const double d1 = p1[1] - pts[idx_p2].y;
    const double d2 = p1[2] - pts[idx_p2].z;
    return d0 * d0 + d1 * d1 + d2 * d2;
  }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the  "if/else's" are actually solved at compile time.
  inline double kdtree_get_pt(const size_t idx, int dim) const {
    if (dim == 0)
      return pts[idx].x;
    else if (dim == 1)
      return pts[idx].y;
    else
      return pts[idx].z;
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned
  // in "bb" so it can be avoided to redo it again.
  //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3
  // for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& bb) const {
    return false;
  }


  void readPointCloud(std::ifstream& f) {
    Point p;

    f >> p.x >> p.y >> p.z >> p.nx >> p.ny >> p.nz;

    unsigned int i = 0;
    xmin = xmax = p.x;
    ymin = ymax = p.y;
    zmin = zmax = p.z;

    while (f) {
      xmax = xmax > p.x ? xmax : p.x;
      ymax = ymax > p.y ? ymax : p.y;
      zmax = zmax > p.z ? zmax : p.z;
      xmin = xmin < p.x ? xmin : p.x;
      ymin = ymin < p.y ? ymin : p.y;
      zmin = zmin < p.z ? zmin : p.z;
      pts.push_back(p);
      ++i;
      f >> p.x >> p.y >> p.z >> p.nx >> p.ny >> p.nz;
    }
  }


  double xmin, ymin, zmin;
  double xmax, ymax, zmax;


  void loadPointCloud(std::vector<Point>& points) {
    pts.clear();
    std::vector<Point>::const_iterator pi = points.begin();
    unsigned int i = 0;
    xmin = xmax = pi->x;
    ymin = ymax = pi->y;
    zmin = zmax = pi->z;

    for (; pi != points.end(); ++pi) {
      xmax = xmax > pi->x ? xmax : pi->x;
      ymax = ymax > pi->y ? ymax : pi->y;
      zmax = zmax > pi->z ? zmax : pi->z;
      xmin = xmin < pi->x ? xmin : pi->x;
      ymin = ymin < pi->y ? ymin : pi->y;
      zmin = zmin < pi->z ? zmin : pi->z;
      pts.push_back(*pi);
      ++i;
    }
  }

  void addPoint(Point& p) {
    if (xmax - xmin < 1e-16) {
      xmin = xmax = p.x;
      ymin = ymax = p.y;
      zmin = zmax = p.z;
    } else {
      xmax = xmax > p.x ? xmax : p.x;
      ymax = ymax > p.y ? ymax : p.y;
      zmax = zmax > p.z ? zmax : p.z;
      xmin = xmin < p.x ? xmin : p.x;
      ymin = ymin < p.y ? ymin : p.y;
      zmin = zmin < p.z ? zmin : p.z;
    }
    pts.push_back(p);
  }

  int npts() const { return (int)pts.size(); }

  // TINIPLY READER
  inline std::vector<uint8_t> read_file_binary(const std::string& pathToFile) {
    std::ifstream file(pathToFile, std::ios::binary);
    std::vector<uint8_t> fileBufferBytes;

    if (file.is_open()) {
      file.seekg(0, std::ios::end);
      size_t sizeBytes = file.tellg();
      file.seekg(0, std::ios::beg);
      fileBufferBytes.resize(sizeBytes);
      if (file.read((char*)fileBufferBytes.data(), sizeBytes)) return fileBufferBytes;
    } else {
      throw std::runtime_error("could not open binary ifstream to path " + pathToFile);
	}  
    return fileBufferBytes;
  }

  struct memory_buffer : public std::streambuf {
    memory_buffer(char const* base, size_t size) {
      char* p(const_cast<char*>(base));
      this->setg(p, p, p + size);
    }
  };

  struct memory_stream : virtual memory_buffer, public std::istream {
    memory_stream(char const* base, size_t size)
        : memory_buffer(base, size), std::istream(static_cast<std::streambuf*>(this)) {}
  };

  struct float2 {
    float x, y;
  };
  struct float3 {
    float x, y, z;
  };
  struct double3 {
    double x, y, z;
  };

  struct unsigned_char3 {
    unsigned char r, g, b;
  };

  struct uint3 {
    uint32_t x, y, z;
  };
  struct uint4 {
    uint32_t x, y, z, w;
  };


  void read_ply_file(const std::string& filepath, const bool preload_into_memory) {
    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try {
      // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a
      // stream is a net win for parsing speed, about 40% faster.
      if (preload_into_memory) {
        byte_buffer = read_file_binary(filepath);
        file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
      } else {
        file_stream.reset(new std::ifstream(filepath, std::ios::binary));
      }

      if (!file_stream || file_stream->fail()) throw std::runtime_error("failed to open " + filepath);

      tinyply::PlyFile file;
      file.parse_header(*file_stream);

      int point_size = 0;
      std::cout << "........................................................................\n";
      for (const auto& c : file.get_comments()) {
        std::cout << "Comment: " << c << std::endl;
      }
      for (const auto& e : file.get_elements()) {
        std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
        if (e.name == "vertex") {
          point_size = e.size;
          std::cout << "Pointset size set to " << point_size << std::endl;
        }
        for (const auto& p : e.properties) {
          std::cout << "\tproperty - " << p.name << " (" << tinyply::PropertyTable[p.propertyType].str << ")"
                    << std::endl;
        }
      }
      for (const auto& i : file.get_info())
		std::cout << i << std::endl;
      std::cout << "........................................................................\n";

	  
      pts.clear();
      pts.resize(point_size);
      ///////////////////////////// MODIFICATION A PARTIR D'ICI /////////////////////////////

      // Tinyply treats parsed data as untyped byte buffers. See below for examples.
      std::shared_ptr<tinyply::PlyData> vertices, colors;

      // The header information can be used to programmatically extract properties on elements
      // known to exist in the header prior to reading the data. For brevity of this sample, properties
      // like vertex position are hard-coded:
      try {
        vertices = file.request_properties_from_element("vertex", {"x", "y", "z"},3);
      } catch (const std::exception& e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
      }

      try {
        colors = file.request_properties_from_element("vertex", {"red", "green", "blue"},3);
      } catch (const std::exception& e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
      }

      // read data from file
      file.read(*file_stream);

      std::vector< std::array<float, 3> > verts;
      std::vector< std::array<unsigned char, 3> > cols;
      // init boundaries
      xmin = ymin = zmin = std::numeric_limits<double>::max();
      xmax = ymax = zmax = std::numeric_limits<double>::min();

      // type casting to your own native types - Option A
      if (vertices) {
        
		// const size_t numVerticesBytes = vertices->buffer.size_bytes();

        //std::cout << "bytes " << vertices->buffer.size_bytes() << std::endl;
        //std::cout << "count " << vertices->count << std::endl;
        //std::cout << "data size " << vertices->buffer.size_bytes() / vertices->count << std::endl;
        //std::cout << "expected data size " << sizeof(float3) << std::endl;
        
		verts.resize(vertices->count);

        std::memcpy(verts.data(), vertices->buffer.get(), vertices->buffer.size_bytes());
        for (const auto& i : verts) {

			// std::cout << i.x << " " << i.y << " " << i.z << std::endl;
        }
      }

      // type casting to your own native types - Option A
      if (colors) {
        const size_t numcolorsBytes = colors->buffer.size_bytes();
        cols.resize(colors->count);
        std::memcpy(cols.data(), colors->buffer.get(), numcolorsBytes);
      }

      Point p;
      for (int i_pts = 0; i_pts < point_size; i_pts++) {

        p.x = verts[i_pts][0];
        p.y = verts[i_pts][1];
        p.z = verts[i_pts][2];
        //std::cout << verts[i_pts].x << std::endl;
        if (colors) {
          p.r = cols[i_pts][0];
          p.g = cols[i_pts][1];
          p.b = cols[i_pts][2];
        } else {
          p.r = 1;
          p.g = 1;
          p.b = 1;
        }

        pts[i_pts] = p;


        xmax = xmax > p.x ? xmax : p.x;
        ymax = ymax > p.y ? ymax : p.y;
        zmax = zmax > p.z ? zmax : p.z;
        xmin = xmin < p.x ? xmin : p.x;
        ymin = ymin < p.y ? ymin : p.y;
        zmin = zmin < p.z ? zmin : p.z;
      }


    } catch (const std::exception& e) {
      std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }
  }
};

#endif
