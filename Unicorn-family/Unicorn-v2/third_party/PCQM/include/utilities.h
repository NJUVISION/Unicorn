#ifndef UTILITIES_H
#define UTILITIES_H

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include "PointSet.h"
#include "nanoflann.hpp"

using namespace nanoflann;

typedef KDTreeSingleIndexAdaptor<
 L2_Simple_Adaptor<double, PointSet > ,
 PointSet,
 3 /* dim */> 
KdTree;
 
static void save(std::vector<double> &scalars, const char *filename)
{
  std::ofstream out;
  out.open(filename);
  
  std::vector<double>::const_iterator it;
  for(it = scalars.begin(); it!=scalars.end();++it)
  {
    out<<*it<<"\n";
  }
  out.close();
} 

static void save(std::vector<Point> &pts, const char *filename)
{
  std::ofstream out;
  out.open(filename);
  
  std::vector<Point>::const_iterator it;
  for(it = pts.begin(); it!=pts.end();++it)
  {
    out<<it->x<<"\t"<<it->y<<"\t"<<it->z<<"\n";
  }
  out.close();
}

#endif
