// Copyright (c) 2020 University of Lyon and CNRS (France).
// All rights reserved.
//
//The code is the implementation of our paper
//"PCQM: A Full-Reference Quality Metric for Colored 3D Point Clouds"
//Gabriel Meynet, Yana Nehme, Julie Digne, Guillaume Lavoué
//Presented at QoMEX 2020.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

/** @file */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "PointSet.h"
//#include <point_cloud_dataset.h>
#include "nanoflann.hpp"
#include <Eigen/Dense>
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include "utilities.h"

//#define WRITE_FEATURES_CSV 1

using std::cerr;
using std::cout;
using std::endl;
using std::string;

using namespace nanoflann;
using namespace Eigen;



using Eigen::JacobiSVD;
using Eigen::Matrix3d;


std::string global_regfile;
std::string global_reffile;


/**
* \fn double interpolate1_computevalue(double x0, double x1, double y0, double y1, double x)
* \brief Find the value of x in the new range
*
* \param x0 : min of X
* \param x1 : max of X
* \param y0 : min of Y
* \param y1 : max of Y
* \param x : Value to interpolate
* \return Returns the interpolated value of x
*/
double interpolate1_computevalue(double x0, double x1, double y0, double y1, double x) {
	return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
}


/**
* \fn double interpolate1_process(std::vector<double>& init_grid, std::vector<double>& val_grid,
* \brief Interpolate
*
* \param init_grid : Range of data
* \param val_grid : New data
* \param x : Value to interpolate
* \return Returns the interpolated value
*/
double interpolate1_process(std::vector<double>& init_grid, std::vector<double>& val_grid,
	double x) 
{

	auto upper = std::upper_bound(init_grid.begin(), init_grid.end(), x);
	int position = std::distance(init_grid.begin(), upper);

	if (position < init_grid.size()) {
		double res = interpolate1_computevalue(init_grid[position - 1], init_grid[position], val_grid[position - 1],
			val_grid[position], x);
		return res;

	}
	else // if upper out of bounds returning maxval
	{
		return val_grid[position - 1];
	}
}



/**
* \fn double interpolate1_computevalue(double x0, double x1, double y0, double y1, double x)
* \brief Find the value of x in the new range
*
* \param x0 : min of X
* \param x1 : max of X
* \param y0 : min of Y
* \param y1 : max of Y
* \param x : Value to interpolate
* \return Returns the interpolated value of x
*/
double interpolate2_computevalue(double q_00, double q_10, double q_01, double q_11, double x0, double x1, double y0,
	double y1, double x, double y) {
	// https://helloacm.com
	double x1x0, y1y0, x1x, y1y, yy0, xx0;

	x1x0 = std::abs(x1 - x0);
	y1y0 = std::abs(y1 - y0);


	x1x = x1 - x;
	y1y = y1 - y;
	yy0 = y - y0;
	xx0 = x - x0;


	return 1.0 / (x1x0 * y1y0) * (q_00 * x1x * y1y + q_10 * xx0 * y1y + q_01 * x1x * yy0 + q_11 * xx0 * yy0);
}


/**
* \fn std::pair<double, double> interpolate2_process(std::vector<std::vector<std::pair<double, double>>>& init_grid_AB,
	double x, double y)
* \brief Given a grid, compute the interpolation of the point defined by x and y
*
* \return Returns the point interpolated
*/
std::pair<double, double> interpolate2_process(std::vector<std::vector<std::pair<double, double>>>& init_grid_AB,
	double x, double y)
{
	// Finding range
	int x_min = std::floor(x);
	int x_max = x_min + 1;
	int y_min = std::floor(y);
	int y_max = y_min + 1;

	// If there is no need to interpolate
	if (x_min == x_max && y_min == y_max) {
		// return the value
		return std::make_pair(init_grid_AB[x_min + 128][y_min + 128].first, init_grid_AB[x_min + 128][y_min + 128].second);
	}

	double a_interpolated = interpolate2_computevalue(
		init_grid_AB[x_min + 128][y_min + 128].first, init_grid_AB[x_max + 128][y_min + 128].first,
		init_grid_AB[x_min + 128][y_max + 128].first, init_grid_AB[x_max + 128][y_max + 128].first, x_min, x_max, y_min,
		y_max, x, y);
	double b_interpolated = interpolate2_computevalue(
		init_grid_AB[x_min + 128][y_min + 128].second, init_grid_AB[x_max + 128][y_min + 128].second,
		init_grid_AB[x_min + 128][y_max + 128].second, init_grid_AB[x_max + 128][y_max + 128].second, x_min, x_max, y_min,
		y_max, x, y);

	return std::make_pair(a_interpolated, b_interpolated);
}

/**
* \fnvoid initMatLABCH(std::vector<double>& init_grid_L, std::vector<double>& grid_L,
	std::vector<std::vector<std::pair<double, double>>>& init_grid_AB)
* \brief Initialize the 1D and 2D grid for interpolation from file
*
*/
void initMatLABCH(
	std::vector<double>& init_grid_L, std::vector<double>& grid_L,
	std::vector<std::vector<std::pair<double, double>>>& init_grid_AB)
{

	int size_L = 100001;
	int size_row = 257;
	int size_col = 257;
	init_grid_L.assign(size_L, 0);
	grid_L.assign(size_L, 0);

	int size_tabAB = 66049;
	init_grid_AB.assign(size_col, std::vector<std::pair<double, double>>(size_row, std::make_pair(0.0, 0.0)));

	std::ifstream data_f;
	std::ifstream data_g;
	std::ifstream data_a;
	std::ifstream data_b;
	data_f.open("L_data.txt", std::ifstream::in);


	if (!data_f.fail()) {
		std::string line1;
		int cpt = 0;
		while (getline(data_f, line1) && cpt < size_L) {
			grid_L[cpt] = std::stod(line1);
			cpt++;
		}
	}
	else
		std::cout << "Unable to open L_data.txt \t";
	data_f.close();

	// init grid L (from matlab code)
	for (int i = 0; i < init_grid_L.size(); i++) {
		init_grid_L[i] = i * 0.001;
	}

	data_f.open("RegularGridInit_0_0_1.txt", std::ifstream::in);
	data_g.open("RegularGridInit_0_0_2.txt", std::ifstream::in);
	data_a.open("RegularGrid_0_0_1.txt", std::ifstream::in);
	data_b.open("RegularGrid_0_0_2.txt", std::ifstream::in);


	if (!data_f.fail() && !data_g.fail() && !data_a.fail() && !data_b.fail()) {
		std::string line1;
		std::string line2;
		std::string line3;
		std::string line4;

		int cpt = 0;
		while (getline(data_f, line1) && getline(data_g, line2) && getline(data_a, line3) && getline(data_b, line4) &&
			cpt <= size_tabAB) {

			init_grid_AB[std::stoi(line1) + 128][std::stoi(line2) + 128].first = std::stod(line3);
			init_grid_AB[std::stoi(line1) + 128][std::stoi(line2) + 128].second = std::stod(line4);

			cpt++;
		}
	}
	else
	{
		std::cerr << "Unable to open RegularGridInit_0_0_1.txt, RegularGridInit_0_0_2.txt, RegularGrid_0_0_1.txt or "
			"RegularGrid_0_0_2.txt Closed \t";
		exit(EXIT_FAILURE);
	}
	
	data_f.close();
	data_g.close();
	data_a.close();
	data_b.close();
}


/**
* \fn void computeProjectionAndCurvature(const Point &origin, const std::vector<Point> &refpoints, std::vector<size_t> indices, Point &proj, double &H)
* \brief Compute the projection of an origin point onto the polynomial approximation of a set of neighbors given by a list of indices.
*
* \param origin : Point to be projected.
* \param refpoints : Contains all points from ref points cloud.
* \param indices : Index of points in refpoints cloud used to compute the projection.
* \param proj : Reference containing the point resulting from the projection.
* \param H : Reference containing the mean curvature of the projected point.
* \return Returns both the projection and the mean curvature (Referenced variables).
*/
void computeProjectionAndCurvature(const Point& origin, const std::vector<Point>& refpoints,
	std::vector<size_t>& indices, Point& proj, double& H) {

	Matrix3d M;
	M.setZero();
	Vector3d mu;
	mu.setZero();
	int nneighbors = indices.size();

	for (int i = 0; i < nneighbors; ++i) {
		Point p = refpoints[indices[i]];
		Vector3d neighbor(p.x, p.y, p.z);
		mu = mu + neighbor;
		M = M + neighbor * neighbor.transpose();
	}

	mu = mu / ((double)nneighbors);
	M = 1. / ((double)nneighbors) * M - mu * mu.transpose();

	// get local frame
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M);


	Eigen::Vector3d t1 = eig.eigenvectors().col(2);
	Eigen::Vector3d t2 = eig.eigenvectors().col(1);
	Eigen::Vector3d n = eig.eigenvectors().col(0);

	MatrixXd A(nneighbors, 6);
	VectorXd B(nneighbors);

	// build linear system
	for (int i = 0; i < nneighbors; ++i) {
		double xglob = refpoints[indices[i]].x - origin.x;
		double yglob = refpoints[indices[i]].y - origin.y;
		double zglob = refpoints[indices[i]].z - origin.z;
		Vector3d v(xglob, yglob, zglob);
		double x = v.transpose() * t1;
		double y = v.transpose() * t2;
		double z = v.transpose() * n;

		A(i, 0) = x * x;
		A(i, 1) = y * y;
		A(i, 2) = x * y;
		A(i, 3) = x;
		A(i, 4) = y;
		A(i, 5) = 1;

		B(i) = z;
	}


	VectorXd coeffs = A.colPivHouseholderQr().solve(B);

	// corresponding point:
	Vector3d delta = coeffs(5) * n;
	proj = origin + delta;

	// corresponding curvature
	double fxx = 2 * coeffs(0);
	double fyy = 2 * coeffs(1);
	double fxy = coeffs(2);
	double fx = coeffs(3);
	double fy = coeffs(4);

	H = 0.5 * ((1 + fx * fx) * fyy + (1 + fy * fy) * fxx - 2 * fxy * fx * fy) / pow(1 + fx * fx + fy * fy, 1.5);
}


/**
* \fn double compute_distance(Point &a, Point &b)
* \brief Compute the Euclidean distance between two points.
*
* \param a : Point a.
* \param b : Point b.
* \return Returns the Euclidean distance between a and b.
*/
double compute_distance(Point& a, Point& b) {
	return std::sqrt((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z));
}


double F(double input) // function f(...), which is used for defining L, a and b
					   // changes within [4/29,1]
{
	if (input > 0.008856)
		return std::cbrt(input); // maximum 1 --- prefer cbrt to pow for cubic root
	else
		return ((double(841.0) / 108.0) * input + double(4.0) / 29.0); // 841/108 = 29*29/36*16
}

/**
* \fn void RGBtoXYZ(int _R, int _G, int _B, double& X, double& Y, double& Z)
* \brief Compute the transformation from RGB colorspace to XYZ colorspace
*
* \param _R : Red input value.
* \param _G : Green input value.
* \param _B : Blue input value.
* \param X : X value computed.
* \param Y : Y value computed.
* \param Z : Z value computed.
*/
void RGBtoXYZ(int _R, int _G, int _B, double& X, double& Y, double& Z) {

	// RGB Working Space: sRGB
	// Reference White: D65
	double R = _R / 255.0;
	double G = _G / 255.0;
	double B = _B / 255.0;

	// INV Gamma correction
	R = ((R > 0.0404482362771076) ? std::pow((R + 0.055) / 1.055, 2.4) : (R / 12.92));
	G = ((G > 0.0404482362771076) ? std::pow((G + 0.055) / 1.055, 2.4) : (G / 12.92));
	B = ((B > 0.0404482362771076) ? std::pow((B + 0.055) / 1.055, 2.4) : (B / 12.92));

	// MATLAB Transform
	X = 0.412396 * R + 0.357583 * G + 0.180493 * B;
	Y = 0.212586 * R + 0.71517 * G + 0.0722005 * B;
	Z = 0.0192972 * R + 0.119184 * G + 0.950497 * B;
}

/**
* \fn void XYZtoLab(double X, double Y, double Z, double& L, double& a, double& b) 
* \brief Compute the transformation from XYZ colorspace to Lab colorspace
*
* \param X : X input value.
* \param Y : Y input value.
* \param Z : Z input value.
* \param L : Lightness value computed.
* \param a : A* value computed.
* \param b : B* value computed.
*/
void XYZtoLab(double X, double Y, double Z, double& L, double& a, double& b) {

	// matlab White point
	const double Xo = 0.950456;
	const double Yo = 1.000000;
	const double Zo = 1.088754;

	L = 116.0 * F(Y / Yo) - 16.0;        // maximum L = 100
	a = 500.0 * (F(X / Xo) - F(Y / Yo)); // maximum
	b = 200.0 * (F(Y / Yo) - F(Z / Zo));
}

/**
* \fn RGBtoLab(double R, double G, double B, double& L, double& a, double& b)
* \brief Compute the transformation from RGB colorspace to Lab colorspace
*
* \param R : Red input value.
* \param G : Green input value.
* \param B : Blue input value.
* \param L : L value computed.
* \param a : a* value computed.
* \param b : b* value computed.
*/
void RGBtoLab(double R, double G, double B, double& L, double& a, double& b) {
	double X, Y, Z;
	RGBtoXYZ(R, G, B, X, Y, Z);
	XYZtoLab(X, Y, Z, L, a, b);
}


/**
* \fn LabtoLCH(double _L, double _a, double _b, double& L, double& C, double& H)
* \brief Compute the transformation from Lab colorspace to LCH colorspace
*
* \param _L : L input value.
* \param _a : a* input value.
* \param _b : b* input value.
* \param L : L value computed.
* \param C : Chroma value computed.
* \param H : Hue input value computed.

*/
void LabtoLCH(double _L, double _a, double _b, double& L, double& C, double& H) {

	double PI = 3.14159265359;
	L = _L;
	C = std::sqrt(_a * _a + _b * _b);
	double temp_h = std::atan2(_b, _a); // std::atan(_b/_a);

	if (temp_h >= 0) {
		H = temp_h;
	}
	else {
		H = temp_h + 360.0;
	}
}


/**
* \fn RGBtoLCH(double R, double G, double B, double& L, double& C, double& H)
* \brief Compute the transformation from RGB colorspace to LCH colorspace
*
* \param R : Red input value.
* \param G : Green input value.
* \param B : Blue input value.
* \param L : L value computed.
* \param C : Chroma value computed.
* \param H : Hue input value computed.

*/
void RGBtoLCH(double R, double G, double B, double& L, double& C, double& H) {

	double _L, _a, _b;
	RGBtoLab(R, G, B, _L, _a, _b);
	LabtoLCH(_L, _a, _b, L, C, H);
}

/**
* \fn std::string remove_extension(const std::string& filename)
* \brief Remove the extension of an input filename
*
* \param filename : filename
* \return The filename's string without extension
*/
std::string remove_extension(const std::string& filename) {
	size_t lastdot = filename.find_last_of(".");
	if (lastdot == std::string::npos) return filename;
	return filename.substr(0, lastdot);
}

/**
* \fn bool write_mean_features(std::vector<double>& Curv_Lumi, std::vector<double>& Curv_Constrast,
	std::vector<double>& Curv_Struct, std::vector<double>& IDF1, std::vector<double>& IDF2,
	std::vector<double>& IDF3, std::vector<double>& IDF4, std::vector<double>& IDF5,
	const string regfile, const string reffile,double PCQM, const string destination))
* \brief Write the mean of each feature and PCQM to CSV
*
* \param std::vector<double>& Curv_Lumi : F1
* \param std::vector<double>& Curv_Constrast : F2
* \param std::vector<double>& Curv_Struct : F3
* \param std::vector<double>& IDF1 : F4
* \param std::vector<double>& IDF2 : F5
* \param std::vector<double>& IDF3 : F6
* \param std::vector<double>& IDF4 : F7
* \param std::vector<double>& IDF5 : F8
* \param const string regfile : registered point cloud filename 
* \param const string reffile : refference point cloud filename
* \param double PCQM : PCQM metric value computed
* \param const string destination : Output filename

* \return True if operation succeeded
*/
bool write_mean_features(std::vector<double>& Curv_Lumi, std::vector<double>& Curv_Constrast,
	std::vector<double>& Curv_Struct, std::vector<double>& IDF1, std::vector<double>& IDF2,
	std::vector<double>& IDF3, std::vector<double>& IDF4, std::vector<double>& IDF5,
	const string regfile, const string reffile,double PCQM, const string destination) {


	int length;
	ifstream filestr;
	filestr.open(destination, ios::binary); // open your file
	filestr.seekg(0, ios::end); // put the "cursor" at the end of the file
	length = filestr.tellg(); // find the position of the cursor
	filestr.close(); // close your file

	std::ofstream PCQM_out_f(destination, std::ios::app);


	if (PCQM_out_f.is_open()) {
		
		if (length == -1) //CSV header
		{
			PCQM_out_f << "reffile"
					<< ";";
			PCQM_out_f << "regfile"
					<< ";";
			PCQM_out_f << "F1"
					<< ";";
			PCQM_out_f << "F2"
					<< ";";
			PCQM_out_f << "F3"
					<< ";";
			PCQM_out_f << "F4"
					<< ";";
			PCQM_out_f << "F5"
					<< ";";
			PCQM_out_f << "F6"
					<< ";";
			PCQM_out_f << "F7"
					<< ";";
			PCQM_out_f << "F8"
				<< ";";
			PCQM_out_f << "PCQM";

		}
		

		double mu_1 = 0.0;
		double mu_2 = 0.0;
		double mu_3 = 0.0;
		double mu_4 = 0.0;
		double mu_5 = 0.0;
		double mu_6 = 0.0;
		double mu_7 = 0.0;
		double mu_8 = 0.0;


		for (int i = 0; i < Curv_Lumi.size(); i++) {
			mu_1 += Curv_Lumi[i];
			mu_2 += Curv_Constrast[i];
			mu_3 += Curv_Struct[i];
			mu_4 += IDF1[i];
			mu_5 += IDF2[i];
			mu_6 += IDF3[i];
			mu_7 += IDF4[i];
			mu_8 += IDF5[i];
		}
		PCQM_out_f << "\n";
		PCQM_out_f << regfile << ";";
		PCQM_out_f << reffile << ";";
		PCQM_out_f << mu_1 / Curv_Lumi.size() << ";";
		PCQM_out_f << mu_2 / Curv_Lumi.size() << ";";
		PCQM_out_f << mu_3 / Curv_Lumi.size() << ";";
		PCQM_out_f << 1.0 - mu_4 / Curv_Lumi.size() << ";";
		PCQM_out_f << 1.0 - mu_5 / Curv_Lumi.size() << ";";
		PCQM_out_f << 1.0 - mu_6 / Curv_Lumi.size() << ";";
		PCQM_out_f << 1.0 - mu_7 / Curv_Lumi.size() << ";";
		PCQM_out_f << 1.0 - mu_8 / Curv_Lumi.size() << ";";
		PCQM_out_f << PCQM << ";";
		PCQM_out_f.close();

		return true;
	}
	else
		cout << "Unable to open " << destination << " for writting";
	PCQM_out_f.close();
	return false;

}

/**
* \fn double compute_color_feature(int index_array, const size_t nMatches_Reg, std::vector<double>& data_proj,
	std::vector<double>& data_me, std::vector<double>& ret_weight_Reg,
	std::vector<double>& ret_weight_Ref,
	std::vector<std::pair<size_t, double>>& ret_matches_Reg, double sum_distances_me,
	double sum_distances_proj, int case_number, double constant_value)
* \brief Compute local feature depending on the case_number
*
* \return Return the local feature value
*/
double compute_color_feature(int index_array, const size_t nMatches_Reg, std::vector<double>& data_proj,
	std::vector<double>& data_me, std::vector<double>& ret_weight_Reg,
	std::vector<double>& ret_weight_Ref,
	std::vector<std::pair<size_t, double>>& ret_matches_Reg, double sum_distances_me,
	double sum_distances_proj, int case_number, double constant_value) {



	// Case 1 => lLuminance | lChroma | lHue
	// Case 2 => LContrast
	// Case 3 => LStructure


	// mean is usefull for each case
	double mu_me_gaussian = 0.0;
	double mu_proj_gaussian = 0.0;
	double mu_me_unweighted = 0.0;
	double mu_proj_unweighted = 0.0;

	double nb_elem = 0.0;


	for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
		size_t index_reg = ret_matches_Reg[cpt_neigh].first;

		mu_me_gaussian += data_me[index_reg] * ret_weight_Reg[cpt_neigh];
		mu_proj_gaussian += data_proj[index_reg] * ret_weight_Ref[cpt_neigh];

		mu_me_unweighted += data_me[index_reg];
		mu_proj_unweighted += data_proj[index_reg];

		nb_elem += 1.0;
	}



	mu_me_gaussian /= sum_distances_me;
	mu_proj_gaussian /= sum_distances_proj;


	mu_me_unweighted /= nb_elem;
	mu_proj_unweighted /= nb_elem;


	switch (case_number) {
	case 1: { // LCH
		return 1.0 / (constant_value * (mu_me_gaussian - mu_proj_gaussian) * (mu_me_gaussian - mu_proj_gaussian) + 1.0);
	}

	case 2: { // Contrast IDF2

		double standard_dev_me = 0.0;
		double standard_dev_proj = 0.0;

		for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
			size_t index_reg = ret_matches_Reg[cpt_neigh].first;

			standard_dev_me += pow(data_me[index_reg] - mu_me_gaussian, 2.0) * ret_weight_Reg[cpt_neigh];
			standard_dev_proj += pow(data_proj[index_reg] - mu_proj_gaussian, 2.0) * ret_weight_Ref[cpt_neigh];

		}

		if (standard_dev_me < 0) {
			standard_dev_me = 0.0;
		}

		if (standard_dev_proj < 0) {
			standard_dev_proj = 0.0;
		}

		standard_dev_me /= sum_distances_me;
		standard_dev_proj /= sum_distances_proj;


		return (2.0 * std::sqrt(standard_dev_me) * std::sqrt(standard_dev_proj) + constant_value) /
			(standard_dev_me + standard_dev_proj + constant_value);
	}

	case 3: { // Structure IDF3
		double standard_dev_me = 0.0;
		double standard_dev_proj = 0.0;
		// Covariance
		double covariance = 0.0;

		for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
			size_t index_reg = ret_matches_Reg[cpt_neigh].first;
			standard_dev_me += pow(data_me[index_reg] - mu_me_unweighted, 2.0);
			standard_dev_proj += pow(data_proj[index_reg] - mu_proj_unweighted, 2.0);
			covariance += ((data_me[index_reg] - mu_me_unweighted) * (data_proj[index_reg] - mu_proj_unweighted));
		}
		standard_dev_me /= nb_elem;
		standard_dev_proj /= nb_elem;

		if (standard_dev_me < 0) {
			standard_dev_me = 0.0;
		}

		if (standard_dev_proj < 0) {
			standard_dev_proj = 0.0;
		}

		covariance /= nb_elem;

		return (covariance + constant_value) / (std::sqrt(standard_dev_me) * std::sqrt(standard_dev_proj) + constant_value);
	}
	}
	return 0;
}

/**
* \fn void compute_geometric_feature(int index_array, const size_t nMatches_Reg, std::vector<double>& data_proj,
	std::vector<double>& data_me, std::vector<double>& ret_weight_Reg,
	std::vector<double>& ret_weight_Ref, std::vector<double>& lightness_field, std::vector<double>& contrast_field, std::vector<double>& structure_field,
	std::vector<std::pair<size_t, double>>& ret_matches_Reg, double sum_distances_me,
	double sum_distances_proj, double constant_value)
* \brief Compute geometric features 
*
*/
void compute_geometric_feature(int index_array, const size_t nMatches_Reg, std::vector<double>& data_proj,
	std::vector<double>& data_me, std::vector<double>& ret_weight_Reg,
	std::vector<double>& ret_weight_Ref, std::vector<double>& lightness_field, std::vector<double>& contrast_field, std::vector<double>& structure_field,
	std::vector<std::pair<size_t, double>>& ret_matches_Reg, double sum_distances_me,
	double sum_distances_proj, double constant_value) {


	// mean is usefull for each case
	double mu_me_gaussian = 0.0;
	double mu_proj_gaussian = 0.0;
	double mu_me_unweighted = 0.0;
	double mu_proj_unweighted = 0.0;

	double nb_elem = 0.0;


	for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
		size_t index_reg = ret_matches_Reg[cpt_neigh].first;

		mu_me_gaussian += data_me[index_reg] * ret_weight_Reg[cpt_neigh];
		mu_proj_gaussian += data_proj[index_reg] * ret_weight_Ref[cpt_neigh];

		mu_me_unweighted += data_me[index_reg];
		mu_proj_unweighted += data_proj[index_reg];

		nb_elem += 1.0;
	}

	mu_me_gaussian /= sum_distances_me;
	mu_proj_gaussian /= sum_distances_proj;


	mu_me_unweighted /= nb_elem;
	mu_proj_unweighted /= nb_elem;


	// Curvature based features

	double luminance = 0.0;
	double contrast = 0.0;
	double structure = 0.0;

	// Variance
	double variance_me = 0.0;
	double variance_proj = 0.0;

	// Covariance
	double covariance = 0.0;

	for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
		size_t index_reg = ret_matches_Reg[cpt_neigh].first;

		variance_me += pow((std::abs(data_me[index_reg]) - mu_me_gaussian), 2.0) * ret_weight_Reg[cpt_neigh];
		variance_proj += pow((std::abs(data_proj[index_reg]) - mu_proj_gaussian), 2.0) * ret_weight_Ref[cpt_neigh];
		covariance +=
			((std::abs(data_me[index_reg]) - mu_me_gaussian) * (std::abs(data_proj[index_reg]) - mu_proj_gaussian)) *
			ret_weight_Reg[cpt_neigh];

	}
	double standard_dev_me = 0.0;
	double standard_dev_proj = 0.0;

	standard_dev_me = std::sqrt(variance_me / sum_distances_me);
	standard_dev_proj = std::sqrt(variance_proj / sum_distances_proj);

	covariance = covariance / sum_distances_me;


	luminance = std::abs(mu_me_gaussian - mu_proj_gaussian) / (std::max(mu_me_gaussian, mu_proj_gaussian) + constant_value);
	contrast = std::abs(standard_dev_me - standard_dev_proj) / (std::max(standard_dev_me, standard_dev_proj) + constant_value);
	structure =
		std::abs(standard_dev_me * standard_dev_proj - covariance) / ((standard_dev_me * standard_dev_proj) + constant_value);

	if (structure > 1.0) structure = 1.0; //Clamping structure field


	// Values are stored in vectors
	lightness_field[index_array] = luminance;
	contrast_field[index_array] = contrast;
	structure_field[index_array] = structure;

}






/**
* \fn void compute_statistics(double radius, const double maxDim, PointSet& regptset, KdTree& m_kdtree2,
	std::vector<Point>& projectedpointsOnRef, std::vector<Point>& projectedpointsOnMe,
	std::vector<double>& meancurvaturesProj, std::vector<double>& meancurvaturesMe,
	std::vector<double>& geom_lightness_field, std::vector<double>& geom_contrast_field,
	std::vector<double>& geom_structure_field, double& PCQM, const double radius_factor,
	std::vector<double>& tab_lstar_me, std::vector<double>& tab_lstar_proj,
	std::vector<double>& tab_astar_me, std::vector<double>& tab_astar_proj, 
	std::vector<double>& tab_bstar_me, std::vector<double>& tab_bstar_proj, 
	std::vector<double>& tab_chroma_me, std::vector<double>& tab_chroma_proj, 
	std::vector<double>& tab_hue_me, std::vector<double>& tab_hue_proj, 
	std::vector<double>& color_lightness_field, std::vector<double>& color_chroma_field,
	std::vector<double>& color_hue_field, std::vector<double>& color_contrast_field, std::vector<double>& color_structure_field,
	 int threshold_knnsearch)
* \brief Compute PCQM and write features with results to file
*/
void compute_statistics(double radius, const double maxDim, PointSet& regptset, KdTree& m_kdtree2,
	std::vector<Point>& projectedpointsOnRef, std::vector<Point>& projectedpointsOnMe,
	std::vector<double>& meancurvaturesProj, std::vector<double>& meancurvaturesMe,
	std::vector<double>& geom_lightness_field, std::vector<double>& geom_contrast_field,
	std::vector<double>& geom_structure_field, double& PCQM, const double radius_factor,
	std::vector<double>& tab_lstar_me, std::vector<double>& tab_lstar_proj,
	std::vector<double>& tab_astar_me, std::vector<double>& tab_astar_proj, 
	std::vector<double>& tab_bstar_me, std::vector<double>& tab_bstar_proj, 
	std::vector<double>& tab_chroma_me, std::vector<double>& tab_chroma_proj, 
	std::vector<double>& tab_hue_me, std::vector<double>& tab_hue_proj, 
	std::vector<double>& color_lightness_field, std::vector<double>& color_chroma_field,
	std::vector<double>& color_hue_field, std::vector<double>& color_contrast_field, std::vector<double>& color_structure_field,
	 int threshold_knnsearch)
{
	// Computing PCQM
	std::cout << "Computing PCQM" << std::endl;
	double f3 = 0.0;
	double f4 = 0.0;
	double f6 = 0.0;

	nanoflann::SearchParams params;
	params.sorted = false;

	//FEATURES COMPUTATION
#pragma omp parallel for
	for (int i = 0; i < regptset.npts(); i++) {

		double search_radius_neighborhood = static_cast<double>(radius * radius_factor);

		Point origin = regptset.pts[i];

		// Structure containing indexes and distances returned from KNN
		std::vector<std::pair<size_t, double>> ret_matches_Reg;

		// Distances
		std::vector<double> ret_distance_Reg;
		std::vector<double> ret_distance_Ref;

		// Weights
		std::vector<double> ret_weight_Reg;
		std::vector<double> ret_weight_Ref;
		double sum_distances_me = 0.0;
		double sum_distances_proj = 0.0;


		double query_pt[3] = { origin.x, origin.y, origin.z };

		// Looking for neighbors of REGISTERED to compute statistics
		const size_t nMatches_Reg =
			m_kdtree2.radiusSearch(&query_pt[0], std::pow(search_radius_neighborhood, 2.0), ret_matches_Reg, params);

		double debug_variance = search_radius_neighborhood / 2.0;


		
		for (size_t cpt_reg = 0; cpt_reg < nMatches_Reg; cpt_reg++) {
			// Get distances for REGISTERED
			ret_distance_Reg.push_back(std::sqrt(ret_matches_Reg[cpt_reg].second));

			// manually computing distance REFERENCE
			Point p_orig_proj = projectedpointsOnRef[i];
			Point p_neigh_proj = projectedpointsOnRef[ret_matches_Reg[cpt_reg].first];


			ret_distance_Ref.push_back(compute_distance(p_orig_proj, p_neigh_proj));

			// Weight computation
			double wi1 =
				1.0 / debug_variance / sqrt(2 * 3.141592) *
				exp(-(ret_distance_Reg[cpt_reg] * ret_distance_Reg[cpt_reg]) / 2.0 / debug_variance / debug_variance);
			double wi2 =
				1.0 / debug_variance / sqrt(2 * 3.141592) *
				exp(-(ret_distance_Ref[cpt_reg] * ret_distance_Ref[cpt_reg]) / 2.0 / debug_variance / debug_variance);


			ret_weight_Reg.push_back(wi1);
			ret_weight_Ref.push_back(wi2);

			// Sum the weight
			sum_distances_me += ret_weight_Reg[cpt_reg];
			sum_distances_proj += ret_weight_Ref[cpt_reg];
		}


		double alpha_1 = 0.0448;
		double constant_curvature = 1.0;
		double constant_1 = 0.002;
		double constant_2 = 0.1;
		double constant_3 = 0.1;
		double constant_4 = 0.002;
		double constant_5 = 0.008;

		compute_geometric_feature(i, nMatches_Reg, meancurvaturesProj, meancurvaturesMe,
			ret_weight_Reg, ret_weight_Ref,
			geom_lightness_field, geom_contrast_field, geom_structure_field,
			ret_matches_Reg, sum_distances_me, sum_distances_proj, constant_curvature);

		color_lightness_field[i] = compute_color_feature(i, nMatches_Reg, tab_lstar_proj, tab_lstar_me, ret_weight_Reg, ret_weight_Ref,
			ret_matches_Reg, sum_distances_me, sum_distances_proj, 1, constant_1);

		color_contrast_field[i] =
			compute_color_feature(i, nMatches_Reg, tab_lstar_proj, tab_lstar_me, ret_weight_Reg, ret_weight_Ref,
				ret_matches_Reg, sum_distances_me, sum_distances_proj, 2, constant_2);

		color_structure_field[i] =
			compute_color_feature(i, nMatches_Reg, tab_lstar_proj, tab_lstar_me, ret_weight_Reg, ret_weight_Ref,
				ret_matches_Reg, sum_distances_me, sum_distances_proj, 3, constant_3);

		color_chroma_field[i] =
			compute_color_feature(i, nMatches_Reg, tab_chroma_proj, tab_chroma_me, ret_weight_Reg, ret_weight_Ref,
				ret_matches_Reg, sum_distances_me, sum_distances_proj, 1, constant_4);

		color_hue_field[i] = compute_color_feature(i, nMatches_Reg, tab_hue_proj, tab_hue_me, ret_weight_Reg, ret_weight_Ref,
			ret_matches_Reg, sum_distances_me, sum_distances_proj, 1, constant_5);

		//PCQM RELATED FEATURES
#pragma omp atomic 
		f3 += geom_structure_field[i];
#pragma omp atomic 
		f4 += color_lightness_field[i];
#pragma omp atomic
		f6 += color_structure_field[i];
		

	}

	double size_tab = (double)color_lightness_field.size();

	//PCQM Formula
	PCQM = f3/ size_tab * 0.0057 + (1.0 - f4/ size_tab) * 0.9771 + (1.0 - f6/ size_tab) * 0.0172;

	//Write features and PCQM to CSV
	write_mean_features(geom_lightness_field, geom_contrast_field, geom_structure_field, color_lightness_field, color_contrast_field, color_structure_field,
		color_chroma_field, color_hue_field, global_regfile, global_reffile, PCQM ,"features_extracted.csv");

	

}


/**
* \fn main(int argc, char** argv)
* \brief Entry point of the program
* \return EXIT_SUCCESS if the code executed successfuly.
*/
int main(int argc, char** argv) {



	// Keep console open if false
	bool fast_quit = false;


	// PCQM params
	double RadiusCurvature = 0.004;
	int threshold_knnsearch = 20;
	double radius_factor = 2.0;


	if (argc < 3) {
		std::cerr << "Usage: " << "REFERENCE.ply DISTORTED.ply (Options) (--fastquit || -fq) (-r radius)(-rx factor) "
			"(-knn nb_point)"
			<< std::endl;
		return -1;
	}

	//As the code compute projection of regfile on reffile we have to inverse input to match paper description (R onto D)
	std::string reffile = argv[2];
	std::string regfile = argv[1];

	std::string inv_check = "";
	for (int i = 3; i < argc; ++i) {

		if (std::string(argv[i]) == "--fastquit" || std::string(argv[i]) == "-fq") {

			fast_quit = true;
			std::cout << "fast_quit set to : " << fast_quit << std::endl;
		}

		if (std::string(argv[i]) == "-r") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				i++;              // Increment 'i' so we don't get the argument as the next argv[i].
				RadiusCurvature = std::stod(std::string(argv[i]));
				std::cout << "Radius set to : " << RadiusCurvature << std::endl;
			}
			else {
				std::cerr << "-r option requires one argument." << std::endl;
				return -1;
			}
		}

		if (std::string(argv[i]) == "-knn") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				i++;
				threshold_knnsearch =
					std::stoi(std::string(argv[i])); // Increment 'i' so we don't get the argument as the next argv[i].

				std::cout << "threshold_knnsearch set to : " << threshold_knnsearch << std::endl;
			}
			else {
				std::cerr << "-knn option requires one argument." << std::endl;
				return -1;
			}
		}

		if (std::string(argv[i]) == "-rx") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				i++;
				radius_factor =
					std::stod(std::string(argv[i])); // Increment 'i' so we don't get the argument as the next argv[i].

				std::cout << "radius_factor set to : " << radius_factor << std::endl;
			}
			else {
				std::cerr << "-rx option requires one argument." << std::endl;
				return -1;
			}
		}
	}

	

	global_regfile = remove_extension(regfile);
	global_reffile = remove_extension(reffile);


	std::cout << "Input reference point set file:  " << reffile << std::endl;
	std::cout << "Input registered point set file:  " << regfile << std::endl;

	//Create point set structure
	PointSet refptset;
	PointSet regptset;

	//Load points clouds (ply files)
	refptset.read_ply_file(reffile, true);
	regptset.read_ply_file(regfile, true);

	//Build KDtree
	KdTree m_kdtree(3, refptset, KDTreeSingleIndexAdaptorParams(10));
	m_kdtree.buildIndex();
	KdTree m_kdtree2(3, regptset, KDTreeSingleIndexAdaptorParams(10));
	m_kdtree2.buildIndex();

	//Color interpolation structures 
	std::vector<double> init_grid_L;
	std::vector<double> grid_L;
	std::vector<std::vector<std::pair<double, double>>> init_grid_AB;

	initMatLABCH(init_grid_L, grid_L, init_grid_AB);



	std::vector<Point> projectedpointsOnRef;
	std::vector<Point> projectedpointsOnMe;
	projectedpointsOnRef.assign(regptset.npts(), Point());
	projectedpointsOnMe.assign(regptset.npts(), Point());

	std::vector<double> meancurvaturesProj;
	std::vector<double> meancurvaturesMe;
	std::vector<double> geom_lightness_field;
	std::vector<double> geom_contrast_field;
	std::vector<double> geom_structure_field;


	std::vector<double> tab_radius;

	meancurvaturesProj.assign(regptset.npts(), 0.0);
	meancurvaturesMe.assign(regptset.npts(), 0.0);
	

	geom_lightness_field.assign(regptset.npts(), 0.0);
	geom_contrast_field.assign(regptset.npts(), 0.0);
	geom_structure_field.assign(regptset.npts(), 0.0);

	tab_radius.assign(regptset.npts(), 0.0);

	//Local color features Storage
	std::vector<double> color_lightness_field, color_chroma_field, color_hue_field, color_contrast_field, color_structure_field;


	//features init
	color_lightness_field.assign(regptset.npts(), 0.0);
	color_chroma_field.assign(regptset.npts(), 0.0);
	color_hue_field.assign(regptset.npts(), 0.0);
	color_contrast_field.assign(regptset.npts(), 0.0);
	color_structure_field.assign(regptset.npts(), 0.0);


	//Point Color data
	std::vector<double> tab_lstar_me, tab_lstar_proj,
						tab_astar_me, tab_astar_proj,
						tab_bstar_me, tab_bstar_proj,
						tab_chroma_me, tab_chroma_proj,
						tab_hue_me, tab_hue_proj;


	tab_lstar_me.assign(regptset.npts(), 0.0);
	tab_lstar_proj.assign(regptset.npts(), 0.0);
	tab_astar_me.assign(regptset.npts(), 0.0);
	tab_astar_proj.assign(regptset.npts(), 0.0);
	tab_bstar_me.assign(regptset.npts(), 0.0);
	tab_bstar_proj.assign(regptset.npts(), 0.0);
	tab_chroma_me.assign(regptset.npts(), 0.0);
	tab_chroma_proj.assign(regptset.npts(), 0.0);
	tab_hue_me.assign(regptset.npts(), 0.0);
	tab_hue_proj.assign(regptset.npts(), 0.0);


	// computation of the bounding box
	double rangeX = std::abs(regptset.xmin) + std::abs(regptset.xmax);
	double rangeY = std::abs(regptset.ymin) + std::abs(regptset.ymax);
	double rangeZ = std::abs(regptset.zmin) + std::abs(regptset.zmax);


	// PCQM params
	double maxDim = std::max(rangeX, std::max(rangeY, rangeZ));
	double radius = RadiusCurvature * maxDim;
	
	nanoflann::SearchParams params;
	params.sorted = true;
	std::cout << "Start curvature computation and color projection" << std::endl;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel for
	for (int i = 0; i < regptset.npts(); ++i) {
		Point origin = regptset.pts[i];

		double H = 0;
		double K = 0;
		Point proj;
		Point projOnMe;

		double query_pt[3] = { origin.x, origin.y, origin.z };

		// KNN_SEARCH
		// Indexes
		std::vector<size_t> ret_index_ref_init(threshold_knnsearch);
		std::vector<size_t> ret_index_reg_init(threshold_knnsearch);
		// Distances
		std::vector<double> out_dist_sqr_ref(threshold_knnsearch);
		std::vector<double> out_dist_sqr_reg(threshold_knnsearch);
		std::vector<double>::iterator result1;
		std::vector<double>::iterator result2;


		// Safe knnSearch on REFERENCE and REGISTER
		m_kdtree.knnSearch(&query_pt[0], threshold_knnsearch, &ret_index_ref_init[0], &out_dist_sqr_ref[0]);
		m_kdtree2.knnSearch(&query_pt[0], threshold_knnsearch, &ret_index_reg_init[0], &out_dist_sqr_reg[0]);
		
		params.sorted = true; //Neighbors are sorted distance wise
		//Radius search results container
		std::vector<std::pair<size_t, double>> ret_matches_Ref;
		std::vector<std::pair<size_t, double>> ret_matches_Me;

		//Radius search
		size_t nMatches_Ref = m_kdtree.radiusSearch(&query_pt[0], radius * radius, ret_matches_Ref, params);
		size_t nMatches_Reg = m_kdtree2.radiusSearch(&query_pt[0], radius * radius, ret_matches_Me, params);

		std::vector<size_t> ret_index_ref(nMatches_Ref);
		std::vector<size_t> ret_index_reg(nMatches_Reg);


		for (size_t cpt_ref = 0; cpt_ref < nMatches_Ref; cpt_ref++) {
			ret_index_ref[cpt_ref] = ret_matches_Ref[cpt_ref].first;
		}

		for (size_t cpt_reg = 0; cpt_reg < nMatches_Reg; cpt_reg++) {
			ret_index_reg[cpt_reg] = ret_matches_Me[cpt_reg].first;
		}

		//////////////////////// Point projection and Curvature computation ////////////////////////

		// We used the neihgborhood computed with a KNN search to compute PROJECTION
		double dummy_double;
		computeProjectionAndCurvature(origin, refptset.pts, ret_index_ref_init, proj, dummy_double);
		computeProjectionAndCurvature(origin, regptset.pts, ret_index_reg_init, projOnMe, dummy_double);


		// We used the neihgborhood computed with a radius search to compute CURVATURE
		Point dummy_point;
		computeProjectionAndCurvature(origin, refptset.pts, ret_index_ref, dummy_point, H);
		computeProjectionAndCurvature(origin, regptset.pts, ret_index_reg, dummy_point, K);


		Point closest_point_from_origin_ref = refptset.pts[ret_index_ref_init[0]];
		Point closest_point_from_origin_reg = regptset.pts[ret_index_reg_init[0]];

		double distance_ori_proj = compute_distance(origin, proj);
		double distance_ori_closest = compute_distance(origin, closest_point_from_origin_ref);


		//Normalizing curvature data with the bounding box
		meancurvaturesProj[i] = std::abs(H) * maxDim;
		meancurvaturesMe[i] = std::abs(K) * maxDim;

		projectedpointsOnRef[i] = (distance_ori_proj <= distance_ori_closest ? proj : closest_point_from_origin_ref);
		projectedpointsOnMe[i] =
			(compute_distance(origin, projOnMe) <= compute_distance(origin, closest_point_from_origin_reg)
				? projOnMe
				: closest_point_from_origin_reg);


		////////////////////////////////////// LAB projection //////////////////////////////////////


		double L_star = 0.0;
		double A_star = 0.0;
		double B_star = 0.0;
		
		RGBtoLab(regptset.pts[i].r, regptset.pts[i].g, regptset.pts[i].b, L_star, A_star, B_star);

		L_star = interpolate1_process(init_grid_L, grid_L, L_star);

		std::pair<double, double> me_a_b = interpolate2_process(init_grid_AB, A_star, B_star);
		
		A_star = me_a_b.first;
		B_star = me_a_b.second;

		tab_lstar_me[i] = L_star;
		tab_astar_me[i] = A_star;
		tab_bstar_me[i] = B_star;


		//refptset.pts[ret_index_ref_init[0]] is the nearest neighboor
		RGBtoLab(refptset.pts[ret_index_ref_init[0]].r, refptset.pts[ret_index_ref_init[0]].g,
			refptset.pts[ret_index_ref_init[0]].b, L_star, A_star, B_star); 


		L_star = interpolate1_process(init_grid_L, grid_L, L_star);
		std::pair<double, double> proj_a_b = interpolate2_process(init_grid_AB, A_star, B_star);
		A_star = proj_a_b.first;
		B_star = proj_a_b.second;

		tab_lstar_proj[i] = L_star;
		tab_astar_proj[i] = A_star;
		tab_bstar_proj[i] = B_star;


		////////////////////////////////////// CHROMA projection //////////////////////////////////////

		tab_chroma_me[i] =
			std::sqrt(tab_astar_me[i] * tab_astar_me[i] + tab_bstar_me[i] * tab_bstar_me[i]);
		tab_chroma_proj[i] = std::sqrt(tab_astar_proj[i] * tab_astar_proj[i] +
			tab_bstar_proj[i] * tab_bstar_proj[i]);


		////////////////////////////////////// HUE projection //////////////////////////////////////

		double delta_HUE =
			(tab_astar_me[i] - tab_astar_proj[i]) * (tab_astar_me[i] - tab_astar_proj[i]) +
			(tab_bstar_me[i] - tab_bstar_proj[i]) * (tab_bstar_me[i] - tab_bstar_proj[i]) -
			(tab_chroma_me[i] - tab_chroma_proj[i]) * (tab_chroma_me[i] - tab_chroma_proj[i]);

		tab_hue_me[i] = (delta_HUE > 0) ? std::sqrt(delta_HUE) : 0.0;
		tab_hue_proj[i] = 0.0;// (Zero here because DeltaHue is already computed above)

		////////////////////////////////////////////////////////////////////////////////////////////


	}

	std::chrono::steady_clock::time_point end_projection = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_projection - begin).count() / 1000000.0 << " sec elpased"
		<< std::endl;

	std::cout << "Start compute_statistics" << std::endl;

	double PCQM = 0.0;

	compute_statistics(radius, maxDim, regptset, m_kdtree2, projectedpointsOnRef, projectedpointsOnMe,
		meancurvaturesProj, meancurvaturesMe, geom_lightness_field, geom_contrast_field, geom_structure_field, PCQM, radius_factor,
		tab_lstar_me, tab_lstar_proj, tab_astar_me, tab_astar_proj, tab_bstar_me, tab_bstar_proj,
		tab_chroma_me, tab_chroma_proj, tab_hue_me, tab_hue_proj, color_lightness_field, color_chroma_field, color_hue_field, color_contrast_field,
		color_structure_field, threshold_knnsearch);


	std::cout << "PCQM value is : " << PCQM << std::endl;
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " sec elpased"
		<< std::endl;

	// if (!fast_quit) {
	// 	std::cout << "Press enter to exit" << std::endl;
	// 	std::getchar();
	// }
	return EXIT_SUCCESS;
}

