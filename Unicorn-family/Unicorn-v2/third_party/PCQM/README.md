# PCQM
---

PCQM, an objective metric for visual quality assessment of colored 3D point clouds.  
This full-reference metric is an optimally-weighted linear combination of geometry-based and color-based features. 

This project is the implementation of our paper "[PCQM: A Full-Reference Quality Metric for Colored 3D Point Clouds](https://hal.archives-ouvertes.fr/hal-02529668v1)"

Here is the list of the parameters : 

* -r   Set a radius for the construction of the neighborhood. As the bounding box is already computed with this program, use proposed value. (default : 0.004) 
* -knn Set the number of points used for the quadric surface construction. (default : 20)
* -rx  Set a radius factor for the statistic computation. (default : 2.0)
* -fq  Keep the console open for single testing

Please verify that those files are in the build folder at the post generation step.  
L_data.txt 
RegularGrid_0_0_1.txt  
RegularGrid_0_0_2.txt  
RegularGridInit_0_0_1.txt  
RegularGridInit_0_0_2.txt  



To use the compiled binary  :

```
Windows : 
	PCQM.exe reference_objet.ply registered_object.ply -r 0.004 -knn 20 -rx 2.0	  
	
Linux : 
	./PCQM reference_objet.ply registered_object.ply -r 0.004 -knn 20 -rx 2.0
		
```

Input files can be binaries or plain text in the .ply format with the following header structure : 

```
ply
format ascii 1.0
element vertex 686061 {number of points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
3.64728 1.3294 -11.1118 124 109 86
3.60579 1.3331 -11.1118 124 110 89
3.5726 1.3294 -11.1118 133 116 84
3.55601 1.3331 -11.1118 130 112 82
3.63898 1.3294 -11.1118 124 110 87
3.5975 1.3331 -11.1118 126 112 89
3.56431 1.3331 -11.1118 133 116 84
3.54771 1.33679 -11.1118 128 110 82
3.61409 1.3294 -11.1118 123 107 86 
3.65557 1.3331 -11.1118 121 107 84 
3.5809 1.3294 -11.1118 130 114 86 
3.63068 1.3294 -11.1118 123 107 86 
3.55601 1.34419 -11.1118 130 112 82 
3.5892 1.3331 -11.1118 128 112 87 
3.65557 1.33679 -11.1118 123 107 84 
3.55601 1.34789 -11.1118 130 114 84 
3.65557 1.34789 -11.1118 124 110 87 
3.5975 1.33679 -11.1118 126 110 87 
3.56431 1.33679 -11.1118 133 116 84 
3.63898 1.33679 -11.1145 126 110 89 
3.55601 1.33679 -11.1118 130 112 82 
3.60579 1.33679 -11.1118 124 110 87 
3.5726 1.33679 -11.1145 133 117 86 
```

Output file is a csv "features_extracted.csv" containing :
```
reffile;regfile;F1;F2;F3;F4;F5;F6;F7;F8;PCQM
reference_objet;registered_object;0.287261;0.883497;0.317393;0.00135188;0.117714;0.308993;0.00394852;0.0387242;0.00844475
```

The documentation can be build using [Doxygen](http://www.doxygen.nl/).

---


Reference : PCQM: A Full-Reference Quality Metric for Colored 3D Point Clouds [Gabriel Meynet](https://liris.cnrs.fr/page-membre/gabriel-meynet/),[Yana Nehme](https://liris.cnrs.fr/page-membre/yana-nehme/), [Julie Digne](https://perso.liris.cnrs.fr/julie.digne/), [Guillaume Lavoué](https://perso.liris.cnrs.fr/guillaume.lavoue/), International Conference on Quality of Multimedia Experience ([QoMEX](http://qomex2020.ie/)), Full paper, Athlone, Ireland, 2020 

Development of this software is part of the [PISCo](https://projet.liris.cnrs.fr/pisco/) research project. 
