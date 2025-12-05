
## PCQM installation guide (Linux and Windows)

## Dependencies
Mandatory dependencies :
 - CMake (External download)
 - Eigen 3 (Included)
 - Nanoflann (Included)
 - Tinyply (Included)

## Windows installation

### Building stage

 - Get PCQM source code using your favourite Git client

 - Run cmake-gui.exe

 - Choose 'Visual Studio 16 2019 Win64' as the compiler version

 - Where is the source code = ".../PCQM/"

 - Where to build the binaries = ".../PCQM/build"

 - Click "Configure"

 - Click "Generate"

 - Open ".../PCQM/build/PCQM.sln" solution with MSVC 2019, select 'Release' mode, then select PCQM as the starting project.


## Documentation

The documentation can be build using [Doxygen](http://www.doxygen.nl/).

## Known issues