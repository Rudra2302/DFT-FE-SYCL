#!/bin/bash
# script to setup and build DFT-FE.

set -e
set -o pipefail

if [ -s CMakeLists.txt ]; then
    echo "This script must be run from the build directory!"
    exit 1
fi

# Path to project source
SRC=`dirname $0` # location of source directory

########################################################################
#Provide paths below for external libraries, compiler options and flags,
# and optimization flag

#Paths for required external libraries
dealiiDir="/home/u4bd86b5d94022cecfc81890caa7faa8/dft/dealii_in/lib/cmake/deal.II"
alglibDir="/home/u4bd86b5d94022cecfc81890caa7faa8/dft/alglib-cpp/src"
libxcDir="/home/u4bd86b5d94022cecfc81890caa7faa8/dft/libxc"
spglibDir="/home/u4bd86b5d94022cecfc81890caa7faa8/dft/spg"
xmlIncludeDir="/home/u4bd86b5d94022cecfc81890caa7faa8/dft/libxml/include/libxml2"
xmlLibDir="/home/u4bd86b5d94022cecfc81890caa7faa8/dft/libxml/lib"
ELPA_PATH="/home/u4bd86b5d94022cecfc81890caa7faa8/dft/elpa_install"
dftdpath=""
numdiffdir=""


#Paths for optional external libraries
# path for NCCL/RCCL libraries
DCCL_PATH=""
mdiPath=""
torchDir=""

#Toggle GPU compilation
withGPU=ON
gpuLang="sycl"     # Use "cuda"/"hip"/"sycl"
gpuVendor="intel" # Use "nvidia/amd/intel"
withGPUAwareMPI=OFF #Please use this option with care
                   #Only use if the machine supports 
                   #device aware MPI and is profiled
                   #to be fast

#Option to link to DCCL library (Only for GPU compilation)
withDCCL=OFF
withMDI=OFF
withTorch=OFF
withCustomizedDealii=OFF

#Compiler options and flags
# cxx_compiler=/opt/intel/oneapi/compiler/2024.1/bin/icpx  #sets DCMAKE_CXX_COMPILER
cxx_compiler=/opt/intel/oneapi/mpi/2021.12/bin/mpiicpx
cxx_flags="-fPIC -fsycl -DMKL_ILP64  -L${MKLROOT}/lib -I/opt/intel/oneapi/mkl/2024.1/include/oneapi -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64 -lmkl_sequential -lmkl_core -lsycl -lpthread -lm -ldl" #sets DCMAKE_CXX_FLAGS
cxx_flagsRelease="-O2" #sets DCMAKE_CXX_FLAGS_RELEASE
# device_flags="-fPIC -fsycl -DMKL_ILP64  -L${MKLROOT}/lib  -qmkl=sequential"
# device_flags="-fPIC -fsycl -DMKL_ILP64 -L${MKLROOT}/lib -lmkl_sycl -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_lp64 -lmkl_sequential -lmkl_core -lsycl -lpthread -lm -ldl" # set DCMAKE_CXX_CUDA/HIP_FLAGS 
device_flags="-fPIC -fsycl -DMKL_ILP64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmpi -lmpi_ilp64 -lmpicxx -lirc -lintlc -lmkl_lapack95_ilp64 -lmkl_blacs_intelmpi_ilp64 -lmkl_scalapack_ilp64 -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lpthread -lm -ldl"
                           #(only applicable for withGPU=ON)
device_architectures="" # set DCMAKE_CXX_CUDA/HIP_ARCHITECTURES 
                           #(only applicable for withGPU=ON)


#Option to compile with default or higher order quadrature for storing pseudopotential data
#ON is recommended for MD simulations with hard pseudopotentials
withHigherQuadPSP=OFF

# build type: "Release" or "Debug"
build_type=Release

testing=OFF
minimal_compile=ON
###########################################################################
#Usually, no changes are needed below this line
#

#if [[ x"$build_type" == x"Release" ]]; then
#  c_flags="$c_flagsRelease"
#  cxx_flags="$c_flagsRelease"
#else
#fi
out=`echo "$build_type" | tr '[:upper:]' '[:lower:]'`

function cmake_configure() {
  if [ "$gpuLang" = "cuda" ]; then
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath -DWITH_TORCH=$withTorch -DTORCH_DIR=$torchDir\
    -DWITH_CUSTOMIZED_DEALII=$withCustomizedDealii\
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH;$dftdpath;$numdiffdir"\
    -DWITH_COMPLEX=$withComplex -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_CUDA_FLAGS="$device_flags" -DCMAKE_CUDA_ARCHITECTURES="$device_architectures"\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  elif [ "$gpuLang" = "hip" ]; then
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath -DWITH_TORCH=$withTorch -DTORCH_DIR=$torchDir\
    -DWITH_CUSTOMIZED_DEALII=$withCustomizedDealii\
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH;$dftdpath;$numdiffdir"\
    -DWITH_COMPLEX=$withComplex -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_HIP_FLAGS="$device_flags" -DCMAKE_HIP_ARCHITECTURES="$device_architectures"\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
    elif [ "$gpuLang" = "sycl" ]; then
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$device_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath -DWITH_TORCH=$withTorch -DTORCH_DIR=$torchDir\
    -DWITH_CUSTOMIZED_DEALII=$withCustomizedDealii\
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH;$dftdpath;$numdiffdir"\
    -DWITH_COMPLEX=$withComplex -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  else
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath -DWITH_TORCH=$withTorch -DTORCH_DIR=$torchDir\
    -DWITH_CUSTOMIZED_DEALII=$withCustomizedDealii\
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH;$dftdpath;$numdiffdir"\
    -DWITH_COMPLEX=$withComplex \
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  fi
}

RCol='\e[0m'
Blu='\e[0;34m';
if [ -d "$out" ]; then # build directory exists
    echo -e "${Blu}$out directory already present${RCol}"
else
    rm -rf "$out"
    echo -e "${Blu}Creating $out ${RCol}"
    mkdir -p "$out"
fi

cd $out

withComplex=OFF
echo -e "${Blu}Building Real executable in $build_type mode...${RCol}"
mkdir -p real && cd real
cmake_configure "$SRC" && make -j8
cd ..

withComplex=ON
echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
mkdir -p complex && cd complex
cmake_configure "$SRC" && make -j8
cd ..


    # -DCMAKE_EXE_LINKER_FLAGS="-fsycl -DMKL_ILP64 -L${MKLROOT}/lib -lmkl_blacs_intelmpi_ilp64 -lmkl_scalapack_ilp64 -Wl,--no-as-needed -lmkl_gf_ilp64 -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lpthread -lm -ldl -I${MKLROOT}/include -L/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin -lirc -limf -lsvml -lipgo" \

echo -e "${Blu}Build complete.${RCol}"
