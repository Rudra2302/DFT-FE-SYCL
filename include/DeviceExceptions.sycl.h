// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
#ifndef dftfeDeviceExceptions_syclh
#define dftfeDeviceExceptions_syclh

#define DEVICE_API_CHECK(event)                                   \
{                                                                 \
    try{                                                          \
        event.wait();                                             \
    }                                                             \
    catch(const dftfe::utils::deviceError_t &e){                  \
        std::cerr<<"SYCL error on or before line number"<<        \
                     __LINE__ <<" in file: "<<                    \
                     __FILE__ <<". Error code: "<<                \
                     e.what()<<".\n";                             \
    }                                                             \
}

#define DEVICEBLAS_API_CHECK(event)                               \
{                                                                 \
    event.wait();                                                 \
    if(event.get_info<dftfe::utils::deviceBlasStatus_t>()         \
                != dftfe::utils::deviceSuccess){                  \
        std::cerr<<"SYCL error on or before line number"<<        \
                     __LINE__ <<" in file: "<<                    \
                     __FILE__ <<".\n";                            \ 
    }                                                             \
}

#endif // dftfeDeviceExceptions_syclh
