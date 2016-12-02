#ifndef CL_HELPER_H
#define CL_HELPER_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

struct no_platform_exception: std::runtime_error
{
        no_platform_exception() :
                        std::runtime_error("No compute platform found")
        {
        }
};

struct no_device_exception: std::runtime_error
{
        no_device_exception() :
                        std::runtime_error("No compute device found")
        {
        }
};

namespace fc_cl {

using Platforms = std::vector<cl::Platform>;
using Devices = std::vector<cl::Device>;

void init(Platforms &platforms, Devices &devices, cl::Context &context)
{
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
            throw(no_platform_exception());
    }

    // Get first available GPU device which supports double precision.

    for(auto p = platforms.begin(); devices.empty() && p != platforms.end(); p++) {
            std::vector<cl::Device> pldev;

            try {
                    p->getDevices(CL_DEVICE_TYPE_CPU, &pldev);

                    for(auto d = pldev.begin(); devices.empty() && d != pldev.end(); d++) {
                            if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

                            std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

                            if (
                                            ext.find("cl_khr_fp64") == std::string::npos &&
                                            ext.find("cl_amd_fp64") == std::string::npos
                                            ) continue;

                            devices.push_back(*d);
                            context = cl::Context(devices);
                    }
            } catch(...) {
                    devices.clear();
            }
    }

    if (devices.empty()) {
        throw(no_device_exception());
    }

    std::cout << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
} // init
} // namespace flexcore_cl
#endif // CL_HELPER_H
