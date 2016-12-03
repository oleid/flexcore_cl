#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include <flexcore/infrastructure.hpp>
#include <flexcore/extended/nodes/region_worker_node.hpp>

#include "cl_helper.h"

struct artificial_input : fc::region_worker_node
{
	   artificial_input(const fc::node_args& args)
		   : region_worker_node([this]() { work(); }, args)
			   , out_data(this)
	   {}

	   void work()
	   {
			std::vector<double> demo_data(10, ++counter);

			std::cerr << "Sending array for counter " << counter << "\n";
			out_data.fire(demo_data);
			std::cerr << "Done sending.\n";
	   }
	   fc::event_source<std::vector<double>> out_data;
	   int counter = 0;
};

struct computation_consumer : fc::owning_base_node
{
	   computation_consumer(const fc::node_args& args)
			: owning_base_node(args)
			, in_result(this, [this](std::vector<double> data) {
							retrieve_data(std::move(data));  }
			  )
	   {}

	   void retrieve_data (std::vector<double> data)
	   {
			std::cerr << "Got result of size " << data.size() << ": ";
			std::copy(data.begin(), data.end(), std::ostream_iterator<double>(std::cerr, " "));
			std::cerr << "\n";
	   }

	   fc::event_sink<std::vector<double>> in_result;
};


struct cl_processing : fc::owning_base_node
{
	   cl_processing(fc_cl::Platforms &,
					 fc_cl::Devices &devices,
					 cl::Context &ctx,
					 const fc::node_args& args)
			: owning_base_node(args)
			, in_data(this, [this](std::vector<double> data) {
				process_data(std::move(data));  }
			  )
			, out_data(this)
			, context(ctx)                  // shallow copy of context
			, queue(context, devices[0])	// Create command queue.
	   {
		   const size_t N = 1 << 20;

		   // Load program from source
		   std::ifstream kernel_file(KERNEL_DIR "/kernels.cl");
		   std::stringstream kernel_buffer;
		   kernel_buffer << kernel_file.rdbuf();

		   const std::string source = kernel_buffer.str();

		   // Compile OpenCL program for found device.
		   cl::Program program(context, source);

		   try {
			   program.build(devices, "-cl-std=CL2.0");
		   } catch (const cl::Error& e) {
			   std::cerr
					   << "OpenCL compilation error" << std::endl
					   << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
					   << e.what()
					   << std::endl;
			   throw(e);
		   }

		   kernel_demo1 =  cl::Kernel(program, "add");
	   }

	   void process_data(std::vector<double> data)
	   {
		   // Prepare input data.
		   a = std::vector<double>(data);
		   b = std::vector<double>(data);
		   c = std::vector<double>(data.size());

		   // Allocate device buffers and transfer input data to device.
		   cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						a.size() * sizeof(double), a.data());

		   cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						b.size() * sizeof(double), b.data());

		   cl::Buffer C(context, CL_MEM_READ_WRITE,
						c.size() * sizeof(double));

		   // Set kernel parameters.
		   kernel_demo1.setArg(0, static_cast<cl_ulong>(a.size() ));
		   kernel_demo1.setArg(1, A);
		   kernel_demo1.setArg(2, B);
		   kernel_demo1.setArg(3, C);

		   // Launch kernel on the compute device.
		   queue.enqueueNDRangeKernel(kernel_demo1, cl::NullRange, a.size(), cl::NullRange);

		   // Get result back to host.
		   queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());

		   out_data.fire(c);
	   }

	   fc::event_sink<std::vector<double>> in_data;
	   fc::event_source<std::vector<double>> out_data;

	   // OpenCL
	   cl::Context context;
	   cl::CommandQueue queue;
	   cl::Kernel kernel_demo1;

	   std::vector<double> a;
	   std::vector<double> b;
	   std::vector<double> c;
};



int main() {

	fc::infrastructure infra;

	auto region_cpu = infra.add_region("region-cpu", fc::thread::cycle_control::fast_tick);
	auto region_cl = infra.add_region("region-cl", fc::thread::cycle_control::fast_tick);

	artificial_input &generator = infra.node_owner().make_child_named<artificial_input>(region_cpu,"artificial_input");
	computation_consumer &consumer = infra.node_owner().make_child_named<computation_consumer>(region_cpu, "output");

	try {
		// Get list of OpenCL platforms.
		fc_cl::Platforms platforms;
		fc_cl::Devices devices;
		cl::Context context;

		// fill platform, device and context with info
		fc_cl::init(platforms, devices, context);

		cl_processing& proc = infra.node_owner().make_child_named<cl_processing>(region_cl,"cl_processor", platforms, devices, context);

		// Connect event ports
		generator.out_data >> proc.in_data;
		proc.out_data >> consumer.in_result;
	} catch (const cl::Error &err) {
		std::cerr
				<< "OpenCL error: "
				<< err.what() << "(" << err.err() << ")"
				<< std::endl;
		return 1;
	}

	infra.start_scheduler();
	infra.iterate_main_loop();
	infra.stop_scheduler();
}
