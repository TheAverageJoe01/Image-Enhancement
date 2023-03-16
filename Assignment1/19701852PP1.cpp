// 19701852PPAssignment1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "CImg.h"
#include "Utils.h"
#include <vector>

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string imageName = "images/test.ppm";
	bool isRGB;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { imageName = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	// taking in user input to create bin size
	string userCommand;
	int bin_num = 10;
	int Bit = 255;
	std::cout << "Enter a bin number in range 0-256" << "\n";
	// while the user hasn't entered a valid number the program will keep running
	while (true)
	{
		// Takes in the suer input 
		getline(std::cin, userCommand);
		// checks input to see if it isn't empty
		if (userCommand == "") { std::cout << "Please enter a number." << "\n"; continue; }
		// convert input into an int 
		try { bin_num = std::stoi(userCommand); }
		catch (...) { std::cout << "Please enter an integer." << "\n"; continue; }
		// checks to see if it is in range of 0 and 256 
		if (bin_num >= 0 && bin_num <= 256) { break; }
		else { std::cout << "Please enter a number in range 0-256." << "\n"; continue; }
	}
	float binSize = (float)256 / (float)bin_num;

	//detect any potential exceptions
	try {

		CImg<unsigned char> tempImage(imageName.c_str());
		CImg<unsigned char> imageInput;
		CImgDisplay disp_input(tempImage, "input");

		CImg<unsigned char> cb, cr;
		// detecting which type of image type is used
		if (tempImage.spectrum() == 1)
		{
			std::cout << "Image is greyscale" << std::endl;
			imageInput = tempImage;
			isRGB = false;
		}
		else if (tempImage.spectrum() == 3)
		{
			std::cout << "Image is an RGB" << std::endl;
			isRGB = true;
			//Converting RBG to Ycbcr
			CImg<unsigned char> YcbcrIMG = tempImage.get_RGBtoYCbCr();
			//extract channels 
			imageInput = YcbcrIMG.get_channel(0);
			cb = YcbcrIMG.get_channel(1);
			cr = YcbcrIMG.get_channel(2);

		}

		// host operations
		// Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		// setting the bin size using the user input 
		typedef int type;
		std::vector<type> H(bin_num);
		std::vector<type> CH(bin_num);
		std::vector<type> L(bin_num);
		std::vector<unsigned char> BP(imageInput.size());
		size_t hist_size = H.size() * sizeof(type);

		//Buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, imageInput.size() * sizeof(imageInput[0]));
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, imageInput.size() * sizeof(imageInput[0]));
		cl::Buffer dev_hist_buffer(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer dev_binSize_buffer(context, CL_MEM_READ_WRITE, sizeof(binSize));
		cl::Buffer dev_cum_buffer(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer dev_look_buffer(context, CL_MEM_READ_WRITE, hist_size);

		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, imageInput.size() * sizeof(imageInput[0]), &imageInput.data()[0]);
		queue.enqueueWriteBuffer(dev_binSize_buffer, CL_TRUE, 0, sizeof(binSize), &binSize);

		cl::Kernel Hist = cl::Kernel(program, "hist_simple");
		Hist.setArg(0, dev_image_input);
		Hist.setArg(1, dev_hist_buffer);
		Hist.setArg(2, dev_binSize_buffer);

		queue.enqueueNDRangeKernel(Hist, cl::NullRange, cl::NDRange(imageInput.size()), cl::NullRange, NULL);
		queue.enqueueReadBuffer(dev_hist_buffer, CL_TRUE, 0, hist_size, &H[0]);
		std::cout << std::endl << H << std::endl;

		//cumalative hist
		cl::Kernel cumHist = cl::Kernel(program, "scan_add");
		cumHist.setArg(0, dev_hist_buffer);
		cumHist.setArg(1, dev_cum_buffer);
		cumHist.setArg(2, cl::Local(hist_size));
		cumHist.setArg(3, cl::Local(hist_size));

		queue.enqueueNDRangeKernel(cumHist, cl::NullRange, cl::NDRange(H.size()), cl::NullRange, NULL);
		queue.enqueueReadBuffer(dev_cum_buffer, CL_TRUE, 0, hist_size, &CH[0]);
		std::cout << std::endl << CH << std::endl;

		//lookupTable
		cl::Kernel lookUp = cl::Kernel(program, "lookupTable");
		lookUp.setArg(0, dev_cum_buffer);
		lookUp.setArg(1, dev_look_buffer);
		lookUp.setArg(2, Bit);


		queue.enqueueNDRangeKernel(lookUp, cl::NullRange, cl::NDRange(H.size()), cl::NullRange, NULL);
		queue.enqueueReadBuffer(dev_look_buffer, CL_TRUE, 0, hist_size, &L[0]);
		std::cout << std::endl << L << std::endl;

		//BackProjection 
		cl::Kernel backprojection = cl::Kernel(program, "backprojection");
		backprojection.setArg(0, dev_image_input);
		backprojection.setArg(1, dev_look_buffer);
		backprojection.setArg(2, dev_image_output);


		queue.enqueueNDRangeKernel(backprojection, cl::NullRange, cl::NDRange(imageInput.size()), cl::NullRange, NULL);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, imageInput.size() * sizeof(imageInput[0]), &BP.data()[0]);

		//
		CImg<unsigned char> imageOutput(BP.data(), imageInput.width(), imageInput.height(), imageInput.depth(), imageInput.spectrum());

		//RBG Output
		if (isRGB == true)
		{
			CImg <unsigned char> RGB_image = imageOutput.get_resize(tempImage.width(), tempImage.height(), tempImage.depth(), tempImage.spectrum());
			for (int x = 0; x < tempImage.width(); x++) {
				for (int y = 0; y < tempImage.height(); y++) {
					RGB_image(x, y, 0) = imageOutput(x, y);
					RGB_image(x, y, 1) = cb(x, y);
					RGB_image(x, y, 2) = cr(x, y);
				}
			}
			imageOutput = RGB_image.get_YCbCrtoRGB();
		}

		// Display the final equalised image
		CImgDisplay disp_output(imageOutput, "output");

		// Close the input image and output image windows if the ESC key is pressed
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC())
		{
			disp_input.wait(1);
			disp_output.wait(1);
		}


	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}