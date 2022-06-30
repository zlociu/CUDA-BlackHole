#pragma once

// to add redshift to accretion disk
//#define REDSHIFT 

// to add gaussian postprocessing
//#define GAUSSIAN
#define GAUSSIAN_ROW_COL

// save output bitmap to default file (.bmp) or (.jpeg)
#define JPEG
#if defined(JPEG)
constexpr auto FILENAME = "blackhole.jpeg";
#else
constexpr auto FILENAME = "blackhole.bmp";
#endif


// output image dimensions							dividers
//												1    2    4    8    16   32    
constexpr auto DIM_X = 3840; //7680; //5760;//4800;//3840; //7680 //3840			// 1280 640  320  160   80   40 
constexpr auto DIM_Y = 2160; //4320; //3840;//3200;//2160; //4320 //2160			//  720 360  180   90   45
constexpr auto BITMAP_MULTIPIER = 4;				// used to create bigger background image             

#define DIM_X_N(n) (DIM_X * n) // makro returning rescaled image width
#define DIM_Y_N(n) (DIM_Y * n) // makro returning rescaled image height

constexpr auto STAR_DENSITY = 128; // 128 // there will be 1 star per STAR_DENSITY pixels;
constexpr auto GAUSS_SIZE = 600; // 600 // size of gaussian convolution kernel (width = 2 * GAUSS_SIZE + 1);
constexpr auto GAUSS_SIZE_SMALL = 100;  // size of small gaussian convolution kernel (width = 2 * GAUSS_SIZE + 1);
#define GAUSS_KERNEL_SIZE ((2 * GAUSS_SIZE + 1) * (2 * GAUSS_SIZE + 1))


#ifdef REDSHIFT
#define ITER 700	// number of iteration each thread
#define STEP 0.08f	// step value in accretion disk iteration
#define SKYDISKRATIO 0.05f //default is 0.05f
#else
#define ITER 700	// number of iteration each thread
#define STEP 0.08f	// step value in accretion disk iteration
#define SKYDISKRATIO 0.08f //default is 0.05f
#endif // RAYTRACING

constexpr auto FOV_TANGENT = 1.5f;
constexpr auto DISK_IN = 3.4f;
constexpr auto DISK_OUT = 16.f;