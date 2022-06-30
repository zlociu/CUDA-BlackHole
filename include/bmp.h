#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct BmpHeader_t
{
	unsigned short	id;				// "BM", 0x424D
	unsigned int	totalFileSize;	// totat file size in bytes
	unsigned int	freeSpace;		// unused buffer, just for space fill
	unsigned int	dataOffset;		// where pixel data starts
};

struct DibHeader_t
{
	unsigned int	numBytesDIBheader;	// length of DIB header in bytes
	unsigned int	width;				// width of image (in pixels)
	unsigned int	height;				// height of image (in pixels)
	unsigned short	colorPlanes;		// must be 1
	unsigned short	bitsPerPixel;		// bits per pixel (1,8,16,24,32)
	unsigned int	comperssion;		// compresion method; 3 if no compression and alpha channel
	unsigned int	imageSize;			// width * height * bytes per pixel
	unsigned int	horizontalRes;		// horizontal resolution in pixels/meter 72DPI = 2835
	unsigned int	verticalRes;		// vertical resolution in pixels/meter
	unsigned int	colorPalette;		// number of colors in palette, default 0
	unsigned int	importantColorUsed; // just ignore it :) [set 0]
};

struct BMPfile_t
{
	BmpHeader_t header;
	DibHeader_t dib;
	unsigned char* data;
};

void CreateBMP(BMPfile_t* bmpFile, int width, int height, unsigned char* dataPointer, unsigned short bitsPerPixel = 24);

void SaveBMPtoFile(BMPfile_t* bitmap, const char* filename);