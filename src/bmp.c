#include "../include/bmp.h"

void CreateBMP(BMPfile_t* bmpFile, int width, int height, unsigned char* dataPointer, unsigned short bitsPerPixel)
{
	bmpFile->header.id = 0x4D42;
	bmpFile->header.totalFileSize =
		14 + 40 + (width * height * (bitsPerPixel / 8));
	bmpFile->header.freeSpace = 0;
	bmpFile->header.dataOffset = 14 + 40;

	bmpFile->dib.numBytesDIBheader = 40;
	bmpFile->dib.width = width;
	bmpFile->dib.height = height;
	bmpFile->dib.colorPlanes = 1;
	bmpFile->dib.bitsPerPixel = bitsPerPixel;
	bmpFile->dib.comperssion = 0;
	bmpFile->dib.imageSize = width * height * bitsPerPixel / 8;
	bmpFile->dib.horizontalRes = 2835;
	bmpFile->dib.verticalRes = 2835;
	bmpFile->dib.colorPalette = 0;
	bmpFile->dib.importantColorUsed = 0;

	bmpFile->data = dataPointer;
}

void SaveBMPtoFile(BMPfile_t* bitmap, const char* filename)
{
	FILE* file;
	if ((file = fopen(filename, "wb")) != 0)
	{
		fwrite(&bitmap->header.id, sizeof(short), 1, file);
		fwrite(&bitmap->header.totalFileSize, 12, 1, file);
		fwrite(&bitmap->dib, 40, 1, file);
		fwrite(&bitmap->data[0], sizeof(unsigned char), bitmap->dib.height * bitmap->dib.width * 3, file);
		fclose(file);
	}
	else
		printf("Error when trying to open file %s", filename);
}