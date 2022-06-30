#pragma once

//constants and useful macros
#define null NULL

#define M_E        2.71828182845904523536f   // e
#define M_LOG2E    1.44269504088896340736f  // log2(e)
#define M_LOG10E   0.434294481903251827651f  // log10(e)
#define M_LN2      0.693147180559945309417f  // ln(2)
#define M_LN10     2.30258509299404568402f   // ln(10)
#define M_PI       3.14159265358979323846f   // pi
#define M_PI_2     1.57079632679489661923f   // pi/2
#define M_PI_3	   1.04719755119659774615f	// pi/3
#define M_PI_4     0.785398163397448309616f  // pi/4
#define M_1_PI     0.318309886183790671538f  // 1/pi
#define M_1_2PI    0.159154943091895335768f	// 1/(2*pi)
#define M_2_PI     0.636619772367581343076f  // 2/pi
#define M_2_SQRTPI 1.12837916709551257390f   // 2/sqrt(pi)
#define M_SQRT2    1.41421356237309504880f   // sqrt(2)
#define M_SQRT1_2  0.707106781186547524401f  // 1/sqrt(2)
#define M_SQRT3_2  1.259921049894873164767f	// sqrt3(2)
#define M_34_LOG_3 0.823959216501f			// (3/4)*log(3)
#define M_12_SQRT3 0.866025403784438646763f  // sqrt(3)/2

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif