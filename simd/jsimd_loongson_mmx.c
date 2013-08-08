/*
 * jidctint.c
 *
 * Copyright (C) 1991-1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains a slow-but-accurate integer implementation of the
 * inverse DCT (Discrete Cosine Transform).  In the IJG code, this routine
 * must also perform dequantization of the input coefficients.
 *
 * A 2-D IDCT can be done by 1-D IDCT on each column followed by 1-D IDCT
 * on each row (or vice versa, but it's more convenient to emit a row at
 * a time).  Direct algorithms are also available, but they are much more
 * complex and seem not to be any faster when reduced to code.
 *
 * This implementation is based on an algorithm described in
 *   C. Loeffler, A. Ligtenberg and G. Moschytz, "Practical Fast 1-D DCT
 *   Algorithms with 11 Multiplications", Proc. Int'l. Conf. on Acoustics,
 *   Speech, and Signal Processing 1989 (ICASSP '89), pp. 988-991.
 * The primary algorithm described there uses 11 multiplies and 29 adds.
 * We use their alternate method with 12 multiplies and 32 adds.
 * The advantage of this method is that no data path contains more than one
 * multiplication; this allows a very simple and accurate implementation in
 * scaled fixed-point arithmetic, with a minimal number of shifts.
 *
 */

#define JPEG_INTERNALS
#include "../jinclude.h"
#include "../jpeglib.h"
#include "../jdct.h"		/* Private declarations for DCT subsystem */
#include "loongson-mmintrin.h"
#ifdef DCT_ISLOW_SUPPORTED

/*
 * This module is specialized to the case DCTSIZE = 8.
 */

#if DCTSIZE != 8
Sorry, this code only copes with 8x8 DCTs. /* deliberate syntax err */
#endif

/*
 * The poop on this scaling stuff is as follows:
 *
 * Each 1-D IDCT step produces outputs which are a factor of sqrt(N)
 * larger than the true IDCT outputs.  The final outputs are therefore
 * a factor of N larger than desired; since N=8 this can be cured by
 * a simple right shift at the end of the algorithm.  The advantage of
 * this arrangement is that we save two multiplications per 1-D IDCT,
 * because the y0 and y4 inputs need not be divided by sqrt(N).
 *
 * We have to do addition and subtraction of the integer inputs, which
 * is no problem, and multiplication by fractional constants, which is
 * a problem to do in integer arithmetic.  We multiply all the constants
 * by CONST_SCALE and convert them to integer constants (thus retaining
 * CONST_BITS bits of precision in the constants).  After doing a
 * multiplication we have to divide the product by CONST_SCALE, with proper
 * rounding, to produce the correct output.  This division can be done
 * cheaply as a right shift of CONST_BITS bits.  We postpone shifting
 * as long as possible so that partial sums can be added together with
 * full fractional precision.
 *
 * The outputs of the first pass are scaled up by PASS1_BITS bits so that
 * they are represented to better-than-integral precision.  These outputs
 * require BITS_IN_JSAMPLE + PASS1_BITS + 3 bits; this fits in a 16-bit word
 * with the recommended scaling.  (To scale up 12-bit sample data further, an
 * intermediate INT32 array would be needed.)
 *
 * To avoid overflow of the 32-bit intermediate results in pass 2, we must
 * have BITS_IN_JSAMPLE + CONST_BITS + PASS1_BITS <= 26.  Error analysis
 * shows that the values given below are the most effective.
 */

#if BITS_IN_JSAMPLE == 8
#define CONST_BITS  13
#define PASS1_BITS  2
#else
#define CONST_BITS  13
#define PASS1_BITS  1		/* lose a little precision to avoid overflow */
#endif

/* Some C compilers fail to reduce "FIX(constant)" at compile time, thus
 * causing a lot of useless floating-point operations at run time.
 * To get around this we use the following pre-calculated constants.
 * If you change CONST_BITS you may want to add appropriate values.
 * (With a reasonable C compiler, you can just rely on the FIX() macro...)
 */

#define DESCALE_P1       (CONST_BITS-PASS1_BITS)
#define DESCALE_P2       (CONST_BITS+PASS1_BITS+3)
#define PD_DECALE_P1     1 << (DESCALE_P1-1)
#define PD_DECALE_P2     1 << (DESCALE_P2-1)
#define CENTERJSAMPLE    128
#if CONST_BITS == 13
#define FIX_0_298  ((short)  2446)	/* FIX(0.298631336) */
#define FIX_0_390  ((short)  3196)	/* FIX(0.390180644) */
#define FIX_0_541  ((short)  4433)	/* FIX(0.541196100) */
#define FIX_0_765  ((short)  6270)	/* FIX(0.765366865) */
#define FIX_0_899  ((short)  7373)	/* FIX(0.899976223) */
#define FIX_1_175  ((short)  9633)	/* FIX(1.175875602) */
#define FIX_1_501  ((short)  12299)	/* FIX(1.501321110) */
#define FIX_1_847  ((short)  15137)	/* FIX(1.847759065) */
#define FIX_1_961  ((short)  16069)	/* FIX(1.961570560) */
#define FIX_2_053  ((short)  16819)	/* FIX(2.053119869) */
#define FIX_2_562  ((short)  20995)	/* FIX(2.562915447) */
#define FIX_3_072  ((short)  25172)	/* FIX(3.072711026) */
#define FIX_0_298631336  ((INT32)  2446)        /* FIX(0.298631336) */
#define FIX_0_390180644  ((INT32)  3196)        /* FIX(0.390180644) */
#define FIX_0_541196100  ((INT32)  4433)        /* FIX(0.541196100) */
#define FIX_0_765366865  ((INT32)  6270)        /* FIX(0.765366865) */
#define FIX_0_899976223  ((INT32)  7373)        /* FIX(0.899976223) */
#define FIX_1_175875602  ((INT32)  9633)        /* FIX(1.175875602) */
#define FIX_1_501321110  ((INT32)  12299)       /* FIX(1.501321110) */
#define FIX_1_847759065  ((INT32)  15137)       /* FIX(1.847759065) */
#define FIX_1_961570560  ((INT32)  16069)       /* FIX(1.961570560) */
#define FIX_2_053119869  ((INT32)  16819)       /* FIX(2.053119869) */
#define FIX_2_562915447  ((INT32)  20995)       /* FIX(2.562915447) */
#define FIX_3_072711026  ((INT32)  25172)       /* FIX(3.072711026) */
#else
#define DECALE(x,n)  (((x)+(1<<((n)-1)))>>(n))
#define FIX_0_298  DECALE( 320652955,30-CONST_BITS)    /* FIX(0.298631336) */
#define FIX_0_390  DECALE( 418953276,30-CONST_BITS)    /* FIX(0.390180644) */
#define FIX_0_541  DECALE( 581104887,30-CONST_BITS)    /* FIX(0.541196100) */
#define FIX_0_765  DECALE( 821806413,30-CONST_BITS)    /* FIX(0.765366865) */
#define FIX_0_899  DECALE( 966342111,30-CONST_BITS)    /* FIX(0.899976223) */
#define FIX_1_175  DECALE(1262586813,30-CONST_BITS)    /* FIX(1.175875602) */
#define FIX_1_501  DECALE(1612031267,30-CONST_BITS)    /* FIX(1.501321110) */
#define FIX_1_847  DECALE(1984016188,30-CONST_BITS)    /* FIX(1.847759065) */
#define FIX_1_961  DECALE(2106220350,30-CONST_BITS)    /* FIX(1.961570560) */
#define FIX_2_053  DECALE(2204520673,30-CONST_BITS)    /* FIX(2.053119869) */
#define FIX_2_562  DECALE(2751909506,30-CONST_BITS)    /* FIX(2.562915447) */
#define FIX_3_072  DECALE(3299298341,30-CONST_BITS)    /* FIX(3.072711026) */
#define FIX_0_298631336  FIX(0.298631336)
#define FIX_0_390180644  FIX(0.390180644)
#define FIX_0_541196100  FIX(0.541196100)
#define FIX_0_765366865  FIX(0.765366865)
#define FIX_0_899976223  FIX(0.899976223)
#define FIX_1_175875602  FIX(1.175875602)
#define FIX_1_501321110  FIX(1.501321110)
#define FIX_1_847759065  FIX(1.847759065)
#define FIX_1_961570560  FIX(1.961570560)
#define FIX_2_053119869  FIX(2.053119869)
#define FIX_2_562915447  FIX(2.562915447)
#define FIX_3_072711026  FIX(3.072711026)
#endif


/* Multiply an INT32 variable by an INT32 constant to yield an INT32 result.
 *  * For 8-bit samples with the recommended scaling, all the variable
 *   * and constant values involved are no more than 16 bits wide, so a
 *    * 16x16->32 bit multiply can be used instead of a full 32x32 multiply.
 *     * For 12-bit samples, a full 32-bit multiplication will be needed.
 *      */

#if BITS_IN_JSAMPLE == 8
#define MULTIPLY(var,const)  MULTIPLY16C16(var,const)
#else
#define MULTIPLY(var,const)  ((var) * (const))
#endif


/* Dequantize a coefficient by multiplying it by the multiplier-table
 *  * entry; produce an int result.  In this module, both inputs and result
 *   * are 16 bits or less, so either int or short multiply will work.
 *    */

#define DEQUANTIZE(coef,quantval)  (((ISLOW_MULT_TYPE) (coef)) * (quantval))


/*
 * Perform dequantization and inverse DCT on one block of coefficients.
 */

#define get_m64_value(addr) (*(__m64 *)addr)
#define test_m64_zero(mm64) (!(*(u64 *)&mm64))

GLOBAL(void)
jsimd_idct_islow_mmx (j_decompress_ptr cinfo, jpeg_component_info * compptr,
		JCOEFPTR coef_block,
		JSAMPARRAY output_buf, JDIMENSION output_col)
{
	__m64 tmp0,tmp1,tmp2,tmp3;
	__m64 z2, z3, z4;
	__m64 t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16;
	__m64 T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11;
	JCOEFPTR inptr;
	ISLOW_MULT_TYPE * quantptr;
	__m64 *wsptr;
	JSAMPROW outptr; //char类型指针 
	int ctr,num;
	__m64 workspace[DCTSIZE]; /* buffers data between passes缓冲区间数据转换 */
	SHIFT_TEMPS

		/* Pass 1: process columns from input, store into work array. */
		/* Note results are scaled up by sqrt(8) compared to a true IDCT; */
		/* furthermore, we scale the results by 2**PASS1_BITS. */
		inptr = coef_block;
	quantptr = (ISLOW_MULT_TYPE *) compptr->dct_table;
	wsptr = workspace;
	//DCTSIZE == 8
	for (ctr = (DCTSIZE >> 2); ctr > 0; ctr--) {
		/* Due to quantization, we will usually find that many of the input
		 * coefficients are zero, especially the AC terms.  We can exploit this
		 * by short-circuiting the IDCT calculation for any column in which all
		 * the AC terms are zero.  In that case each output is equal to the
		 * DC coefficient (with scale factor as needed).
		 * With typical images and quantization tables, half or more of the
		 * column DCT calculations can be simplified this way.
		 */
#if 1
		__m64 inptra0 = _mm_unpacklo_pi16(inptr[DCTSIZE*0 + 0],inptr[DCTSIZE*0 + 1]);
		__m64 inptra1 = _mm_unpacklo_pi16(inptr[DCTSIZE*0 + 2],inptr[DCTSIZE*0 + 3]);
		__m64 inptra = _mm_unpacklo_pi16(inptra0,inptra1);	
		__m64 inptrb0 = _mm_unpacklo_pi16(inptr[DCTSIZE*1 + 0],inptr[DCTSIZE*1 + 1]);
		__m64 inptrb1 = _mm_unpacklo_pi16(inptr[DCTSIZE*1 + 2],inptr[DCTSIZE*1 + 3]);
		__m64 inptrb = _mm_unpacklo_pi16(inptrb0,inptrb1);

		__m64 inptrc0 = _mm_unpacklo_pi16(inptr[DCTSIZE*2 + 0],inptr[DCTSIZE*2 + 1]);
		__m64 inptrc1 = _mm_unpacklo_pi16(inptr[DCTSIZE*2 + 2],inptr[DCTSIZE*2 + 3]);
		__m64 inptrc = _mm_unpacklo_pi16(inptrc0,inptrc1);

		__m64 inptrd0 = _mm_unpacklo_pi16(inptr[DCTSIZE*3 + 0],inptr[DCTSIZE*3 + 1]);
		__m64 inptrd1 = _mm_unpacklo_pi16(inptr[DCTSIZE*3 + 2],inptr[DCTSIZE*3 + 3]);
		__m64 inptrd = _mm_unpacklo_pi16(inptrd0,inptrd1);

		__m64 inptre0 = _mm_unpacklo_pi16(inptr[DCTSIZE*4 + 0],inptr[DCTSIZE*4 + 1]);
		__m64 inptre1 = _mm_unpacklo_pi16(inptr[DCTSIZE*4 + 2],inptr[DCTSIZE*4 + 3]);
		__m64 inptre = _mm_unpacklo_pi16(inptre0,inptre1);

		__m64 inptrf0 = _mm_unpacklo_pi16(inptr[DCTSIZE*5 + 0],inptr[DCTSIZE*5 + 1]);
		__m64 inptrf1 = _mm_unpacklo_pi16(inptr[DCTSIZE*5 + 2],inptr[DCTSIZE*5 + 3]);
		__m64 inptrf = _mm_unpacklo_pi16(inptrf0,inptrf1);

		__m64 inptrg0 = _mm_unpacklo_pi16(inptr[DCTSIZE*6 + 0],inptr[DCTSIZE*6 + 1]);
		__m64 inptrg1 = _mm_unpacklo_pi16(inptr[DCTSIZE*6 + 2],inptr[DCTSIZE*6 + 3]);
		__m64 inptrg = _mm_unpacklo_pi16(inptrg0,inptrg1);

		__m64 inptrh0 = _mm_unpacklo_pi16(inptr[DCTSIZE*7 + 0],inptr[DCTSIZE*7 + 1]);
		__m64 inptrh1 = _mm_unpacklo_pi16(inptr[DCTSIZE*7 + 2],inptr[DCTSIZE*7 + 3]);
		__m64 inptrh = _mm_unpacklo_pi16(inptrh0,inptrh1);

		inptrb0 = _mm_or_si64(inptrb0,inptrc0);
		inptrb = _mm_or_si64(inptrb,inptrd);
		inptrc = _mm_or_si64(inptrc,inptre);
		inptrb = _mm_or_si64(inptrb,inptrf);
		inptrc = _mm_or_si64(inptrc,inptrg);
		inptrb = _mm_or_si64(inptrb,inptrh);
		inptrc = _mm_or_si64(inptrc,inptrb); 
#endif
		__m64 mm1 = get_m64_value(&inptr[DCTSIZE*1]);
		__m64 mm2 = get_m64_value(&inptr[DCTSIZE*2]);
		__m64 mm3 = get_m64_value(&inptr[DCTSIZE*3]);
		mm1 = _mm_or_si64(mm1, mm3);
		__m64 mm4 = get_m64_value(&inptr[DCTSIZE*4]);
		mm2 = _mm_or_si64(mm2, mm4);
		__m64 mm5 = get_m64_value(&inptr[DCTSIZE*5]);
		mm1 = _mm_or_si64(mm1, mm5);
		__m64 mm6 = get_m64_value(&inptr[DCTSIZE*6]);
		mm2 = _mm_or_si64(mm2, mm6);
		__m64 mm7 = get_m64_value(&inptr[DCTSIZE*7]);
		mm1 = _mm_or_si64(mm1, mm7);

		mm1 = _mm_or_si64(mm1, mm2);

		//if (inptrb0 == 0 && inptrc == 0) {
		if (test_mm64_zero(mm1)) {
			/* AC terms all zero */

			__m64 inptra0 = _mm_unpacklo_pi16(inptr[DCTSIZE*0 + 0],inptr[DCTSIZE*0 + 1]);
			__m64 inptra1 = _mm_unpacklo_pi16(inptr[DCTSIZE*0 + 2],inptr[DCTSIZE*0 + 3]);
			__m64 inptra = _mm_unpacklo_pi16(inptra0,inptra1);

			__m64 quantptra0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*0 + 0],quantptr[DCTSIZE*0 + 1]);
			__m64 quantptra1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*0 + 2],quantptr[DCTSIZE*0 + 3]);
			__m64 quantptra = _mm_unpacklo_pi16(quantptra0,quantptra1);       	
			__m64 dcval0 = _mm_mullo_pi16(quantptra,inptra);
			__m64 dcval = _mm_slli_pi16(dcval0,PASS1_BITS);		//dcval=in0=(00 01 02 03) 

			__m64 dcvalL = _mm_unpacklo_pi16(dcval,dcval);		//dcvalL=in0=(00 01 01 01)
			__m64 dcvalH = _mm_unpackhi_pi16(dcval,dcval);		//dcvalL=in0=(02 02 03 03)

			__m64 dcvalLL = _mm_unpacklo_pi32(dcvalL,dcvalL);	//dcvalL0=in0=(00 00 00 00)
			__m64 dcvalLH = _mm_unpackhi_pi32(dcvalL,dcvalL);	//dcvalL1=in0=(01 01 01 01)
			__m64 dcvalHL = _mm_unpacklo_pi32(dcvalH,dcvalH);	//dcvalH0=in0=(02 02 02 02)
			__m64 dcvalHH = _mm_unpackhi_pi32(dcvalH,dcvalH);	//dcvalH1=in0=(03 03 03 03)

			for (num = 0; num++; num <= ((DCTSIZE >> 1) -1)) {
				wsptr[DCTSIZE*0 + num] = _mm_extract_pi16(dcvalLL,(int64_t) num);
				wsptr[DCTSIZE*1 + num] = _mm_extract_pi16(dcvalLL,(int64_t) num);
				wsptr[DCTSIZE*2 + num] = _mm_extract_pi16(dcvalLH,(int64_t) num);
				wsptr[DCTSIZE*3 + num] = _mm_extract_pi16(dcvalLH,(int64_t) num);
				wsptr[DCTSIZE*4 + num] = _mm_extract_pi16(dcvalHL,(int64_t) num);
				wsptr[DCTSIZE*5 + num] = _mm_extract_pi16(dcvalHL,(int64_t) num);
				wsptr[DCTSIZE*6 + num] = _mm_extract_pi16(dcvalHH,(int64_t) num);
				wsptr[DCTSIZE*7 + num] = _mm_extract_pi16(dcvalHH,(int64_t) num);
			}	

			inptr += (DCTSIZE >> 1);			/* advance pointers to next column */
			quantptr += (DCTSIZE >> 1);
			wsptr += (DCTSIZE >> 1);
			continue;
		}

		/* Even part: reverse the even part of the forward DCT. */
		/* The rotator is sqrt(2)*c(-6). */

		/*(Original)
		 * z1 = (z2 + z3) * 0.541196100;
		 * tmp2 = z1 + z3 * -1.847759065;
		 * tmp3 = z1 + z2 * 0.765366865;
		 *
		 * (This implementation)
		 * tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065);
		 * tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100;
		 */

		__m64 quantptra0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*1 + 0],quantptr[DCTSIZE*1 + 1]);
		__m64 quantptra1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*1 + 2],quantptr[DCTSIZE*1 + 3]);
		__m64 quantptra = _mm_unpacklo_pi16(quantptra0,quantptra1);
		__m64 dcval0 = _mm_mullo_pi16(quantptra,inptra);

		__m64 quantptrc0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*2 + 0],quantptr[DCTSIZE*2 + 1]);
		__m64 quantptrc1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*2 + 2],quantptr[DCTSIZE*2 + 3]);
		__m64 quantptrc = _mm_unpacklo_pi16(quantptrc0,quantptrc1);	
		__m64 dcval2 = _mm_mullo_pi16(quantptrc,inptrc);

		__m64 quantptre0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*4 + 0],quantptr[DCTSIZE*4 + 1]);
		__m64 quantptre1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*4 + 2],quantptr[DCTSIZE*4 + 3]);
		__m64 quantptre = _mm_unpacklo_pi16(quantptre0,quantptre1);	
		__m64 dcval4= _mm_mullo_pi16(quantptre,inptre);

		__m64 quantptrg0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*6 + 0],quantptr[DCTSIZE*6 + 1]);
		__m64 quantptrg1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*6 + 2],quantptr[DCTSIZE*6 + 3]);
		__m64 quantptrg = _mm_unpacklo_pi16(quantptrg0,quantptrg1);	
		__m64 dcval6 = _mm_mullo_pi16(quantptrg,inptrg);

		// (0,0,0,FIX_0_541) (00 00 00 00 ) --> (FIX_0_541，FIX_0_5410，FIX_0_541，FIX_0_541)
		//	--> (FIX_0_541，0，FIX_0_541，0)
		t1 = _mm_shuffle_pi16((__m64) FIX_0_541,_mm_setzero_si64());
		t4 = _mm_loadlo_pi16(t1);
		//	(0,0,0,FIX_0_541+FIX_0_765) (00 00 00 00 ) --> (FIX_0_541+FIX_0_765,FIX_0_541+FIX_0_765,FIX_0_541+FIX_0_765，FIX_0_541+FIX_0_765)
		//						   --> (0,FIX_0_541+FIX_0_765,0，FIX_0_541+FIX_0_765)
		t2 = _mm_shuffle_pi16((__m64) (FIX_0_541 + FIX_0_765),_mm_setzero_si64());
		t2 = _mm_loadlo_pi16_f(t2);
		T1 = _mm_add_pi16(t4,t2);
		// (0,0,0,FIX_0_541 - FIX_1_847) (00 00 00 00 ) --> (FIX_0_541 - FIX_1_847，FIX_0_541 - FIX_1_847，FIX_0_541 - FIX_1_847)
		//                                  --> (FIX_0_541 - FIX_1_847，0，FIX_0_541 - FIX_1_847，0)
		t3 = _mm_shuffle_pi16((__m64) (FIX_0_541 - FIX_1_847),_mm_setzero_si64());
		t3 = _mm_loadlo_pi16(t3);
		t4 = _mm_loadlo_pi16_f(t1);
		T2 = _mm_add_pi16(t3,t4);

		z2 = dcval2;
		z3 = dcval6;
		__m64 z23L = _mm_unpacklo_pi16(z2,z3);
		__m64 z23H = _mm_unpackhi_pi16(z2,z3);

		__m64 tmp3L = _mm_madd_pi16(z23L,T1);	//tmp3L
		__m64 tmp3H = _mm_madd_pi16(z23H,T1);	//tmp3H	

		__m64 tmp2L = _mm_madd_pi16(z23L,T2);	//tmp2L
		__m64 tmp2H = _mm_madd_pi16(z23H,T2);	//tmp2H

		z2 = dcval0;
		z3 = dcval4;
		__m64 z23a = _mm_add_si64(z2,z3);
		__m64 z23s = _mm_sub_si64(z2,z3);

		__m64 tmp0L = _mm_loadlo_pi16(z23a);	//tmp0L
		__m64 tmp0H = _mm_loadhi_pi16(z23a);	//tmp0H

		tmp0L = _mm_srai_pi32(tmp0L,(16-CONST_BITS));	//tmp0L
		tmp0H = _mm_srai_pi32(tmp0H,(16-CONST_BITS));	//tmp0H

		__m64 tmp10L = _mm_add_pi32(tmp0L,tmp3L);	//tmp10L
		__m64 tmp13L = _mm_sub_pi32(tmp0L,tmp3L);	//tmp13L

		__m64 tmp10H = _mm_add_pi32(tmp0H,tmp3H);	//tmp10H
		__m64 tmp13H = _mm_sub_pi32(tmp0H,tmp3H);	//tmp13H

		__m64 tmp1L = _mm_loadlo_pi16(z23s);	//tmp1L
		__m64 tmp1H = _mm_loadhi_pi16(z23s);	//tmp1H

		tmp1L = _mm_srai_pi32(tmp1L,(16-CONST_BITS));	//tmp1L
		tmp1H = _mm_srai_pi32(tmp1H,(16-CONST_BITS));	//tmp2H

		__m64 tmp11L = _mm_add_pi32(tmp1L,tmp2L);	//tmp11L
		__m64 tmp12L = _mm_sub_pi32(tmp1L,tmp2L);	//tmp12L

		__m64 tmp11H = _mm_add_pi32(tmp1H,tmp2H);	//tmp11H
		__m64 tmp12H = _mm_sub_pi32(tmp1H,tmp2H);	//tmp12H

		/* Odd part per figure 8; the matrix is unitary and hence its
		 * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
		 */	

		__m64 quantptrh0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*7 + 0],quantptr[DCTSIZE*7 + 1]);
		__m64 quantptrh1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*7 + 2],quantptr[DCTSIZE*7 + 3]);
		__m64 quantptrh = _mm_unpacklo_pi16(quantptrh0,quantptrh1);	
		__m64 dcval7 = _mm_mullo_pi16(quantptrh,inptrh);

		__m64 quantptrf0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*5 + 0],quantptr[DCTSIZE*5 + 1]);
		__m64 quantptrf1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*5 + 2],quantptr[DCTSIZE*5 + 3]);
		__m64 quantptrf = _mm_unpacklo_pi16(quantptrf0,quantptrf1);	
		__m64 dcval5= _mm_mullo_pi16(quantptrf,inptrf);

		__m64 quantptrd0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*3 + 0],quantptr[DCTSIZE*3 + 1]);
		__m64 quantptrd1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*3 + 2],quantptr[DCTSIZE*3 + 3]);
		__m64 quantptrd = _mm_unpacklo_pi16(quantptrd0,quantptrd1);	
		__m64 dcval3 = _mm_mullo_pi16(quantptrd,inptrd);

		__m64 quantptrb0 = _mm_unpacklo_pi16(quantptr[DCTSIZE*1 + 0],quantptr[DCTSIZE*1 + 1]);
		__m64 quantptrb1 = _mm_unpacklo_pi16(quantptr[DCTSIZE*1 + 2],quantptr[DCTSIZE*1 + 3]);
		__m64 quantptrb = _mm_unpacklo_pi16(quantptrb0,quantptrb1);	
		__m64 dcval1 = _mm_mullo_pi16(quantptrb,inptrb);

		tmp0 = dcval7;
		tmp1 = dcval5;
		tmp2 = dcval3;
		tmp3 = dcval1;

		z3 = _mm_add_pi16(tmp0,tmp2);
		z4 = _mm_add_pi16(tmp1,tmp3);
		/*
		 * (Original)
		 * z5 = (z3 + z4) * 1.175875602;
		 * z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
		 * z3 += z5;  z4 += z5;
		 *
		 * (This implementation)
		 * z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
		 * z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);
		 */
		t5 = _mm_shuffle_pi16((__m64) FIX_1_175,_mm_setzero_si64());
		t8 = _mm_loadlo_pi16(t5);

		t6 = _mm_shuffle_pi16((__m64) (FIX_1_175 - FIX_1_961),_mm_setzero_si64());
		t6 = _mm_loadlo_pi16_f(t6);

		T3 = _mm_add_pi16(t8,t6);

		t7 = _mm_shuffle_pi16((__m64) (FIX_1_175 - FIX_0_390),_mm_setzero_si64());
		t7 = _mm_loadlo_pi16(t7);
		t8 = _mm_loadlo_pi16_f(t5);

		T4 = _mm_add_pi16(t7,t8);

		__m64 z34L = _mm_unpacklo_pi16(z3,z4);
		__m64 z34H = _mm_unpackhi_pi16(z3,z4);

		__m64 z3L = _mm_madd_pi16(z34L,T3);
		__m64 z3H = _mm_madd_pi16(z34H,T3);

		__m64 z4L = _mm_madd_pi16(z34L,T4);
		__m64 z4H = _mm_madd_pi16(z34H,T4);
		/*
		 * (Original)
		 * z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
		 * tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
		 * tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
		 * z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
		 * tmp0 += z1 + z3;  tmp1 += z2 + z4;
		 * tmp2 += z2 + z3;  tmp3 += z1 + z4;
		 *
		 * (This implementation)
		 * tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
		 * tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
		 * tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
		 * tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
		 * tmp0 += z3;  tmp1 += z4;
		 * tmp2 += z3;  tmp3 += z4;
		 */
		t9 = _mm_shuffle_pi16((__m64) -FIX_0_899,_mm_setzero_si64());
		t12 = _mm_loadlo_pi16(t9);

		t10 = _mm_shuffle_pi16((__m64) (FIX_0_298 - FIX_0_899),_mm_setzero_si64());
		t10 = _mm_loadlo_pi16_f(t10);

		T5 = _mm_add_pi16(t12,t10);

		t11 = _mm_shuffle_pi16((__m64) (FIX_1_501 - FIX_0_899),_mm_setzero_si64());
		t11 = _mm_loadlo_pi16(t11);
		t12 = _mm_loadlo_pi16_f(t9);

		T6 = _mm_add_pi16(t11,t12);

		__m64 tmp03L = _mm_unpacklo_pi16(tmp0,tmp3);
		__m64 tmp03H = _mm_unpackhi_pi16(tmp0,tmp3);

		tmp0L = _mm_madd_pi16(tmp03L,T5);
		tmp0H = _mm_madd_pi16(tmp03H,T5);

		tmp3L = _mm_madd_pi16(tmp03L,T6);
		tmp3H = _mm_madd_pi16(tmp03H,T6);   	

		tmp0L =  _mm_add_pi32(tmp0L,z3L);
		tmp0H = _mm_add_pi32(tmp0H,z3H);
		tmp3L =  _mm_add_pi32(tmp0L,z4L);
		tmp3H = _mm_add_pi32(tmp0H,z4H);
		//
		t13 = _mm_shuffle_pi16((__m64) -FIX_2_562,_mm_setzero_si64());
		t16 = _mm_loadlo_pi16(t13);

		t14 = _mm_shuffle_pi16((__m64) (FIX_2_053 - FIX_2_562),_mm_setzero_si64());
		t14 = _mm_loadlo_pi16_f(t14);

		T7 = _mm_add_pi16(t16,t14);

		t15 = _mm_shuffle_pi16((__m64) (FIX_3_072 - FIX_2_562),_mm_setzero_si64());
		t15 = _mm_loadlo_pi16(t15);
		t16 = _mm_loadlo_pi16_f(t13);

		T8 = _mm_add_pi16(t15,t16);

		tmp12L = _mm_unpacklo_pi16(tmp1,tmp2);
		tmp12H = _mm_unpackhi_pi16(tmp1,tmp2);

		tmp1L = _mm_madd_pi16(tmp12L,T5);
		tmp1H = _mm_madd_pi16(tmp12H,T5);

		tmp2L = _mm_madd_pi16(tmp12L,T6);
		tmp2H = _mm_madd_pi16(tmp12H,T6);

		tmp1L = _mm_add_pi32(tmp1L,z4L);
		tmp1H = _mm_add_pi32(tmp1H,z4H);
		tmp2L = _mm_add_pi32(tmp2L,z3L);
		tmp2H = _mm_add_pi32(tmp2H,z3H);

		/* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

		T9 = _mm_set1_pi32(PD_DECALE_P1);//[PD_DECALE_P1,PD_DECALE_P1]

		__m64 data0L = _mm_add_pi32(tmp10L,tmp3L);  //data0L
		__m64 data0H = _mm_add_pi32(tmp10H,tmp3H);  //data0H
		__m64 data7L = _mm_sub_pi32(tmp10L,tmp3L);  //data7L
		__m64 data7H = _mm_sub_pi32(tmp10H,tmp3H);  //data7H

		data0L = _mm_add_pi32(data0L,T9);
		data0H = _mm_add_pi32(data0H,T9);	
		data0L = _mm_srai_pi32(data0L,DESCALE_P1);
		data0H = _mm_srai_pi32(data0H,DESCALE_P1);

		data7L = _mm_add_pi32(data7L,T9);
		data7H = _mm_add_pi32(data7H,T9);
		data7L = _mm_srai_pi32(data7L,DESCALE_P1);
		data7H = _mm_srai_pi32(data7H,DESCALE_P1);

		__m64 data0 = _mm_packs_pi32(data0L,data0H);  //data0 =(00 01 02 03)
		__m64 data7 = _mm_packs_pi32(data7L,data7H);  //data7 =(70 71 72 73)

		__m64 data1L = _mm_add_pi32(tmp11L,tmp2L);  //data1L
		__m64 data1H = _mm_add_pi32(tmp11H,tmp2H);  //data1H
		__m64 data6L = _mm_sub_pi32(tmp11L,tmp2L);  //data6L
		__m64 data6H = _mm_sub_pi32(tmp11H,tmp2H);  //data6H

		data1L = _mm_add_pi32(data1L,T9);
		data1H = _mm_add_pi32(data1H,T9);
		data1L = _mm_srai_pi32(data1L,DESCALE_P1);
		data1H = _mm_srai_pi32(data1H,DESCALE_P1);

		data6L = _mm_add_pi32(data6L,T9);
		data6H = _mm_add_pi32(data6H,T9);
		data6L = _mm_srai_pi32(data6L,DESCALE_P1);
		data6H = _mm_srai_pi32(data6H,DESCALE_P1);

		__m64 data1 = _mm_packs_pi32(data1L,data1H);  //data1 =(10 11 12 13)
		__m64 data6 = _mm_packs_pi32(data6L,data6H);  //data6 =(60 61 62 63)

		__m64 data01L = _mm_unpacklo_pi16(data0,data1); //data01L = (00 10 01 11)
		__m64 data01H = _mm_unpackhi_pi16(data0,data1); //data01H = (02 12 03 13)
		__m64 data67L = _mm_unpacklo_pi16(data6,data7); //data67L = (60 70 61 71)
		__m64 data67H = _mm_unpackhi_pi16(data6,data7); //data67H = (62 72 63 73)

		__m64 data2L = _mm_add_pi32(tmp12L,tmp1L);  //data2L
		__m64 data2H = _mm_add_pi32(tmp12H,tmp1H);  //data2H
		__m64 data5L = _mm_sub_pi32(tmp12L,tmp1L);  //data5L
		__m64 data5H = _mm_sub_pi32(tmp12H,tmp1H);  //data5H

		data2L = _mm_add_pi32(data2L,T9);
		data2H = _mm_add_pi32(data2H,T9);
		data2L = _mm_srai_pi32(data2L,DESCALE_P1);
		data2H = _mm_srai_pi32(data2H,DESCALE_P1);

		data5L = _mm_add_pi32(data5L,T9);
		data5H = _mm_add_pi32(data5H,T9);
		data5L = _mm_srai_pi32(data5L,DESCALE_P1);
		data5H = _mm_srai_pi32(data5H,DESCALE_P1);

		__m64 data2 = _mm_packs_pi32(data2L,data2H);  //data2 =(20 21 22 23)
		__m64 data5 = _mm_packs_pi32(data5L,data5H);  //data5 =(50 51 52 53)

		__m64 data3L = _mm_add_pi32(tmp13L,tmp0L);  //data3L
		__m64 data3H = _mm_add_pi32(tmp13H,tmp0H);  //data3H

		__m64 data4L = _mm_sub_pi32(tmp13L,tmp0L);  //data4L
		__m64 data4H = _mm_sub_pi32(tmp13H,tmp0H);  //data4H

		data3L = _mm_add_pi32(data3L,T9);
		data3H = _mm_add_pi32(data3H,T9);
		data3L = _mm_srai_pi32(data3L,DESCALE_P1);
		data3H = _mm_srai_pi32(data3H,DESCALE_P1);

		data4L = _mm_add_pi32(data4L,T9);
		data4H = _mm_add_pi32(data4H,T9);
		data4L = _mm_srai_pi32(data4L,DESCALE_P1);
		data4H = _mm_srai_pi32(data4H,DESCALE_P1);

		__m64 data3 = _mm_packs_pi32(data3L,data3H);  //data3 =(30 31 32 33)
		__m64 data4 = _mm_packs_pi32(data4L,data4H);  //data4 =(40 41 42 43)

		__m64 data23L = _mm_unpacklo_pi16(data2,data3); //data23L = (20 30 21 31)
		__m64 data23H = _mm_unpackhi_pi16(data2,data3); //data23H = (22 32 23 33)
		__m64 data45L = _mm_unpacklo_pi16(data4,data5); //data45L = (40 50 41 51)
		__m64 data45H = _mm_unpackhi_pi16(data4,data5); //data45H = (42 52 43 53)

		//data01L = (00 10 01 11) punpcklwd data23L = (20 30 21 31)
		//data01L = (00 10 01 11) punpckhwd data23L = (20 30 21 31)
		__m64 data0123LL = _mm_unpacklo_pi32(data01L,data23L); //(00 10 20 30)
		__m64 data0123LH = _mm_unpackhi_pi32(data01L,data23L); //(01 11 21 31)
		//data01H = (02 12 03 13) punpcklwd data23H = (22 32 23 33)
		//data01H = (02 12 03 13) punpckhwd data23H = (22 32 23 33)
		__m64 data0123HL = _mm_unpacklo_pi32(data01L,data23L); //(02 12 22 32)
		__m64 data0123HH = _mm_unpackhi_pi32(data01H,data23H); //(03 13 23 33)

		//data45L = (40 50 41 51) punpcklwd data67L = (60 70 61 71)
		//data45L = (40 50 41 51) punpckhwd data67L = (60 70 61 71)
		__m64 data4567LL = _mm_unpacklo_pi32(data45L,data67L); //(40 50 60 70)
		__m64 data4567LH = _mm_unpackhi_pi32(data45L,data67L); //(41 51 61 71)
		//data45H = (42 52 43 53) punpcklwd data67H = (62 72 63 73)
		//data45H = (42 52 43 53) punpckhwd data67H = (62 72 63 73) 
		__m64 data4567HL = _mm_unpacklo_pi32(data45L,data67L); //(42 52 62 72)
		__m64 data4567HH = _mm_unpackhi_pi32(data45H,data67H); //(43 53 63 73)
		/* wsptr[DCTSIZE*0 + 0] = (00 00 00 00)	wsptr[DCTSIZE*0 + 1] = (00 00 00 10) ...
		   wsptr[DCTSIZE*1 + 0] = (00 00 00 40) wsptr[DCTSIZE*1 + 1] = (00 00 00 50) ...
		   wsptr[DCTSIZE*2 + 0] = (00 00 00 01) wsptr[DCTSIZE*2 + 1] = (00 00 00 11) ...
		   wsptr[DCTSIZE*3 + 0] = (00 00 00 41) wsptr[DCTSIZE*3 + 1] = (00 00 00 51) ...
		   wsptr[DCTSIZE*4 + 0] = (00 00 00 02) wsptr[DCTSIZE*4 + 1] = (00 00 00 12) ...
		   wsptr[DCTSIZE*5 + 0] = (00 00 00 42) wsptr[DCTSIZE*5 + 1] = (00 00 00 52) ...
		   wsptr[DCTSIZE*6 + 0] = (00 00 00 03) wsptr[DCTSIZE*6 + 1] = (00 00 00 13) ...
		   wsptr[DCTSIZE*7 + 0] = (00 00 00 43)	wsptr[DCTSIZE*7 + 1] = (00 00 00 53) ...
		   */
		for(num = 0;num++;num <= ((DCTSIZE >> 1) -1)){
			wsptr[DCTSIZE*0 + num] = _mm_extract_pi16(data0123LL,(int64_t) num);
			wsptr[DCTSIZE*1 + num] = _mm_extract_pi16(data4567LL,(int64_t) num);
			wsptr[DCTSIZE*2 + num] = _mm_extract_pi16(data0123LH,(int64_t) num);
			wsptr[DCTSIZE*3 + num] = _mm_extract_pi16(data4567LH,(int64_t) num);
			wsptr[DCTSIZE*4 + num] = _mm_extract_pi16(data0123HL,(int64_t) num);
			wsptr[DCTSIZE*5 + num] = _mm_extract_pi16(data4567HL,(int64_t) num);
			wsptr[DCTSIZE*6 + num] = _mm_extract_pi16(data0123HH,(int64_t) num);
			wsptr[DCTSIZE*7 + num] = _mm_extract_pi16(data4567HH,(int64_t) num);
		}
		inptr += (DCTSIZE >> 1);			/* advance pointers to next column */
		quantptr += (DCTSIZE >> 1);
		wsptr += (DCTSIZE >> 1);
	}

	/* Pass 2: process rows from work array, store into output array. */
	/* Note that we must descale the results by a factor of 8 == 2**3, */
	/* and also undo the PASS1_BITS scaling. */

	wsptr = workspace;
	for (ctr = 0; ctr < (DCTSIZE >> 2); ctr++) {
		outptr = output_buf[ctr] + output_col;	

		/* Even part: reverse the even part of the forward DCT. */
		/* The rotator is sqrt(2)*c(-6). 
		 * (Original)
		 * z1 = (z2 + z3) * 0.541196100;
		 * tmp2 = z1 + z3 * -1.847759065;
		 * tmp3 = z1 + z2 * 0.765366865;
		 *
		 * (This implementation)
		 * tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065);
		 * tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100;
		 */	
		__m64 wsptra0 = _mm_unpacklo_pi16(wsptr[0],wsptr[8]);
		__m64 wsptra1 = _mm_unpacklo_pi16(wsptr[16],wsptr[24]);
		__m64 wsptra = _mm_unpacklo_pi16(wsptra0,wsptra1);

		__m64 wsptrb0 = _mm_unpacklo_pi16(wsptr[1],wsptr[9]);
		__m64 wsptrb1 = _mm_unpacklo_pi16(wsptr[17],wsptr[25]);
		__m64 wsptrb = _mm_unpacklo_pi16(wsptrb0,wsptrb1);

		__m64 wsptrc0 = _mm_unpacklo_pi16(wsptr[2],wsptr[10]);
		__m64 wsptrc1 = _mm_unpacklo_pi16(wsptr[18],wsptr[26]);
		__m64 wsptrc = _mm_unpacklo_pi16(wsptrc0,wsptrc1);

		__m64 wsptrd0 = _mm_unpacklo_pi16(wsptr[3],wsptr[11]);
		__m64 wsptrd1 = _mm_unpacklo_pi16(wsptr[19],wsptr[27]);
		__m64 wsptrd = _mm_unpacklo_pi16(wsptrd0,wsptrd1);

		__m64 wsptre0 = _mm_unpacklo_pi16(wsptr[4],wsptr[12]);
		__m64 wsptre1 = _mm_unpacklo_pi16(wsptr[20],wsptr[28]);
		__m64 wsptre = _mm_unpacklo_pi16(wsptre0,wsptre1);

		__m64 wsptrf0 = _mm_unpacklo_pi16(wsptr[5],wsptr[13]);
		__m64 wsptrf1 = _mm_unpacklo_pi16(wsptr[21],wsptr[29]);
		__m64 wsptrf = _mm_unpacklo_pi16(wsptrf0,wsptrf1);

		__m64 wsptrg0 = _mm_unpacklo_pi16(wsptr[6],wsptr[14]);
		__m64 wsptrg1 = _mm_unpacklo_pi16(wsptr[22],wsptr[30]);
		__m64 wsptrg = _mm_unpacklo_pi16(wsptrg0,wsptrg1);

		__m64 wsptrh0 = _mm_unpacklo_pi16(wsptr[7],wsptr[15]);
		__m64 wsptrh1 = _mm_unpacklo_pi16(wsptr[23],wsptr[31]);
		__m64 wsptrh = _mm_unpacklo_pi16(wsptrh0,wsptrh1);

		z2 = wsptrc;
		z3 = wsptrg;

		__m64 z23L = _mm_unpacklo_pi16(z2,z3);
		__m64 z23H = _mm_unpackhi_pi16(z2,z3);

		__m64 tmp3L = _mm_madd_pi16(z23L,T1);	//tmp3L
		__m64 tmp3H = _mm_madd_pi16(z23H,T1);	//tmp3H
		__m64 tmp2L = _mm_madd_pi16(z23L,T2);	//tmp2L
		__m64 tmp2H = _mm_madd_pi16(z23H,T2);	//tmp2H

		z2 = wsptra;
		z3 = wsptre;

		__m64 z23a = _mm_add_si64(z2,z3);
		__m64 z23s = _mm_sub_si64(z2,z3);

		__m64 tmp0L = _mm_loadlo_pi16(z23a);
		__m64 tmp0H = _mm_loadhi_pi16(z23a);

		tmp0L = _mm_srai_pi32(tmp0L,(16-CONST_BITS));
		tmp0H = _mm_srai_pi32(tmp0H,(16-CONST_BITS));

		__m64 tmp10L = _mm_add_pi32(tmp0L,tmp3L);
		__m64 tmp13L = _mm_sub_pi32(tmp0L,tmp3L);

		__m64 tmp10H = _mm_add_pi32(tmp0H,tmp3H);
		__m64 tmp13H = _mm_sub_pi32(tmp0H,tmp3H);

		__m64 tmp1L = _mm_loadlo_pi16(z23s);
		__m64 tmp1H = _mm_loadhi_pi16(z23s);

		tmp1L = _mm_srai_pi32(tmp1L,(16-CONST_BITS));
		tmp1H = _mm_srai_pi32(tmp1H,(16-CONST_BITS));

		__m64 tmp11L = _mm_add_pi32(tmp1L,tmp2L);
		__m64 tmp12L = _mm_sub_pi32(tmp1L,tmp2L);

		__m64 tmp11H = _mm_add_pi32(tmp1H,tmp2H);
		__m64 tmp12H = _mm_sub_pi32(tmp1H,tmp2H);

		/* Odd part per figure 8; the matrix is unitary and hence its
		 * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
		 */
		/*   ; (Original)
		 * z5 = (z3 + z4) * 1.175875602;
		 * z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
		 * z3 += z5;  z4 += z5;
		 *
		 * (This implementation)
		 * z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
		 *z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);
		 */
		tmp0 = wsptrh;
		tmp1 = wsptrf;
		tmp2 = wsptrd;
		tmp3 = wsptrb;

		z3 = _mm_add_pi16(tmp0,tmp2);	//z3 = tmp0 + tmp2
		z4 = _mm_add_pi16(tmp1,tmp3);	//z4 = tmp1 + tmp3

		__m64 z34L = _mm_unpacklo_pi16(z3,z4);
		__m64 z34H = _mm_unpackhi_pi16(z3,z4);

		__m64 z3L = _mm_madd_pi16(z34L,T3);	//z3L
		__m64 z3H = _mm_madd_pi16(z34H,T3);	//z3H

		__m64 z4L = _mm_madd_pi16(z34L,T4);	//z4L
		__m64 z4H = _mm_madd_pi16(z34H,T4);	//z4H

		/*
		 * (Original)
		 * z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
		 * tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
		 * tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
		 * z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
		 * tmp0 += z1 + z3;  tmp1 += z2 + z4;
		 * tmp2 += z2 + z3;  tmp3 += z1 + z4;
		 *
		 * (This implementation)
		 * tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
		 * tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
		 * tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
		 * tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
		 * tmp0 += z3;  tmp1 += z4;
		 * tmp2 += z3;  tmp3 += z4;
		 */
		__m64 tmp03L = _mm_unpacklo_pi16(tmp0,tmp3);
		__m64 tmp03H = _mm_unpackhi_pi16(tmp0,tmp3);

		tmp0L = _mm_madd_pi16(tmp03L,T5);
		tmp0H = _mm_madd_pi16(tmp03H,T5);

		tmp3L = _mm_madd_pi16(tmp03L,T6);
		tmp3H = _mm_madd_pi16(tmp03H,T6);

		tmp0L =  _mm_add_pi32(tmp0L,z3L);
		tmp0H = _mm_add_pi32(tmp0H,z3H);
		tmp3L =  _mm_add_pi32(tmp0L,z4L);
		tmp3H = _mm_add_pi32(tmp0H,z4H);

		tmp12L = _mm_unpacklo_pi16(tmp1,tmp2);
		tmp12H = _mm_unpackhi_pi16(tmp1,tmp2);

		tmp1L = _mm_madd_pi16(tmp12L,T5);
		tmp1H = _mm_madd_pi16(tmp12H,T5);

		tmp2L = _mm_madd_pi16(tmp12L,T6);
		tmp2H = _mm_madd_pi16(tmp12H,T6);

		tmp1L = _mm_add_pi32(tmp1L,z4L);
		tmp1H = _mm_add_pi32(tmp1H,z4H);
		tmp2L = _mm_add_pi32(tmp2L,z3L);
		tmp2H = _mm_add_pi32(tmp2H,z3H);

		/* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

		T10 = _mm_set1_pi32(PD_DECALE_P2);//[PD_DECALE_P2,PD_DECALE_P2]

		__m64 data0L = _mm_add_pi32(tmp10L,tmp3L);  //data0L
		__m64 data0H = _mm_add_pi32(tmp10H,tmp3H);  //data0H
		__m64 data7L = _mm_sub_pi32(tmp10L,tmp3L);  //data7L
		__m64 data7H = _mm_sub_pi32(tmp10H,tmp3H);  //data7H

		data0L = _mm_add_pi32(data0L,T9);
		data0H = _mm_add_pi32(data0H,T9);
		data0L = _mm_srai_pi32(data0L,DESCALE_P2);
		data0H = _mm_srai_pi32(data0H,DESCALE_P2);

		data7L = _mm_add_pi32(data7L,T10);
		data7H = _mm_add_pi32(data7H,T10);
		data7L = _mm_srai_pi32(data7L,DESCALE_P2);
		data7H = _mm_srai_pi32(data7H,DESCALE_P2);

		__m64 data0 = _mm_packs_pi32(data0L,data0H);  //data0 =(00 10 20 30)
		__m64 data7 = _mm_packs_pi32(data7L,data7H);  //data7 =(07 17 27 37)

		__m64 data1L = _mm_add_pi32(tmp11L,tmp2L);  //data1L
		__m64 data1H = _mm_add_pi32(tmp11H,tmp2H);  //data1H
		__m64 data6L = _mm_sub_pi32(tmp11L,tmp2L);  //data6L
		__m64 data6H = _mm_sub_pi32(tmp11H,tmp2H);  //data6H

		data1L = _mm_add_pi32(data1L,T10);
		data1H = _mm_add_pi32(data1H,T10);
		data1L = _mm_srai_pi32(data1L,DESCALE_P2);
		data1H = _mm_srai_pi32(data1H,DESCALE_P2);

		data6L = _mm_add_pi32(data6L,T10);
		data6H = _mm_add_pi32(data6H,T10);
		data6L = _mm_srai_pi32(data6L,DESCALE_P2);
		data6H = _mm_srai_pi32(data6H,DESCALE_P2);

		__m64 data1 = _mm_packs_pi32(data1L,data1H);  //data1 =(01 11 21 31)
		__m64 data6 = _mm_packs_pi32(data6L,data6H);  //data6 =(06 16 26 36)

		__m64 data06 = _mm_packs_pi16(data0,data6);	//data06 = (00 10 20 30 06 16 26 36)
		__m64 data17 = _mm_packs_pi16(data1,data7);	//data17 = (01 11 21 31 07 17 27 37)

		__m64 data2L = _mm_add_pi32(tmp12L,tmp1L);  //data2L
		__m64 data2H = _mm_add_pi32(tmp12H,tmp1H);  //data2H
		__m64 data5L = _mm_sub_pi32(tmp12L,tmp1L);  //data5L
		__m64 data5H = _mm_sub_pi32(tmp12H,tmp1H);  //data5H

		data2L = _mm_add_pi32(data2L,T10); 
		data2H = _mm_add_pi32(data2H,T10);
		data2L = _mm_srai_pi32(data2L,DESCALE_P2);
		data2H = _mm_srai_pi32(data2H,DESCALE_P2);

		data5L = _mm_add_pi32(data5L,T10); 
		data5H = _mm_add_pi32(data5H,T10);
		data5L = _mm_srai_pi32(data5L,DESCALE_P2);
		data5H = _mm_srai_pi32(data5H,DESCALE_P2);

		__m64 data2 = _mm_packs_pi32(data2L,data2H);  //data2 =(02 12 22 32)
		__m64 data5 = _mm_packs_pi32(data5L,data5H);  //data5 =(05 15 25 35)

		__m64 data3L = _mm_add_pi32(tmp13L,tmp0L);  //data3L
		__m64 data3H = _mm_add_pi32(tmp13H,tmp0H);  //data3H
		__m64 data4L = _mm_sub_pi32(tmp13L,tmp0L);  //data4L
		__m64 data4H = _mm_sub_pi32(tmp13H,tmp0H);  //data4H

		data3L = _mm_add_pi32(data3L,T10);
		data3H = _mm_add_pi32(data3H,T10);
		data3L = _mm_srai_pi32(data3L,DESCALE_P2);
		data3H = _mm_srai_pi32(data3H,DESCALE_P2);

		data4L = _mm_add_pi32(data4L,T10);
		data4H = _mm_add_pi32(data4H,T10);
		data4L = _mm_srai_pi32(data4L,DESCALE_P2);
		data4H = _mm_srai_pi32(data4H,DESCALE_P2);

		__m64 data3 = _mm_packs_pi32(data3L,data3H);  //data3 =(03 13 23 33)
		__m64 data4 = _mm_packs_pi32(data4L,data4H);  //data4 =(04 14 24 34)

		__m64 data24 = _mm_packs_pi16(data2,data4);		//data24 = (02 12 22 32 04 14 24 34)
		__m64 data35 = _mm_packs_pi16(data3,data5); 	//data35 = (03 13 23 33 05 15 25 35)

		//	CENTERJSAMPLE = 128 = [0000 0000 0000 0080] --> [0080] = [0000 0000 1000 0000]
		//	                ==> [0000 0000 0000 8080] --> [8080] = [1000 0000 1000 0000]
		//					==>	[1000 1000 1000 1000]
		T11 = _mm_unpacklo_pi8((__m64) CENTERJSAMPLE,(__m64) CENTERJSAMPLE);	//CENTERJSAMPLE = 128
		T11 = _mm_shuffle_pi16(T11,_mm_setzero_si64());
		//	data06 = (00 10 20 30 06 16 26 36)	
		//	data17 = (01 11 21 31 07 17 27 37)
		data06 = _mm_add_pi8(data06,T11);
		data17 = _mm_add_pi8(data17,T11); 	
		data24 = _mm_add_pi8(data24,T11);
		data35 = _mm_add_pi8(data35,T11);
		/* transpose coefficients(phase 1) */	
		__m64 dataAL = _mm_unpacklo_pi8(data06,data17);		//data0617L = (00 01 10 11 20 21 30 31)
		__m64 dataAH = _mm_unpackhi_pi8(data06,data17); 	//data0617H = (06 07 16 17 26 27 36 37)
		__m64 dataBL = _mm_unpacklo_pi8(data24,data35); 	//data2435L = (02 03 12 13 22 23 32 33)
		__m64 dataBH = _mm_unpackhi_pi8(data24,data35);		//data2435H = (04 05 14 15 24 25 34 35)
		/* transpose coefficients(phase 2) */
		__m64 dataLL = _mm_unpacklo_pi16(dataAL,dataBL);	//dataLL = (00 01 02 03 10 11 12 13)
		__m64 dataLH = _mm_unpackhi_pi16(dataAL,dataBL);  	//dataLH = (20 21 22 23 30 31 32 33)
		__m64 dataHL = _mm_unpacklo_pi16(dataBH,dataAH);  	//dataHL = (04 05 06 07 14 15 16 17)
		__m64 dataHH = _mm_unpackhi_pi16(dataBH,dataAH);  	//dataHH = (24 25 26 27 34 35 36 37)
		/* transpose coefficients(phase 3) */
		__m64 dataHLL = _mm_unpacklo_pi32(dataLL,dataHL);	//dataHLL = (00 01 02 03 04 05 06 07)
		__m64 dataHLH = _mm_unpackhi_pi32(dataLL,dataHL);	//dataHLH = (10 11 12 13 14 15 16 17)
		__m64 dataHHL = _mm_unpacklo_pi32(dataLH,dataHH);	//dataHHL = (20 21 22 23 24 25 26 27)
		__m64 dataHHH = _mm_unpackhi_pi32(dataLH,dataHH);	//dataHHH = (30 31 32 33 34 35 36 37)

		/*  dataHLL = (00 01 02 03 04 05 06 07) punpcklbh 0000 -->(0000 0001 0002 0003)  -->
		 *					punpckhbh 0000 -->(0004 0005 0006 0007)	 -->	
		 *	pextrh --> (0000 0000 0000 0000) (0000 0000 0000 0001) (0000 0000 0000 0002) (0000 0000 0000 0003)
		 *	pextrh --> (0004 0000 0000 0004) (0000 0000 0000 0005) (0000 0000 0000 0006) (0000 0000 0000 0007)
		 *	outptr[DCTSIZE*0 + 0] = (00) 
		 * */	
		for (num = 0; num++; num <= (DCTSIZE >> 1)-1){
			outptr[DCTSIZE*0 + num + 0] = (char) _mm_extract_pi16(_mm_loadlo_pi8_f(dataHLL),(__m64) num);
			outptr[DCTSIZE*0 + num + 4] = (char) _mm_extract_pi16(_mm_loadhi_pi8_f(dataHLL),(__m64) num);
			outptr[DCTSIZE*1 + num + 0] = (char) _mm_extract_pi16(_mm_loadlo_pi8_f(dataHLH),(__m64) num);
			outptr[DCTSIZE*1 + num + 4] = (char) _mm_extract_pi16(_mm_loadhi_pi8_f(dataHLH),(__m64) num);
			outptr[DCTSIZE*2 + num + 0] = (char) _mm_extract_pi16(_mm_loadlo_pi8_f(dataHHL),(__m64) num);
			outptr[DCTSIZE*2 + num + 4] = (char) _mm_extract_pi16(_mm_loadhi_pi8_f(dataHHL),(__m64) num);
			outptr[DCTSIZE*3 + num + 0] = (char) _mm_extract_pi16(_mm_loadlo_pi8_f(dataHHH),(__m64) num);
			outptr[DCTSIZE*3 + num + 4] = (char) _mm_extract_pi16(_mm_loadhi_pi8_f(dataHHH),(__m64) num);

			wsptr += (DCTSIZE >> 1);		/* advance pointer to next row */
		}
	}
}

#endif /* DCT_ISLOW_SUPPORTED */
