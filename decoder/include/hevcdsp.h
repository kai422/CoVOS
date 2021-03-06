/*
 * HEVC video decoder
 *
 * Copyright (C) 2012 - 2013 Guillaume Martres
 * Copyright (C) 2013 - 2014 Pierre-Edouard Lepere
 *
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "hevc.h"

#ifndef AVCODEC_HEVCDSP_H
#define AVCODEC_HEVCDSP_H
struct AVFrame;
struct UpsamplInf;
struct HEVCWindow;


#define NTAPS_LUMA 8
#define NTAPS_CHROMA 4
#define US_FILTER_PREC  6

#define MAX_EDGE  4
#define MAX_EDGE_CR  2
#define N_SHIFT (20-8)
#define I_OFFSET (1 << (N_SHIFT - 1))


typedef struct HEVCDSPContext {
    void (*put_pcm)(uint8_t *_dst, ptrdiff_t _stride, int width, int height,
                    struct GetBitContext *gb, int pcm_bit_depth);

    void (*transform_add[4])(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride);

    void (*transform_skip)(int16_t *coeffs, int16_t log2_size);

    void (*transform_rdpcm)(int16_t *coeffs, int16_t log2_size, int mode);

    void (*idct_4x4_luma)(int16_t *coeffs);

    void (*idct[4])(int16_t *coeffs, int col_limit);

    void (*idct_dc[4])(int16_t *coeffs);

    void (*sao_band_filter)( uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride_dst, ptrdiff_t _stride_src, struct SAOParams *sao, int *borders, int width, int height, int c_idx);

    void (*sao_edge_filter[2])(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride_dst, ptrdiff_t _stride_src,  struct SAOParams *sao, int *borders, int _width, int _height, int c_idx, uint8_t *vert_edge, uint8_t *horiz_edge, uint8_t *diag_edge);

#if COM16_C806_EMT
    void (*idct_emt)(int16_t *coeffs, int16_t *dst, int log2_trafo_size, int TRANSFORM_MATRIX_SHIFT, int nLog2SizeMinus2, int maxLog2TrDynamicRange, int bitDepth, int ucMode, int intra_pred_mode, int emt_tu_idx);
#endif

    void (*put_hevc_qpel[10][2][2])(int16_t *dst, ptrdiff_t dststride, uint8_t *src, ptrdiff_t srcstride,
                                    int height, intptr_t mx, intptr_t my, int width);
    void (*put_hevc_qpel_uni[10][2][2])(uint8_t *dst, ptrdiff_t dststride, uint8_t *src, ptrdiff_t srcstride,
                                        int height, intptr_t mx, intptr_t my, int width);
    void (*put_hevc_qpel_uni_w[10][2][2])(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                          int height, int denom, int wx, int ox, intptr_t mx, intptr_t my, int width);

    void (*put_hevc_qpel_bi[10][2][2])(uint8_t *dst, ptrdiff_t dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                       int16_t *src2, ptrdiff_t src2stride,
                                       int height, intptr_t mx, intptr_t my, int width);
    void (*put_hevc_qpel_bi_w[10][2][2])(uint8_t *dst, ptrdiff_t dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                         int16_t *src2, ptrdiff_t src2stride,
                                         int height, int denom, int wx0, int wx1,
                                         int ox0, int ox1, intptr_t mx, intptr_t my, int width);
    void (*put_hevc_epel[10][2][2])(int16_t *dst, ptrdiff_t dststride, uint8_t *src, ptrdiff_t srcstride,
                                    int height, intptr_t mx, intptr_t my, int width);

    void (*put_hevc_epel_uni[10][2][2])(uint8_t *dst, ptrdiff_t dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                        int height, intptr_t mx, intptr_t my, int width);
    void (*put_hevc_epel_uni_w[10][2][2])(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                          int height, int denom, int wx, int ox, intptr_t mx, intptr_t my, int width);
    void (*put_hevc_epel_bi[10][2][2])(uint8_t *dst, ptrdiff_t dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                       int16_t *src2, ptrdiff_t src2stride,
                                       int height, intptr_t mx, intptr_t my, int width);
    void (*put_hevc_epel_bi_w[10][2][2])(uint8_t *dst, ptrdiff_t dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                         int16_t *src2, ptrdiff_t src2stride,
                                         int height, int denom, int wx0, int ox0, int wx1,
                                         int ox1, intptr_t mx, intptr_t my, int width);


    void (*hevc_h_loop_filter_luma)(uint8_t *_pix, ptrdiff_t _stride, int _beta, int *_tc, uint8_t *_no_p, uint8_t *_no_q);
    void (*hevc_v_loop_filter_luma)(uint8_t *_pix, ptrdiff_t _stride, int _beta, int *_tc, uint8_t *_no_p, uint8_t *_no_q);
    void (*hevc_h_loop_filter_chroma)(uint8_t *_pix, ptrdiff_t _stride, int *_tc, uint8_t *_no_p, uint8_t *_no_q);
    void (*hevc_v_loop_filter_chroma)(uint8_t *_pix, ptrdiff_t _stride, int *_tc, uint8_t *_no_p, uint8_t *_no_q);
    void (*hevc_h_loop_filter_luma_c)(uint8_t *_pix, ptrdiff_t _stride, int _beta, int *_tc, uint8_t *_no_p, uint8_t *_no_q);
    void (*hevc_v_loop_filter_luma_c)(uint8_t *_pix, ptrdiff_t _stride, int _beta, int *_tc, uint8_t *_no_p, uint8_t *_no_q);
    void (*hevc_h_loop_filter_chroma_c)(uint8_t *_pix, ptrdiff_t _stride, int *_tc, uint8_t *_no_p, uint8_t *_no_q);
    void (*hevc_v_loop_filter_chroma_c)(uint8_t *_pix, ptrdiff_t _stride, int *_tc, uint8_t *_no_p, uint8_t *_no_q);

    void (*upsample_base_layer_frame)  (struct AVFrame *FrameEL, struct AVFrame *FrameBL, short *Buffer[3], const struct HEVCWindow *Enhscal, struct UpsamplInf *up_info, int channel);
    void (*upsample_filter_block_luma_h[3])(
                                         int16_t *dst, ptrdiff_t dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                         int x_EL, int x_BL, int block_w, int block_h, int widthEL,
                                         const struct HEVCWindow *Enhscal, struct UpsamplInf *up_info);
    void (*upsample_filter_block_luma_v[3])(
                                         uint8_t *dst, ptrdiff_t dststride, int16_t *_src, ptrdiff_t _srcstride,
                                         int y_BL, int x_EL, int y_EL, int block_w, int block_h, int widthEL, int heightEL,
                                         const struct HEVCWindow *Enhscal, struct UpsamplInf *up_info);
    void (*upsample_filter_block_cr_h[3])(
                                           int16_t *dst, ptrdiff_t dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                           int x_EL, int x_BL, int block_w, int block_h, int widthEL,
                                           const struct HEVCWindow *Enhscal, struct UpsamplInf *up_info);
    void (*upsample_filter_block_cr_v[3])(
                                           uint8_t *dst, ptrdiff_t dststride, int16_t *_src, ptrdiff_t _srcstride,
                                           int y_BL, int x_EL, int y_EL, int block_w, int block_h, int widthEL, int heightEL,
                                           const struct HEVCWindow *Enhscal, struct UpsamplInf *up_info);
} HEVCDSPContext;

void ff_hevc_dsp_init(HEVCDSPContext *hpc, int bit_depth);

extern const int8_t ff_hevc_epel_filters[7][4];
extern const int8_t ff_hevc_qpel_filters[3][16];

#if COM16_C806_EMT
// ******************************************** Mode intra et SubSet ********************************************
extern const int emt_Tr_Set_H[35];
extern const int emt_Tr_Set_V[35];
extern const int g_aiTrSubSetIntra[3][2];
extern const int g_aiTrSubSetInter[2];
// ************************************************* Initialisation du tableau fastInvTrans *************************************************
typedef void InvTrans(int16_t*, int16_t*, int, int, int, int, int, int);
// ************************************************* Initialisation des transform??es *************************************************
extern int16_t g_aiTr4 [8][ 4][ 4];
extern int16_t g_aiTr8 [8][ 8][ 8];
extern int16_t g_aiTr16[8][16][16];
extern int16_t g_aiTr32[8][32][32];
// ****************************************************************** DCT_II ******************************************************************
void fastInverseDCT2_B4(int16_t *src, int16_t *dst, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT2_B8(int16_t *src, int16_t *dst, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT2_B16(int16_t *src, int16_t *dst, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT2_B32(int16_t *src, int16_t *dst, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT2_B64(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
// ****************************************************************** DCT_V ******************************************************************
void fastInverseDCT5_B4(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT5_B8(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT5_B16(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT5_B32(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
// ****************************************************************** DCT_VIII ******************************************************************
void fastInverseDCT8_B4(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT8_B8(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT8_B16(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDCT8_B32(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
// ****************************************************************** DST_I ******************************************************************
void fastInverseDST1_B4(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDST1_B8(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDST1_B16(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDST1_B32(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
// ****************************************************************** DST_VII ******************************************************************
void fastInverseDST7_B4(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDST7_B8(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDST7_B16(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
void fastInverseDST7_B32(int16_t *coeff, int16_t *block, int shift, int line, int zo, int use, int outputMinimum, int outputMaximum);
// *********************************************************************************************************************************************
#endif

void ff_hevcdsp_init_x86(HEVCDSPContext *c, const int bit_depth);
void ff_hevcdsp_init_arm(HEVCDSPContext *c, const int bit_depth);
#endif /* AVCODEC_HEVCDSP_H */
