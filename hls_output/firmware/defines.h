#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 64
#define N_INPUT_2_1 64
#define N_INPUT_3_1 1
#define OUT_HEIGHT_30 66
#define OUT_WIDTH_30 66
#define N_CHAN_30 1
#define OUT_HEIGHT_2 64
#define OUT_WIDTH_2 64
#define N_FILT_2 8
#define OUT_HEIGHT_2 64
#define OUT_WIDTH_2 64
#define N_FILT_2 8
#define OUT_HEIGHT_5 32
#define OUT_WIDTH_5 32
#define N_FILT_5 8
#define OUT_HEIGHT_31 34
#define OUT_WIDTH_31 34
#define N_CHAN_31 8
#define OUT_HEIGHT_6 32
#define OUT_WIDTH_6 32
#define N_FILT_6 4
#define OUT_HEIGHT_6 32
#define OUT_WIDTH_6 32
#define N_FILT_6 4
#define OUT_HEIGHT_9 16
#define OUT_WIDTH_9 16
#define N_FILT_9 4
#define N_SIZE_0_10 1024
#define N_LAYER_11 32
#define N_LAYER_11 32
#define N_LAYER_14 1024
#define N_LAYER_14 1024
#define N_SIZE_1_29 16
#define N_SIZE_2_29 16
#define N_SIZE_3_29 4
#define OUT_HEIGHT_32 18
#define OUT_WIDTH_32 18
#define N_CHAN_32 4
#define OUT_HEIGHT_18 16
#define OUT_WIDTH_18 16
#define N_FILT_18 4
#define OUT_HEIGHT_18 16
#define OUT_WIDTH_18 16
#define N_FILT_18 4
#define OUT_HEIGHT_21 32
#define OUT_WIDTH_21 32
#define N_CHAN_21 4
#define OUT_HEIGHT_33 34
#define OUT_WIDTH_33 34
#define N_CHAN_33 4
#define OUT_HEIGHT_22 32
#define OUT_WIDTH_22 32
#define N_FILT_22 8
#define OUT_HEIGHT_22 32
#define OUT_WIDTH_22 32
#define N_FILT_22 8
#define OUT_HEIGHT_25 64
#define OUT_WIDTH_25 64
#define N_CHAN_25 8
#define OUT_HEIGHT_34 66
#define OUT_WIDTH_34 66
#define N_CHAN_34 8
#define OUT_HEIGHT_26 64
#define OUT_WIDTH_26 64
#define N_FILT_26 1
#define OUT_HEIGHT_26 64
#define OUT_WIDTH_26 64
#define N_FILT_26 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<8,1>, 1*1> input_t;
typedef nnet::array<ap_fixed<8,1>, 1*1> layer30_t;
typedef ap_fixed<8,1> model_default_t;
typedef nnet::array<ap_fixed<8,1>, 8*1> layer2_t;
typedef ap_fixed<8,2> weight2_t;
typedef ap_fixed<8,2> bias2_t;
typedef nnet::array<ap_fixed<8,1>, 8*1> layer4_t;
typedef ap_fixed<18,8> enc_act1_table_t;
typedef nnet::array<ap_fixed<8,1>, 8*1> layer5_t;
typedef nnet::array<ap_fixed<8,1>, 8*1> layer31_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer6_t;
typedef ap_fixed<8,2> weight6_t;
typedef ap_fixed<8,2> bias6_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer8_t;
typedef ap_fixed<18,8> enc_act2_table_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer9_t;
typedef nnet::array<ap_fixed<8,1>, 32*1> layer11_t;
typedef ap_fixed<8,2> weight11_t;
typedef ap_fixed<8,2> bias11_t;
typedef ap_uint<1> layer11_index;
typedef nnet::array<ap_fixed<8,1>, 32*1> layer13_t;
typedef ap_fixed<18,8> latent_act_table_t;
typedef nnet::array<ap_fixed<8,1>, 1024*1> layer14_t;
typedef ap_fixed<8,2> weight14_t;
typedef ap_fixed<8,2> bias14_t;
typedef ap_uint<1> layer14_index;
typedef nnet::array<ap_fixed<8,1>, 1024*1> layer16_t;
typedef ap_fixed<18,8> dec_dense_act_table_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer29_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer32_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer18_t;
typedef ap_fixed<8,2> weight18_t;
typedef ap_fixed<8,2> bias18_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer20_t;
typedef ap_fixed<18,8> dec_act1_table_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer21_t;
typedef nnet::array<ap_fixed<8,1>, 4*1> layer33_t;
typedef nnet::array<ap_fixed<8,1>, 8*1> layer22_t;
typedef ap_fixed<8,2> weight22_t;
typedef ap_fixed<8,2> bias22_t;
typedef nnet::array<ap_fixed<8,1>, 8*1> layer24_t;
typedef ap_fixed<18,8> dec_act2_table_t;
typedef nnet::array<ap_fixed<8,1>, 8*1> layer25_t;
typedef nnet::array<ap_fixed<8,1>, 8*1> layer34_t;
typedef nnet::array<ap_fixed<8,1>, 1*1> layer26_t;
typedef ap_fixed<8,2> weight26_t;
typedef ap_fixed<8,2> bias26_t;
typedef nnet::array<ap_fixed<8,1>, 1*1> result_t;
typedef ap_fixed<18,8> dec_out_act_table_t;

#endif
