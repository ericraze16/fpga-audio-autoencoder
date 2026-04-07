#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<result_t> &layer28_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer28_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 72>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight6_t, 288>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 4>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight11_t, 32768>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 32>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight14_t, 32768>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 1024>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight18_t, 144>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 4>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight22_t, 288>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 8>(b22, "b22.txt");
        nnet::load_weights_from_txt<weight26_t, 72>(w26, "w26.txt");
        nnet::load_weights_from_txt<bias26_t, 1>(b26, "b26.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS STREAM variable=layer30_out depth=4356
    nnet::zeropad2d_cl<input_t, layer30_t, config30>(input_1, layer30_out); // zp2d_enc_conv1

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=4096
    nnet::conv_2d_cl<layer30_t, layer2_t, config2>(layer30_out, layer2_out, w2, b2); // enc_conv1

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=4096
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // enc_act1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=1024
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // enc_pool1

    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS STREAM variable=layer31_out depth=1156
    nnet::zeropad2d_cl<layer5_t, layer31_t, config31>(layer5_out, layer31_out); // zp2d_enc_conv2

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=1024
    nnet::conv_2d_cl<layer31_t, layer6_t, config6>(layer31_out, layer6_out, w6, b6); // enc_conv2

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1024
    nnet::relu<layer6_t, layer8_t, relu_config8>(layer6_out, layer8_out); // enc_act2

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=256
    nnet::pooling2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // enc_pool2

    auto& layer10_out = layer9_out;
    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::dense<layer9_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11); // latent_space

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1
    nnet::relu<layer11_t, layer13_t, relu_config13>(layer11_out, layer13_out); // latent_act

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1
    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14); // dec_dense

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=1
    nnet::relu<layer14_t, layer16_t, relu_config16>(layer14_out, layer16_out); // dec_dense_act

    hls::stream<layer29_t> layer29_out("layer29_out");
    #pragma HLS STREAM variable=layer29_out depth=256
    nnet::repack_stream<layer16_t, layer29_t, 1024>(layer16_out, layer29_out); // repack_dec_reshape

    hls::stream<layer32_t> layer32_out("layer32_out");
    #pragma HLS STREAM variable=layer32_out depth=324
    nnet::zeropad2d_cl<layer29_t, layer32_t, config32>(layer29_out, layer32_out); // zp2d_dec_conv1

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=256
    nnet::conv_2d_cl<layer32_t, layer18_t, config18>(layer32_out, layer18_out, w18, b18); // dec_conv1

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=256
    nnet::relu<layer18_t, layer20_t, relu_config20>(layer18_out, layer20_out); // dec_act1

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=1024
    nnet::resize_nearest<layer20_t, config21>(layer20_out, layer21_out); // dec_up1

    hls::stream<layer33_t> layer33_out("layer33_out");
    #pragma HLS STREAM variable=layer33_out depth=1156
    nnet::zeropad2d_cl<layer21_t, layer33_t, config33>(layer21_out, layer33_out); // zp2d_dec_conv2

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=1024
    nnet::conv_2d_cl<layer33_t, layer22_t, config22>(layer33_out, layer22_out, w22, b22); // dec_conv2

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=1024
    nnet::relu<layer22_t, layer24_t, relu_config24>(layer22_out, layer24_out); // dec_act2

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=4096
    nnet::resize_nearest<layer24_t, config25>(layer24_out, layer25_out); // dec_up2

    hls::stream<layer34_t> layer34_out("layer34_out");
    #pragma HLS STREAM variable=layer34_out depth=4356
    nnet::zeropad2d_cl<layer25_t, layer34_t, config34>(layer25_out, layer34_out); // zp2d_dec_out_conv

    hls::stream<layer26_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=4096
    nnet::conv_2d_cl<layer34_t, layer26_t, config26>(layer34_out, layer26_out, w26, b26); // dec_out_conv

    nnet::relu<layer26_t, result_t, relu_config28>(layer26_out, layer28_out); // dec_out_act

}
