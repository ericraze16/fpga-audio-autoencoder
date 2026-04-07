//Numpy array shape [8]
//Min -0.031250000000
//Max 0.031250000000
//Number of zeros 1

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[8];
#else
bias2_t b2[8] = {-0.031250, 0.000000, -0.031250, 0.031250, 0.031250, 0.015625, 0.031250, -0.015625};
#endif

#endif
