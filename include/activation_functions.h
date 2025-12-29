#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

// Existing functions...
double swish(double x);
double leaky_relu(double x, double alpha);
// ... other existing functions

// Add novel conversation functions
float* softmax(float* logits, int size);
void softmax_2d(float* scores, int rows, int cols);

#endif
