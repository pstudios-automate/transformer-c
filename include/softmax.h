#ifndef SOFTMAX_H
#define SOFTMAX_H

float* softmax(float* logits, int size);
void softmax_2d(float* scores, int rows, int cols);

#endif
