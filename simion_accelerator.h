#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include <xmmintrin.h>

void set_npas(unsigned int);
void add_pa(unsigned int, double *, double);
void set_pasize(unsigned int, unsigned int, double, double);

void free();

void getFieldGradient(unsigned int, double *, double *, double *);
void fastAdjustAll(double *);
