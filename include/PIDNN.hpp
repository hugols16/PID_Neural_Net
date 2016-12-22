#include <stdio.h>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

#define WINDOW_SIZE 30
#define DZ_TOL 1e-20

class PIDNN
{
	public:

		// To avoid division by zero
		float DivByZero( float value);

		// Neuron Acitvation Functions
		float ptransfer( float value );
		float itransfer( float value, float accum, float ts );
		float dtransfer( float cur_value, float past_value, float ts );
		
		float predict( float measured, float expected ); // Forward Propagation

		void BackProp( float dr, float dy, float d_ow[3], float d_hw[2][3], int i ); // Back Propagation
		
		void learn( void ); // PIDNN Learn and Update

		PIDNN( float l_rate, float weight_change, float tolerance, float timestep ); // Constructor
		~PIDNN( void ); // Destructor
 
	private:

		// Neural Net Parameters that can be tuned
		float n; // Learning Rate
		float max_change; // Maximum change in weight
		float tol; // Error Tolerance
		float ts; // Time Step
		float d_tol; // Maximum Tolerance for weight change
		int pos = 0;

		// Starting weights
		float hw[2][3] = {{-1, -1, -1},{1, 1, 1}};
		float ow[3] = {0.2, 0.1, 0.1};

		// Arrays to keep the current and past value
		float *y;
		float *r;
		float *i_y;
		float *i_r;
		float *u_p;
		float *u_i;
		float *u_d;
		float *x_p;
		float *x_i;
		float *x_d;
		float *v;
};