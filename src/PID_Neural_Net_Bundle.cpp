#include <PIDNN.hpp>
using namespace std;

float weighted_mov_avg( float r[], int i, float size ) {
	float value  = 0;
    float weights[20] = { 0.0023597,
                          0.0051437,
                          0.0102821,
                          0.0188492,
                          0.0316884,
                          0.0488547,
                          0.0690735,
                          0.0895602,
                          0.1064922,
                          0.1161232,
                          0.1161232,
                          0.1064922,
                          0.0895602,
                          0.0690735,
                          0.0488547,
                          0.0316884,
                          0.0188492,
                          0.0102821,
                          0.0051437,
                          0.0023597};
	int j = 0;
	while (j < 20) {
	    for (int k = (i - size/2) + j*size/20; k < (i - size/2) + (j+1)*size/20; k++) {
	    value += weights[j]*r[k]*20/size;
	    }
	    j++;
	}
	return value;
}

int main() {

	printf("Initializing Variables and Creating File\n");
	// Create the file
	ofstream file;
	file.open("data.dat");
    file << "#x y" << std::endl;

    // Simulated System Variables
	float ts = 0.01;
	float tend = 2100;
	float freq = 1/6.28; // Hz

	// for square wave
	float period_length = 30;
	float number_period = floor(tend/period_length);
	float plateau_length = 7;
	float moving_avg_size = (period_length/2.0 - plateau_length)/ts;

	int time_num = tend/ts;
	float time_l[time_num];
	float r[time_num];
	float expected[time_num];
	float measured[time_num];
	float output = 0;

	// Dynamic System Variables
	float xddot = 0, xdot = 0, x = 0;
	float G = 1, a = 0.1, b = 1, c = 0.1;

	// Neural Net Variables
	float learning_rate = 1e-1, max_change = 0.2, tolerance = 1e-4;

	printf("Creating Square Wave Function Function\n");
	int k = 0;
	while(k < number_period) {
		for (int i = 0 + k*period_length/ts; i < k*period_length/ts+ period_length/(2*ts); i++) {
			time_l[i] = i*ts;
			expected[i] = 0;
			r[i] = 0;
		}
		for (int j = period_length/(2*ts) + k*period_length/ts; j < (k+1)*period_length/(ts); j++) {
			time_l[j] = j*ts;
			expected[j] = 0.5;
			r[j] = 0.5;
		}
		k++;
	}
	for (int m = moving_avg_size/(2); m < time_num - moving_avg_size/(2); m++) {
		expected[m] = weighted_mov_avg(r, m, moving_avg_size);
	}

    printf("Initializing Neural Net\n");
    PIDNN* NeuralNet = new PIDNN(learning_rate, max_change, tolerance, ts);

    // Initial Conditions
    measured[0] = 0;

    //Print first line of file
	file << time_l[0] << ' ' << expected[0] << ' ' << measured[0] << endl;

    printf("Simulating the Neural net response\n");
    for (int i = 1; i < time_num; i++) {
    	output = NeuralNet->predict(measured[i-1], expected[i]);
    	xddot = (output/a) - xdot*b - measured[i-1]*c;
  		xdot = xdot + xddot*ts;
  		measured[i] = measured[i-1] + xdot*ts;
		file << time_l[i] << ' ' << expected[i] << ' ' << measured[i] << endl;

    }
    file.close();

    printf("Creating Image\n");
    system("gnuplot plot.plt");

    printf("Opening Image\n");
    system("open -a Google\\ Chrome plot.svg");

    printf("Done\n");

	return 0;
}