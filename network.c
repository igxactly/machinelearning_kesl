#include "network.h"
#include <stdio.h>
#include <stdlib.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist/mnist.h"
#include "timeutils.h"
#include <math.h>

#include <cilk/cilk.h>

double randn(void);
static void print_error(struct network *net, enum DATA_T t, int layer);
static double sigmoid(double z);
static double sigmoid_prime(double z);
static void print_arr(enum DATA_T t, struct network *net, char *func, int line);
char *read_conf_file(char *conf_name);

void run(struct network *net, char *conf_file_path)
{

	// Initialze from configuration file
	initializer(net, conf_file_path);

	// read and fill up network input later
	reader(net);

	// run the training
	update(net);

	// report to the file
	report(net);
}

/* Init network struct from configuration file */
void initializer(struct network *net, char *conf_fname)
{
	int i,j,k;
	int before_ac_weights = 0;
	int before_ac_neurals = 0;
	char *conf_str = read_conf_file(conf_fname);

	net->best_recog = 0.0;

	net->tokens = json_parsing(conf_str, &net->nr_tokens);
	net->num_layer = atoi((char *) parse_value(net->tokens, conf_str, "num_layer", net->nr_tokens));
	net->layer_size = (int *) parse_value(net->tokens, conf_str, "layer_size", net->nr_tokens);
	net->learning_rate = strtod((char *) parse_value(net->tokens, conf_str, "learning_rate", net->nr_tokens), NULL);
	net->mini_batch_size = atoi((char *) parse_value(net->tokens, conf_str, "mini_batch_size", net->nr_tokens));
	net->epoch = atoi((char *) parse_value(net->tokens, conf_str, "epoch", net->nr_tokens));

	net->ac_weight = (int *) malloc(sizeof(double) * net->num_layer);
	net->ac_neuron = (int *) malloc(sizeof(double) * net->num_layer);

	net->train_q_name = (char *) parse_value(net->tokens, conf_str, "train_q", net->nr_tokens);
	net->train_a_name = (char *) parse_value(net->tokens, conf_str, "train_a", net->nr_tokens);
	net->test_q_name = (char *) parse_value(net->tokens, conf_str, "test_q", net->nr_tokens);
	net->test_a_name = (char *) parse_value(net->tokens, conf_str, "test_a", net->nr_tokens);
	net->report_file = (char *) parse_value(net->tokens, conf_str, "report_file", net->nr_tokens);


	for (i = 0; i < net->num_layer; i++) {
		net->ac_neuron[i] = net->layer_size[i] + before_ac_neurals;
		before_ac_neurals = net->ac_neuron[i];

		if (i == net->num_layer-1)
			continue;

		net->ac_weight[i] = net->layer_size[i] * net->layer_size[i+1] + before_ac_weights;
		before_ac_weights = net->ac_weight[i];
	}

	net->neuron = (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net));
	net->zs = (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net));
	net->error =  (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net));
	net->bias = (double *) malloc(sizeof(double) * TOTAL_NEURONS(net));
	net->weight = (double *) malloc(sizeof(double) * TOTAL_WEIGHTS(net));

	// init weight with bias with random values
	for (i = 0; i < TOTAL_WEIGHTS(net); i++) {
		net->weight[i] = randn();
	}

	for (i = 0; i < TOTAL_NEURONS(net); i++) {
		net->bias[i] = randn();
	}

	free(conf_str);
}

/* Read and fillup the first layer of neurons */
void reader(struct network *net)
{
	int i, j, k, l;
	mnist_data *train_data, *test_data;
	int first_layer_size = AC_NEURONS(net, 0);
	int last_layer_size = net->layer_size[net->num_layer-1];

	// Reading train data
	if (mnist_load(net->train_q_name, net->train_a_name, &train_data, &net->nr_train_data) < 0)
		printf("Mnist train data reading error occured\n");

	net->train_q = (double *) malloc(net->layer_size[0] * net->nr_train_data * sizeof(double));
	net->train_a = (int *) calloc(net->nr_train_data, sizeof(int));

	// copy train imgae&label
	for (i = 0; i < net->nr_train_data; i++) {
		for (j = 0; j < first_layer_size; j++) {
			DATA_TRAIN_Q(net, i, j) = train_data[i].data[j/28][j%28];
		}
		DATA_TRAIN_A(net, i) = train_data[i].label;
	}

	// Reading test data
	if (mnist_load(net->test_q_name, net->test_a_name, &test_data, &net->nr_test_data) < 0)
		printf("Mnist test data reading error occured\n");

	net->test_q = (double *) malloc(net->layer_size[0] * net->nr_test_data * sizeof(double));
	net->test_a = (int *) calloc(net->nr_test_data, sizeof(int));

	// copy test imgae&label
	for (i = 0; i < net->nr_test_data; i++) {
		for (j = 0; j < first_layer_size; j++) {
			DATA_TEST_Q(net, i, j) = test_data[i].data[j/28][j%28];
		}

		DATA_TEST_A(net, i) = test_data[i].label;
	}

	free(train_data);
	free(test_data);
}

// run the training
void update(struct network *net)
{
	int i, j, k, l;
	int nr_train = net->nr_train_data;
	int nr_loop = (int)(net->nr_train_data/net->mini_batch_size);
	int first_layer_size = AC_NEURONS(net, 0);
	int last_layer_size = net->layer_size[net->num_layer-1];
	int recog = 0;


	// initialize the first input layer of neuron
	for (i = 0; i < net->epoch; i++) {
		for (j = 0; j < nr_loop; j++) {

			// copy input and output for SGD
			for (k = 0; k < net->mini_batch_size; k++) {
				int s_index = (k+j*19)%nr_train;
				// copy input to first layer of neuron array
				for (l = 0; l < first_layer_size; l++)
					NEURON(net, 0, k, l) = DATA_TRAIN_Q(net, s_index, l);

				// copy output to error array
				ERROR(net, net->num_layer-1, k, DATA_TRAIN_A(net, s_index)) = 1.0;
			}
			// feedforward + backpropagation
			learner(net);
		}
		// test per every epoch
		recog = evaluator(net);
		if (recog > net->best_recog)
			net->best_recog = recog;
		printf("%dth epoch %d / %d\n", i, recog, net->nr_test_data);
	}
}

/* Operation like backpropagation */
void learner(struct network *net)
{
	int i, j, k, l;
	double sum = 0.0;

	// feedforward
	START_TIME(net->t_feedforward);
	sum = 0.0;
	for (i = 0; i < net->num_layer-1; i++) {
#pragma cilk grainsize = 50
		cilk_for (j = 0; j < net->mini_batch_size; j++) {
			for (k = 0; k < net->layer_size[i+1]; k++) {
#pragma simd
				for (l = 0; l < net->layer_size[i]; l++) {
					sum = sum + NEURON(net, i, j, l) * WEIGHT(net, i, l, k);
				}

				ZS(net, i+1, j, k) = sum + BIAS(net, i+1, k);
				NEURON(net, i+1, j, k) = sigmoid(ZS(net, i+1, j, k));
				sum = 0.0;
			}
		}
	}
	END_TIME(net->t_feedforward);

	START_TIME(net->t_back_pass);
	// calculate delta
#pragma cilk grainsize = 50
	cilk_for (i = 0; i < net->mini_batch_size; i++) {
		for (j = 0; j < net->layer_size[net->num_layer-1]; j++) {
			//	calculate delta in last output layer
			ERROR(net, net->num_layer-1, i, j) =
				(NEURON(net, net->num_layer-1, i, j)-ERROR(net, net->num_layer-1, i, j)) *
				sigmoid_prime(ZS(net, net->num_layer-1, i, j));
		}
	}

	sum = 0.0;
	for (i = net->num_layer-2; i > 0; i--) {
#pragma cilk grainsize = 50
		cilk_for (j = 0; j < net->mini_batch_size; j++) {
			for (k = 0; k < net->layer_size[i]; k++) {

				#pragma simd
				for (l = 0; l < net->layer_size[i+1]; l++) {
					//	calculate delta from before layer
					sum = sum + ERROR(net, i+1, j, l) * WEIGHT(net, i, k, l);
				}

				ERROR(net, i, j, k) = sum * sigmoid_prime(ZS(net, i, j, k));
				sum = 0.0;
			}
		}
	}
	END_TIME(net->t_back_pass);

	double delta_sum = 0.0;
	double eta = net->learning_rate;
	double mini = (double) net->mini_batch_size;

	START_TIME(net->t_backpropagation);
	// update bias
	delta_sum = 0.0;
	for (i = 1; i < net->num_layer; i++) {
#pragma cilk grainsize = 50
		cilk_for (j = 0; j < net->layer_size[i]; j++) {
			for (k = 0; k < net->mini_batch_size; k++) {
				delta_sum += ERROR(net, i, k, j);
			}
			BIAS(net, i, j) -= (eta/mini)*delta_sum;
			delta_sum = 0.0;
		}
	}

	// update weight
	delta_sum = 0.0;
	for (i = 0; i < net->num_layer-1; i++) {
#pragma cilk grainsize = 50
		cilk_for (j = 0; j < net->layer_size[i]; j++) {
			for (k = 0; k < net->layer_size[i+1]; k++) {
#pragma simd
				for (l = 0; l < net->mini_batch_size; l++) {
					//	calculate delta from before layer
					delta_sum  += (NEURON(net, i, l, j) * ERROR(net, i+1, l, k));
				}
				WEIGHT(net, i, j, k) -= (eta/mini)*delta_sum;
				delta_sum = 0.0;
			}
		}
	}
	END_TIME(net->t_backpropagation);
}

static double sigmoid(double z)
{
	return (1/(1 + exp(-z)));
}

static double sigmoid_prime(double z)
{
	return sigmoid(z)*(1-sigmoid(z));
}

/* evaluator to show how it works well */
int evaluator(struct network *net)
{
	int nr_true = 0;

	int i, j, k, l;
	double sum = 0.0;
	int nr_loop = (int)(net->nr_test_data);
	int first_layer_size = AC_NEURONS(net, 0);
	int last_layer_size = net->layer_size[net->num_layer-1];

	for (i = 0; i < nr_loop; i++) {
		// copy input to first layer of neuron array
		for (j = 0; j < first_layer_size; j++) {
			NEURON(net, 0, 0, j) = DATA_TEST_Q(net, i, j);
		}

		//feedforward
		sum = 0.0;
		for (j = 0; j < net->num_layer-1; j++) {
#pragma cilk grainsize = 50
			cilk_for (k = 0; k < net->layer_size[j+1]; k++) {
#pragma simd
				for (l = 0; l < net->layer_size[j]; l++) {
					sum = sum + NEURON(net, j, 0, l) * WEIGHT(net, j, l, k);
				}

				ZS(net, j+1, 0, k) = sum + BIAS(net, j+1, k);
				NEURON(net, j+1, 0, k) = sigmoid(ZS(net, j+1, 0, k));
				sum = 0.0;
			}
		}

		double max = NEURON(net, net->num_layer-1, 0, 0);
		int max_idx = 0;

		for (j = 0; j < last_layer_size; j++) {
			if (NEURON(net, net->num_layer-1, 0, j) > max) {
				max = NEURON(net, net->num_layer-1, 0, j);
				max_idx = j;
			}
		}

		if (DATA_TEST_A(net, i) == max_idx)
			nr_true ++;
	}

	return nr_true;
}

void report(struct network *net)
{

	int i = 0;
	FILE *f = fopen(net->report_file, "w");
	if (f == NULL) {
		printf("%s open failed\n", net->report_file);
		exit(1);
	}
	fprintf( f, "SGD \n");
	fprintf( f, "mini_batch_size : %d\n", net->mini_batch_size);
	fprintf( f, "num_layer : %d\n", net->num_layer);
	fprintf( f, "layers : [");
	for (i = 0; i < net->num_layer; i++) {
		fprintf( f, "%d ", net->layer_size[i]);
	}
	fprintf( f, "]\n");
	fprintf( f, "epoch : %d\n", net->epoch);
	fprintf( f, "learning_rate : %f\n", net->learning_rate);
	fprintf( f, "recognization rate : %d/%d\n", net->best_recog, net->nr_test_data);

	fprintf( f, "========TIMES======\n");

	fprintf( f, "feedforward : %f sec\n", TOTAL_SEC_TIME(net->t_feedforward));
	fprintf( f, "back_pass : %f sec\n", TOTAL_SEC_TIME(net->t_back_pass));
	fprintf( f, "backpropagation : %f sec\n", TOTAL_SEC_TIME(net->t_backpropagation));
	fprintf( f, "total : %f sec\n", 
			TOTAL_SEC_TIME(net->t_feedforward) + 
			TOTAL_SEC_TIME(net->t_back_pass) +
			TOTAL_SEC_TIME(net->t_backpropagation));

	fclose(f);
}

char *read_conf_file(char *conf_name)
{
	FILE *fp;
	long lSize;
	char *buffer;

	if ((fp = fopen ( conf_name , "rb" )) == NULL) {
		printf("%s fopen failed\n", conf_name);
		exit(1);
	}

	fseek( fp , 0L , SEEK_END);
	lSize = ftell( fp );
	rewind( fp );

	/* allocate memory for entire content */
	if ((buffer = calloc( 1, lSize+1 )) == NULL) {
		fclose(fp);
		printf("buffer memory alloc fails\n");
		exit(1);
	}

	/* copy the file into the buffer */
	if( 1!=fread( buffer , lSize, 1 , fp) ) {
		fclose(fp);
		free(buffer);
		printf("%s entire read fails\n", conf_name);
		exit(1);
	}

	return buffer;
}

static void print_arr(enum DATA_T t, struct network *net, char *func, int line)
{
	int i, j, k;

	if (t == BIAS) {
		printf("BIAS %s[%d]\n", func, line);
		for (i = 1; i < net->num_layer; i++) {
			for (j = 0; j < net->layer_size[i]; j++) {

				printf("%3.2f ",BIAS(net, i, j));
			}
			printf("\n");
		}
	} else if(t == WEIGHT) {
		printf("WEIGHT %s[%d]\n", func, line);
		for (i = 0; i < net->num_layer; i++) {
			for (j = 0; j < net->layer_size[i]; j++) {
				for (k = 0; k < net->layer_size[i+1]; k++) {

					printf("%3.2f ",WEIGHT(net, i, j, k));
				}
			}
			printf("\n");
		}printf("\n");
	} else if(t == ERROR) {
		printf("ERROR %s[%d]\n", func, line);
		for (i = 0; i < net->num_layer; i++) {
			for (j = 0; j < net->mini_batch_size; j++) {
				for (k = 0; k < net->layer_size[i]; k++) {

					printf("%3.2f ",ERROR(net, i, j, k));
				}
			}
			printf("\n");
		}printf("\n");
	} else if(t == ZS) {
		printf("ZS %s[%d]\n", func, line);
		for (i = 0; i < net->num_layer; i++) {
			for (j = 0; j < net->mini_batch_size; j++) {
				for (k = 0; k < net->layer_size[i]; k++) {

					printf("%3.2f ",ZS(net, i, j, k));
				}
			}
			printf("\n");
		}printf("\n");
	} else if(t == NEURON) {
		printf("NEURON %s[%d]\n", func, line);
		for (i = 0; i < net->num_layer; i++) {
			for (j = 0; j < net->mini_batch_size; j++) {
				for (k = 0; k < net->layer_size[i]; k++) {

					printf("%3.2f ",NEURON(net, i, j, k));
				}
			}
			printf("\n");
		}printf("\n");
	}
}

static void print_error(struct network *net, enum DATA_T t, int layer)
{
	int i, j;

	if (t == ERROR) {
		printf("ERROR\n");
		for (i = 0 ; i < net->mini_batch_size; i++) {
			for (j = 0; j < net->layer_size[layer]; j++) {
				printf("%1.2f ", ERROR(net, layer, i , j));
			}
			printf("\n");
		}
	} else if(t == NEURON){
		printf("NEURON\n");
		for (i = 0 ; i < net->mini_batch_size; i++) {
			for (j = 0; j < net->layer_size[layer]; j++) {
				printf("%1.2f ", NEURON(net, layer, i , j));
			}
			printf("\n");
		}
	} else if(t == ZS){
		printf("ZS\n");
		for (i = 0 ; i < net->mini_batch_size; i++) {
			for (j = 0; j < net->layer_size[layer]; j++) {
				printf("%1.2f ", ZS(net, layer, i , j));
			}
			printf("\n");
		}
	} else if(t == WEIGHT){
		printf("WEIGHT\n");
		for (i = 0 ; i < net->layer_size[layer]; i++) {
			for (j = 0; j < net->layer_size[layer+1]; j++) {
				printf("%1.2f ", WEIGHT(net, layer, i , j));
			}
			printf("\n");
		}
	} else if(t == BIAS){
		printf("BIAS\n");
		for (i = 0 ; i < net->layer_size[layer]; i++) {
			printf("%1.2f ", BIAS(net, layer, i));
		}
		printf("\n");
	}

}

double randn(void)
{
	double v1, v2, s;

	do {
		v1 =  2 * ((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
		v2 =  2 * ((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
		s = v1 * v1 + v2 * v2;
	} while (s >= 1 || s == 0);

	s = sqrt( (-2 * log(s)) / s );

	return v1 * s;
}
