#include "network.h"
#include <stdio.h>
#include <stdlib.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist/mnist.h"
#include "timeutils.h"
#include <math.h>

#include <jsmn/jsmn.h>
#include <cilk/cilk.h>

#define MAX_CPU         256

static void feedforward(struct network *net, int thread);
static void back_pass(struct network *net, int thread1, int thread2);
static void backpropagation(struct network *net, int thread1, int thread2);

static double randn(void);
static double sigmoid(double z);
static double sigmoid_prime(double z);

/* Init network struct from configuration file */
void init(struct network *net, char *conf_str)
{
	int i,j,k;
	int before_ac_weights = 0;
	int before_ac_neurals = 0;

    timeutils *feedforward = &net->t_feedforward;
    timeutils *back_pass = &net->t_back_pass;
    timeutils *backpropagation = &net->t_backpropagation;

	net->best_recog = 0.0;
    TIMER_INIT(feedforward);
    TIMER_INIT(back_pass);
    TIMER_INIT(backpropagation);

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
void train(struct network *net, void *threads)
{
	int i, j, k, l;
	int nr_train = net->nr_train_data;
	int nr_loop = (int)(net->nr_train_data/net->mini_batch_size);
	int first_layer_size = AC_NEURONS(net, 0);
	int last_layer_size = net->layer_size[net->num_layer-1];
	int recog = 0;
    int *thread = (int *) threads;


	// initialize the first input layer of neuron
	for (i = 0; i < net->epoch; i++) {
		for (j = 0; j < nr_loop; j++) {

			// copy input and output for SGD
			for (k = 0; k < net->mini_batch_size; k++) {
                int s_index = (int) rand()%nr_train;
				// copy input to first layer of neuron array
				for (l = 0; l < first_layer_size; l++)
					NEURON(net, 0, k, l) = DATA_TRAIN_Q(net, s_index, l);

                for (l = 0; l < last_layer_size; l++)
                    ERROR(net, net->num_layer-1, k, l) = 0.0;
				// copy output to error array
				ERROR(net, net->num_layer-1, k, DATA_TRAIN_A(net, s_index)) = 1.0;
			}
            // feedforward + back_pass
            feedforward(net, thread[0]);
            back_pass(net, thread[1], thread[2]);
            backpropagation(net, thread[3], thread[4]);
		}
		// test per every epoch
		recog = predict(net);
		if (recog > net->best_recog)
			net->best_recog = recog;
		printf("%dth epoch %d / %d\n", i, recog, net->nr_test_data);
	}
}

void feedforward(struct network *net, int thread)
{
	int i, j, k, l, m;
	double sum = 0.0;
    timeutils *feedforward = &net->t_feedforward;

	// feedforward
	START_TIME(feedforward);
    sum = 0.0;
    int nr_chunk = thread;
    int chunk_size = (int) (net->mini_batch_size/nr_chunk);

/* OpenMP : with collapse */
    for (i = 0; i < net->num_layer-1; i++) {
        for (j = 0; j < nr_chunk; j++) {
            cilk_for(m = 0; m < chunk_size; m++) {
                for(k = 0; k < net->layer_size[i+1]; k++) {
                    #pragma simd
                    for(l = 0; l < net->layer_size[i]; l++) {
                        sum = sum + NEURON(net, i, j*chunk_size+m, l) * WEIGHT(net, i, l, k);
                    }

                    ZS(net, i+1, j*chunk_size+m, k) = sum + BIAS(net, i+1, k);
                    NEURON(net, i+1, j*chunk_size+m, k) = sigmoid(ZS(net, i+1, j*chunk_size+m, k));
                    sum = 0.0;
                }
            }
        }
    }
	END_TIME(feedforward);
}

void back_pass(struct network *net, int thread1, int thread2)
{
	int i, j, k, l;
	double sum = 0.0;
    timeutils *back_pass = &net->t_back_pass;

	START_TIME(back_pass);
	// calculate delta
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
	END_TIME(back_pass);
}

/* Operation like backpropagation */
void backpropagation(struct network *net, int thread1, int thread2)
{
	int i, j, k, l;
    timeutils *backpropagation = &net->t_backpropagation;
	double eta = net->learning_rate;
	double mini = (double) net->mini_batch_size;

	START_TIME(backpropagation);
	// update bias
	cilk_for (i = 1; i < net->num_layer; i++) {
		for (j = 0; j < net->layer_size[i]; j++) {
			#pragma simd
			for (k = 0; k < net->mini_batch_size; k++) {
				BIAS(net, i, j) -= (eta/mini)*ERROR(net, i, k, j);
			}
		}
	}

	// update weight
	for (i = 0; i < net->num_layer-1; i++) {
		cilk_for (j = 0; j < net->layer_size[i]; j++) {
			for (k = 0; k < net->layer_size[i+1]; k++) {
				#pragma simd
				for (l = 0; l < net->mini_batch_size; l++) {
					//	calculate delta from before layer
					WEIGHT(net, i, j, k) -= (eta/mini)*(NEURON(net, i, l, j) * ERROR(net, i+1, l, k));
				}
			}
		}
	}
	END_TIME(backpropagation);
}

double sigmoid(double z)
{
	return (1/(1 + exp(-z)));
}

double sigmoid_prime(double z)
{
	return sigmoid(z)*(1-sigmoid(z));
}

int predict(struct network *net)
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
			cilk_for (k = 0; k < net->layer_size[j+1]; k++) {
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

void report(struct network *net, void *threads)
{
    int *thread = (int *) threads;
    timeutils *feedforward = &net->t_feedforward;
    timeutils *back_pass = &net->t_back_pass;
    timeutils *backpropagation = &net->t_backpropagation;
    timeutils t;
    timeutils *total = &t;

    TIMER_INIT(total);

	int i = 0;
	FILE *f = fopen(net->report_file, "a+");
	FILE *f_json = fopen("./result/dump_json", "a+");
	if (f == NULL || f_json == NULL) {
		printf("%s open failed\n", net->report_file);
		printf("%s open failed\n", "dump_json");
		exit(1);
	}
	fprintf( f, "\n=======================REPORT=======================\n");
//	fprintf( f, "==================HYPER-PARAMETERS================== \n");
//	fprintf( f, "mini_batch_size : %d\n", net->mini_batch_size);
//	fprintf( f, "num_layer : %d\n", net->num_layer);
//	fprintf( f, "layers : [");
//	for (i = 0; i < net->num_layer; i++) {
//		fprintf( f, "%d ", net->layer_size[i]);
//	}
//	fprintf( f, "]\n");
	fprintf( f, "epoch : %d\n", net->epoch);
	fprintf( f, "learning_rate : %f\n", net->learning_rate);
	fprintf( f, "recognization rate : %d/%d\n", net->best_recog, net->nr_test_data);
	fprintf( f, "=======================THREADS======================\n");
	fprintf( f, "feedforward thread : %d\n", thread[0]);
	fprintf( f, "back_pass thread1 : %d\n", thread[1]);
	fprintf( f, "back_pass thread2 : %d\n", thread[2]);
	fprintf( f, "backpropagation thread1 : %d\n", thread[3]);
	fprintf( f, "backpropagation thread2 : %d\n", thread[4]);
	fprintf( f, "========================TIME========================\n");
	fprintf( f, "feedforward : %ld.%d sec\n", TOTAL_SEC_TIME(feedforward), TOTAL_SEC_UTIME(feedforward));
	fprintf( f, "back_pass : %ld.%d sec\n", TOTAL_SEC_TIME(back_pass), TOTAL_SEC_UTIME(back_pass));
	fprintf( f, "backpropagation : %ld.%d sec\n", TOTAL_SEC_TIME(backpropagation), TOTAL_SEC_UTIME(backpropagation));

    TIMER_ADD(feedforward, total);
    TIMER_ADD(back_pass, total);
    TIMER_ADD(backpropagation, total);
	fprintf( f, "total : %ld.%d sec\n", TOTAL_SEC_TIME(total), TOTAL_SEC_UTIME(total));

    fprintf( f_json, "{\n");
    fprintf( f_json, "\"feedforward_thread\":%d,\n",thread[0]);
    fprintf( f_json, "\"back_pass_thread1\":%d,\n",thread[1]);
    fprintf( f_json, "\"back_pass_thread2\":%d,\n",thread[2]);
    fprintf( f_json, "\"backpropagation_thread1\":%d,\n",thread[3]);
    fprintf( f_json, "\"backpropagation_thread2\":%d,\n",thread[4]);
	fprintf( f_json, "\"feedforward_time\": %ld.%d,\n", TOTAL_SEC_TIME(feedforward), TOTAL_SEC_UTIME(feedforward));
	fprintf( f_json, "\"back_pass_time\": %ld.%d,\n", TOTAL_SEC_TIME(back_pass), TOTAL_SEC_UTIME(back_pass));
	fprintf( f_json, "\"backpropagation\": %ld.%d,\n", TOTAL_SEC_TIME(backpropagation), TOTAL_SEC_UTIME(backpropagation));
	fprintf( f_json, "\"total_time\": %ld.%d\n", TOTAL_SEC_TIME(total), TOTAL_SEC_UTIME(total));
    fprintf( f_json, "},");

	fclose(f);
	fclose(f_json);
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
