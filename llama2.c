#define _GNU_SOURCE // keep this line
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 * 
 * Matrix-vector multiplication, used in QKV Mapping and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is 
 * independent of each other, so we can use parallel computing for acceleration.
 * 
 * Please use <pthread.h> and your favorite control method,
 * semaphore (please #include <semaphore.h>) / mutex lock + conditional variable
 * 
 * A sequential version is provided below, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE

// additional header file
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>

// Structs
typedef struct {
    int start;
    int end;
} Range;

struct rusage main_usage;        // get usage for main thread

// Global variables for thread
int thread_count;

pthread_t* threads;  // Dynamic array to hold threads
sem_t mainMutex;     // to let the main thread wait for all the subthreads to finish
sem_t* workerMutexs;
Range** rangeArr;   // to hold thread arguments

int terminate = 0;  // to indicate the threads to terminate

// global variables for matrix vector multiplication
float* Out;
float* Vec;
float* Mat;
int Col;
int Row;

// thread function declaration
void *thr_func(void *arg);

// TODO:
// Creates n threads; each with a unique ID (i.e., 0 to n-1)
// Initializes necessary variables
int create_mat_vec_mul(int thr_count) {
    thread_count = thr_count;
    
    // init mainMutex
    if (sem_init(&mainMutex, 0, 0) == -1) {
        printf("Error: mainMutex sem_init failed\n");
        return -1;
    }

    // init all worker thread semaphores
    workerMutexs = (sem_t*) malloc(thr_count * sizeof(sem_t));
    for (int i = 0; i < thr_count; i++) {
        if (sem_init(&workerMutexs[i], 0, 0) == -1) {
            printf("Error: workerMutexs sem_init failed\n");
            return -1;
        }
    }

    // init all threads
    threads = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
    for (int i = 0; i < thread_count; i++) {
        // create thread
        if (pthread_create(&threads[i], NULL, thr_func, (void*)(intptr_t)i) != 0) {
            printf("Error: pthread_create failed\n");
            return -1;
        }
    }

    // init the rangeArr
    rangeArr = (Range**) malloc(thread_count * sizeof(Range*));
    if (rangeArr == NULL) {
        printf("Error allocating memory for array\n");
        return -11;
    }
    // Initialize each array entry
    for (int i = 0; i < thread_count; i++) {
        // Allocate memory for the (start, end) pair
        rangeArr[i] = (Range*) malloc(sizeof(Range));
        if (rangeArr[i] == NULL) {
            printf("Error allocating memory for entry %d\n", i);
            return -1;
        }
    }
}

void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    //  in the case of a Matrix with d rows and n threads:
    //  if d is divisible by n, the k-th thread (k = 0 , 1 , ... , n − 1) will handle the rows from [k * d / n] to [(k + 1) * d / n - 1] .
    //  If is not divisible by n, we can assign first n − 1 threads (k = 0 , 1 , … , n − 2 ) with ceil(d / n) rows, while the last thread handles remaining rows.
    int start_row = 0;
    int end_row = 0;

    if (row % thread_count != 0) {   // row is not divisible by thread_count
        int rows_per_thread = row / thread_count + (row % thread_count != 0); // ceil(row / thread_count)
        // first n-1 threads
        for (int k = 0; k < thread_count - 1; k++) {
            start_row = k * rows_per_thread;
            end_row = (k + 1) * rows_per_thread - 1;
            // assign the parameters to the threads
            rangeArr[k]->start = start_row;
            rangeArr[k]->end = end_row;
        }
        // last thread: thread id: thread_count - 1
        start_row = (thread_count - 1) * rows_per_thread;
        end_row = row - 1;
        // assign the parameters to the threads
        rangeArr[thread_count - 1]->start = start_row;
        rangeArr[thread_count - 1]->end = end_row;

    } else {        // row is divisible by thread_count
        for (int k = 0; k < thread_count; k++) {
            start_row = k * row / thread_count;
            end_row = (k + 1) * row / thread_count - 1;
            // assign the parameters to the threads
            rangeArr[k]->start = start_row;
            rangeArr[k]->end = end_row;
        }
    }
    // assign all the corresponding global variables
    Out = out;
    Vec = vec;
    Mat = mat;
    Col = col;
    Row = row;

    // sem wait all subthread semaphores (when subthreads finish, they will post their semaphores)
    for (int i = 0; i < thread_count; i++) {
        sem_post(&workerMutexs[i]);
    }

    // wait for all the subthreads to finish
    for (int i = 0; i < thread_count; i++) {
        sem_wait(&mainMutex);
    }
}


int destroy_mat_vec_mul() {
    // print out self usage information for main thread
    getrusage(RUSAGE_SELF, &main_usage);
    printf("main thread - user: %.4f s, system: %.4f s\n",
    (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec/1000000.0),
    (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec/1000000.0));
    // print out usage info for worker threads
    terminate = 1; // to indicate the threads to terminate
    for (int i = 0; i < thread_count; i++) {
        sem_post(&workerMutexs[i]);        // wake up and inform threads to terminate
    }

    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);   // wait for all threads
    }

    // free all the memory allocated
    for (int i = 0; i < thread_count; i++) {
        sem_destroy(&workerMutexs[i]);
    }
    sem_destroy(&mainMutex);
    free(workerMutexs);
    free(threads);
}


void *thr_func(void *arg) {
    struct rusage thread_usage;
    // Obtain the thread's ID
    intptr_t threadID = (intptr_t)arg;

    while (1) {
        sem_wait(&workerMutexs[threadID]);      // Wait for main thread's wake-up signal

        if (terminate == 1) {
            getrusage(RUSAGE_SELF, &main_usage);
            printf("Thread %ld has completed - user: %.4f s, system: %.4f s\n",
            threadID,
            (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec/1000000.0),
            (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec/1000000.0));
            pthread_exit(NULL);
        } else {
            int start_row = rangeArr[threadID]->start;
            int end_row = rangeArr[threadID]->end;
            // W (d,n) @ x (n,) -> xout (d,)
            for (int i = start_row; i <= end_row; i++) {
                float val = 0.0f;
                for (int j = 0; j < Col; j++) {
                    val += Mat[i * Col + j] * Vec[j];
                }
                Out[i] = val;
            }
            sem_post(&mainMutex);   // wake up the main thread
        }
    }
}


// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    create_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    destroy_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}