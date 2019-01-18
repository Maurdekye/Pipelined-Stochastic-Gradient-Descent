#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define TRAIN_IMAGES_FILE "train-images-idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels-idx1-ubyte"
#define MNIST_DIM 28

#define INPUT_SIZE 784 // MNIST_DIM * MNIST_DIM
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 10

#define STEP_SIZE 0.1
#define NORMALIZE_FACTOR 100
#define BATCH_ITERS 8
#define BATCH_SIZE 16

#define MULTITHREADING
#define NTHREADS 16

typedef struct {
  double input_biases[INPUT_SIZE];
  double hidden_biases[HIDDEN_SIZE];
  double output_biases[OUTPUT_SIZE];
  double input_to_hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
  double hidden_to_output_weights[HIDDEN_SIZE][OUTPUT_SIZE];
} neural_network;

typedef struct {
  char label;
  char image[MNIST_DIM][MNIST_DIM];
} mnist_image;

int big_endian(unsigned char bytes[4])
{
  return bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3];
}

double* format_image_for_nn(char image[MNIST_DIM][MNIST_DIM])
{
  double* inputs = malloc(sizeof(double) * MNIST_DIM * MNIST_DIM);
  for (int x = 0; x < MNIST_DIM; x++)
  {
    for (int y = 0; y < MNIST_DIM; y++)
    {
      double pixel = ((double)image[x][y])/255.0;
      inputs[y * MNIST_DIM + x] = pixel;
    }
  }
  return inputs;
}

double* format_label_for_nn(char label)
{
  double* target = calloc(sizeof(double), 10);
  target[label] = 1.0;
  return target;
}

void initialize_network(neural_network* network, double value)
{
  for (int i = 0; i < INPUT_SIZE; i++) { 
    network->input_biases[i] = value;
    for (int h = 0; h < HIDDEN_SIZE; h++)
      network->input_to_hidden_weights[i][h] = value;
  }
  for (int h = 0; h < HIDDEN_SIZE; h++) {
    network->hidden_biases[h] = value;
    for (int o = 0; o < OUTPUT_SIZE; o++)
      network->hidden_to_output_weights[h][o] = value;
  }
  for (int o = 0; o < OUTPUT_SIZE; o++)
    network->output_biases[o] = value;
}

void scale_network(neural_network* network, double value)
{
  for (int i = 0; i < INPUT_SIZE; i++) { 
    network->input_biases[i] *= value;
    for (int h = 0; h < HIDDEN_SIZE; h++)
      network->input_to_hidden_weights[i][h] *= value;
  }
  for (int h = 0; h < HIDDEN_SIZE; h++) {
    network->hidden_biases[h] *= value;
    for (int o = 0; o < OUTPUT_SIZE; o++)
      network->hidden_to_output_weights[h][o] *= value;
  }
  for (int o = 0; o < OUTPUT_SIZE; o++)
    network->output_biases[o] *= value;
}

double sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

double sign(double x)
{
  if (x < 0)
    return -1.0;
  if (x > 0)
    return 1.0;
  return 0.0;
}

double* evaluate(neural_network* network, double* inputs, int tweak_layer, int tweak_id, double tweak_amt)
{
  // Apply input layer biases
  double input_layer[INPUT_SIZE]; 
  for (int i = 0; i < INPUT_SIZE; i++)
  {
    double newval = inputs[i];
    newval *= network->input_biases[i];
    input_layer[i] = sigmoid(newval);
  }

  // Compute hidden layer
  double hidden_layer[HIDDEN_SIZE];
  
  for (int h = 0; h < HIDDEN_SIZE; h++)
  {
    double node_sum = 0;
    for (int i = 0; i < INPUT_SIZE; i++)
    {
      if (tweak_layer == 0 && tweak_id == (h * INPUT_SIZE + i)) 
        node_sum += input_layer[i] * (network->input_to_hidden_weights[i][h] + tweak_amt);
      else
        node_sum += input_layer[i] * network->input_to_hidden_weights[i][h];
    }
    node_sum += network->hidden_biases[h];
    hidden_layer[h] = sigmoid(node_sum);
  }

  // Compute output layer
  double* output_layer = malloc(sizeof(double) * OUTPUT_SIZE);

  for (int o = 0; o < OUTPUT_SIZE; o++)
  {
    double node_sum = 0;
    for (int h = 0; h < HIDDEN_SIZE; h++)
    {
      if (tweak_layer == 1 && tweak_id == (o * HIDDEN_SIZE + h)) 
        node_sum += hidden_layer[h] * (network->hidden_to_output_weights[h][o] + tweak_amt);
      else
        node_sum += hidden_layer[h] * network->hidden_to_output_weights[h][o];
    }
    node_sum += network->output_biases[o];
    output_layer[o] = sigmoid(node_sum);
  }

  return output_layer;
}

double error(neural_network* network, double* inputs, double* ground_truth)
{
  double* output = evaluate(network, inputs, -1, 0, 0);
  double sqr_sum = 0;
  for (int o = 0; o < OUTPUT_SIZE; o++)
  {
    double err = output[o] - ground_truth[o];
    sqr_sum += err*err;
  }
  free(output);
  return sqr_sum;
}

double error_with_tweak(neural_network* network, double* inputs, double* ground_truth, int tweak_layer, int tweak_id, double tweak_amt)
{
  double* output = evaluate(network, inputs, tweak_layer, tweak_id, tweak_amt);
  double sqr_sum = 0;
  for (int o = 0; o < OUTPUT_SIZE; o++)
  {
    double err = output[o] - ground_truth[o];
    sqr_sum += err*err;
  }
  free(output);
  return sqr_sum;
}

typedef struct {
  int id;
  double base_err;
  double step_distance;
  neural_network* network;
  neural_network* new_network;
  neural_network* gradient;
  double (*train_batch_inputs)[BATCH_SIZE][INPUT_SIZE];
  double (*train_batch_outputs)[BATCH_SIZE][OUTPUT_SIZE];
} thread_local_data;

void* train_itoh_thread(void* varargp)
{
  thread_local_data* data = (thread_local_data*) varargp;

  int range = INPUT_SIZE / NTHREADS;
  for (int i = range * data->id; i < range * (data->id + 1); i++)
  {
    for (int h = 0; h < HIDDEN_SIZE; h++)
    {
      double step = data->gradient->input_to_hidden_weights[i][h] * STEP_SIZE;
      data->step_distance += step*step;
      data->new_network->input_to_hidden_weights[i][h] = data->network->input_to_hidden_weights[i][h] + step;

      double new_err = 0;
      for (int k = 0; k < BATCH_SIZE; k++)
        new_err += error_with_tweak(data->network, data->train_batch_inputs[k][0], data->train_batch_outputs[k][0], 0, h * INPUT_SIZE + i, step);

      data->gradient->input_to_hidden_weights[i][h] = sign(data->gradient->input_to_hidden_weights[i][h]) * (data->base_err - new_err);
    }
  }
}

void* train_htoo_thread(void* varargp)
{
  thread_local_data* data = (thread_local_data*) varargp;

  int range = HIDDEN_SIZE / NTHREADS;
  for (int h = range * data->id; h < range * (data->id + 1); h++)
  {
    for (int o = 0; o < OUTPUT_SIZE; o++)
    {
      double step = data->gradient->hidden_to_output_weights[h][o] * STEP_SIZE;
      data->step_distance += step*step;
      data->new_network->hidden_to_output_weights[h][o] = data->network->hidden_to_output_weights[h][o] + step;

      double new_err = 0;
      for (int k = 0; k < BATCH_SIZE; k++)
        new_err += error_with_tweak(data->network, data->train_batch_inputs[k][0], data->train_batch_outputs[k][0], 1, o * HIDDEN_SIZE + h, step);

      data->gradient->hidden_to_output_weights[h][o] = sign(data->gradient->hidden_to_output_weights[h][o]) * (data->base_err - new_err);
    }
  }
}

int main(int argc, char** argv)
{
  printf("Loading MNIST dataset\n");

  FILE* mnist_imagef = fopen(TRAIN_IMAGES_FILE, "rb");
  FILE* mnist_labelf = fopen(TRAIN_LABELS_FILE, "rb");

  int data_size;
  mnist_image* data;

  if (!mnist_imagef || !mnist_labelf)
  {
    printf("Training files '%s' and '%s' not found! Aborting...\n", TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE);
    return 1;
  }
  else
  {
    // check magic numbers
    unsigned char magic_n[4];
    fread(&magic_n, 4, 1, mnist_imagef);
    
    if (big_endian(magic_n) != 2051)
    {
      printf("Magic number on image file is %d, should be 2051; aborting...\n", big_endian(magic_n));
      return 1;
    }

    fread(&magic_n, 4, 1, mnist_labelf);
    if (big_endian(magic_n) != 2049)
    {
      printf("Magic number on label file is %d, should be 2049; aborting...\n", big_endian(magic_n));
      return 1;
    }

    // read sizes
    unsigned char imcountb[4], lbcountb[4];
    fread(&imcountb, 4, 1, mnist_imagef);
    fread(&lbcountb, 4, 1, mnist_labelf);
    int imcount = big_endian(imcountb);
    int lbcount = big_endian(lbcountb);
    if (imcount != lbcount)
    {
      printf("Image count is %d, but label count is %d: must be the same! Aborting...\n", imcount, lbcount);
      return 1;
    }
    else if (imcount < 0)
    {
      printf("Error interpreting image count, read value %d. Aborting...\n", imcount);
      return 1;
    }

    data_size = imcount;

    // read image dimensions
    unsigned char heightb[4], widthb[4];
    fread(&heightb, 4, 1, mnist_imagef);
    fread(&widthb, 4, 1, mnist_imagef);
    int height = big_endian(heightb);
    int width = big_endian(widthb);
    if (height != MNIST_DIM && width != MNIST_DIM)
    {
      printf("Error: files contain %dx%d images, but program is configured for %dx%d images. Aborting...\n", width, height, MNIST_DIM, MNIST_DIM);
      return 1;
    }

    // read data
    data = malloc(sizeof(mnist_image) * data_size);
    for (int i = 0; i < data_size; i++)
    {
      fread(&(data[i].image), MNIST_DIM, MNIST_DIM, mnist_imagef);
      fread(&(data[i].label), 1, 1, mnist_labelf);
    }

    fclose(mnist_imagef);
    fclose(mnist_labelf);
  }

  printf("Loaded %d images at %dx%d\n", data_size, MNIST_DIM, MNIST_DIM);

  printf("Initializing neural network\n");

  neural_network* network = calloc(sizeof(neural_network),1);
  neural_network* gradient = calloc(sizeof(neural_network),1);

  initialize_network(gradient, 0.01);
  
  double train_batch_inputs[BATCH_SIZE][INPUT_SIZE];
  double train_batch_outputs[BATCH_SIZE][OUTPUT_SIZE];

  for(int epoch = 0; 1; epoch++)
  {
    //printf("Preparing training examples for Epoch #%d\n", epoch+1);

    // fetch new training examples
    for (int i = 0; i < BATCH_SIZE; i++)
    {
      int ind = rand() % data_size;
      mnist_image item = data[ind];
      train_batch_inputs[i][0] = *format_image_for_nn(item.image);
      train_batch_outputs[i][0] = *format_label_for_nn(item.label);
    }
    
    printf("Training Epoch #%d\n", epoch+1);

    for (int batch = 0; batch < BATCH_ITERS; batch++)
    {
      
      printf("Training iteration %d/%d\n", batch+1, BATCH_ITERS);

      neural_network new_network;
      double total_step_distance = 0;

      double base_err = 0;
      for (int k = 0; k < BATCH_SIZE; k++)
        base_err += error(network, train_batch_inputs[k], train_batch_outputs[k]);
      
      thread_local_data thread_data[NTHREADS];
      pthread_t thread_ids[NTHREADS];

      for (int t = 0; t < NTHREADS; t++)
      {
        thread_data[t].id = t;
        thread_data[t].base_err = base_err;
        thread_data[t].step_distance = 0;
        thread_data[t].network = network;
        thread_data[t].new_network = &new_network;
        thread_data[t].gradient = gradient;
        thread_data[t].train_batch_inputs = &train_batch_inputs;
        thread_data[t].train_batch_outputs = &train_batch_outputs;
      }

      printf("Computing gradient for input biases\n");

      // compute gradient for input biases
      for (int i = 0; i < INPUT_SIZE; i++)
      {
        double orig_value = network->input_biases[i];
        double step = gradient->input_biases[i] * STEP_SIZE;
        network->input_biases[i] += step;
        total_step_distance += step*step;
        new_network.input_biases[i] = network->input_biases[i];
       
        //printf("    -- Input node #%d\n", i+1);
        double new_err = 0;
        for (int k = 0; k < BATCH_SIZE; k++)
          new_err += error(network, train_batch_inputs[k], train_batch_outputs[k]);
        
        network->input_biases[i] = orig_value;
        gradient->input_biases[i] = sign(gradient->input_biases[i]) * (base_err - new_err);
      }

      printf("Computing gradient for hidden biases\n");

      // compute gradient for hidden biases
      for (int h = 0; h < HIDDEN_SIZE; h++)
      {
        double orig_value = network->hidden_biases[h]; 
        double step = gradient->hidden_biases[h] * STEP_SIZE;
        network->hidden_biases[h] += step;
        total_step_distance += step*step;
        new_network.hidden_biases[h] = network->hidden_biases[h];
        
        double new_err = 0;
        for (int k = 0; k < BATCH_SIZE; k++)
          new_err += error(network, train_batch_inputs[k], train_batch_outputs[k]);
        
        network->hidden_biases[h] = orig_value;
        gradient->hidden_biases[h] = sign(gradient->hidden_biases[h]) * (base_err - new_err);
      }

      printf("Computing gradient for output biases\n");

      // compute gradient for output biases
      for (int o = 0; o < OUTPUT_SIZE; o++)
      {
        double orig_value = network->output_biases[o]; 
        double step = gradient->output_biases[o] * STEP_SIZE;
        network->output_biases[o] += step;
        total_step_distance += step*step;
        new_network.output_biases[o] = network->output_biases[o];

        double new_err = 0;
        for (int k = 0; k < BATCH_SIZE; k++)
          new_err += error(network, train_batch_inputs[k], train_batch_outputs[k]);

        network->output_biases[o] = orig_value;
        gradient->output_biases[o] = sign(gradient->output_biases[o]) * (base_err - new_err);
      }

#ifdef MULTITHREADING
      printf("Initializing multithreaded input-hidden weight gradient compute\n");

      // compute gradient for input to hidden weights
      for (int t = 0; t < NTHREADS; t++)
      {
        int errcode = pthread_create(&thread_ids[t], NULL, train_itoh_thread, (void*)&thread_data[t]);
        if (errcode != 0)
        {
          printf("Recieved error code %d! Aborting...\n", errcode);
          return 1;
        }
      }
      for (int t = 0; t < NTHREADS; t++)
      {
        pthread_join(thread_ids[t], NULL);
        total_step_distance += thread_data[t].step_distance;
      }

      printf("Initializing multithreaded hidden-output weight gradient compute\n");

      // compute gradient for hidden to output weights
      for (int t = 0; t < NTHREADS; t++)
      {
        int errcode = pthread_create(&thread_ids[t], NULL, train_htoo_thread, (void*)&thread_data[t]);
        if (errcode != 0)
        {
          printf("Recieved error code %d! Aborting...\n", errcode);
          return 1;
        }        
      }
      for (int t = 0; t < NTHREADS; t++)
      {
        pthread_join(thread_ids[t], NULL);
        total_step_distance += thread_data[t].step_distance;
      }

#else      
      printf("Computing gradient for input-hidden weights\n");

      // compute gradient for input to hidden weights
      for (int i = 0; i < INPUT_SIZE; i++)
      {
        for (int h = 0; h < HIDDEN_SIZE; h++)
        {
          double orig_value = network->input_to_hidden_weights[i][h];
          double step = gradient->input_to_hidden_weights[i][h] * STEP_SIZE;
          network->input_to_hidden_weights[i][h] += step;
          total_step_distance += step*step;
          new_network.input_to_hidden_weights[i][h] = network->input_to_hidden_weights[i][h];

          double new_err = 0;
          for (int k = 0; k < BATCH_SIZE; k++)
            new_err += error(network, train_batch_inputs[k], train_batch_outputs[k]);

          network->input_to_hidden_weights[i][h] = orig_value;
          gradient->input_to_hidden_weights[i][h] = sign(gradient->input_to_hidden_weights[i][h]) * (base_err - new_err);
        }
      }

      printf("Computing gradient for hidden-output weights\n");

      // compute gradient for hidden to output weights
      for (int h = 0; h < HIDDEN_SIZE; h++)
      {
        for (int o = 0; o < OUTPUT_SIZE; o++)
        {
          double orig_value = network->hidden_to_output_weights[h][o];
          double step = gradient->hidden_to_output_weights[h][o] * STEP_SIZE;
          network->hidden_to_output_weights[h][o] += step;
          total_step_distance += step*step;
          new_network.hidden_to_output_weights[h][o] = network->hidden_to_output_weights[h][o];

          double new_err = 0;
          for (int k = 0; k < BATCH_SIZE; k++)
            new_err += error(network, train_batch_inputs[k], train_batch_outputs[k]);

          network->hidden_to_output_weights[h][o] = orig_value;
          gradient->hidden_to_output_weights[h][o] = sign(gradient->hidden_to_output_weights[h][o]) * (base_err - new_err);
        }
      }
#endif

      // apply gradient shift
      *network = new_network;

      // scale gradient to keep a specific step size
      double factor = (NORMALIZE_FACTOR * base_err) / sqrt(total_step_distance);
      scale_network(gradient, factor);

      printf("Stepped a total distance of %.4f; scaling by a factor of %.4f\n", sqrt(total_step_distance), factor);
      printf("Error after iteration %d: %.16lf\n", batch+1, base_err);
      
    }
  }

  return 0;
}
