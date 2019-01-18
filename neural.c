#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRAIN_IMAGES_FILE "train-images-idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels-idx1-ubyte"
#define MNIST_DIM 28

#define INPUT_SIZE 784
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 10

#define STEP_SIZE 0.1
#define NORMALIZE_FACTOR 100
#define IMAGE_ITERS 1

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

double* evaluate(neural_network* network, double* inputs)
{
  // Apply input layer biases
  double input_layer[INPUT_SIZE]; 
  for (int i = 0; i < INPUT_SIZE; i++)
  {
    double newval = inputs[i] * network->input_biases[i];
    input_layer[i] = sigmoid(newval);
  }

  // Compute hidden layer
  double hidden_layer[HIDDEN_SIZE];
  
  for (int h = 0; h < HIDDEN_SIZE; h++)
  {
    double node_sum = 0;
    for (int i = 0; i < INPUT_SIZE; i++)
    {
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
      node_sum += hidden_layer[h] * network->hidden_to_output_weights[h][o];
    }
    node_sum += network->output_biases[o];
    output_layer[o] = sigmoid(node_sum);
  }

  return output_layer;
}

double error(neural_network* network, double* inputs, double* ground_truth)
{
  double* output = evaluate(network, inputs);
  double sqr_sum = 0;
  for (int o = 0; o < OUTPUT_SIZE; o++)
  {
    double err = output[o] - ground_truth[o];
    sqr_sum += err*err;
  }
  free(output);
  return sqr_sum;
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

  for(int epoch = 0; 1; epoch++)
  {

    printf("Training Epoch #%d\n", epoch+1);

    // fetch new training examples
    mnist_image datapoint = data[epoch%data_size];
    double* inputs = format_image_for_nn(datapoint.image);
    double* outputs = format_label_for_nn(datapoint.label);

    for (int j = 0; j < IMAGE_ITERS; j++)
    {
      printf("Training iteration %d/%d\n", j+1, IMAGE_ITERS);

      neural_network new_network;
      double total_step_distance = 0;

      double base_err = error(network, inputs, outputs);

      // compute gradient for input biases
      for (int i = 0; i < INPUT_SIZE; i++)
      {
        double orig_value = network->input_biases[i];
        double step = gradient->input_biases[i] * STEP_SIZE;
        network->input_biases[i] += step;
        total_step_distance += step*step;
        new_network.input_biases[i] = network->input_biases[i];
        double new_err = error(network, inputs, outputs);
        network->input_biases[i] = orig_value;
        gradient->input_biases[i] = sign(gradient->input_biases[i]) * (base_err - new_err);
      }

      // compute gradient for hidden biases
      for (int h = 0; h < HIDDEN_SIZE; h++)
      {
        double orig_value = network->hidden_biases[h]; 
        double step = gradient->hidden_biases[h] * STEP_SIZE;
        network->hidden_biases[h] += step;
        total_step_distance += step*step;
        new_network.hidden_biases[h] = network->hidden_biases[h];
        double new_err = error(network, inputs, outputs);
        network->hidden_biases[h] = orig_value;
        gradient->hidden_biases[h] = sign(gradient->hidden_biases[h]) * (base_err - new_err);
      }

      // compute gradient for output biases
      for (int o = 0; o < OUTPUT_SIZE; o++)
      {
        double orig_value = network->output_biases[o]; 
        double step = gradient->output_biases[o] * STEP_SIZE;
        network->output_biases[o] += step;
        total_step_distance += step*step;
        new_network.output_biases[o] = network->output_biases[o];
        double new_err = error(network, inputs, outputs);
        network->output_biases[o] = orig_value;
        gradient->output_biases[o] = sign(gradient->output_biases[o]) * (base_err - new_err);
      }

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
          double new_err = error(network, inputs, outputs);
          network->input_to_hidden_weights[i][h] = orig_value;
          gradient->input_to_hidden_weights[i][h] = sign(gradient->input_to_hidden_weights[i][h]) * (base_err - new_err);
        }
      }

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
          double new_err = error(network, inputs, outputs);
          network->hidden_to_output_weights[h][o] = orig_value;
          gradient->hidden_to_output_weights[h][o] = sign(gradient->hidden_to_output_weights[h][o]) * (base_err - new_err);
        }
      }

      // apply gradient shift
      *network = new_network;

      // scale gradient to keep a specific step size
      double factor = (NORMALIZE_FACTOR * base_err) / sqrt(total_step_distance);
      //if (factor > 10000) factor = 10000;
      //if (factor < 0.0001) factor = 0.0001;
      scale_network(gradient, factor);

      printf("Stepped a total distance of %.4f; scaling by a factor of %.4f\n", sqrt(total_step_distance), factor);
      printf("Error after iteration %d: %.16lf\n", j+1, base_err);
      
    }

    free(inputs);
    free(outputs);
  }

  return 0;
}
