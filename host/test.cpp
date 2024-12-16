#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

using namespace std;

#define N1 128
#define N2 128
#define N_OUT 10
#define LEARNING_RATE 1e-3
#define EPSILON 1e-3
#define HEIGHT 28
#define WIDTH 28
#define N_IN 784 //N_IN=HEIGH*WIDTH
#define NTESTING 10000

double input[N_IN];
double expected[N_OUT];

//percepton
double **w1, b1[N1];
double layer1[N1];
double out_layer1[N1];
double **w2, b2[N2];
double layer2[N2];
double out_layer2[N2];
double **w3, b3[N_OUT];
double in_out[N_OUT];
double output[N_OUT];

//back propagation
double delta3[N_OUT];
double delta2[N2];
double delta1[N1];

//I/O
ifstream image;
ifstream label;
ofstream report;
string MODEL= "model.dat";
string testing_label_fn="mnist/t10k-labels-idx1-ubyte";
string testing_image_fn="mnist/t10k-images-idx3-ubyte";


void softmax(double *in_out, double *ouput, int n)
{
  double d=0;
  for (int i=0;i<n;i++){
    output[i]=exp(in_out[i]);
    d+=output[i];
  }
  for (int i=0;i<n;i++){
    output[i]/=d;
    output[i]+=1e-5;
  }
}
void ReLU(double *layer, double *out_layer, int n)
{
  for (int i=0;i<n;i++){
    out_layer[i]=(layer[i]>0)?layer[i]:0;
  }
}

void perceptron()
{
  for (int i=0;i<N1;i++){ //layer 1
    layer1[i]=b1[i];
    for (int j=0;j<N_IN;j++){
      layer1[i]+=w1[i][j]*input[j];
    }
  }
  
  ReLU(layer1,out_layer1,N1); //out of layer 1
  
  for (int i=0;i<N2;i++){ //layer 2
    layer2[i]=b2[i];
    for (int j=0;j<N1;j++){
      layer2[i]+=w2[i][j]*out_layer1[j];
    }
  }

  ReLU(layer2,out_layer2,N2); //out of layer 2

  for (int i=0;i<N_OUT;i++){ //indirecr of output
    in_out[i]=b3[i];
    for (int j=0;j<N2;j++){
      in_out[i]+=w3[i][j]*out_layer2[j];
    }
  }

  softmax(in_out,output,N_OUT);
}

void load_matrix(ifstream &file,double **weight,double *bias,int r, int c)
{
  for (int i = 0; i<r; ++i) {
    for (int j = 0; j<c; ++j) {
      file >> weight[i][j];
    }
  }

  for (int i=0;i<r;i++){
    file>>bias[i];
  }
}

void load_model(string file_name) {
  ifstream file(file_name.c_str(), ios::in);

  if (file.fail()) {
    cout<<"Cannot open model!\n";
    exit(EXIT_FAILURE);
  }
  
  load_matrix(file,w1,b1,N1,N_IN);
  load_matrix(file,w2,b2,N2,N1);
  load_matrix(file,w3,b3,N_OUT,N2);
  
  file.close();
}


int next_sample()
{
  char number;
  for (int i=0;i<HEIGHT;i++){
    for (int j=0;j<WIDTH;j++){
      image.read(&number, sizeof(char));
      input[i*WIDTH+j]=number/255.0;
    }
  }
  
  label.read(&number, sizeof(char));
  for (int i = 0; i < N_OUT; ++i) {
    expected[i] = 0.0;
  }
  expected[number] = 1.0;
  return (int) number;
}

int main()
{
  image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
  label.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

  if (image.fail() || label.fail()) {
    cout<< "Cannot open a file!\n";
    return 1;
  }
  // Reading file headers
  char number;
  for (int i = 0; i < 16; ++i) {
    image.read(&number, sizeof(char));
  }
  for (int i = 0; i < 8; ++i) {
    label.read(&number, sizeof(char));
  }

  
  w1=(double**)malloc(N1*sizeof(double*));
  w2=(double**)malloc(N2*sizeof(double*));
  w3=(double**)malloc(N_OUT*sizeof(double*));

  for (int i=0;i<N1;i++){
    w1[i]=(double*)malloc(N_IN*sizeof(double));
  }
  for (int i=0;i<N2;i++){
    w2[i]=(double*)malloc(N1*sizeof(double));
  }
  for(int i=0;i<N_OUT;i++){
    w3[i]=(double*)malloc(N2*sizeof(double));
  }

  
  load_model(MODEL);

  int correct=0;
  for (int sample=1;sample<=NTESTING;sample++){
    int label=next_sample();
    perceptron();
    int predict=0;
    for (int i=1;i<= N_OUT;i++){
      if (output[i]>output[predict]){
	      predict=i;
      }
      predict--;
      if (label==predict){
	      correct++;
      }
    }
  }

  printf("Accuracy: %0.2lf\n",(double)correct/NTESTING*100.0);
  image.close();
  label.close();

  for (int i=0;i<N1;i++){
    free(w1[i]);
  }
  for (int i=0;i<N2;i++){
    free(w2[i]);
  }
  for(int i=0;i<N_OUT;i++){
    free(w3[i]);
  }
  free(w1);
  free(w2);
  free(w3);

  return 0;
}
