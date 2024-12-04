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
#define EPOCHS 5
#define EPSILON 1e-3
#define HEIGHT 28
#define WIDTH 28
#define N_IN 784 //N_IN=HEIGH*WIDTH
#define NTESTING 10000

#define MODEL "model.dat"

char input[N_IN];
double expected[N_OUT];

//percepton
double w1[N_IN][N1], b1[N1];
double layer1[N1];
int out_layer1[N1];
double w2[N1][N2], b2[N2];
double layer2[N2];
int  out_layer2[N2];
double w3[N2][N_OUT], b3[N_OUT];
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
string testing_label_fn="../mnist/";
string testing_image_fn="../mnist/";


void softmax(double *in_out, double *ouput, int n)
{
  double d=0;
  for (int i=0;i<n;i++){
    output[i]=exp(in_out[i]);
    d+=output[i];
  }
  for (int i=0;i<n;i++){
    output[i]/=d;
  }
}
void ReLU(double *layer, int *out_layer, int n)
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
      layer1[i]+=w1[j][i]*input[j];
    }
  }
  
  ReLU(layer1,out_layer1,N1); //out of layer 1
  
  for (int i=0;i<N2;i++){ //layer 2
    layer2[i]=b2[i];
    for (int j=0;j<N1;j++){
      layer2[i]+=w2[j][i]*out_layer1[j];
    }
  }

  ReLU(layer2,out_layer2,N2); //out of layer 2

  for (int i=0;i<N_OUT;i++){ //indirecr of output
    in_out[i]=b3[i];
    for (int j=0;j<N2;j++){
      in_out[i]+=w3[j][i]*out_layer2[j];
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

  for (int i=0;i<c;i++){
    file>>bias[i];
  }
}

void load_model(string file_name) {
  ifstream file(file_name.c_str(), ios::in);
	
  load_matrix(file,(double **)w1,b1,N_IN,N1);
  load_matrix(file,(double **)w2,b2,N1,N2);
  load_matrix(file,(double **)w3,b3,N2,N_OUT);
  
  file.close();
}


int next_label()
{
  char number;
  for (int i=0;i<HEIGHT;i++){
    for (int j=0;j<WIDTH;j++){
      image.read(&number, sizeof(char));
      input[i*WIDTH+j]=(number!=0);
    }
  }
  
  label.read(&number, sizeof(char));
  for (int i = 0; i < N_OUT; ++i) {
    expected[i] = 0.0;
  }
  expected[number] = 1.0;
  return (int) number;
}

double square_error()
{
  double sq_err=0;
  for (int i=0;i<N_OUT;i++){
    sq_err+=(output[i]-expected[i])*(output[i]-expected[i]);
  }
  return sq_err*0.5;
}

int main()
{
  image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
  label.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

  // Reading file headers
  char number;
  for (int i = 0; i < 16; ++i) {
    image.read(&number, sizeof(char));
  }
  for (int i = 0; i < 8; ++i) {
    label.read(&number, sizeof(char));
  }

  load_model(MODEL);

  int correct=0;
  for (int sample=1;sample<=NTESTING;sample++){
    int label=next_label();
    perceptron();
    int predict=0;
    for (int i=1;i<N_OUT;i++){
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
  return 0;
}
