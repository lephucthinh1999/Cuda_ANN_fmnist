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
#define EPOCHS 3
#define EPSILON 1e-3
#define HEIGHT 28
#define WIDTH 28
#define N_IN 784 //N_IN=HEIGH*WIDTH
#define NTRAINING 60000


char input[N_IN];
double expected[N_OUT];

//percepton
double **w1, b1[N1];
double layer1[N1];
double out_layer1[N1];
double **w2, b2[N2];
double layer2[N2];
double  out_layer2[N2];
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
string training_label_fn= "../mnist/train-labels-idx3-ubyte";
string training_image_fn= "../mnist/train-images-idx3-ubyte";


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

void back_propagration()
{
  for (int i=0;i<N_OUT;i++){//delta 3
    delta3[i]=output[i]-expected[i];
  }
  
  for (int i=0;i<N_OUT;i++){//update w3
    for (int j=0;j<N2;j++){
      w3[i][j]-=LEARNING_RATE*delta3[i]*out_layer2[j];
    }
  }

  for (int i=0;i<N_OUT;i++){//update b3
    b3[i]-=LEARNING_RATE*delta3[i];
  }
  
  for (int i=0;i<N2;i++){//delta 2
    delta2[i]=0;
    for (int j=0;j<N_OUT;j++){
      delta2[i]+=w3[j][i]*delta3[j];
    }
    delta2[i]*=(layer2[i]>0);
  }

  for (int i=0;i<N2;i++){//update w2
    for (int j=0;j<N1;j++){
      w2[i][j]-=LEARNING_RATE*delta2[i]*out_layer1[j];
    }
  }

  for (int i=0;i<N2;i++){//update b2
    b2[i]-=LEARNING_RATE*delta2[i];
  }
  
  for (int i=0;i<N1;i++){//delta 1
    delta1[i]=0;
    for (int j=0;j<N2;j++){
      delta1[i]+=w2[j][i]*delta2[j];
    }
    delta1[i]*=(layer1[i]>0)?1:0;
  }

  for (int i=0;i<N1;i++){//update w1
    for (int j=0;j<N_IN;j++){
      w1[i][j]-=LEARNING_RATE*delta1[i]*input[j];
    }
  }

  for (int i=0;i<N1;i++){//update b1
    b1[i]-=LEARNING_RATE*delta1[i];
  }
}

double cross_entropy()
{
  double cr_entr=0;
  for (int i=0;i<N_OUT;i++){
    cr_entr+=expected[i]*log(output[i]);
  }
  return -cr_entr;
}

int learning()
{
  for (int i=0;i<EPOCHS;i++){
    perceptron();
    back_propagration();
    if (cross_entropy()<EPSILON)
      return i;
  }
  return EPOCHS;
}

void init(double **weight,double *bias,int r,int c) //length of bias is r
{
  for (int i=0;i<r;i++){
    for (int j=0;j<c;j++){
      int sign=rand()%2; //+-
      weight[i][j]=(rand()%6)/10.0;
      if (sign)
	weight[i][j]=-weight[i][j];
    }
  }

  for (int i=0;i<r;i++){
    int sign=rand()%2;
    bias[i]=(rand()%10+1)/(10.0+c);
    if (sign)
      bias[i]=-bias[i];
  }
}

void next_label()
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

}

void write_matrix(ofstream &file, double **weight, double *bias, int r, int c)
{
  for (int i=0;i<r;i++){
    for (int j=0;j<c;j++){
      file << weight[i][j] << " ";
    }
    file<<endl;
  }
  
  for (int i=0;i<c;i++){
    file<<bias[i]<<" ";
  }
  file<<endl;
}

void write_model(string filename)
{
  ofstream file(filename.c_str(),ios::out);
  write_matrix(file,w1,b1,N1,N_IN);
  write_matrix(file,w2,b2,N2,N1);
  write_matrix(file,w3,b3,N_OUT,N2);
  file.close();
}

int main(int argc, char *argv[])
{
  image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
  label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

  //Time begins
  clock_t begin = clock();
  // Reading file headers
  char number;
  for (int i = 0; i < 16; i++) {
    image.read(&number, sizeof(char));
  }
  
  for (int i = 0; i < 8; i++) {
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
  
  init(w1,b1,N1,N_IN);
  init(w2,b2,N2,N1);
  init(w3,b3,N_OUT,N2);

  for (int i=0;i<EPOCHS;i++){
    for (int sample=1;sample<=NTRAINING;sample++){
      printf("Sample: %d\n", sample);
      next_label();
      learning();
      printf("Cross entropy: %0.6lf\n", cross_entropy());
      if (sample%100==0){
	write_model(MODEL);
      }
    }
  }
  
  clock_t end = clock();
  double elapsed_time = double(end-begin)/CLOCKS_PER_SEC;
  cout<<"Time elapsed: "<<elapsed_time<<" seconds"<<"\n";
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
