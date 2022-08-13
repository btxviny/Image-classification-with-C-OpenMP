//time spent: 707.018000
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define IN 784
#define L1 100
#define L2 10
#define num_of_test_data 10000

double input[IN];
double output[L2];
double WL1[L1][IN+1];
double WL2[L2][L1+1];
double OL1[L1];
double OL2[L2];
double Error=0.0f;

double test_in[num_of_test_data][IN];
double test_out[num_of_test_data][L2];


//######################################
void ImportWeights(double WL1[L1][IN+1], double WL2[L2][L1+1]){
	int i,j;
	FILE* f1;
	FILE* f2;
	
	f1 =fopen("./DAT/model_l1_par.dat","r");
	for (i=0; i<L1; i++){
		for (j = 0; j<IN+1; ++j){
			fscanf(f1,"%lf", &WL1[i][j]);
		}
	}
	fclose(f1);
	//printf("Imported Layer1 Weights\n");

    f2 =fopen("./DAT/model_l2_par.dat","r");
    for (i=0; i<L2; i++){
		for (j = 0; j<L1+1; ++j){
			fscanf(f2,"%lf", &WL2[i][j]);
		}
	}
	fclose(f2);
	//printf("Imported Layer2 Weights\n");
	
	
}

//#######################################################################
void ImportData(){
    int i,j;
    FILE* f3;   
    FILE* f4;
    f3 =fopen("./DAT/Test/test_inputs.dat","r");
	for (i=0; i< num_of_test_data; i++){
		for (j=0; j<IN; j++){
			fscanf(f3,"%lf", &test_in[i][j]);
		}
	}
	fclose(f3);
	printf("Imported training inputs\n");
	
	f4 =fopen("./DAT/Test/test_outputs.dat","r");
	for (i=0; i< num_of_test_data; i++){
		for (j=0; j<L2; j++){
			fscanf(f4,"%lf", &test_out[i][j]);
		}
	}
	fclose(f4);
	printf("Imported desired outputs\n");
	
}
//######################################
// Activation function and its derivative
double sigmoid(double x) { return 1 / (1 + exp(-x)); };
double dSigmoid(double x) { return x * (1 - x); };

void activateNN(double *Vector){
    memset(OL1,0,sizeof(OL1));
    memset(OL2,0,sizeof(OL2));
	int i,j;
	for (i=0; i<L1; i++){
		for (j=0; j<IN; j++){
			OL1[i]+=Vector[j]*WL1[i][j];
            
		}
		OL1[i]+=WL1[i][IN];
		OL1[i]=sigmoid(OL1[i]);
	}
	for (i=0; i<L2; i++){
		for (j=0; j<L1; j++){
			OL2[i]+=OL1[j]*WL2[i][j];
            
		}
		OL2[i]+=WL2[i][L1];
		OL2[i]=sigmoid(OL2[i]);
	}
}
//#####################################
int main(){
    int i,j;
    Error=0.0f;
    ImportData();
	ImportWeights(WL1,WL2);
	double begin= clock();
    for(i=0; i<num_of_test_data; i++){
        
        memcpy(input,test_in[i],sizeof(input));
        memcpy(output,test_out[i],sizeof(output));
        activateNN(input);
        for (j=0; j<L2; j++){
			Error+=(output[j]-OL2[j])*(output[j]-OL2[j]);
		}
        //printf("Error: %lf\n",Error/(L2*num_of_test_data));
    }
    printf("Error: %.10lf\n",Error/(num_of_test_data*L2));
}


