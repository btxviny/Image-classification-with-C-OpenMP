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
#define num_of_training_data 60000
#define num_of_epochs 100
#define bias 0.1
#define lr 0.1
#define NUM_THREADS 8

int chunk_size=num_of_training_data/NUM_THREADS;
double chunk=num_of_training_data/NUM_THREADS;

double input[IN];
double output[L2];

double global_WL1[L1][IN+1];
double global_WL2[L2][L1+1];
double global_OL1[L1];
double global_OL2[L2];

double training_in[num_of_training_data][IN];
double training_out[num_of_training_data][L2];
double deltaHidden[L1];
double deltaOutput[L2];
double Error[num_of_epochs]={0};

//######################################
void shuffle(){
	for (size_t i = 0; i <num_of_training_data; i++)
	{
		int r= rand()%num_of_training_data;
		for (size_t j = 0; j <IN; j++)
		{
			int temp=training_in[i][j];
			training_in[i][j]=training_in[r][j];
			training_in[r][j]=temp;

		}
		for (size_t j = 0; j <L2; j++)
		{
			int temp=training_out[i][j];
			training_out[i][j]=training_out[r][j];
			training_out[r][j]=temp;

		}
		
	}
	
}
//######################################
//Load Random Weights
void Import(double global_WL1[L1][IN+1], double global_WL2[L2][L1+1]){
	int i,j;
	FILE* f1;
	FILE* f2;
	
	f1 =fopen("./DAT/layer1_weights.dat","r");
	for (i=0; i<L1; i++){
		for (j = 0; j<IN+1; ++j){
			fscanf(f1,"%lf", &global_WL1[i][j]);
		}
	}
	fclose(f1);
    f2 =fopen("./DAT/layer2_weights.dat","r");
    for (i=0; i<L2; i++){
		for (j = 0; j<L1+1; ++j){
			fscanf(f2,"%lf", &global_WL2[i][j]);
		}
	}
	fclose(f2);
	printf("Model initialized with random weights\n");
	
}

//#######################################################################
//Load Training Data
void ImportTraining(){
    int i,j;
    FILE* f3;   
    FILE* f4;
    f3 =fopen("./DAT/Train/inputs.dat","r");
	for (i=0; i<num_of_training_data; i++){
		for (j=0; j<IN; j++){
			fscanf(f3,"%lf", &training_in[i][j]);
		}
	}
	fclose(f3);
	printf("Imported training inputs\n");
	
	f4 =fopen("./DAT/Train/cat_labels.dat","r");
	for (i=0; i<num_of_training_data; i++){
		for (j=0; j<L2; j++){
			fscanf(f4,"%lf", &training_out[i][j]);
		}
	}
	fclose(f4);
	printf("Imported training outputs\n");
	
}
//#######################################################################
//Save Model
void Extract() {
	int i,j;
	FILE* f1;
    FILE* f2;
	f1 =fopen("./DAT/Model/model_l1_par.dat","w");
	for (i=0; i<L1; i++){
        for (j=0; j<IN+1; j++){
            fprintf(f1,"%lf\t",global_WL1[i][j]);
        }
        fprintf(f1,"\n"); 
    }
	fclose(f1);
	printf("Extracted layer 1 weights\n");
    f2 =fopen("./DAT/Model/model_l2_par.dat","w");
	for (i=0; i<L2; i++){
        for (j=0; j<L1+1; j++){
            fprintf(f2,"%lf\t",global_WL2[i][j]);
        }
        fprintf(f2,"\n");
    }
	fclose(f2);
	printf("Extracted layer 2 weights\n");
}
//#################################################################

//######################################
// Activation function and its derivative
double sigmoid(double x) { return 1 / (1 + exp(-x)); };
double dSigmoid(double x) { return x * (1 - x); };


//#####################################
void Train(double input[IN],double output[L2],double WL1[L1][IN+1],double WL2[L2][L1+1],double OL1[L1],double OL2[L2]){
    double deltaHidden[L1], deltaOutput[L2];
    double dError;
    int i,j;
    //ACTIVATION
    for (i=0; i<L1; i++){
		for (j=0; j<IN; j++){
			OL1[i]+=input[j]*WL1[i][j];
            
		}
		OL1[i]+=WL1[i][IN];
		OL1[i]=sigmoid(OL1[i]);
	}
	for (i=0; i<L2; i++){
		for (j=0; j<L1; j++){
			OL2[i]+=OL1[j]*WL2[i][j];
            
		}
		OL2[i]+=WL2[i][IN];
		OL2[i]=sigmoid(OL2[i]);
	}
    //Compute change in output weights
     for (i=0; i<L2;i++){
        dError=(output[i]-OL2[i]);
        deltaOutput[i]= dError*dSigmoid(OL2[i]);
    }
    // Compute change in hidden weights
    for (i=0; i<L1; i++){
        dError=0.0f;
        for (j=0; j<L2; j++){
            dError+=deltaOutput[j]*WL2[j][i];
        }
        deltaHidden[i]=dError*dSigmoid(OL1[i]);
    }
    //Apply change in output weights
    for (i=0; i<L2; i++){
        for (j=0; j<L1; j++){
            WL2[i][j]+=OL1[j]*deltaOutput[i]*lr;
        }
        //change bias
        WL2[i][L1]+=deltaOutput[i]*lr;
    }
    // Apply change in hidden weights
    for (i=0; i<L1;i++){
        for (j=0;j<IN;j++){
            WL1[i][j]+=input[j]*deltaHidden[i]*lr;
        }
         //change bias
        WL1[i][IN]+=deltaHidden[i]*lr;
    }
}
//#####################################
int main(){
    int i,j,k;
    double begin= clock();
    ImportTraining();
    Import(global_WL1,global_WL2);
    for(i=0; i<num_of_epochs;i++){
       #pragma omp parallel num_threads(NUM_THREADS) shared(global_WL1,global_WL2,Error) private(j,k,input,output) 
       {    // kathe thread pairnei ena kommati training data kai ypologizei ta W
            double WL1[L1][IN+1], WL2[L2][L1+1];
            double OL1[L1], OL2[L2];
            double error=0.0f;
            #pragma omp critical
            {
                memcpy(WL1,global_WL1,sizeof(WL1));
                memcpy(WL2,global_WL2,sizeof(WL2));
                
            }
            #pragma omp single
            {
                memset(global_WL1,0,sizeof(global_WL1));
                memset(global_WL2,0,sizeof(global_WL2));
            }

            #pragma omp for schedule(static,chunk_size) 
                for (j=0; j<num_of_training_data; j++){
                    memset(OL1,0,sizeof(OL1));
                    memset(OL2,0,sizeof(OL2));
                    memcpy(input,training_in[j],sizeof(input));
                    memcpy(output,training_out[j],sizeof(output));
                    Train(input,output,WL1,WL2,OL1,OL2);
                    for (k=0; k<L2; k++){
                        error+=(output[k]-OL2[k])*(output[k]-OL2[k]);
                    }
                }
            
            //ypologise to athroisma gia kathe weight tou layer1 gia ola ta data
            for (j=0; j<L1; j++){
                for (k=0; k<IN+1; k++){
                    #pragma omp critical
                    {
                        global_WL1[j][k]+=WL1[j][k];
                    }
                }
            }
            //ypologise to athroisma gia kathe weight tou layer2 gia ola ta data
            for (j=0; j<L2; j++){
                for (k=0; k<L1+1; k++){
                    #pragma omp critical
                    {
                        global_WL2[j][k]+=WL2[j][k];
                    }
                }
            }
            #pragma omp critical
            {
                Error[i]+=error;
            }
                    
            #pragma omp single
            {
                for (j=0; j<L1; j++){
                    for (k=0; k<IN+1; k++){
                        global_WL1[j][k]/NUM_THREADS;
                    }
                }
                for (j=0; j<L2; j++){
                    for (k=0; k<L1+1; k++){
                        global_WL2[j][k]/=NUM_THREADS;
                    }
                }
            }
        }
      
        printf("Epoch %d completed, Error--->%.6lf\n",i,Error[i]/(num_of_training_data*L2));
		shuffle();
    }

   Extract();
   double end=clock();
   printf("NN3 time spent: %f\n",(end-begin)/CLOCKS_PER_SEC);
}


