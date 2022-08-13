//initialize random model weights
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define IN 784
#define L1 100
#define L2 10
#define bias 0.5

double WL1[L1][IN+1];
double WL2[L2][L1+1];


void Extract(){
	int i,j;
	FILE* f1;
	FILE* f2;
	
	f1 =fopen("./DAT/layer1_weights.dat","w");
	f2=fopen("./DAT/layer2_weights.dat","w");
	for (i=0; i<L1; i++){
		for (j = 0; j<IN+1; ++j){
			fprintf(f1,"%lf\t",WL1[i][j]);
		}
		fprintf(f1, "\n");
	}
	fclose(f1);
	for (i=0; i<L2; i++){
		for (j = 0; j<L1+1; ++j){
			fprintf(f2,"%lf\t",WL2[i][j]);
		}
		fprintf(f2, "\n");
	}
	fclose(f2);
}
//#################################################################
//#################################################################
int main(){
	srand(time(NULL));
    //Layer1
    for (int i = 0; i < L1; i++)
    {
        for (int j=0; j<IN+1; j++){
           double temp=(double)(rand() % 6) / 10.0;
           WL1[i][j]=temp;
        }
     }
    //Layer2
    for (int i = 0; i < L2; i++)
    {
        for (int j=0; j<L1+1; j++){
            double temp=(double)(rand() % 6) / 10.0;//(double)rand()/(double)RAND_MAX;////
            WL2[i][j]=temp;
        }
     }
     Extract();
}