#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<math.h>
#include <string.h>

#define IN 785
#define num_of_training_data 10000
#define num_of_outputs 10

int label[num_of_training_data];
double input[num_of_training_data][IN-1];
double output[num_of_training_data][num_of_outputs];

void Import(){
	int i,j;
	FILE* f1;
	FILE* f2;
	f1 =fopen("./DAT/Test/labels.dat","r");
	for (i=0; i<num_of_training_data; i++){
		fscanf(f1,"%d",&label[i]);
	}
	fclose(f1);
	/*
	f2 =fopen("./DAT/clothes_inputs.dat","r");
	for (i=0; i<num_of_training_data; i++){
		for (j=0; j<IN-1; j++){
			fscanf(f2,"%lf",&input[i][j]);
		}
	}
	fclose(f2);
	*/
}
//######################################
void Extract() {
	int i,j;
	FILE* f1;
	f1 =fopen("./DAT/Test/cat_labels.dat","w");
	for (i=0; i<num_of_training_data; i++){
        for (j=0; j<num_of_outputs; j++){
            fprintf(f1,"%lf\t",output[i][j]);
        }
        fprintf(f1,"\n"); 
    }
	fclose(f1);
}
//#################################################################
int main() {
	int i,j;
	memset(output,0,sizeof(output));
    Import();
	for (i=0; i<num_of_training_data; i++){
		for (j=0; j<num_of_outputs; j++){
			if (j==label[i]){
				output[i][j]=1.0f;
			}
		}
		
	}
	Extract();
}