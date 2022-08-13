#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<math.h>
#include <string.h>

#define IN 785
#define num_of_training_data 60000
#define num_of_outputs 10

int label[num_of_training_data];
double input[num_of_training_data][IN-1];
double output[num_of_training_data][num_of_outputs];

void Import(){
	int i,j;
	FILE* f1;
	FILE* f2;
	f1 =fopen("./DAT/Train/labels.dat","r");
	for (i=0; i<num_of_training_data; i++){
		fscanf(f1,"%d",&label[i]);
	}
	fclose(f1);
}
//######################################
void Extract() {
	int i,j;
	FILE* f1;
	f1 =fopen("./DAT/Train/cat_labels.dat","w");
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