#include<stdio.h>

int main(){
    int size = 3 ;
    int A[] ={1,2,3,4,5,6,7,8,9};
    int B[] ={5,6,7,8,1,2,3,4,5};
    int C[9] ={0,0,0,0};
/*
    for (int j = 0; j < size; j++) {
    for (int k = 0; k < size; k++) {
        int val = B[j * size + k];
        for (int i = 0; i < size; i++) {
            C[i * size + j] += val * A[k * size + i];
        }
    }
}*/

for (int i = 0; i < size; i++) {
    for (int k = 0; k < size; k++) {
        double val = A[i * size + k];   // Correct: A[j][k]
        for (int j = 0; j < size; j++) {
            C[i * size + j] += val * B[k * size + j]; // Correct: A[k][i], C[j][i]
        }
    }
}



for(int i=0 ;i <9; i++){
    printf("%d  ",C[i]);
}

}