/*
 *  Copyright (C) 2010 by Vitsios Dimitrios
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

/*************************************************
*						 *   
*  Description: a "game of life" implementation  *
*						 *				    
*		        ~ using CUDA ~  	 *
*						 *            
*************************************************/


#include <stdio.h>
#include <sys/types.h>
#include <cuda.h>
#include <time.h>

#define BLOCK_SIZE 256


char *host_board;
int  n, t;



__global__ void make_move(char *dev_board, int n)
{
           
		   __shared__ char block_brd[324], int sum[256], flags[256];
		   int i, j;
		   int local_Idx  = threadIdx.x + threadIdx.y * 16; 
		   flags[local_Idx]=0;
		   
		  
           
           int ix = blockIdx.x * blockDim.x + threadIdx.x;
           int iy = blockIdx.y * blockDim.y + threadIdx.y;
           int index = ix + iy * n;
           
	       int th_idx=threadIdx.x + 1 + (threadIdx.y + 1) * 18;
           
           block_brd[th_idx] = dev_board[index]; //fill in (a part of) the matrix 'block board' with the '256' elements to process 
           
           
		   //fill in with the remaining '68' elements
           
           int ix_ul = blockIdx.x * blockDim.x;  //coordinates for the upper left corner of the quadratic table...
           int iy_ul = blockIdx.y * blockDim.y;  //...(size 16x16) containing the 256 elements designated for processing   
           int upperLeftCorner = (ix_ul == 0 && iy_ul == 0)? (n*n-1): ix_ul - 1 + (iy_ul - 1)*n;
           block_brd[0] = dev_board[upperLeftCorner];
           
           int ix_ur = ix_ul+15;  //coordinates for the upper right corner...
           int iy_ur = iy_ul;	  //...of the quadratic table 
           int upperRightCorner = (ix_ur == n-1 && iy_ur == 0)? n*(n-1): ix_ur + 1 + (iy_ur - 1)*n; 
           block_brd[17] = dev_board[upperRightCorner];
           
           int ix_bl = ix_ul;     //coordinates for the bottom left corner...
           int iy_bl = iy_ul+15;  //...of the quadratic table
           int bottomLeftCorner = (ix_bl == 0 && iy_bl == n-1)? n-1: ix_bl - 1 + (iy_bl + 1)*n;
           block_brd[306] = dev_board[bottomLeftCorner];
           
           int ix_br = ix_ul+15;   //coordinates for the bottom right corner...
           int iy_br = iy_ul+15;   //...of the quadratic table
           int bottomRightCorner = (ix_br == n-1 && iy_br == n-1)? 0: ix_br + 1 + (iy_br + 1)*n; 
           block_brd[323] = dev_board[bottomRightCorner];
           
           
           //Upper Row
           for(int k=0; k<16;k++){
				
				int urIdx1 = (iy_ul == 0)? n*(n-1)+ix_ul+k: (ix_ul+k) + (iy_ul-1) * n;		
				
				block_brd[k+1] = dev_board[urIdx1];
		   }		
           
           //Right Column
           for(int k=0, i=35; k<16; k++, i+=18){
				
				int urIdx2 = (ix_ur == n-1)? n*(iy_ul+k):(ix_ur+1) + (iy_ul+k) * n;		
				
				block_brd[i] = dev_board[urIdx2];
		   }
           
           //Bottom Row
           for(int k=0, i=307; k<16;k++, i++){
				
				int urIdx3 = (iy_bl == n-1)? ix_ul+k: (ix_bl+k) + (iy_bl+1) * n;		
				
				block_brd[i] = dev_board[urIdx3];
		   }
		   
		   //Left Column
           for(int k=0, i=18; k<16;k++, i+=18){
				
				int urIdx4 = (ix_ul == 0)? n*(iy_ul+1+k)-1 :(ix_ul-1) + (iy_ul+k) * n;		
				
				block_brd[i] = dev_board[urIdx4];
		   } 
		   
		   
           if ( index < n*n ){
				
			 
           
           
              sum[local_Idx]   =	  (block_brd[threadIdx.x + threadIdx.y * 18])
									 +(block_brd[threadIdx.x + 1 + threadIdx.y * 18])
								     +(block_brd[threadIdx.x + 2 + threadIdx.y * 18])
								     +(block_brd[threadIdx.x + (threadIdx.y + 1) * 18])
								     +(block_brd[threadIdx.x + 2 + (threadIdx.y + 1) * 18])
								     +(block_brd[threadIdx.x + (threadIdx.y + 2) * 18])
								     +(block_brd[threadIdx.x + 1 + (threadIdx.y + 2) * 18])
								     +(block_brd[threadIdx.x + 2 + (threadIdx.y + 2) * 18]);
						 
						 
						 
           if(block_brd[th_idx]==0 && sum[local_Idx]==3)
               flags[local_Idx]=1;
           if(block_brd[th_idx]==1 && (sum[local_Idx]<2 || sum[local_Idx]>3))
               flags[local_Idx]=2;
                  
           __syncthreads();
		   
           if(flags[local_Idx] == 1)
	       	dev_board[index]=1;
           
		   if(flags[local_Idx] == 2)
            dev_board[index]=0;
			
           }
	
}




int main(int argc, char* argv[]){
    
	FILE *Data_File;
	
	int *br,i,j;
	char inFile[256], *inFileName=inFile, test_ch, outFileName[256], *dev_board;;
	int ncount=0;
	time_t start, end;
	
	if (argc != 3 && argc !=1) {
		printf("Insufficient parameters!\n");
		exit(1);
	}
	else{
		if (argc == 1){
			printf("Type the number of iterations: ");
			scanf("%d",&t);
			printf("\nType the name of the data file: ");
			scanf("%s",inFile);
			printf("\n\n");
		}
		else{
				t=atoi(argv[1]);
				inFileName=argv[2];
		}
	}
	Data_File=fopen(inFileName,"r");

	do{
		fscanf(Data_File, "%c", &test_ch);
		ncount++;
	}while(test_ch!='\n');
	n=ncount/2;   // in d.txt: ncounter = 600 --> 300, the numbers (0,1) and 300, the spaces. So: n=ncounter/2 
	fseek(Data_File,0,SEEK_SET);

	
	int size = n * n *sizeof(char);
	host_board=(char *)malloc(size);
	
	for(i=0;i<n;i++){
	        for(j=0;j<n;j++){
	                fscanf(Data_File,"%c ",&host_board[i+j*n]);
					host_board[i+j*n]-=48;
	        }
	}
	                
    printf("Reading done\n\n");
	fclose(Data_File);
	
	//Start timer...
    time(&start);
    
	cudaMalloc((void**)&dev_board,size);

	cudaMemcpy( dev_board, host_board, size, cudaMemcpyHostToDevice );
	
	printf("Transfer done\n\n");
	
	
    dim3 dimBlock(16,16);
    dim3 dimGrid( (n/dimBlock.x) , (n/dimBlock.y) );


    for(int r=0; r<t; r++)
    {       

			make_move<<< dimGrid, dimBlock>>>(dev_board, n);
			
    }
    
    cudaMemcpy(host_board, dev_board, size, cudaMemcpyDeviceToHost);
    
    printf("GPU PROCESSING COMPLETE!\n\n");

	//Stop timer;
    time(&end);
    
    
	//Writing to the output data file
	i=0;
	do{
		outFileName[i]=inFileName[i];
		i++;
	}while(inFileName[i]!=0);
	outFileName[i]='.';
	outFileName[i+1]='o';
	outFileName[i+2]='u';
	outFileName[i+3]='t';
	outFileName[i+4]=0;
	printf("Output File \''%s\'' was created!\n",outFileName);
	Data_File=fopen(outFileName,"w");

	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			fprintf(Data_File,"%i",host_board[i+j*n]);
			if(j!=n-1)
				fprintf(Data_File," ");
		}
		if(i!=n-1)
			fprintf(Data_File,"\n");
	}
    cudaFree(dev_board);
    free(host_board);
    
    double dif=difftime(end,start);
    printf("\n*******************************************************************************");
    printf("\nTotal time elapsed for transfering the data and computing in GPU: %.2lf seconds",dif);
    
    scanf("%d",&i);
    return EXIT_SUCCESS;
}


