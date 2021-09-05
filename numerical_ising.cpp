//code written by Jan Zimbelmann
//this files is originally saved as './numerical_ising.cpp'
//////

//libraries
//////
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <ctime>
#include <sys/stat.h>
#include <stdio.h>
#include<bits/stdc++.h>
using namespace std;

//initializing variabes
//////

const int L0 = 16; //1D system length
double J = 1; //coupling constant
const int sizeTemp = 49; //inverse temperature array size
double factor = 0.05; //beta is later on multiplied by factor
double Tc =  2.26918531421; //critical temperature at infinite system size
double bias = Tc-0.8; //starting point of the simulations
double T; //actual temperatuer for an iteration
double B; //actual inverse temperatuer for an iteration
int x; //random x position
int y; //random y position
double r; //uniform random number
double condition ;

//observables
int beforeE = 0; //local energy before spin flip 
int afterE = 0; //local energy after spin flip
int difE = 0; //energy difference of previous two energies
int E = 0; //total energy
int M = 0; //total magnetization
double savedE = 0; //summed energy over all iterations
double savedM = 0; //summed magnetization of all iterations

//functions
//////

int add1(int target, int L){ //custom +1; adition function
	if(target<(L-1)){return target + 1;}
	else{return 0;}
}

int sub1(int target, int L){ //custom -1; subtraction function
	if(target>0){return target - 1;}
	else{return L-1;}
}

void printDisplay(vector<vector<int>> spin, int L){ //printing a colored spin configuration
	for(int i = 0; i < L; ++i){
		for(int j = 0; j < L; ++j){
			int s = spin[i][j];
			if(s==1){//color green
				cout << "\033[1;32m"<< " " << s << "\033[0m";
			}
			else{//color red
				cout << "\33[;31m" << s << "\033[0m";
			}
		}
		cout << endl;
	}
	cout << endl;	
}


double totE(vector<vector<int>> spin, int L){ //calculating the total energy
	double energ = 0;
	for(int i = 0; i<L; ++i){
		for(int j = 0; j<L; ++j){
			energ += spin[i][j]*spin[add1(i,L)][j];
			energ += spin[i][j]*spin[i][add1(j,L)];
		}
	}
	energ *= -J;
	return energ;
}

double localE(vector<vector<int>> spin, int x, int y, int L){ //calculating the local energy
	int dif = 0;
	dif += spin[y][x]*spin[y][add1(x,L)];
	dif += spin[y][x]*spin[y][sub1(x,L)];
	dif += spin[y][x]*spin[add1(y,L)][x];
	dif += spin[y][x]*spin[sub1(y,L)][x];
	dif = -dif * J;
	return dif;
}

double totM(vector<vector<int>> spin, int L){ //calculating the total magnetization
	int m = 0;
	for(int i = 0; i<L; ++i){
		for(int j = 0; j<L; ++j){
			m += spin[i][j];
		}
	}
	return m;
}


//start the Metropolis MC algortihm for creating the solutions of the
//Ising model. This is for the original and the decimated case
//////

int main(){
	//begin timer
	clock_t begin = clock();
	srand(time(NULL));
	//create a folder under linux
	string folder = "numerical/";
	mkdir("numerical", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);	
	
	ofstream mcFile;
	string name = string("Mc") + string("L") + to_string(L0) + string(".csv");	
	mcFile.open(folder+name);
	ofstream rgFile;
	name = string("Rg") + string("L") + to_string(L0) + string(".csv");	
	rgFile.open(folder+name);

	for(int step=0; step < 2; step+=1){
		cout << "Enlargening step counter is currently at: " << step << "."<< endl;
		//step dependent boundary conditions
		const int L = L0 / pow(2,1-step); //system length	
		int N = L*L; //number of lattice sites
		vector<vector<int>> spin(L, vector<int> (L,0)); //initial spin configuration
		start = N * 1000; //starting the procedure after a set amount of iterations
		
		iteration = (sweeps * N)+start; //total amount of spin flip iterations
		//randomize spins, plotting and calculating observables
		//for(int i=0; i<L; ++i){for(int j=0; j<L; ++j){spin[i][j]=((round((float)rand()/((float)RAND_MAX)))-0.5)*2;}} //hot start
		for(int i=0; i<L; ++i){for(int j=0; j<L; ++j){spin[i][j]=1;}} //cold start
		
		printDisplay(spin,L); //plot the spin configuration

		//initiate observables
		E = totE(spin,L); //total Energy
		M = totM(spin,L); //total magnetization
		for(int t = 0; t<sizeTemp; ++t){
			T = (t*factor)+bias; //temperature
			B = 1/T; //inverse temperature
			
			counter = 0; //counter for surveiling the configuration size
			savedM = 0; //later saved magnetization
			//for loop over the iteration amount
			for(int i = 0; i < iteration; ++i){
				//random spin position
				x=rand()%L; //random x value
				y=rand()%L; //random y value
				//calculate Energy difference and flip spin
				beforeE = localE(spin, x, y, L);
				spin[y][x] *= -1; //spin flip
				afterE = localE(spin, x, y, L);
				difE = afterE - beforeE; //calculate the energy difference
				//check spin acceptance condition
				r = ((double) rand() / (RAND_MAX)); //random double between 0 and 1
                                condition = exp(-difE * B); //the probability for a spin flip
                                if(difE < 0||r <= condition){ //accepting the spin flip
                                        E += difE;
					M += 2*spin[y][x];
                                }
                                else{ //rejecting the spin flip
                                        spin[y][x] *= -1;
                                }
                                //store observables
				if(i>= start){
					counter += 1;
					//storing the observable for the original spin configurations
					if(step == 0){
						savedM += abs(M);
					}
					
					//storing the observables for the decimated spin configurations
					else{//step == 1 here
						//decimating the spin configurations to blocks of 2x2
						int blockM = 0;
						for(int p = 0; p<L; p = p+2){
							for(int q=0; q<L; q = q+2){
								int blockSpin = 0;
								blockSpin += spin[p][q];
								blockSpin += spin[p][q+1];
								blockSpin += spin[p+1][q];
								blockSpin += spin[p+1][q+1];
								blockM += (blockSpin > 0) ? 1 : ((blockSpin < 0) ? -1 : spin[p][q]);
							}
						}
						savedM += abs(blockM);
					}
				}
			}
			//average stored observables
			savedM /= counter;
			//saving observables
			if(step==0){
				mcFile << T << "," << savedM << endl;	
			}
			if(step==1){
				rgFile << T << "," << savedM << endl;	
			}
			cout << "T : " << T << " M : " << savedM << endl;
			printDisplay(spin,L);
			//closing configuration files
		}
		cout << M << " = " << totM(spin,L) << endl; 
		cout << "Counter has reached: " << counter << ". It was expected to reach: " << sweeps * N << "." << endl;
		//closing observable file
		mcFile.close();
	}
	//end timer
	clock_t end = clock();
	double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << elapsed_secs << "seconds." << endl;
	return 0;
}
