//code written by Jan Zimbelmann
//it is required to have the following file:
//a file for the numeric results of the temperature transformation
//with the name './transformT.csv'
//the spin configurations are stored as 0 and 1
//this files is originally saved as './solutions_ising.cpp'
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

//initialize variables
//////

const int L0 = 16; //1D system length
const int steps = 2; //enlarging steps
double J = 1; //coupling constant
int data_points;
double T; //actual inverse temperatuer for an iteration
double B; //actual inverse temperatuer for an iteration
double r = 0; // uniform random number
int x = 0; // random x position
int y = 0; // random y position
int intE; //indexation for the histograms of the energy
int intM; //indexation for the histograms of the magnetization
double condition = 0; //spin flip condition

//observables
int beforeE = 0; //local energy before spin flip
int afterE = 0; //local energy after spin flip
int difE = 0; //energy difference of of previous two energies
int E = 0; //total energy
int M = 0; //total magnetization
double beforeG = 0; //previous two point spin correlation function
double G = 0; //total two point spin correlation sum
double savedE = 0; //summed energy over all iterations
double savedM = 0; //summed magnetization "
double savedG = 0; //summed G             "
int distG = 3; //distance of G
int iteration; //total iterations of spin flips
int start; //spin flip iteration conter at which the observable calculation starts
int sweeps = 20000; //amount of total sweeps over the entire latice
int N; //system particle number

//functions
//////

int add1(int target, int L){ //custom +1; addition funciton
        if(target<(L-1)){return target + 1;}
        else{return 0;}
}
int addX(int target, int L, int distance){ //custom +X; addition function
        int newTarget = target;
        for(int i = 0; i<distance; ++i){
                newTarget = add1(newTarget,L);
        }
        return newTarget;
}
int sub1(int target, int L){ //custom -1; subtraction function
        if(target>0){return target - 1;}
        else{return L-1;}
}
int subX(int target, int L, int distance){ //custom -X; subtraction function
        int newTarget = target;
        for(int i = 0; i<distance; ++i){
                newTarget = sub1(newTarget,L);
        }
        return newTarget;
}


void printDisplay(vector<vector<int>> spin, int L){ //printing a colored spin configuration
	for(int i = 0; i < L; ++i){
		for(int j = 0; j < L; ++j){
			int s = spin[i][j];
			if(s==1){
				//color green
				cout << "\033[1;32m"<< " " << s << "\033[0m";
			}
			else{
				//color red
				cout << "\33[;31m" << s << "\033[0m";
			}
		}
		cout << endl;
	}
	cout << endl;	
}


//functions for calculating the observables
double totE(vector<vector<int>> spin, int L){ //calculate the total energy
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

double localE(vector<vector<int>> spin, int x, int y, int L){ //calculate the local energy
	int dif = 0;
	dif += spin[y][x]*spin[y][add1(x,L)];
	dif += spin[y][x]*spin[y][sub1(x,L)];
	dif += spin[y][x]*spin[add1(y,L)][x];
	dif += spin[y][x]*spin[sub1(y,L)][x];
	dif = -dif * J;
	return dif;
}

double totM(vector<vector<int>> spin, int L){ //calculate the local energy
	int m = 0;
	for(int i = 0; i<L; ++i){
		for(int j = 0; j<L; ++j){
			m += spin[i][j];
		}
	}
	return m;
}

template <typename locGT> //multidimensional template
double localG(locGT& spin, int x, int y, int L, int distance){ 
        double dif = 0;
        dif += spin[y][x]*spin[y][addX(x,L,distance)];
        dif += spin[y][x]*spin[y][subX(x,L,distance)];
        dif += spin[y][x]*spin[addX(y,L,distance)][x];
        dif += spin[y][x]*spin[subX(y,L,distance)][x];
        return dif/(2*L*L);
}

template <typename totGT>
double totG(totGT& spin, int L, int distance){
        double g = 0;
        for(int i = 0; i<L; ++i){
                for(int j = 0; j<L; ++j){
                        g+= spin[j][i] * spin[j][addX(i,L,distance)];
                        g+= spin[j][i] * spin[addX(j,L,distance)][i];
                }
        }
        return g/(2*L*L);
}

//start the Metropolis MC algorithm for calculating the solutions to the Ising 
//spin configurations for different system sizes
//////

int main(){
	//begin timer
	clock_t begin = clock();
	srand(time(NULL));

	//load the numerical solutions for the temperatures
        ifstream f;
        f.open("transformT.csv");
        vector<vector<double>> temp;
        string line, val;
        while(getline(f, line)){
                vector<double> v;
                stringstream s (line);
                while(getline(s, val, ','))
                        v.push_back(stod(val));
                temp.push_back(v);
        }

        data_points = temp[0].size(); //calculate the amount of initial temperatures
        cout << "total amount of data points "<< data_points << endl;
        f.close();

	//create a folder under linux
	string folderSol = "solutions/";
	mkdir("solutions", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);	
	string folderHist = "mc_histogram/";
	mkdir("mc_histogram", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);	
	
	//iterate over every super resolution step, as well as the different system sizes
	for(int step=0; step <= steps; step+=1){
		ofstream mcFile;
		string name = string("z") + to_string(step) + string("Mc") + string("L") + to_string(L0) + string(".csv");	
		mcFile.open(folderSol+name);
		cout << "Enlargening step counter is currently at: " << step << "."<< endl;
		//step dependent boundary conditions
		const int L = L0 * pow(2,step);	
		N = L*L;
		vector<vector<int>> spin(L, vector<int> (L,0));
		start = N *5000;
		iteration = (sweeps * N)+start;
		//randomize spins, plotting and calculating observables
		//for(int i=0; i<L; ++i){for(int j=0; j<L; ++j){spin[i][j]=((round((float)rand()/((float)RAND_MAX)))-0.5)*2;}} //hot start
		for(int i=0; i<L; ++i){for(int j=0; j<L; ++j){spin[i][j]=1;}} //cold start
		
		printDisplay(spin,L); //plot the spins

		//initiate observables
		M = totM(spin,L); //calculate total magnetization
		E = totE(spin,L); //calculate total energy
		G = totG(spin, L, distG); //calculate total two point spin corelation function
		//loop over all temperatures of interest
		for(int t = 0; t<data_points; ++t){
			
                        //initiate histogram files
                        vector<vector<int>> histM; //create a vector for the magnetization histogram
                        histM.resize((N/2)+1); //resize the previos histogram to the possible counters
                        vector<vector<int>> histE; //create a vector for the magnetization histogram
                        histE.resize(N+1); //resize the previos histogram to the possible counters
                        for(int i=0; i<=N/2; ++i){ //iterate over all possible magnetizations
                                histM[i].push_back(i*2);
                                histM[i].push_back(0);
                        }
                        for(int i=0; i<=N; ++i){ //iterate over all possible energies
                                histE[i].push_back((i*4)-(2*N));
                                histE[i].push_back(0);
			}
                        ofstream histM_file;
                        ofstream histE_file;
                        name = string("z") + to_string(step) + string("HistM") + to_string(t) + string("L") + to_string(L0) + string(".csv");
                        histM_file.open(folderHist + name);
                        name = string("z") + to_string(step) + string("HistE") + to_string(t) + string("L") + to_string(L0) + string(".csv");
                        histE_file.open(folderHist + name);

			T = temp[step][t]; //temperature
			cout << "T: " << T << endl;

			B = 1/T; //inverse temperature

			//renormalization temperature
			//B = rgBeta(B, L, step);
			savedM = 0; //later saved expected magnetization
			savedE = 0; //later saved expected energy
			savedG = 0; //later saved expected two point spin correlation
			
			//for loop over the iteration amount
			for(int i = 0; i < iteration; ++i){
				//random spin position
				x=rand()%L; //random x position
				y=rand()%L; //random y position
				//calculate Energy difference and flip spin
				beforeE = localE(spin, x, y, L);
				beforeG = localG(spin, x, y, L, distG); //also for G
				spin[y][x] *= -1; //execute the spin flip
				afterE = localE(spin, x, y, L);
				difE = afterE - beforeE; //calculate the energy difference from before and after
				//check spin acceptance condition
				r = ((double) rand() / (RAND_MAX)); //random double number between 0 and 1
                                condition = exp(-difE * B); //acceptance or rejection rate of the spin flip
                                if(difE < 0||r <= condition){ //accept the spin flip
                                        E += difE;
					M += 2 * spin[y][x];
					G += localG(spin,x,y,L,distG)-beforeG;
                                }
                                else{ //reject the spin flip
                                        spin[y][x] *= -1;
                                }
                                //store observables
				if(i>=start){
					savedM += abs(M);
					savedE += E;
					savedG += G;
					intM = abs(M)/2;
					intE = (E+(N*2))/4;
					histE[intE][1]+=1;
					histM[intM][1]+=1;
				}	
			}
			//for security reasons printing out the calculated observables
			//if they actually do not equate eqach other, there should be a bug in the code
			cout <<
                        E << " = " << totE(spin, L) << " , " <<
                        M << " = " << totM(spin, L) << " , " <<
                        G << " = " << totG(spin, L, distG)
                        << endl;
			//storing the histograms
			for(int i=0; i<=N/2; ++i){
                                histM_file << histM[i][0] << "," << histM[i][1] << endl;
                        }
                        for(int i=0; i<=N; ++i){
                                histE_file << histE[i][0] << "," << histE[i][1] << endl;
			}
                        histM_file.close();
                        histE_file.close();
			//average stored observables
			savedM /= iteration-start;
			savedE /= iteration-start;
			savedG /= iteration-start;
			//saving observables
			mcFile << T << "," << savedM << "," << savedE << "," << savedG << endl;	
			cout << "T : " << T << " M : " << savedM << " E : " << savedE << " G : " << savedG << endl;	

		}
		//closing observable file
		mcFile.close();
	}
	//end timer
	clock_t end = clock();
	double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << elapsed_secs << "seconds." << endl;
	return 0;
}

