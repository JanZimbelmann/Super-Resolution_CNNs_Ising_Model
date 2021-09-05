//code written by Jan Zimbelmann
//it is required to have the following file:
//a file for the numeric results of the temperature transformation
//with the name './transformT.csv'
//the spin configurations are stored as 0 and 1
//this files is originally saved as './configurations_ising.cpp'
//////

//libraries
//////
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <ctime>
#include <sys/stat.h>
#include <sstream>

using namespace std;

//initializing variables
//////

const int L0 = 16; //1D system length
const int steps = 2; //enlarging steps
double J = 1; //coupling constant

int iteration; //number of total spin flip iterations
int start; //starting iteration to count the observable
int outSize = 20000; //amount of the configurations stored
int mult = 30; //every N*mult, one iteration is stored
int counter; //security counter, later used

//information on the spin lattice
int x; // random position in first dimension
int y; // random position in second dimension
double r; // uniform random number
double condition;
int blockSpin; //for the majority rule decimation

//observables
int beforeE = 0; //local energy before spinflip
int afterE = 0; //local energy after spinflip
int difE = 0;//energy differene of previous two energies
int E = 0; //total energy
int M = 0; //total magnetization
double G = 0; //total two point spin correlation function 
double beforeG = 0; //local G before spinflip
double savedE = 0; //summed energy over all iterations
double savedM = 0; //summed magnetization "
double savedG = 0; //summed G "
int distG = 3; //distance of G 
int data_points; //size of the loaded temperature array


//functions
//////

int add1(int target, int L){ //custom +1 ; addition function
	if(target<(L-1)){return target + 1;}
	else{return 0;}
}
int addX(int target, int L, int distance){ //custom +X ;addition function
	int newTarget = target;
	for(int i = 0; i<distance; ++i){
		newTarget = add1(newTarget,L);
	}
	return newTarget;
}
int sub1(int target, int L){ //custom -1 ; subtraction function
	if(target>0){return target - 1;}
	else{return L-1;}
}
int subX(int target, int L, int distance){ //custom -X ; subtraction function
	int newTarget = target;
	for(int i = 0; i<distance; ++i){
		newTarget = sub1(newTarget,L);
	}
	return newTarget;
}

template <typename spinT> //multidimensional template
void printDisplay(spinT& spin, int L){ //printing a colored spin configuration
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

template <typename totET> //multidimensional template
//functions for calculating the observables
double totE(totET& spin, int L){ //calculate the total energy
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

template <typename locET> //multidimensional template
int localE(locET& spin, int x, int y, int L){ //calculated the local energy
	int dif = 0;
	dif += spin[y][x]*spin[y][add1(x,L)];
	dif += spin[y][x]*spin[y][sub1(x,L)];
	dif += spin[y][x]*spin[add1(y,L)][x];
	dif += spin[y][x]*spin[sub1(y,L)][x];
	dif = -dif * J;
	return dif;
}

template <typename totMT> //multidimensional template
int totM(totMT& spin, int L){ //calculate the total magnetization
	int m = 0;
	for(int i = 0; i<L; ++i){
		for(int j = 0; j<L; ++j){
			m += spin[i][j];
		}
	}
	return m;
}

template <typename locGT> //multidimensional template
double localG(locGT& spin, int x, int y, int L, int distance){ //calculate the local G
	double dif = 0;
	dif += spin[y][x]*spin[y][addX(x,L,distance)];
	dif += spin[y][x]*spin[y][subX(x,L,distance)];
	dif += spin[y][x]*spin[addX(y,L,distance)][x];
	dif += spin[y][x]*spin[subX(y,L,distance)][x];
	return dif/(2*L*L);
}

template <typename totGT> //multidimensional template
double totG(totGT& spin, int L, int distance){ //calculate the total G
	double g = 0;
	for(int i = 0; i<L; ++i){
		for(int j = 0; j<L; ++j){
			g+= spin[j][i] * spin[j][addX(i,L,distance)];
			g+= spin[j][i] * spin[addX(j,L,distance)][i];
		}
	}
	return g/(2*L*L);
}

//start the Metropolis MC algorithm for creating the Ising spin configurations
//////

int main(){
	//begin timer
	clock_t begin = clock();
	srand(time(NULL));

	//load the numeric solutions for the temperatures
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
	cout << "total amount of data points " << data_points << endl;
	f.close();
	//create a folder under linux
	string folder = "configurations/"; //folder for storing the spin configurations
	mkdir("configurations", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);	
	//iterate over every super resolution step, as well as the initial system size
	for(int step=0; step <= steps; step+=1){
		cout << "super-resolution step is currently at: " << step << endl;
		//step dependent boundary conditions
		const int L = L0; //system size is gonna only be calculated at the original system size
		int N = L*L; //numer of lattice sites
		int spin[L][L]; //create a original spin lattice
		int rgSpin[L][L]; //create the decimated spin lattice
		int period = N * mult; //period of storing a new spin configuration
		start = N * 3000; //starting the procedure after a set amount of iterations
		iteration = (outSize * period)+start; //total amount of iterations
		//randomize spins, plotting and calculating observables
		//for(int i=0; i<L; ++i){for(int j=0; j<L; ++j){spin[i][j]=((round((float)rand()/((float)RAND_MAX)))-0.5)*2;}} //hot start
		for(int i=0; i<L; ++i){for(int j=0; j<L; ++j){spin[i][j]=1;}} //cold start, all spins pointing at +1
		printDisplay(spin,L); //plot the spins
		
		//initiate observable/configuration files 
		ofstream mcConfFile; //file for the original spin configuration
		ofstream rgConfFile; //file for the decimated spin configuration
		ofstream mcFile; //file for the observables of the original spin configurations
		string name = string("z") + to_string(step) + string("Mc") + string("L") + to_string(L) + string(".csv");	
		mcFile.open(folder+name);
		//initiate observables
		E = totE(spin,L); //calculate total energy
		M = totM(spin,L); //calculate total magnetization
		G = totG(spin,L,distG); //calculate total two point spin correlation function
		//loop over all temperatures of interest
		for(int t = 0; t<data_points; ++t){
			double T = temp[step][t]; //set the current Temperature
			cout << "T: " << T <<endl;
			//initiate spin configuration files
			string name = string("z") + to_string(step) + string("Mc") + to_string(t) + string("L") + to_string(L0) + string(".csv");
			mcConfFile.open(folder + name); //original spin configuration file
			name = string("z") + to_string(step) + string("Rg") + to_string(t) + string("L") + to_string(L0) + string(".csv");
			rgConfFile.open(folder + name); //decimated spin configuration file
			//surveilance of the configuration size with a counter
			counter = 0;

			savedM = 0; //later saved magnetization M
			savedE = 0; //later saved expected energy E
			savedG = 0; //later saved total spin correlation G

			//for loop over the iteration amount and and store the spin configurations & observables
			for(int i = 0; i < iteration; ++i){		
				//random spin position
				x=rand()%L; //random x value
				y=rand()%L; //random y value
				//calculate the local energy difference before the flip spin
				beforeE = localE(spin, x, y, L);
				beforeG = localG(spin, x, y, L, distG); //also the two point spin correlation function
				spin[y][x] *= -1; //execute the spin flip
				//calculate Energy difference/two point correlation before the flip spin
				afterE = localE(spin, x, y, L);
				difE = afterE - beforeE; // calculate the energy difference from before and after
				//check spin acceptance condition
				r = ((double) rand() / (RAND_MAX)); //random double number between 0 and 1
				condition = exp(-difE * (1/T)); //acceptance/rejection rate of the spin flip
				if(difE < 0||r <= condition){ //accept the spin flip
					E += difE;
					M += 2*spin[y][x];
					G += localG(spin,x,y,L,distG) - beforeG;
				}
				else{ //reject the spin flip
					spin[y][x] *= -1;
				}
				//summing observables for every iteration
				if(i>=start){
					savedE += E;
					savedM += abs(M);
					savedG += G;
				}
				
				//decimating the spin configurations and storing both configurations
				if(i>= start && i%period ==0 && counter < outSize){//calculated only with after every certain period
					counter += 1; //counter for security reasons
					//decimation of the spin configurations
					//now iterating over every block
					for(int p = 0; p<L; p = p+2){
						for(int q=0; q<L; q = q+2){
							//majority rule of the block decimation
							blockSpin = 0;
							blockSpin += spin[p][q];
							blockSpin += spin[p][q+1];
							blockSpin += spin[p+1][q];
							blockSpin += spin[p+1][q+1];
							blockSpin = (blockSpin > 0) ? 1 : ((blockSpin < 0) ? -1 : spin[p][q]);
							//after the majority rule & now the simple upscaling is applied
							//resulting in a deflated spin configuration
							rgSpin[p][q] = blockSpin;
							rgSpin[p][q+1] = blockSpin;
							rgSpin[p+1][q] = blockSpin;
							rgSpin[p+1][q+1] = blockSpin;
						}
					}
					//storing the original and deflated spin configuration
					//the 2 dimensional array is converted into a 1 dimensional string of 0s and 1s
					for(int p = 0; p<L; ++p){
						for(int q=0; q<L; ++q)
							if(not(p == L-1 and q == L-1)){
								mcConfFile << (spin[p][q]+1)/2 << ",";
								rgConfFile << (rgSpin[p][q]+1)/2 << ",";
							}
							else{
								mcConfFile << (spin[p][q]+1)/2 << endl;
								rgConfFile << (rgSpin[p][q]+1)/2 << endl;
							}	
					}
				}
			}
			//for security reasons printing out the calculated energy, magnetization and two point correlation function
			//if they actually do not equate to each other, there is a bug in the code
			cout <<
			M << " = " << totM(spin, L) << " , " <<
			E << " = " << totE(spin, L) << " , " <<
			G << " = " << totG(spin, L, distG)
			<< endl;
			//store the expectation values of the observables
			savedM /= iteration-start; //magnetization
			savedE /= iteration-start; //energy
			savedG /= iteration-start; //two point spin correlation function
			//saving observables
			mcFile << T << "," << savedM << "," << savedE << "," << savedG << endl;	
			//printing the calculated expectations of the observables
			cout << "T : " << T << endl << "E : " << savedE << endl << "M : " << savedM << endl;
			//closing spin configuration configuration files
			mcConfFile.close();
			rgConfFile.close();
		}
		//plotting the final spin configuration
		printDisplay(spin,L);
		//checking the counter for security reasons
		cout << "Counter has reached: " << counter << ". It was expected to reach: " << outSize << "." << endl;
		//closing the observables file
		mcFile.close();
	}
	//end timer
	clock_t end = clock();
	double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << elapsed_secs << " seconds." << endl; //print the run time
	return 0;
}
