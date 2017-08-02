#include <iostream>
#include<math.h>
#include<ctime>
#include<set>
#include<cstdlib>
#include<vector>
#include "mpi.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace std;

double fy(double, double);
double fh(double);
double ff(double, int, double);
double pe(double);
double random(double);
double random_normal();
void resampling(int N, double * q, int *);
void RRP(int *, int, int, int *, int *, vector<int>*, int, int & supSize, int & regSize);
void sizes(int surpSize, int regSize, int * surpProcess, int * surpIndex, int * regProcess, int * regIndex, int * shorProcess, vector<int>*, int t);
double SIR()
{
	/*double P0 = 5.0;
	double qSum = 0;
	int N = 1000;
	double * x = new double[N];
	double * e = new double[N];
	double * q = new double[N];
	double * y = new double[N];
	double * h = new double[N];
	for (int i = 0; i < N; i++)
	{
	//	x[i] = rand() % (sqrt(P0)); //make array
	x[i] = random(sqrt(P0));
	y[i] = fy(x[i]); // + e(t) <- What exactly is this e(t)? It was in the formula for calculating y(t)
	h[i] = fh(x[i]);
	}
	for (int t = 0; t < 100; t++)
	{
	for (int i = 0; i < N; i++)
	{
	e[i] = y[i] - h[i];
	q[i] = pe(e[i]);
	for (int i = 0; i < N; i++)
	qSum += q[i];
	for (int i = 0; i < N; i++)
	{
	q[i] = q[i] / qSum;
	}
	int * zeros = new int [N];
	resampling(N, q, zeros);

	double * newX = new double[N];
	for (int i = 0; i < N; i++)
	{
	newX[i] = x[zeros[i]];
	cout << newX[i] << ' ';
	}
	cout << "done";
	}
	double * meanArray = new double[100];
	for (int i = 0; i < N; i++)
	{
	meanArray[t] = newX[i];
	}
	meanArray[t] = meanArray[t] / N;

	}*/
	return 0;
}
double pe(double x) {
	double ma = pow((2 * M_PI), (-1 / 2))*exp(-x*x / 2);
	return ma;
}

double fy(double x, double Q)
{
	return (pow(x, 2)) / 20 + random(sqrt(Q));
}


double ff(double x, int t, double Q) {
	return (x / 2 + 25 * x / (1 + pow(x, 2)) + 8 * cos(1.2*t)) + random(sqrt(Q));
}
double random(double sigma)
{
	//rand() % sqrt(R);
	return sigma*random_normal();
}
double drand()
{
	return (rand() + 1.0) / (RAND_MAX + 1.0);
}
double random_normal() {
	return sqrt(-2 * log(drand())) * cos(2 * M_PI*drand());
}
double fh(double x)
{
	return (pow(x, 2)) / 20;
}

void resampling(int N, double * q, int * zeros) {
	//cout << q[0] << "qqqqqqqqqqqqqqq";
	double * p = new double[N];
	p[0] = q[0];
	//cout << p[0] << " ";
	for (int i = 1; i < N; i++) // cumsum
	{
		p[i] = p[i - 1] + q[i];
		//	cout << p[i]<<"  ";
	}
	double * randArray = new double[N];
	for (int i = 0; i < N; i++) // Initialize the random array
	{
		randArray[i] = ((double)rand() / (RAND_MAX));
		//	cout << randArray[i] << "  ";
	}
	for (int i = N; i >= 1; i--)
	{
		randArray[i - 1] = pow(randArray[i - 1], (1.0 / i));
		//	cout << randArray[i-1] << "  ";
	}
	double * u = new double[N];
	double * ut = new double[N];
	u[0] = randArray[0];
	//cout << u[0] << " ";
	for (int i = 1; i < N; i++) // cumprod
	{
		u[i] = u[i - 1] * randArray[i];
		//cout << "u is " << endl;
		//	cout << u[i] << " ";
	}
	for (int i = N - 1; i >= 0; i--) // fliplr
	{
		ut[i] = u[N - 1 - i];
		//	cout << ut[i] << " ";
		//cout << u[i] << "this is u ";

	}
	int k = 0;
	for (int j = 0; j < N; j++)
	{
		while (p[k] < ut[j])
		{
			k = k + 1;
			//printf("k number %d", k);
		}
		zeros[j] = k;
		//printf("this is the zeros %d\n",zeros[j]);
	//cout << "testing success" << endl;
	}
}

void RRP(int * zeros,  int blockSize, int Npid, int * quotient, int * remainder, vector<int>*v, int N, int & supSize, int & regSize)
{
	MPI_Status * status;
	int counter = 0;
	//cout << "hi" << endl;
	vector <int> surplusProcess;
	vector <int> surplusIndex;
	vector <int> shortageProcess;
	vector <int> shortageIndex;
	//vector <int> quotientShortage;
	//int * quotientShortage = new int[Npid];
	//printf("tis the size so far %i\n", (int) surplusProcess.size());
	//srand(time(null));
	int RandIndex;
	//vector<int> *v;
	//v = new vector<int>[4];
	//v[0].at(0) = 5;
	//v[0].resize(sizeof(v[0]) + 1);
	//array<array<int, 5>, 5> values;
	int * survivorSum = new int[Npid];
	int * shortageSurvivorSum = new int[Npid];
	int * tempSurvivorSum = new int[N];
	int * tempRemainder = new int[N];
	for (int i = 0; i < Npid; i++)
	{
		survivorSum[i] = 0;
		shortageSurvivorSum[i] = 0;
		//quotientShortage[i] = 0;
	}
	for (int i = 0; i < N; i++)
	{
		tempSurvivorSum[i] = 0;
		tempRemainder[i] = 0;
	}
	for (int i = 0; i < N; i++)
	{
		div_t answer = div(zeros[i], blockSize);
		quotient[i] = answer.quot;
		remainder[i] = answer.rem;
		survivorSum[quotient[i]] += 1;
		shortageSurvivorSum[quotient[i]] = survivorSum[quotient[i]];
		tempSurvivorSum[i] = quotient[i];
		tempRemainder[i] = remainder[i];
		//printf( "Processor number %d index number %d \n", quotient, remainder);
		//MPI_Send(remainder, 1, MPI_DOUBLE, quotient, 02, MPI_COMM_WORLD);
		//MPI_Recv(localZeros, 1 ,MPI_INT, 0, 02, MPI_COMM_WORLD, status);
		//printf("localzero is %d \n", zeros[i]);
		//printf("quotient is %d and remainder number is %d \n", quotient[i], remainder[i]);
		//printf("The survivors for %d are %d and this is process number %d\n", i, survivorSum[quotient[i]], quotient[i]);
		//MPI_Bcast(quotient, sizeof(quotient), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		//MPI_Bcast(remainder, sizeof(quotient), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	//for (int i = 0; i < N; i++)
		//printf("this is temp remainder %i\n", tempRemainder[i]);
	//for (int i = 0; i < Npid; i++)
		//printf("pid %i has %i survivors \n", i, survivorSum[i]);
	//for (int i = 0; i < N; i++)
		//printf("quotient is %i\n", quotient[i]);
	//int empty = sizeof(quotient);
	for (int i = 0; i < Npid; i++)
		for (int j = 0; j < blockSize; j++)
		{
			if (shortageSurvivorSum[i] < blockSize)
			{
				shortageSurvivorSum[i] += 1;
				shortageProcess.push_back(i);
				//printf("%i has been pushed back into the process vector\n", shortageProcess.at(j));
				//shortageIndex.push_back(remainder[i]);
			}
		}
	//printf("shortage survivor sum for process 0 %i \n", shortageSurvivorSum[0]);
	//for (int i = 0; i < N - blockSize; i++)
		//cout << quotientShortage[i] << endl;
	//for (int i = 0; i < 4; i++)
		//printf("Processor %i has %i survivors\n", i, survivorSum[i]);
	///for (int i = 0; i < sizeof(quotient); i++)
		//printf("quotients %i and remainders %i \n", quotient[i], remainder[i]);
	for (int i = 0; i < Npid; i++)
		for (int j = 0; j < N; j++)
		{
			if (survivorSum[i] > blockSize && tempSurvivorSum[j] != -1 && i == quotient[j])
			{
				//if (survivorSum[i] == blockSize)
					//break;
				//surplusProcess.resize(surplusProcess.size() + 1);
				//surplusIndex.resize(surplusIndex.size() + 1);
				surplusProcess.push_back(quotient[j]);
				surplusIndex.push_back(remainder[j]);
				survivorSum[i] -= 1;
				//printf("%i has been added to the list with its index %i and survivor sum is now %i in iteration %i and i number %i\n", quotient[j], remainder[j], survivorSum[i], j, i);
				tempSurvivorSum[j] = -1;
				tempRemainder[j] = -1;
				//RandIndex = rand() % surplusProcess.size();
				//cout << 1;
				//printf("surplus process is %d and surplus index is %d at iteration %d random %d size of is %i\n", surplusProcess.at(i), surplusIndex.at(i), i, RandIndex, (int)surplusProcess.size());
			}
			//else if (survivorSum[i] < blockSize && tempSurvivorSum[j] != -1 && i == quotient[j])
		}
	//for (int i = 0; i < shortageProcess.size(); i++)
		//printf("process %i quotient %i\n", shortageProcess.at(i), quotient[i]);
		//for (int i = 0; i < shortageProcess.size(); i++)
			//if (shortageProcess.at(i) == quotient[i])
			//{
			//	shortageIndex.push_back(remainder[shortageProcess.at(i)]);
				//printf("this remainder has been inserted %i size %i\n ", remainder[i], (int) shortageProcess.size());
				//shortageProcess.resize(sizeof(shortageProcess) + 1);
				//shortageIndex.resize(sizeof(shortageIndex) + 1);
				//shortageProcess.push_back(quotientShortage[i]);
				//shortageIndex.push_back(remainder[i]);
				//printf("shortage process is %d at iteration %d\n", shortageProcess.at(i), i);
				//cout << 2;
			//}

	//return;
	 //vector array - sender, local x, receiver, copies
	//cout << "bye" << endl;
	int surplusCounter = surplusProcess.size();
	for (int i = 0; i < surplusCounter; i++)
	{
		RandIndex = rand() % surplusProcess.size();
		v[0].push_back(surplusProcess[RandIndex]);
		v[1].push_back(surplusIndex[RandIndex]);
			//v[0].at(i), v[1].at(i), (int) surplusProcess.size());
		surplusProcess.erase(surplusProcess.begin() + RandIndex);
		surplusIndex.erase(surplusIndex.begin() + RandIndex);
	//cout << surplusProcess.size() << i << endl;
		//printf("After surplus process is %d and shortage index is %d at iteration %d\n", shortageProcess.at(i), shortageIndex.at(i), i);
		//printf("iteration number surplus%i\n", surplusProcess.);
		//printf("%d has been deleted and it local index %d has been deleted \n", )

	}
	int shortageCounter = shortageProcess.size();
	//printf("shortage counter %i\n", (int) shortageProcess.size());
	for (int i = 0; i < shortageCounter; i++)
	{
		RandIndex = rand() % shortageProcess.size();
		v[2].push_back(shortageProcess[RandIndex]);
		//v[3].push_back(shortageIndex[RandIndex]);
		shortageProcess.erase(shortageProcess.begin() + RandIndex);
		//shortageIndex.erase(shortageIndex.begin() + RandIndex);
		//printf("iteration number %i and now at size %i\n", i, (int) shortageProcess.size());
	//	printf("selected numbers are %d and %d this is size of surplus process %i \n", v[2].at(i), v[3].at(i), (int)shortageProcess.size());
	}

		supSize = v[0].size();
		//shortSize = v[2].size();
		/*surpProcess = new int[supSize];
		surpIndex = new int[supSize];
		shortProcess = new int[v[2].size()];
		regProcess = new int[v[2].size()];
		regIndex = new int[v[2].size()];
		//regSize = v[2].size();
		//int surpProcess [dsupSize];
		//cout << "supsize is finally " << supSize << endl;*/
		//for (int i = 0; i < Npid; i++)
			//printf("survivorsum is %i\n", survivorSum[i]);
		//for (int i = 0; i < N; i++)
		//{
			//printf("tempSurvivorsum is %i and quotient is %i and tempRemainder is %i\n", tempSurvivorSum[i], tempRemainder[i], quotient[i]);
		//}
		//for (int i = 0; i < Npid; i++)
		regSize = 0;
			for (int j = 0; j < N; j++)
			//{
				if (tempSurvivorSum[j] != -1)
				{
					//printf("came in %i\n", j);
					v[3].push_back(tempSurvivorSum[j]);
					v[4].push_back(remainder[j]);
					//regIndex[j] = tempRemainder[j];
					tempSurvivorSum[j] = -1;
					regSize++;
				}
			//for (int i = 0; i < regSize; i++)
					//printf("process %i with index remainder %i at iteration %i\n", v[3].at(i), v[4].at(i), i);
			//cout << "reg size is " << regSize << endl;
			//}
	/*	for (int i = 0; i < supSize; i++)
		{
			surpProcess[i] = 0;
		}*/
	/*for (int i = 0; i < supSize; i++)
	{
		surpProcess[i] = v[0].at(i);
		surpIndex[i] = v[1].at(i);
		//cout << "surp process i is " << surpProcess[i] << "and this is the size " << supSize << endl;
		printf("this is the value %i and this pid %i and this iteration %i and this size %i beforeeeeeeeeeeee\n", surplusProcess[i], 0, i, supSize);

	}
		//return;
	for (int i = 0; i < v[2].size(); i++)
	{
		shortProcess[i] = v[2].at(i);
		//shortIndex[i] = v[3].at(i);
		//printf("this is the shortage value %i and this is the shortage index %i at iteration number %i \n", shortProcess[i], shortIndex[i], i);
	}*/
	for (int i = 0; i < N; i++)
	{
		///printf("temp survivor sum is %i \n", tempSurvivorSum[i]);
	}
	//cout << "this is the size of surplus" << v[0].size() << endl;
	//printf("Blocksize is %d and size of zeros is %d", blockSize, sizeof(zeros));
	//cout << blockSize << " " << sizeof(zeros);
		/*for (int i = 0; i < Npid; i++)
			for (int j = 0; j < blockSize; j++)
			{
				routes[i][j] = zeros[counter];
				counter++;
				//printf("This is the  routes quotient %d this is the zeros %d iterayion number %d \n", routes[i][j], zeros[counter - 1], counter - 1);
			}*/
	//cout << "size of v[0].size() << endl;
	//cout << supSize << endl;
}
void sizes(int surpSize, int regSize, int * surpProcess, int * surpIndex, int * shortProcess, int * regProcess, int * regIndex, vector<int>*v, int t)
{
	for (int i = 0; i < surpSize; i++)
	{
		surpProcess[i] = v[0].at(i);
		surpIndex[i] = v[1].at(i);
		//cout << "surp process i is " << surpProcess[i] << "and this is the size " << supSize << endl;
		//printf("this is the value %i and this pid %i and this iteration %i and this size %i beforeeeeeeeeeeee\n", surpProcess[i], 0, i, surpSize);

	}
	for (int i = 0; i < surpSize; i++)
	{
		shortProcess[i] = v[2].at(i);
		//shortIndex[i] = v[3].at(i);
		//printf("this is the shortage value %i and this is the shortage index %i at iteration number %i \n", shortProcess[i], shortIndex[i], i);
	}
	//cout << "v3 size is " << v[3].size() << "v4 size is " << v[4].size() << endl;
	for (int i = 0; i < regSize; i++)
	{
		regProcess[i] = v[3].at(i);
		regIndex[i] = v[4].at(i);
		//printf("reg process is %i and reg index is %i\n", regProcess[i], regIndex[i]);
	}
	//printf("this has been reached\n");
	//if(t == 2)
	//return;
}
int main(int argc, char * argv[])
{
	MPI_Init(&argc, &argv);
vector<double> xNew (1);
	int Npid;
	int pid;
	MPI_Comm_size(MPI_COMM_WORLD,&Npid);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	//srand(time(0));
	srand(time(NULL) + pid);
	double P0 = 5.0;
	double Q = 10.0;
	double R = 1.0;
	//int surPro[];
	double qSum = 0;
	int timeStep = 100;
	int tag = 01;
	int N = 8;
	int number;
	int blockSize = N / Npid;
	//double * x = new double[N];
	MPI_Status * status;
	double * x = new double [N];
	
	//int * localZeros = new int [blockSize];
	double * localE = new double[blockSize];
	double * localQ = new double[blockSize];
	int num;
	int * shortProcess;
	int * shortIndex;
	int * surplusProcess;
	int * surplusIndex;
	int * regProcess;
	int * regIndex;
	int surpSize = 0;
	int regSize = 0;
	int shortSize = 0;
	vector<int> *v;
	v = new vector<int>[5];
	double * y = new double[N];
	double * h = new double[N];
	double * yy = new double[blockSize];
	double * hh = new double[blockSize];
	double * e = new double[blockSize];
	double * q = new double[blockSize];
	double * qGlobal = new double[N];
	double * sample = new double [20];
	double * localX = new double[blockSize];
	double * newX = new double[N];
	double localMeanArray;
	double rootQSum = 0.0;
	double * meanArray = new double[timeStep];
	int * zeros = new int[N];
	int * quotient = new int[N];
	int * remainder = new int[N];
	int ** routes = new int *[Npid];
	for (int i = 0; i != Npid; ++i)
		routes[i] = new int[blockSize];
	//set<int, int> routes;
	double * globalSum = new double[Npid];
	double * tempMean = new double[Npid];
	meanArray[0] = 0;
	/*for (int i = 0; i < sizeof(localZeros); i++)
	{
		localZeros[i] = 0;
	}*/
	//if (pid == 0)
	{
		//for (int i = 0; i < N; i++)
			for (int i = 0; i < blockSize; i++)
		{
			//x.resize(i + 1);
			//x.at(i) = random(sqrt(P0));
			//x[i] = random(sqrt(P0));
			//y[i] = fy(x[i], sqrt(R)); // + e(t) <- What exactly is this e(t)? It was in the formula for calculating y(t)
			//h[i] = fh(x[i]);
			//meanArray[0] += x[i];
			//cout << x.at(i) << ' ';
			localX[i] = random(sqrt(P0));
			yy[i] = fy(localX[i], sqrt(R)); // + e(t) <- What exactly is this e(t)? It was in the formula for calculating y(t)
			hh[i] = fh(localX[i]);
			meanArray[0] += x[i];
			//printf("localX before the process is %f for pid %i\n", localX[i], pid);
		}
		meanArray[0] = meanArray[0] / N;
	}
	///MPI_Scatter(x, N/Npid, MPI_DOUBLE, localX, N/Npid,
		///MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Scatter(y, N / Npid, MPI_DOUBLE, yy, N / Npid,
		//MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Scatter(h, N / Npid, MPI_DOUBLE, hh, N / Npid,
		//MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//for (int i = 0; i < blockSize; i++)
	//{
		//printf("localX before the process is %f for pid %i\n", localX[i], pid);
	//}
	/*for (int i = 0; i < blockSize; i++)
	{
		e[i] = 0;
		q[i] = 0;
	}*/
	for (int t = 1; t < timeStep; t++)
	{
		//	cout << "In the " << t << " iteration" << endl;
		//for (int i = 0; i < blockSize; i++)
		//{
			//printf("Local X fot t = %i is %f on pid %i\n", t, localX[i], pid);
		//}
	/*	if(t > 1)
		{
			for (int i = 0; i < blockSize; i++)
			{
				yy[i] = fy(localX[i], sqrt(R)); // + e(t) <- What exactly is this e(t)? It was in the formula for calculating y(t)
				hh[i] = fh(localX[i]);
			}
		}*/
		//if(t == 1)
			for (int i = 0; i < blockSize; i++)
			{
				e[i] = yy[i] - hh[i];
				q[i] = pe(e[i]);
				//printf("yy[i] %f hh[i] %f e[i] %f for pid %i and t number %i\n", yy[i], hh[i], e[i], pid, t);
				//printf("q[i] %f for pid %i\n", q[i], pid);
			}
		//	cout << "before call resampling q" << q[0];
		qSum = 0.0;
		for (int i = 0; i < blockSize; i++)
		{
			qSum += q[i];
		}
		//cout << "before gather" << endl;
		MPI_Gather(&qSum, 1, MPI_DOUBLE, globalSum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		//printf("qSum for pid %i is %f\n", pid, qSum);
		//if(pid == 0)
		//for (int i = 0; i < Npid; i++)
		//printf("globalSum is %f\n", globalSum[i]);
		//cout << "after gather" << endl;
		if (pid == 0)
		{
		rootQSum = 0.0;
			for (int i = 0; i < Npid; i++)
			{
				rootQSum += globalSum[i];
				//cout << globalSum[i] << " " << i <<endl;
			}
			//printf("final qSum is %f\n", rootQSum);
		}
		MPI_Bcast(&rootQSum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			for (int i = 0; i < blockSize; i++)
			{
				q[i] = q[i] / rootQSum;
				//printf("q[i] %f for pid %i afterrrrrrrrr %f\n", q[i], pid, rootQSum);
			}
			MPI_Gather(q, blockSize, MPI_DOUBLE, qGlobal, blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			//return 0;
		if (pid == 0)
		{
			//	cout << "this is before resampling" << endl;
			resampling(N, qGlobal, zeros);
			RRP(zeros, blockSize, Npid, quotient, remainder, v, N, surpSize, regSize);
			surplusProcess = new int[surpSize];
			surplusIndex = new int[surpSize];
			shortProcess = new int[surpSize];
			regProcess = new int[regSize];
			regIndex = new int[regSize];
			sizes(surpSize, regSize, surplusProcess, surplusIndex, shortProcess, regProcess, regIndex, v, t);
		}
		//if(t == 2)
		//return 0;
			MPI_Bcast(&surpSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
			//MPI_Bcast(&shortSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&regSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

			if (pid != 0)
			{
				surplusProcess = new int[surpSize];
				surplusIndex = new int[surpSize];
				shortProcess = new int[surpSize];
				regProcess = new int[regSize];
				regIndex = new int[regSize];
			}
			MPI_Bcast(surplusProcess, surpSize, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(surplusIndex, surpSize, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(shortProcess, surpSize, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(regProcess, regSize, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(regIndex, regSize, MPI_INT, 0, MPI_COMM_WORLD);
			double * newLocalX = new double[blockSize];
			for (int i = 0; i < blockSize; i++)
			{
				newLocalX[i] = localX[i];
			}
			int top = 0;
			for (int i = 0; i < regSize; i++)
			{
				//make a new array and store in there
				if (pid == regProcess[i])
				{
					newLocalX[top] = localX[regIndex[i]];
					top++;
				}
			}
			//printf("top is %i from pid %i\n", top, pid);
				//	if (pid == 0)
			/*if (pid == 0)
			{
				for (int i = 0; i < surpSize; i++)
					printf("surplus process is %i on pid %i\n", surplusProcess[i], pid);
				for (int i = 0; i < surpSize; i++)
					printf("shortage process is %i on pid %i\n", shortProcess[i], pid);
			}*/
		for (int i = 0; i < surpSize; i++)
		{
			//if(pid == v[0].at(i))
				///MPI_Send(&localX[v[1].at(i)], 1, MPI_DOUBLE, v[2].at(i), tag, MPI_COMM_WORLD);
			//else if (pid == v[2].at(i))
				//MPI_Recv(&localX[v[3].at(i)], 1, MPI_DOUBLE, v[0].at(i), tag, MPI_COMM_WORLD, status);
			if (pid == surplusProcess[i])
			{
				MPI_Send(&localX[surplusIndex[i]], 1, MPI_DOUBLE, shortProcess[i], 1, MPI_COMM_WORLD);
				//printf("Process %i sent %f to %i\n", surplusProcess[i], localX[surplusIndex[i]], shortProcess[i]);
			}
			if (pid == shortProcess[i])
			{
				MPI_Recv(&newLocalX[top], 1, MPI_DOUBLE, surplusProcess[i], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				top++;
				//printf("top is now %i for pid %i\n", top, pid);
			}
			//else
				//cout << "recv failed\n" << endl;
				//printf("iteration %i \n", i);
		}
		//for (int i = 0; i < blockSize; i++)
			//printf("on process %i the value at index %i is %f and the value of old x is %f\n", pid, i, newLocalX[i], localX[i]);
			//printf("on process %i the value at index %i is %f\n", pid, i, newLocalX[i]);
		localMeanArray = 0;
		for (int i = 0; i < blockSize; i++)
		{
			//printf("this is local x number %i on pid %i for iteration %i : %f\n", i, pid, i, newLocalX[i]);
			localMeanArray += newLocalX[i];

		}
		MPI_Gather(&localMeanArray, 1, MPI_DOUBLE, tempMean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (pid == 0)
		{
			for (int i = 0; i < Npid; i++)
				meanArray[t] += tempMean[i];
				meanArray[t] = meanArray[t] / N;
				printf("Mean Value %i %f\n", t, meanArray[t]);
		}
		//cout << meanArray[t] << endl;
		//return 0;
		//for (int i = 0; i < blockSize; i++)
			//printf("localX is %f for pid %i\n", newLocalX[i], pid);
		//delete[] yy;
		//delete[]hh;
		//yy = new double[blockSize];
		//hh = new double[blockSize];
		for (int i = 0; i < blockSize; i++)
		{
			yy[i] = 0;
			hh[i] = 0;
		}
		for (int i = 0; i < blockSize; i++)
		{
			localX[i] = ff(newLocalX[i], t, Q);
			//localX[i] = newLocalX[i];
			yy[i] = fy(localX[i], sqrt(R)); // + e(t) <- What exactly is this e(t)? It was in the formula for calculating y(t)
			hh[i] = fh(localX[i]);
			//e[i] = yy[i] - hh[i];
			//q[i] = pe(e[i]);
			//printf("yy[i] %f hh[i] %f e[i] %f for pid %i and t number %i\n", yy[i], hh[i], e[i], pid, t);
		}
		//return 0;
		delete [] surplusProcess;
		delete[] surplusIndex;
		delete[] shortProcess;
		delete[] regProcess;
		delete[] regIndex;
		v[0].clear();
		v[1].clear();
		v[2].clear();
		v[3].clear();
		v[4].clear();
		//if(pid == 0)
		//printf("Executed %i times\n", t);
	}
		delete[] e;
		delete[] q;
		delete[] yy;
		delete[] hh;
		if (pid == 0)
		{
			int t = 0;
			double * trueX = new double[blockSize];
			trueX[0] = random(sqrt(P0));
			printf("True X number 0 %f\n", trueX[0]);
			for (t = 1; t < timeStep; t++)
			{
				trueX[t] = (trueX[t - 1] / 2 + 25 * trueX[t - 1] / (1 + pow(trueX[t - 1], 2)) + 8 * cos(1.2*(t))) + random(sqrt(Q));
				printf("True X number %i %f\n", t, trueX[t]);
				//std::cout << trueX[t] << endl;
			}
		}
	MPI_Finalize();
	return 0;
}
/*function[xhat] = SIR(y, f, h, pe, Q, P0, N)
x = sqrt(P0)*randn(1, N); % STEP 1, Initialize the particles
for t = 1:100
e = repmat(y(t), 1, N) - h(x); % STEP 2, Calculate weights
q = feval(pe, e); % The likelihood function
q = q / sum(q); % Normalize the importance weights
ind = resampling(q); % STEP 3, Measurement update
x = x(:, ind); % The new particles
xhat(t) = mean(x); % Compute the estimate
x = feval(f, x, t) + sqrt(Q)*randn(1, N); % STEP 4, Time update
end
function[i] = resampling(q)
P = cumsum(q); N = length(q);
u = cumprod(rand(1, N). ^ (1. / (N:-1 : 1)));
ut = fliplr(u); k = 1; i = zeros(1, N);
for j = 1:N
while (P(k)<ut(j))
k = k + 1;
end;
i(j) = k;
end;*/