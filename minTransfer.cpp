#include <iostream>
#include<math.h>
#include<ctime>
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
void MTRP(int t, int *, int, int, int *, int *, int, int & supSize, int & regSize, vector <int> &, vector <int> &, vector <int> &, vector <int> &, vector <int> &, vector <int> &, vector <int> &);
void sizes(int * copies, int surpSize, int * surpProcess, int * surpIndex, int * shorProcess, int t, vector <int>, vector <int>, vector <int>, vector <int>, vector <int>, vector <int>, vector <int> , int regSize, int * regProcess, int * regIndex, int * regCopies);

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
	double * p = new double[N];
	p[0] = q[0];
	//cout << p[0] << " ";
	for (int i = 1; i < N; i++) // cumsum
	{
		p[i] = p[i - 1] + q[i];
	}
	double * randArray = new double[N];
	for (int i = 0; i < N; i++) // Initialize the random array
	{
		randArray[i] = ((double)rand() / (RAND_MAX));
	}
	for (int i = N; i >= 1; i--)
	{
		randArray[i - 1] = pow(randArray[i - 1], (1.0 / i));
	}
	double * u = new double[N];
	double * ut = new double[N];
	u[0] = randArray[0];
	//cout << u[0] << " ";
	for (int i = 1; i < N; i++)
	{
		u[i] = u[i - 1] * randArray[i];
	}
	for (int i = N - 1; i >= 0; i--)
	{
		ut[i] = u[N - 1 - i];

	}
	int k = 0;
	for (int j = 0; j < N; j++)
	{
		while (p[k] < ut[j])
		{
			k = k + 1;
		}
		zeros[j] = k;
	}
}

void MTRP(int t, int * zeros, int blockSize, int Npid, int * quotient, int * remainder, int N, int & supSize, int & regSize, vector <int> & senderProc, vector <int> & senderInd, vector <int> & senderCopies, vector <int> & receivingProc, vector <int> & regProc, vector <int> & regInd, vector <int> & regCopies)
{
	int counter = 0;
	vector <int> surplusProcess;
	vector <int> surplusIndex;
	vector <int> shortageProcess;
	vector <int> shortageIndex;
	vector <int> copiesProc;
	vector <int> copiesInd;
	vector <int> indVal;
	int highestProc = 0;
	int highestInd = 0;
	int highestCopies = 0;
	int highest =0;
	int * survivorSum = new int[Npid];
	int * shortageSurvivorSum = new int[Npid];
	int * tempQuotient = new int[N];
	int * tempRemainder = new int[N];
	int * particles = new int[Npid];
	int jLocal = 0;
	for (int i = 0; i < Npid; i++)
	{
		survivorSum[i] = 0;
		shortageSurvivorSum[i] = 0;
		particles[i] = 0;
	}
	for (int i = 0; i < N; i++)
	{
		tempQuotient[i] = 0;
		tempRemainder[i] = 0;
	}
	for (int i = 0; i < N; i++)
	{
		div_t answer = div(zeros[i], blockSize);
		quotient[i] = answer.quot;
		remainder[i] = answer.rem;
		survivorSum[quotient[i]] += 1;
		shortageSurvivorSum[quotient[i]] = survivorSum[quotient[i]];
		tempQuotient[i] = quotient[i];
		tempRemainder[i] = remainder[i];
		if (i == 0)
		{
			copiesProc.push_back(quotient[i]);
			copiesInd.push_back(1);
			indVal.push_back(remainder[i]);
		}
		if(i > 0)
		if (quotient[i] == quotient[i - 1] && remainder[i] == remainder[i - 1])
		{
			copiesInd.at(jLocal) = copiesInd.at(jLocal) + 1;
		}
		else
		{
			indVal.push_back(remainder[i]);
			copiesInd.push_back(1);
			copiesProc.push_back(quotient[i]);
			jLocal++;
		}
	}
	int lowest = 0;
	for (int i = 0; i < Npid; i++)
	{
		for (int j = 0; j < blockSize; j++)
		{
			if (shortageSurvivorSum[i] < blockSize)
			{
				shortageSurvivorSum[i] += 1;
				shortageProcess.push_back(i);
				particles[i]++;
			}
		}
	}
	int rank = 0;
	int top = 0;
	bool entered = false;
	highest = 0;
	int valid = 0;
	bool finished = false;
	while(finished == false)
	{
		valid = 0;
		for (int f = 0; f < Npid; f++)
		{
			if (particles[f] > particles[lowest])
				lowest = f;
		}
		for (int k = 0; k < copiesInd.size(); k++)
		{
			if ((survivorSum[copiesProc.at(k)] > blockSize))
				if ((copiesInd.at(k) >= copiesInd.at(highest)))
					if((copiesInd.at(k) > 0))
			{
				highest = k;
			}
		}
		entered = false;
				while (survivorSum[copiesProc.at(highest)] > blockSize && copiesInd.at(highest) > 0 && particles[lowest] > 0)
			{
				if (entered == false)
				{
					senderProc.push_back(copiesProc.at(highest));
					senderInd.push_back(indVal.at(highest));
					senderCopies.push_back(0);
					receivingProc.push_back(lowest);
					entered = true;
				}
				senderCopies.at(top) = senderCopies.at(top) + 1;
				copiesInd.at(highest) = (copiesInd.at(highest) - 1);
				particles[lowest]--;
				survivorSum[copiesProc.at(highest)]--;
				survivorSum[lowest]++;
			}
			top++;
			for (int f = 0; f < Npid; f++)
			{
				valid += particles[f];
			}
				if (valid == 0)
				{
					finished = true;
				}
	}
	supSize = senderProc.size();
	regSize = 0;
	for (int i = 0; i < copiesInd.size(); i++)
	{
		if (copiesInd.at(i) > 0)
		{
			regProc.push_back(copiesProc.at(i));
			regInd.push_back(indVal.at(i));
			regCopies.push_back(copiesInd.at(i));
			regSize++;
		}
	}
	surplusProcess.clear();
	surplusIndex.clear();
	shortageProcess.clear();
	shortageIndex.clear();
	copiesProc.clear();
	copiesInd.clear();
	indVal.clear();
	return;
}
void sizes(int * copies, int surpSize, int * surpProcess, int * surpIndex, int * shortProcess, int t, vector <int> senderProc, vector <int> senderInd, vector <int> senderCopies, vector <int> receivingProc, vector <int> regProc, vector <int> regInd, vector <int> regCop, int regSize, int * regProcess, int * regIndex, int * regCopies)
{
	for (int i = 0; i < surpSize; i++)
	{
		surpProcess[i] = senderProc.at(i);
		surpIndex[i] = senderInd.at(i);
		copies[i] = senderCopies.at(i);
		shortProcess[i] = receivingProc.at(i);

	}
	for (int i = 0; i < regSize; i++)
	{
		regProcess[i] = regProc.at(i);
		regIndex[i] = regInd.at(i);
		regCopies[i] = regCop.at(i);
		//printf("reg process is %i and reg index is %i and number of copies is %i\n", regProcess[i], regIndex[i], regCopies[i]);
	}
}
int main(int argc, char * argv[])
{
	MPI_Init(&argc, &argv);
	vector<double> xNew(1);
	int Npid;
	int pid;
	MPI_Comm_size(MPI_COMM_WORLD, &Npid);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	srand(time(NULL) + pid);
	double P0 = 5.0;
	double Q = 10.0;
	double R = 1.0;
	double qSum = 0;
	int timeStep = 100;
	int tag = 01;
	int N = 8;
	int number;
	int blockSize = N / Npid;
	MPI_Status * status;
	double * x = new double[N];
	vector <int> senderProc;
	vector <int> senderInd;
	vector <int> senderCopies;
	vector <int> receivingProc;
	vector <int> regProc;
	vector <int> regInd;
	vector <int> regCop;
	double * localE = new double[blockSize];
	double * localQ = new double[blockSize];
	int num;
	int * shortProcess;
	int * surplusProcess;
	int * surplusIndex;
	int * regProcess;
	int * regIndex;
	int * regCopies;
	int * copies;
	int surpSize = 0;
	int regSize = 0;
	double * y = new double[N];
	double * h = new double[N];
	double * yy = new double[blockSize];
	double * hh = new double[blockSize];
	double * e = new double[blockSize];
	double * q = new double[blockSize];
	double * qGlobal = new double[N];
	double * localX = new double[blockSize];
	double localMeanArray;
	double rootQSum = 0.0;
	double * meanArray = new double[timeStep];
	int * zeros = new int[N];
	int * quotient = new int[N];
	int * remainder = new int[N];
	double * globalSum = new double[Npid];
	double * tempMean = new double[Npid];
	//meanArray[0] = 0;
	localMeanArray = 0;
		for (int i = 0; i < blockSize; i++)
		{
			localX[i] = random(sqrt(P0));
			yy[i] = fy(localX[i], sqrt(R));
			hh[i] = fh(localX[i]);
			//meanArray[0] += x[i];
			localMeanArray += localX[i];
		}
		MPI_Gather(&localMeanArray, 1, MPI_DOUBLE, tempMean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (pid == 0)
		{
			for (int i = 0; i < Npid; i++)
				meanArray[0] += tempMean[i];
			meanArray[0] = meanArray[0] / N;
			printf("Mean Value %i %f\n", 0, meanArray[0]);
		}
		//meanArray[0] = meanArray[0] / N;
	for (int t = 1; t < timeStep; t++)
	{
		for (int i = 0; i < blockSize; i++)
		{
			e[i] = yy[i] - hh[i];
			q[i] = pe(e[i]);
		}
		qSum = 0.0;
		for (int i = 0; i < blockSize; i++)
		{
			qSum += q[i];
		}
		MPI_Gather(&qSum, 1, MPI_DOUBLE, globalSum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (pid == 0)
		{
			rootQSum = 0.0;
			for (int i = 0; i < Npid; i++)
			{
				rootQSum += globalSum[i];
			}
		}
		MPI_Bcast(&rootQSum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		for (int i = 0; i < blockSize; i++)
		{
			q[i] = q[i] / rootQSum;
		}
		MPI_Gather(q, blockSize, MPI_DOUBLE, qGlobal, blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		//return 0;
		if (pid == 0)
		{
			resampling(N, qGlobal, zeros);
		if (pid == 0)
			MTRP(t, zeros, blockSize, Npid, quotient, remainder, N, surpSize, regSize, senderProc, senderInd, senderCopies, receivingProc, regProc, regInd, regCop);
			//return 0;
			surplusProcess = new int[surpSize];
			surplusIndex = new int[surpSize];
			shortProcess = new int[surpSize];
			copies = new int[surpSize];
		//return 0;
			regProcess = new int[regSize];
			regIndex = new int[regSize];
			regCopies = new int[regSize];
			sizes(copies, surpSize, surplusProcess, surplusIndex, shortProcess, t, senderProc, senderInd, senderCopies, receivingProc, regProc, regInd, regCop, regSize, regProcess, regIndex, regCopies);
		}
		MPI_Bcast(&surpSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&regSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (pid != 0)
		{
			surplusProcess = new int[surpSize];
			surplusIndex = new int[surpSize];
			shortProcess = new int[surpSize];
			copies = new int[surpSize];
			regProcess = new int[regSize];
			regIndex = new int[regSize];
			regCopies = new int[regSize];
		}
		MPI_Bcast(surplusProcess, surpSize, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(surplusIndex, surpSize, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(shortProcess, surpSize, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(copies, surpSize, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(regProcess, regSize, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(regIndex, regSize, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(regCopies, regSize, MPI_INT, 0, MPI_COMM_WORLD);
		double * newLocalX = new double[blockSize];
		for (int i = 0; i < blockSize; i++)
		{
			newLocalX[i] = localX[i];
		}
		int top = 0;
		for (int i = 0; i < regSize; i++)
		{
			if (pid == regProcess[i])
			{
				for (int j = 0; j < regCopies[i]; j++)
				{
					newLocalX[top] = localX[regIndex[i]];
					top++;
				}
			}
		}
		for (int i = 0; i < surpSize; i++)
		{
			if (pid == surplusProcess[i])
			{
				MPI_Send(&localX[surplusIndex[i]], 1, MPI_DOUBLE, shortProcess[i], 1, MPI_COMM_WORLD);
			}
			if (pid == shortProcess[i])
			{
				MPI_Recv(&newLocalX[top], 1, MPI_DOUBLE, surplusProcess[i], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				top++;
				for (int k = 1; k < copies[i]; k++)
				{
					newLocalX[top] = newLocalX[top - 1];
					top++;
				}
			}
		}
		localMeanArray = 0;
		for (int i = 0; i < blockSize; i++)
		{
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
		for (int i = 0; i < blockSize; i++)
		{
			yy[i] = 0;
			hh[i] = 0;
		}
		for (int i = 0; i < blockSize; i++)
		{
			localX[i] = ff(newLocalX[i], t, Q);
			yy[i] = fy(localX[i], sqrt(R));
			hh[i] = fh(localX[i]);
		}
		delete[] surplusProcess;
		delete[] surplusIndex;
		delete[] shortProcess;
		delete[] regProcess;
		delete[] regIndex;
		delete[] copies;
		delete[] regCopies;
		senderProc.clear();
		senderInd.clear();
		senderCopies.clear();
		receivingProc.clear();
		regProc.clear();
		regInd.clear();
		regCop.clear();
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
		}
	}
	MPI_Finalize();
	return 0;
}