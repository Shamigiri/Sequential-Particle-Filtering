#include <iostream>
#include<math.h>
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
	double ma =  pow((2 * M_PI) , (-1 / 2))*exp(-x*x/2);
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
	double * p = new double [N];
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
		randArray [i] = ((double)rand() / (RAND_MAX));
	//	cout << randArray[i] << "  ";
	}
	for (int i = N; i >=1; i--)
	{
		randArray[i - 1] = pow(randArray[i - 1], (1.0/i));
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
			k = k + 1;
		zeros[j] = k;
	//	cout << zeros[j] << ' ';
	}
}
void main()
{
	double P0 = 5.0;
	double Q = 10.0;
	double R = 1.0;
	double qSum = 0;
	int N = 1000;
	double * x = new double[N];
	double * e = new double[N];
	double * q = new double[N];
	double * y = new double[N];
	double * h = new double[N];
	double * newX = new double[N];
	double * meanArray = new double[100];
		int * zeros = new int[N];

	for (int i = 0; i < N; i++)
	{
		//	x[i] = rand() % (sqrt(P0)); //make array
		x[i] = random(sqrt(P0));
		y[i] = fy(x[i], sqrt(R)); // + e(t) <- What exactly is this e(t)? It was in the formula for calculating y(t)
		h[i] = fh(x[i]);
		meanArray[0] += x[i];
	}
	meanArray[0] = meanArray[0] / N;
	cout << "X Mean number " << 0 << ' ' << meanArray[0] << endl;
	//cout << "Initialized Variables" << endl;
	for (int t = 1; t < 100; t++)
	{
		//	cout << "In the " << t << " iteration" << endl;
		for (int i = 0; i < N; i++)
		{
			e[i] = y[i] - h[i];
			q[i] = pe(e[i]);
		}
	//	cout << "before call resampling q" << q[0];
		qSum = 0.0;
			for (int i = 0; i < N; i++)
				qSum += q[i];
			for (int i = 0; i < N; i++)
			{
				q[i] = q[i] / qSum;
		     //	cout << q[i] << "        ";
			}
			for (int i = 0; i < N; i++)
				zeros[i] = 0;
		resampling(N, q, zeros);

		for (int i = 0; i < N; i++)
		{
			//cout << zeros[i] << "   ";
		}
		for (int i = 0; i < N; i++)
		{
				newX[i] = x[zeros[i]];
				//cout << newX[i] << ' ';
		}
		meanArray[t] = 0;
		for(int i = 0; i < N; i++)
		{
			meanArray[t] += newX[i];
			
		}
		meanArray[t] = meanArray[t] / N;
		cout << "X Mean number " << t << ' ' << meanArray[t] << endl;
		for (int i = 0; i < N; i++)
		{
			x[i] = ff(newX[i], t, Q);
			y[i] = fy(x[i], sqrt(R)); // + e(t) <- What exactly is this e(t)? It was in the formula for calculating y(t)
			h[i] = fh(x[i]);
		}

	}
	int t = 0;
	double * trueX = new double[100];
	trueX[0] = random(sqrt(P0));
	std::cout << "True X number " << 0 << ' ' << trueX[0] << endl;
	for (t = 1; t < 100; t++)
	{
		trueX[t] = (trueX[t - 1] / 2 + 25 * trueX[t - 1] / (1 + pow(trueX[t - 1], 2)) + 8 * cos(1.2*(t)))+random(sqrt(Q));
		std::cout << "True X number " << t << ' ' << trueX[t] <<endl;
	}
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