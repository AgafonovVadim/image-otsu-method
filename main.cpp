#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
using namespace std;
int main(int argc, char* argv[])
{
	if (argc != 4) {
		cerr << "Incorrect number of arguments.";
		return -1;
	}
	int num_threads = stoi(argv[1]);
	bool withOpenMp = true;

	if (num_threads == -1)
	{
		num_threads = 0;
		withOpenMp = false;
	}
	else if (num_threads == 0)
	{
		num_threads = omp_get_max_threads();
	}
	else if (num_threads < -1) {
		cerr << "incorrect number of threads";
		return -1;
	}
	string inputFIleName = argv[2];
	ifstream img;
	img.open(inputFIleName, ios::binary);
	string type;
	img >> type;
	if (type != "P5")
	{
		cout << "incorrect format of file";
		exit(-1);
	}
	int width, height, colorRange;
	img >> width >> height >> colorRange;
	int all = width * height;
	unsigned char* image = new unsigned char[all] {0};
	img.get();
	colorRange++;
	char* quick = new char[width];
	for (int i = 0; i < height; i++)
	{
		img.read(quick, width);
		for (int j = 0; j < width; j++)
		{
			image[i * height + j] = (unsigned char) quick[j];
		}
	}
	img.close();
	double* prepropability = new double[colorRange] { 0 };
	double* curpropability = new double[colorRange] { 0 };
	double* pre_u = new double[colorRange] { 0 };
	int* colors = new int[colorRange] {0};
	int f0;
	int f1;
	int f2;
	int pix;
	double ans = 0;
	double start = omp_get_wtime();

	//Histogram
#pragma omp parallel num_threads(num_threads) if(withOpenMp)
	{
		int* personalThreadColors = new int[colorRange] {0};
#pragma omp for nowait
		for (int i = 0; i < all; i++) {
			personalThreadColors[image[i]]++;
		}
#pragma omp critical
		{
			for (int i = 0; i < colorRange; i++)
				colors[i] += personalThreadColors[i];
		}
	}
	for (int i = 0; i < colorRange; i++)
	{
		curpropability[i] = (double)colors[i] / all;
	}
	prepropability[0] = curpropability[0];
	pre_u[0] = 0;
	for (int i = 1; i < colorRange; i++)
	{
		prepropability[i] += curpropability[i] + prepropability[i - 1];
		pre_u[i] = curpropability[i] * i + pre_u[i - 1];
	}




#pragma omp parallel num_threads(num_threads) if(withOpenMp)
	{
		int tmp0;
		int tmp1;
		int tmp2;
		double best_sigma_in_thread = 0;
#pragma omp for schedule(static, 16) nowait
		for (int threshold0 = 1; threshold0 < colorRange - 2; threshold0++)
		{
			for (int threshold1 = threshold0 + 1; threshold1 < colorRange - 1; threshold1++)
			{
				for (int threshold2 = threshold1 + 1; threshold2 < colorRange; threshold2++)
				{
					double u1 = pre_u[threshold0];
					double u2 = pre_u[threshold1] - pre_u[threshold0];
					double u3 = pre_u[threshold2] - pre_u[threshold1];
					double u4 = pre_u[colorRange - 1] - pre_u[threshold2];
					double q1 = prepropability[threshold0];
					double q2 = prepropability[threshold1] - prepropability[threshold0];
					double q3 = prepropability[threshold2] - prepropability[threshold1];
					double q4 = prepropability[colorRange - 1] - prepropability[threshold2];
					double cur = u1 * u1 / q1 + u2 * u2 / q2 + u3 * u3 / q3 + u4 * u4 / q4;
					if (cur > best_sigma_in_thread) {
						best_sigma_in_thread = cur;
						tmp0 = threshold0;
						tmp1 = threshold1;
						tmp2 = threshold2;
					}
				}
			}
		}
		if (ans < best_sigma_in_thread)
		{
#pragma omp critical
			{
				if (ans < best_sigma_in_thread)
				{
					ans = best_sigma_in_thread;
					f0 = tmp0;
					f1 = tmp1;
					f2 = tmp2;
				}
			}
		}
#pragma omp for
		for (int i = 0; i < all; i++) {
			pix = image[i];
			if (pix <= f0) image[i] = 0;
			else if (pix <= f1) image[i] = 84;
			else if (pix <= f2) image[i] = 170;
			else image[i] = 255;
		}
	}
	double finish = omp_get_wtime();
	printf("Time (%i thread(s)): %g ms\n", num_threads, (finish - start) * 1000);
	printf("%u %u %u\n", f0, f1, f2);
	ofstream out(argv[3]);
	out << type << "\n";
	out << width << " " << height << "\n" << colorRange - 1 << "\n";
	for (int i = 0; i < all; i++) {
		out << image[i];
	}
	out.close();
}