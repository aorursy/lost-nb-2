#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:



#include <cv.h>       // opencv general include file

#include <ml.h>		  // opencv machine learning include file

#include <stdio.h>
#include <conio.h>

using namespace cv; // OpenCV API is in the C++ "cv" namespace
					******************************************************************************(/)
					(/, global, definitions, (for, speed, and, ease, of, use))
					(/手写体数字识别)
#define NUMBER_OF_TRAINING_SAMPLES 15120//训练集样本数

#define ATTRIBUTES_PER_SAMPLE 54//特征数

#define NUMBER_OF_TESTING_SAMPLES 565892//测试集样本数

#define NUMBER_OF_CLASSES 7//分类

					(/, N.B., classes, are, integer, handwritten, digits, in, range, 0-9)

					******************************************************************************(/)
					(/, loads, the, sample, database, from, file, (which, is, a, CSV, text, file))

int read_data_from_csv(const char* filename, Mat data, Mat classes,
	int n_samples)//读取csv文件数据
{
	float tmp;
	// if we can't read the input file then return 0
	FILE* f = fopen(filename, "r");
	if (!f)
	{
		printf("ERROR: cannot read file %s\n", filename);
		return 0; // all not OK
	}
	(/, for, each, sample, in, the, file)
	for (int line = 0; line < n_samples; line++)
	{
		// for each attribute on the line in the file
		for (int attribute = 0; attribute< (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
		{
				if (attribute < ATTRIBUTES_PER_SAMPLE)
				{
					// first 64 elements (0-63) in each line are the attributes
					fscanf(f, "%f,", &tmp);//fscanf(文件指针，格式字符串，输入列表);
					data.at<float>(line, attribute) = tmp;//特征项
														  (/printf("%f,",, data.at<float>(line,, attribute-1));)
				}
				else if (attribute == ATTRIBUTES_PER_SAMPLE)
				{
					// attribute 65 is the class label {0 ... 9}
					fscanf(f, "%f,", &tmp);
					classes.at<float>(line, 0) = tmp;//结果项
													 //printf("%f\n", classes.at<float>(line, 0));
				}
			}
			
		}
		(/printf("Testing, train, Sample, at, %i, \n",, classes.at<float>(line,, 0));)
	fclose(f);
	return 1; // all OK
}

******************************************************************************(/)
int main(int argc, char** argv)
{
	for (int i = 0; i< argc; i++)
		std::cout << argv[i] << std::endl;
	// lets just check the version first
	printf("OpenCV version %s (%d.%d.%d)\n",
		CV_VERSION,
		CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
	FILE* fp = fopen("sample.csv", "w");
	if (fp == NULL)
	{
		printf("无法打开文件\n!");
		exit(0);
	}

	//定义训练数据与标签矩阵
	Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
	Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);
	//定义测试数据矩阵与标签
	Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
	Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);
	// define all the attributes as numerical
	// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
	// that can be assigned on a per attribute basis
	Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U);
	var_type.setTo(Scalar(CV_VAR_NUMERICAL)); // all inputs are numerical

											  // this is a classification problem (i.e. predict a discrete number of class

											  // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

	var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;
	double result; // value returned from a prediction
				   //加载训练数据集和测试数据集
	if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
		read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
	{
		/*for (int i = 0; i < NUMBER_OF_TRAINING_SAMPLES; i++) {
			printf("Testing train Sample %f at %i \n", training_classifications.at<float>(i, 0),i);
		}*/
		/*for (int i = 0; i < NUMBER_OF_TESTING_SAMPLES; i++) {
			printf("Testing test Sample %f \n", testing_classifications.at<float>(i,0));
		}*/
		/********************************步骤1：定义初始化Random Trees的参数******************************/
		float priors[] = { 1,1,1,1,1,1,1 };  // weights of each classification for classes
		CvRTParams params = CvRTParams(25, // max depth 较低不符合，较高会过拟合
			5, // min sample count 叶节点上需要分割的最小样本数
			0, // regression accuracy: N/A here 回归树的终止条件当前节点上所有样本的真实值和预测值之间的差小于这个数值时，停止生产这个节点，并将其作为叶子节点
			false, // compute surrogate split, no missing data 是否使用代理
			15, // max number of categories (use sub-optimal algorithm for larger numbers)k均值的最大分类数
			priors, // the array of priors先验知识 分配权重数量较少样本类的分类正确率也不会太低
			false,  // calculate variable importance 计算变量重要性
			4,       // number of variables randomly selected at node and used to find the best split(s).每个树节点上随机选择的要素子集的大小，用于查找最佳分割
			100,	 // max number of trees in the forest随机森林中树的最大颗数
			0.01f,				// forrest accuracy 
			CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria CV_TERMCRIT_ITER通过max_num_of_trees_in_the_forest终止学习;CV_TERMCRIT_EPS通过forest_accuracy终止学习;CV_TERMCRIT_ITER | CV_TERMCRIT_EPS使用两个终止条件。
		);

		****************************步骤2(：训练, Random, Decision, Forest(RDF)分类器*********************/)
		printf("\nUsing training database: %s\n\n", argv[1]);
		CvRTrees* rtree = new CvRTrees;
		rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
			Mat(), Mat(), var_type, Mat(), params);//训练
		(/, perform, classifier, testing, and, report, results)
		Mat test_sample;
		int correct_class = 0;
		int wrong_class = 0;
		int false_positives[NUMBER_OF_CLASSES] = { 0,0,0,0,0,0,0,};
		printf("\nUsing testing database: %s\n\n", argv[2]);
		for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
		{
			// extract a row from the testing matrix
			test_sample = testing_data.row(tsample);
			/********************************步骤3：预测*********************************************/
			result = rtree->predict(test_sample, Mat());//预测
			fprintf(fp, "%f \n", result);
			(/, if, the, prediction, and, the, (true), testing, classification, are, the, same)
			(/, (N.B., openCV, uses, a, floating, point, decision, tree, implementation!))
			if (fabs(result - testing_classifications.at<float>(tsample, 0))
				>= FLT_EPSILON)//预测结果与实际结果相差大于某个值
			{
				//printf("Testing wrong Sample %f ->result %f\n", testing_classifications.at<float>(tsample, 0), result);
				// if they differ more than floating point error => wrong class
				wrong_class++;
				false_positives[(int)result]++;//在错误分类的样本数加1
			}
			else
			{
				//printf("Testing correct Sample %f \n->result %f\n", testing_classifications.at<float>(tsample, 0), result);
				// otherwise correct
				correct_class++;
			}
		}
		printf("\nResults on the testing database: %s\n"
			"\tCorrect classification: %d (%g%%)\n"
			"\tWrong classifications: %d (%g%%)\n",
			argv[2],
			correct_class, (double)correct_class * 100 / NUMBER_OF_TESTING_SAMPLES,
			wrong_class, (double)wrong_class * 100 / NUMBER_OF_TESTING_SAMPLES);
		for (int i = 0; i < NUMBER_OF_CLASSES; i++)
		{
			printf("\tClass (digit %d) false postives 	%d (%g%%)\n", i,
				false_positives[i],
				(double)false_positives[i] * 100 / NUMBER_OF_TESTING_SAMPLES);
		}
		(/, all, matrix, memory, free, by, destructors)
		(/, all, OK, :, main, returns, 0)
		_getch();
		return 0;
	}
	(/, not, OK, :, main, returns, -1)
	return -1;
}

******************************************************************************(/)

