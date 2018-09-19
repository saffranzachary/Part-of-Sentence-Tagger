import sys
import math
import csv
import matplotlib.pyplot as pp

#assume data has just been read from getArrFromTSV
#for dense, the last index is the label and the rest is just one-hot
#for sparse model1, index 0 is label, index 1 is curr
#for sparse model2, index 0 is label, index 1 is prev, 2 is curr, 3 is next
def getDenseAndSparseFromArr(data, modeltype, columns=[]):
	denseArr = []
	sparseArr = []
	colnames = []
	labels = []
	data_length = len(data)
	for i in range(data_length):
		if len(data[i]) == 0:
			if modeltype == 2:
				colnames.append("prev:" + str(data[i-1][0]))
				colnames.append("next:" + str(data[i+1][0]))
			continue
		labels.append(data[i][1])
		if modeltype == 1:
			colnames.append("curr:" + str(data[i][0]))
		if modeltype == 2:
			colnames.append("curr:" + str(data[i][0]))
			if i - 1 == 0 or len(data[i - 1]) == 0:
				colnames.append('prev:BOS')
			else:
				colnames.append('prev:' + str(data[i-1][0]))
			if i + 1 == data_length or len(data[i + 1]) == 0:
				colnames.append('next:EOS')
			else:
				colnames.append('next:' + str(data[i+1][0]))
	labels = list(set(labels))
	colnames = list(set(colnames))
	if len(columns) != 0: colnames = columns
	#colnames.append('y_label')
	colnames_dict = {}
	colnames_length = len(colnames)
	for i in range(colnames_length):
		colnames_dict[colnames[i]] = i
	for i in range(data_length):
		if len(data[i]) == 0: continue
		sparse = []
		dense = [0] * colnames_length
		if modeltype == 1:
			colname = "curr:" + str(data[i][0])
			index = colnames_dict[colname]
			dense[index] = 1
			dense[-1] = data[i][1]
			sparse.append(data[i][1])
			sparse.append(index)
		elif modeltype == 2:
			sparse.append(data[i][1])
			dense[-1] = data[i][1]
			if i - 1 == 0 or len(data[i - 1]) == 0:
				colname = "prev:BOS"
				index = colnames_dict[colname]
				dense[index] = 1
				sparse.append(index)
			else:
				colname = "prev:" + data[i-1][0]
				index = colnames_dict[colname]
				dense[index] = 1
				sparse.append(index)
			colname = "curr:" + str(data[i][0])
			index = colnames_dict[colname]
			dense[index] = 1
			sparse.append(index)
			if i + 1 == data_length or len(data[i + 1]) == 0:
				colname = "next:EOS"
				index = colnames_dict[colname]
				dense[index] = 1
				sparse.append(index)
			else:
				colname = "next:" + str(data[i + 1][0])
				index = colnames_dict[colname]
				dense[index] = 1
				sparse.append(index)
		denseArr.append(dense)
		sparseArr.append(sparse)
	return [denseArr, sparseArr, labels, colnames]

#creates 2d array from data with 0th index data, 1th index label
def getArrFromTSV(file):
	data = list(csv.reader(open(file), delimiter='\t'))
	return data

#x is a single datapoint
#index is the row in theta
def thetaTx(index, theta, x):
	returnval = 0
	for i in x[1:]:
		returnval += theta[index][i]
	returnval += theta[index][-2]
	return returnval

#theta is the weight vector
#theta[-1] is the label
#theta[-2] is the bias
#index is the index at hand
#data is the sparse array
def stepSGD(index, theta, data, learningrate):
	dotproduct = []
	for i in range(len(theta)):
		dotproduct.append(math.exp(float(thetaTx(i, theta, data[index]))))
	denominator = sum(dotproduct)
	for i in range(len(theta)):
		indicator = int(data[index][0] == theta[i][-1])
		fraction = dotproduct[i]/denominator
		gradient = -(indicator - fraction)
		for j in range(1, len(data[index])):
			theta[i][data[index][j]] -= learningrate * gradient
		theta[i][-2] -= learningrate * gradient
	return theta

#data is fed as sparse array
def getError(theta, data):
	errorcount = 0
	totalcount = len(data)
	for i in data:
		if getLabel(theta, i) != i[0]:
			errorcount += 1
	return float(errorcount)/totalcount

#data is a single data point
def getLabel(theta, data):
	labels = []
	probabilities = []
	for i in range(len(theta)):
		prob = thetaTx(i, theta, data)
		probabilities.append(prob)
		labels.append(theta[i][-1])
	maxval = 0
	label = ''
	for j in range(len(probabilities)):
		if label == '':
			maval = probabilities[j]
			label = labels[j]
		if probabilities[j] > maxval:
			maxval = probabilities[j]
			label = labels[j]
		elif probabilities[j] == maxval:
			label = min(label, labels[j])
	return label

#returns the negative log likelihood of a sparse dataset
def getLikelihood(theta, data):
	output = 0
	for i in data:
		numerators = []
		indicator = []
		for j in range(len(theta)):
			indicator.append(int(i[0] == theta[j][-1]))
			numerator = thetaTx(j, theta, i)
			numerator = math.exp(numerator)
			numerators.append(numerator)
		numeratorsum = sum(numerators)
		for j in range(len(theta)):
			output += indicator[j]*math.log(float(numerators[j])/numeratorsum)
	return -(1/len(data))*output

#dense and sparse contain the dense and sparse representations of the
#training data
#labels is the list of unique labels 
#metric is the metric filepath

def getTheta(dense, sparse, labels, num_epoch, val_sparse):
	theta = []
	numfeatures = len(dense[0])
	outputstr = ''
	for i in labels:
		row = [0.0] * (numfeatures + 1)
		row.append(i)
		theta.append(row)
	for i in range(num_epoch):
		for index in range(len(sparse)):
			theta = stepSGD(index, theta, sparse, 0.5)
		tr = str(getLikelihood(theta, sparse))
		val = str(getLikelihood(theta, val_sparse))
		outputstr += 'epoch=' + str(i+1) + ' likelihood(train): ' + tr
		outputstr += '\n'
		outputstr += 'epoch=' + str(i+1) + ' likelihood(validation): ' + val
		outputstr += '\n'
	return theta, outputstr


def main():
	inputs = sys.argv[-8:]
	train_input = getArrFromTSV(inputs[0])
	val_input = getArrFromTSV(inputs[1])
	test_input = getArrFromTSV(inputs[2])
	train_out = inputs[3]
	test_out = inputs[4]
	metrics_out = inputs[5]
	num_epoch = int(inputs[6])
	modeltype = int(inputs[7])
	[tr_dense, tr_sparse, labels, cols] = getDenseAndSparseFromArr(train_input,
															 modeltype)
	[val_dense, val_sparse, na, na2] = getDenseAndSparseFromArr(val_input,
															 modeltype,
															 cols)
	[te_dense, te_sparse, na, na2] = getDenseAndSparseFromArr(test_input, 
															modeltype,
															cols)
	[theta, metrics] = getTheta(tr_dense, tr_sparse, labels,
					 num_epoch, val_sparse)
	trerror = getError(theta, tr_sparse)
	teerror = getError(theta, te_sparse)
	metrics += 'error(train): ' + str(trerror) + '\n'
	metrics += 'error(test): ' + str(teerror)
	print(metrics)
	tr_str = ''
	te_str = ''
	count = 0
	with open(train_out, 'a') as writefile:
		for i in range(len(train_input)):
			if len(train_input[i]) == 0:
				writefile.write('\n')
			else:
				writefile.write(str(getLabel(theta, tr_sparse[count])))
				if i != len(train_input): writefile.write('\n')
				count+=1
	count = 0
	with open(test_out, 'a') as writefile:
		for i in range(len(test_input)):
			if len(test_input[i]) == 0:
				writefile.write('\n')
			else:
				writefile.write(str(getLabel(theta, te_sparse[count])))
				if i != len(test_input): writefile.write('\n')
				count+=1
	open(metrics_out, 'w').write(metrics)

main()