# coding:utf-8

import numpy as np
# There we use the liblinear with instances weights, so the imported packege is liblinear-weights
from liblinearutil import svm_read_problem, train, predict

train_file = '/home/lan/Packages/libsvm-3.20/cn_07to20.txt'
test_file = '/home/lan/Packages/libsvm-3.20/cn_07to20.txt'

# Prepare data
print "read training  data.."
ytrain, Xtrain = svm_read_problem(train_file)
print "read testing  data.."
# ytest, Xtest = svm_read_problem(test_file)
ytest = ytrain
Xtest = Xtrain

num_train = len(Xtrain)
num_test = len(Xtest)
P = len(Xtrain[0]) # P features in each instance
W = np.ones(num_train)/num_train # Initial Wi
M = 20 # The num of week classifiers we want to learn
alpha = np.zeros(shape = (M,))

weak_classifiers = []

print "Begin to train...Please wait for a while."
for m in xrange(M):
	print "Iteration %d" %m
	# Train week classifier, use different instances weights in every iteration.
	Sm = train(W.tolist(), ytrain, Xtrain, '-c 5 -w1 10 -w2 20 -s 2')
	ytr_pred, p_acc, p_vals = predict(ytrain, Xtrain, Sm)

	# Compute error_rate, weight of this week classifier
	binaries = np.array(ytrain) != np.array(ytr_pred) # these cases classified wrong!
	e = np.sum(W[binaries])
	# if e > 0.5:
	# 	break
	alpha_m = (1-e)/e

	# Update W
	z = W*np.exp(alpha_m*binaries)
	W = z/np.sum(z)

	# Store results
	alpha[m] = alpha_m
	print "current alpha: %f, current e: %f" %(alpha_m, e)
	weak_classifiers.append(Sm)

	print "Training Finished."

	# Test
	te_pred_result = np.zeros(shape = (num_train, m))
	for i in range(m):
		p_labels, p_acc, p_vals = predict(ytest, Xtest, weak_classifiers[i])
		te_pred_result[:,i] = p_labels
		# G_final += alpha[m] * p_labels

	te_pred_labels = []
	# Decide test lables with multilabels
	for i in xrange(num_test):
		scores = []
		for c in range(3): # 0,1,2 is the class label
			score_c = np.sum(alpha[np.where(te_pred_result[i,:] == c)])
			scores.append(score_c)
		te_pred_labels.append(np.argmax(scores))

	print "T = %d, test accuracy : %f" %(m, np.mean(np.array(ytest) == np.array(te_pred_labels))) 










