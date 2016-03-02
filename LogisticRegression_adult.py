import random
import numpy as np
from numpy import log as ln 
from numpy import exp as exp
from copy import deepcopy,copy
from sys import argv,exit

def isFloat(strng): # returns True if the string can be converted to float
    try:
        float(strng)
    except ValueError:
        return False
    else:
        return True

        

        # Load Data:
        
myfile = open("adult.data")
AllData = myfile.read()
lines = AllData.splitlines()
MyData = [] # A list holding all of the Data features

        # Pre Process the data:
print "Preparing data for Logistic Regression Training ...",
for q in lines:
    v = q.rsplit(', ')
    p = [1 if v[-1] == '>50K' else -1 ] + v[:-1]
    MyData.append(p)
    # convert numbers to floats and also normalize numerical attributes to range [0,1]
    MyData[-1][:] = map(lambda atr: float(atr) if isFloat(atr) else atr.strip() , MyData[-1])  # convert numerical attributes to floats
atIsNumerical = map( lambda h: 1 if isinstance( h , (int , float))  else 0 , MyData[0])  # scans the data to distinguish categorical and numerical attributes
OutputLabels = [ -1 , 1]
# determine how many attributes are necessary for categorical ones ( and normalize Numerical data at the same time):
category_list = []
data_for_LR = deepcopy(MyData)  # data for Logistic Regression) will manipulate original data to make it suitable for Logistic Regression
numAt_max = np.zeros_like(atIsNumerical) # will store the max for numerical attributes 
for i in range(len(MyData[0])):
 if atIsNumerical[i]==0:
    category_list.append([])
    for p in MyData:
        if p[i] not in category_list[-1]:
            category_list[-1].append(p[i])
 else: # attribute is numerical
    numAt_max[i] = max(q[i] for q in MyData)
cat_index = [i for i in range(len(atIsNumerical)) if  atIsNumerical[i]==0]

# for logistic regression implementation: code categorical attributes into binary based on number of possible categories in data:
for i in reversed(range(len(MyData[0]))):
    for entry_index,q in enumerate(MyData):
        if atIsNumerical[i] == 0 :
            # code categorical attributes:
            cat_num = category_list[cat_index.index(i)].index(q[i])
            cat_code = [1. if z==cat_num else 0. for z in range(len(category_list[cat_index.index(i)]))] 
            data_for_LR[entry_index].pop(i)
            # data_for_LR[entry_index].insert(i , np.array(cat_code))
            for bit in cat_code:
                data_for_LR[entry_index].insert(i , bit)
        else: # attribute is numerical (performing normalization here)
            if numAt_max[i]: # to avoid division by zero
                data_for_LR[entry_index][i] /= numAt_max[i]

# randomly, split data into test and training for k-fold cross validation ( k=10 as a rule of thumb):
# random.shuffle(data_for_LR)
all = len(data_for_LR)
k = 10  # number of folds
flds = int(round(all/k))  # 1/kth of data for each fold
folds = []
for i in range(k):
    folds.append(data_for_LR[i*flds:(i+1)*flds])
# create k lists , each leaving out one out of k folds:
TrainList = []
for i in range(k):
    TrainList.append([])
    for j in range(k):
        if i!=j:        # TrainList[i] will have folds[i] left out
            TrainList[i] .extend( folds[j] )
print "Done with preprocessing the data!"
# Now data is ready for Training Logistic Regression

class LogistRegres(object):
 
    def __init__(self , train_data , train_mode='Batch' , use_regularizer = False):
    # randomly initialize weights of our logistic function:
        self.weights = np.random.rand(len(train_data[0]))
        self.train( train_data , train_mode , use_regularizer )

    def train(self , train_data , mode , use_regularizer):  # mode can be Batch or SGA (for stochastic gradient ascent) , setting use_regularizer flag to true will add a L2 regularizer term to both modes of training
        iters = 0
        W_cur = self.weights
        eta = ETA # learning rate
        if mode == 'Batch':   # Batch Training ( Step 2 in assignment)
            X = np.array(train_data) # feed in all the data in each iteration (BATCH Training)
            Y = copy(X[:,0])[None]
            N = len(X)
            # weights[0] will be the bias, therefore inputs[0] will be set to 1 for all entries, ( it was output label in our data which is stored in Ys)
            X[:,0] = 1.
            if len(X.shape) != 2 or len(Y.shape)!=2:
                raise Exception('LogistRegres.train(): incorrect train data formatting')
            LL_cur = -( ln( 1.+exp(-Y*(np.dot(X,W_cur))).T ) ).sum(0)  # LogLikelihood
            print "Starting Batch Training with learning rate of " , eta , 'and Max iteration of ' , MAX_ITERS, '...'     
            while iters < MAX_ITERS: # add some other criteria  dw < threshold
                LL_grad = ( Y.T*X/(1.+exp(Y*(np.dot(X,W_cur))).T) ).sum(0)/N# LogLikelihood gradient
                # if LL_grad.shape != W_cur.shape:
                    # raise Exception("Error in calculation of Gradient (batch training)")
                W_next = W_cur + eta*LL_grad
                # optoinal: check loglikelihood increase
                LL_next = -( ln( 1.+exp(-Y*(np.dot(X,W_next))).T ) ).sum(0)  # LogLikelihood
                LL_dif = LL_next - LL_cur
                print "Loglikelihood at iteration {}: {} , LL difference:{} ,max LL_Grad :".format(iters,LL_cur,LL_dif) , max(abs(LL_grad))
                if LL_dif <= 0:
                    print "decrease in loglikelihood => shrinking step size"
                    W_next = W_cur + eta*LL_grad/5 # shrink step size
                if use_regularizer:
                    W_next -= eta*W_cur/SIGMA**2
                if max( abs(W_next - W_cur)) < TH_MIN_W_UPDATE :
                    print "\n --> Min weight update criteria triggered, training stopped <-- \n"
                    break
                W_cur = W_next
                LL_cur = LL_next
                iters += 1
                if iters == MAX_ITERS:
                    print "\n --> reached maximum number of iterations, training stopped <-- \n"
            print "Batch Training converged!Log Likelihood = {},iterations = {}".format(LL_next,iters)
            self.weights = W_cur
        
        elif mode == 'SGA':   # Stochastic Gradient Ascent Training
            N = SGA_SAMPLE_SIZE
            Xall = np.array(train_data) # All the training data for loglikelihood report
            Yall = copy(Xall[:,0])[None]
            Nall = len(Xall)
            # weights[0] will be the bias, therefore inputs[0] will be set to 1 for all entries, ( it was output label in our data which is stored in Ys)
            Xall[:,0] = 1.
            # LL_cur = -( ln( 1.+exp(-Yall*(np.dot(Xall,W_cur))).T ) ).sum(0)  # LogLikelihood
            # SGA_MAX_ITERS = MAX_ITERS*Nall/SGA_SAMPLE_SIZE
            SGA_MAX_ITERS = MAX_ITERS*5
            print "Starting Stochastic Gradient Ascent training with learning rate of " , eta , ',sample size for each iteration:',SGA_SAMPLE_SIZE,'and Max iterations of ' , SGA_MAX_ITERS , '...'     
            LL_cur = -50000 # just for initialization
            while iters < SGA_MAX_ITERS: # add some other criteria  dw < threshold
                X = random.sample(train_data,SGA_SAMPLE_SIZE) # randomly sample training data for each iteration
                X = np.array(X)
                Y = copy(X[:,0])[None]
                # weights[0] will be the bias, therefore inputs[0] will be set to 1 for all entries, ( it was output label in our data which is stored in Ys)
                X[:,0] = 1.
                # if len(X.shape) != 2 or len(Y.shape)!=2:
                    # raise Exception('LogistRegres.train(): incorrect train data formatting')
                LL_grad = ( Y.T*X/(1.+exp(Y*(np.dot(X,W_cur))).T) ).sum(0)/N# LogLikelihood gradient
                # if LL_grad.shape != W_cur.shape:
                    # raise Exception("Error in calculation of Gradient (batch training)")
                W_next = W_cur + eta*LL_grad
                # optoinal: check loglikelihood increase and shrink step size if it's decreasing:
                LL_next = -( ln( 1.+exp(-Y*(np.dot(X,W_next))).T ) ).sum(0)  # LogLikelihood
                LL_dif = LL_next - LL_cur
                # print "Loglikelihood at iteration {}: {} , LL difference:{} ,max LL_Grad :".format(iters,LL_cur,LL_dif) , max(abs(LL_grad))     
                if LL_dif <= 0:
                    # print "decrease in loglikelihood => shrinking step size"
                    W_next = W_cur + eta*LL_grad/5 # shrink step size
                if use_regularizer:
                    W_next -= eta*W_cur/SIGMA**2
                LL_cur_all = -( ln( 1.+exp(-Yall*(np.dot(Xall,W_cur))).T ) ).sum(0)  # LogLikelihood
                if max( abs(W_next - W_cur)) < TH_MIN_W_UPDATE/Nall and LL_cur_all > -10000 : # weight until a minimum level of likelihood on overall data is reached before terminating training
                    print "\n --> minimum weight update criteria triggered, training stopped <-- \n"
                    break
                W_cur = W_next
                LL_cur = LL_next
                iters += 1
                if iters == MAX_ITERS:
                    print "\n --> reached maximum number of iterations, training stopped <-- \n"
            print " calculating Overall loglikelihood on training data. biggest weight= ", max(abs(self.weights))
            LL_cur_All = -( ln( 1.+exp(-Yall*(np.dot(Xall,W_cur))).T ) ).sum(0)  # LogLikelihood
            print "Stochastic Training converged! Log Likelihood = {},iterations = {}\n\n".format(LL_cur_All,iters)
            self.weights = W_cur
            
    def classify(self , entry):     # this method can be used (after training) to determine output label for an entry based on loglikelihood ratio
        _entry = copy(entry)
        if _entry[0]!= 1:
            _entry[0] = 1 # set the bias attribute to 1
        if   np.dot(_entry , self.weights) > 0 :       #LogLikelihood ratio
            return 1
        else:
            return -1

            
def evalResult( estimate , ground_truth):    # function to evaluate classification results by extracting true positive , false positive , etc
    if estimate.size != ground_truth.size:
        raise Exception("evalResult: dimensions don't match")
    TP = np.count_nonzero( (estimate==1)*(ground_truth==1) ) # True Positive
    TN = np.count_nonzero( (estimate==-1)*(ground_truth==-1) ) # True Negative
    FP = np.count_nonzero( (estimate==1)*(ground_truth==-1) ) # False Positive
    FN = np.count_nonzero( (estimate==-1)*(ground_truth==1) ) # False Negative
    N = estimate.size
    TPR = float(TP)/(TP+FN) # true positive rate ( or sensitivity)
    TNR = float(TN)/(TN+FP) # true negative rate ( or specificity)
    FPR = float(FP) / (FP+TN) # false positive rate ( or fall-out)
    CCR = float(TP+TN)/N # correct classification rate ( or accuracy)
    return CCR , TPR ,TNR, FPR

    
MAX_ITERS = 300 # stop criteria 1
TH_MIN_W_UPDATE = .003  # stop criteria 2
SGA_SAMPLE_SIZE = 4  # number of entries sampled from training data in each iteration of stochastic gradient ascent algorithm
SIGMA = 70 # sigma for L2 regularizer ( only used in step 4 of assignment)
if len(argv) >  1:
    try:
        ETA = float(argv[1])
    except ValueError,e:
        print e,", Correct syntax is: python script.py learning_rate(should be a float)"
        exit(0)
else:
    ETA = 5 # learning rate

# weights = np.random.rand(len(data_for_LR[0]) - 1)
LR = LogistRegres(TrainList[0] , train_mode='SGA' , use_regularizer = True)
q = np.array(TrainList[0]) # train data
w = np.array(folds[0]) # test data
# ground truth labels:
gt_q = q[:,0] 
gt_w = w[:,0]
# evaluate results (on train and test data):

est_q = np.array( map(LR.classify , q) )
est_w = np.array( map(LR.classify , w) )    
    
print '\n','*'*30 , "Evaluation Results:" , '*'*30 , '\n'
a,b,c,d = evalResult(est_q , gt_q)
print "Train data evaluation result: Accuracy={} , TruePositiveRate={} , TrueNegativeRate={} , FalsePostiveRate={}  ".format(a,b,c,d)
a,b,c,d = evalResult(est_w , gt_w)
print "Test data evaluation result: Accuracy={} , TruePositiveRate={} , TrueNegativeRate={} , FalsePostiveRate={}  ".format(a,b,c,d)
