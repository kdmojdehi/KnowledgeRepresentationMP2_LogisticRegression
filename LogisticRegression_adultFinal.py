import random
import numpy as np
from numpy import log as ln 
from numpy import exp as exp
from copy import deepcopy,copy
import matplotlib.pyplot as plt
from sys import argv,exit

def isFloat(strng): # returns True if the string can be converted to float
    try:
        float(strng)
    except ValueError:
        return False
    else:
        return True

class LogistRegres(object):
 
    def __init__(self , train_data ,test_data=[], train_mode='Batch' , use_regularizer = False):
    # randomly initialize weights of our logistic function:
        self.weights = np.random.rand(len(train_data[0]))
        self.iter_tr_error = [] # ccr in each iteration for train data
        self.iter_tst_error = [] # ccr in each iteration for test data
        self.iter_LL = [] # Loglikelihood in each iteration
        self.train( train_data , train_mode , use_regularizer , test_data)

    def train(self , train_data , mode , use_regularizer , test_data = []):  # mode can be Batch or SGA (for stochastic gradient ascent) , setting use_regularizer flag to true will add a L2 regularizer term to both modes of training
        iters = 0
        W_cur = self.weights
        eta = ETA # learning rate
        self.iter_error = []
        if test_data:
            X_test = np.array(test_data) # arrange test data for plot data across iterations
            Y_test = copy(X_test[:,0])[None]
            N_test = len(X_test)
            X_test[:,0] = 1.
        if mode == 'Batch':   # Batch Training ( Step 2 in assignment)
            X = np.array(train_data) # feed in all the data in each iteration (BATCH Training)
            Y = copy(X[:,0])[None]
            N = len(X)
            # weights[0] will be the bias, therefore inputs[0] will be set to 1 for all entries, ( it was output label in our data which is stored in Ys)
            X[:,0] = 1.
            if len(X.shape) != 2 or len(Y.shape)!=2:
                raise Exception('LogistRegres.train(): incorrect train data formatting')
            LL_cur = -( ln( 1.+exp(-Y*(np.dot(X,W_cur))).T ) ).sum(0)/N  # 
            if use_regularizer:
                LL_cur -=  float( np.dot(W_cur,W_cur)) /(2*SIGMA**2) # add the regularization term
            print "Starting Batch Training with learning rate of " , eta , 'and Max iteration of ' , MAX_ITERS, '...'     
            while iters < MAX_ITERS: # add some other criteria  dw < threshold
                LL_grad = ( Y.T*X/(1.+exp(Y*(np.dot(X,W_cur))).T) ).sum(0)/N# LogLikelihood gradient
                # if LL_grad.shape != W_cur.shape:
                    # raise Exception("Error in calculation of Gradient (batch training)")
                W_next = W_cur + eta*LL_grad
                # optoinal: check loglikelihood increase
                LL_next = -( ln( 1.+exp(-Y*(np.dot(X,W_next))).T ) ).sum(0)/N  # LogLikelihood
                LL_dif = LL_next - LL_cur
                # print "Loglikelihood at iteration {}: {} , LL difference:{} ,max LL_Grad :".format(iters,LL_cur,LL_dif) , max(abs(LL_grad))
                if LL_dif <= 0:
                    # print "decrease in loglikelihood => shrinking step size"
                    W_next = W_cur + eta*LL_grad/5 # shrink step size
                if use_regularizer:
                    W_next -= eta*W_cur/SIGMA**2
                    LL_next -=  float( np.dot(W_cur,W_cur)) /(2*SIGMA**2) # add the regularization term
                if max( abs(W_next - W_cur)) < TH_MIN_W_UPDATE :
                    print "\n --> Min weight update criteria triggered, training stopped <-- \n"
                    break
                if PLOT:  # set global Name PLOT to true to save plot results
                    estimate = lambda entry:1 if np.dot(entry , W_cur) > 0 else -1 # labels an entry with current weights
                    Y_cur_est_tr = np.array( map(estimate , X ) ) # current estimate of label for train data
                    Y_cur_est_tst = np.array( map(estimate , X_test ) ) # current estimate of label for test data
                    cur_CCR_tr = float(np.count_nonzero(Y==Y_cur_est_tr))/N # correct classification rate at current iteration for train set
                    cur_CCR_tst = float(np.count_nonzero(Y_test==Y_cur_est_tst))/N_test # correct classification rate at current iteration for test set
                    self.iter_tr_error.append (1. - cur_CCR_tr)
                    self.iter_tst_error.append (1. - cur_CCR_tst)
                    self.iter_LL.append(LL_cur)
                LL_cur = LL_next
                W_cur = W_next
                iters += 1
                if iters == MAX_ITERS:
                    print "\n --> reached maximum number of iterations, training stopped <-- \n"
            print "Batch Training converged!Log Likelihood = {},iterations = {}".format(LL_next,iters)
            if PLOT:
                plt.plot(range(iters) , self.iter_LL) , plt.title("LogLikelihood") , plt.savefig('figures/LLBatch_iter{}_eta{}_numIns{}_sig{}.png'.format(iters,ETA , NUMOF_INSTANCES,SIGMA)) , plt.figure()
                plt.plot(range(iters) , self.iter_tr_error , 'b' , label="Training Error") , plt.title("Training Error") , #plt.savefig('figures/trainError_Batch_iter{}_eta{}.png'.format(iters,ETA)) ,
                plt.plot(range(iters) , self.iter_tst_error , 'r' , label = "Test Error") , plt.title("Classification Error") , plt.legend() , plt.savefig('figures/testError_Batch_iter{}_eta{}_numIns{}_sig{}.png'.format(iters , ETA , NUMOF_INSTANCES,SIGMA))
            self.weights = W_cur
            # report error on test set(which is our validation set) for fine tuning of regluarizer:
            if use_regularizer:
                estimate = lambda entry:1 if np.dot(entry , W_cur) > 0 else -1 # labels an entry with current weights
                Y_cur_est_tst = np.array( map(estimate , X_test ) ) # current estimate of label for test data
                cur_CCR_valid = float(np.count_nonzero(Y_test==Y_cur_est_tst))/N_test # correct classification rate at current iteration for test set
                valid_error = 1. - cur_CCR_valid
                return valid_error
        
        elif mode == 'SGA':   # Stochastic Gradient Ascent Training
            N = SGA_SAMPLE_SIZE
            Xall = np.array(train_data) # All the training data for loglikelihood report
            Yall = copy(Xall[:,0])[None]
            Nall = len(Xall)
            # weights[0] will be the bias, therefore inputs[0] will be set to 1 for all entries, ( it was output label in our data which is stored in Ys)
            Xall[:,0] = 1.
            eta = SGA_ETA
            # LL_cur = -( ln( 1.+exp(-Yall*(np.dot(Xall,W_cur))).T ) ).sum(0)  # LogLikelihood
            # SGA_MAX_ITERS = MAX_ITERS*Nall/SGA_SAMPLE_SIZE
            SGA_MAX_ITERS = MAX_ITERS*5
            print "Starting Stochastic Gradient Ascent training with learning rate of " , eta , ',sample size for each iteration:',SGA_SAMPLE_SIZE,'and Max iterations of ' , SGA_MAX_ITERS , '...'     
            LL_cur_all = -( ln( 1.+exp(-Yall*(np.dot(Xall,W_cur))).T ) ).sum(0)/Nall  # LogLikelihood
            if use_regularizer:
                LL_cur_all -=  float( np.dot(W_cur,W_cur)) /(2*SIGMA**2) # add the regularization term
            LL_cur = LL_cur_all
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
                LL_cur_all = -( ln( 1.+exp(-Yall*(np.dot(Xall,W_cur))).T ) ).sum(0)/Nall  # LogLikelihood
                if use_regularizer:
                    W_next -= eta*W_cur/SIGMA**2
                    LL_cur_all -=  float( np.dot(W_cur,W_cur)) /(2*SIGMA**2) # add the regularization term
                if max( abs(W_next - W_cur)) < TH_MIN_W_UPDATE/Nall and LL_cur_all > -10000/Nall : # weight until a minimum level of likelihood on overall data is reached before terminating training
                    print "\n --> minimum weight update criteria triggered, training stopped <-- \n"
                    break
                if PLOT:  # set global Name PLOT to true to save plot results
                    estimate = lambda entry:1 if np.dot(entry , W_cur) > 0 else -1 # labels an entry with current weights
                    Y_cur_est_tr = np.array( map(estimate , Xall ) ) # current estimate of label for train data
                    Y_cur_est_tst = np.array( map(estimate , X_test ) ) # current estimate of label for test data
                    cur_CCR_tr = float(np.count_nonzero(Yall==Y_cur_est_tr))/Nall # correct classification rate at current iteration for train set
                    cur_CCR_tst = float(np.count_nonzero(Y_test==Y_cur_est_tst))/N_test # correct classification rate at current iteration for test set
                    self.iter_tr_error.append (1. - cur_CCR_tr)
                    self.iter_tst_error.append (1. - cur_CCR_tst)
                    # LL_cur_All = -( ln( 1.+exp(-Yall*(np.dot(Xall,W_cur))).T ) ).sum(0)/Nall  # LogLikelihood
                    self.iter_LL.append(LL_cur_all)
                W_cur = W_next
                LL_cur = LL_next
                iters += 1
                if iters == MAX_ITERS:
                    print "\n --> reached maximum number of iterations, training stopped <-- \n"
            # print " calculating Overall loglikelihood on training data. biggest weight= ", max(abs(self.weights))
            LL_cur_All = -( ln( 1.+exp(-Yall*(np.dot(Xall,W_cur))).T ) ).sum(0)/Nall  # LogLikelihood
            print "Stochastic Training converged! Log Likelihood = {},iterations = {}\n\n".format(LL_cur_All,iters)
            if PLOT:
                plt.plot(range(iters) , self.iter_LL) , plt.title("SGALogLikelihood") ,plt.yticks(np.arange(-1,0.1,.1)),plt.ylim(-1,0), plt.savefig('figures/LLSGA_iters{}_samplesize{}_eta{}_numIns{}_sig{}.png'.format(iters,SGA_SAMPLE_SIZE, SGA_ETA, NUMOF_INSTANCES,SIGMA)) , plt.figure()
                plt.plot(range(iters) , self.iter_tr_error, 'b' , label="Training Error") , plt.title("SGATraining Error") , #plt.savefig('figures/trainError_SGD_samplesize{}_eta{}.png'.format(SGA_SAMPLE_SIZE, SGA_ETA)) , plt.figure()
                plt.plot(range(iters) , self.iter_tst_error, 'r' , label="Test Error") , plt.title("SGA Classification Error") , plt.legend(), plt.savefig('figures/testError_SGA_iters{}_samplesize{}_eta{}_numIns{}_sig{}.png'.format(iters,SGA_SAMPLE_SIZE, SGA_ETA, NUMOF_INSTANCES,SIGMA))
            self.weights = W_cur
            # report error on test set(which is our validation set) for fine tuning of regluarizer:
            if use_regularizer:
                estimate = lambda entry:1 if np.dot(entry , W_cur) > 0 else -1 # labels an entry with current weights
                Y_cur_est_tst = np.array( map(estimate , X_test ) ) # current estimate of label for test data
                cur_CCR_valid = float(np.count_nonzero(Y_test==Y_cur_est_tst))/N_test # correct classification rate at current iteration for test set
                valid_error = 1. - cur_CCR_valid
                return valid_error
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


        

        # Load Data:
NUMOF_INSTANCES = 32000
K = 2 # number of folds
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
# since data set is too large ,randomly sample take a fraction of it for our task:

# MyDataFrac = random.sample(MyData, NUMOF_INSTANCES )
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
def subsample_data(data_for_LR):
    # randomly, split data into test and training for k-fold cross validation ( k=10 as a rule of thumb):
    data_sub_LR = random.sample(data_for_LR, NUMOF_INSTANCES)
    all = len(data_sub_LR)
    k = K   # number of folds
    flds = int(round(all/k))  # 1/kth of data for each fold
    folds = []
    for i in range(k):
        folds.append(data_sub_LR[i*flds:(i+1)*flds])
    # create k lists , each leaving out one out of k folds:
    TrainList = []
    for i in range(k):
        TrainList.append([])
        for j in range(k):
            if i!=j:        # TrainList[i] will have folds[i] left out
                TrainList[i] .extend( folds[j] )
    print "Done with preprocessing the data!"
    return TrainList , folds
TrainList,folds = subsample_data(data_for_LR)
# Now data is ready for Training Logistic Regression

    
MAX_ITERS = 1002 # stop criteria 1
TH_MIN_W_UPDATE = .0003  # stop criteria 2
SGA_SAMPLE_SIZE = 40  # number of entries sampled from training data in each iteration of stochastic gradient ascent algorithm
SIGMA = 70 # sigma for L2 regularizer ( only used in step 4 of assignment)
PLOT = True # set to true to save plot results in figures/ folder( you need to have a figures folder in the same directory as the code)

if len(argv) >  1:
    try:
        ETA = float(argv[1])
        SGA_ETA = float(argv[2])
    except ValueError,e:
        print e,", Correct syntax is: python script.py ETA SGA_ETA(should be a float)"
        exit(0)
else:
    ETA = 5 # learning rate
    SGA_ETA = .3
    

# def Inference(i): # trains a logistic regressor on i'th fold of our dataset and reports performance of inference results on train and test darta
i=0
LR = LogistRegres(TrainList[i] , train_mode='Batch' , use_regularizer = True , test_data = folds[i])

# interface of LR model:
# mode can be 'SGA' for stochastic gradient ascent or 'Batch' for batch learning
# setting use_regilarizer to True will add the regularizer term to log likelihood and update rule in each iteration
# global variables for fine Tuning are: ETA, SGA_ETA, MAX_ITERS, TH_MIN_W_UPDATE, SGA_SAMPLE_SIZE, SIGMA (refer to report for explanation of these constants) 
# setting global variable PLOT to True will slow down the code, but will ask it to store log likelihood and error over iterations figures in the figures/ folder
q = np.array(TrainList[i]) # train data
w = np.array(folds[i]) # test data
# ground truth labels:
gt_q = q[:,0] 
gt_w = w[:,0]
# evaluate results (on train and test data):

est_q = np.array( map(LR.classify , q) )
est_w = np.array( map(LR.classify , w) )    
    
print '\n','*'*30 , "Evaluation Results: i = %d" %i , '*'*30 , '\n'
a,b,c,d = evalResult(est_q , gt_q)
print "Train data evaluation result: Accuracy={} , TruePositiveRate={} , TrueNegativeRate={} , FalsePostiveRate={}  ".format(a,b,c,d)
a,b,c,d = evalResult(est_w , gt_w)
print "Test data evaluation result: Accuracy={} , TruePositiveRate={} , TrueNegativeRate={} , FalsePostiveRate={}  ".format(a,b,c,d)
print '*'*77 , '\n'
plt.show()
'''
# fine tune sigma (for regularization) using validation set; for each Trailist[i], folds[i] will be the validation set:

PLOT = False
reg_error = []
for SIGMA in range(1,100):
    temp_valid_errList = []
    #for each sigma, run 10 distinct runs:
    for i in range(10):
        TrainList,folds = subsample_data(data_for_LR)
        # print "len trains list  ,folds:" , len(TrainList[0]) , len(folds[0])
        LR.weights = np.random.rand(len(TrainList[0][0]))
        temp_valid_err = LR.train( TrainList[0], 'SGA' , use_regularizer=True , test_data = folds[0])
        temp_valid_errList.append(temp_valid_err)
    if temp_valid_errList:
        reg_error.append(sum(temp_valid_errList)/len(temp_valid_errList))
plt.close(),plt.figure(),plt.plot(range(1,100) , reg_error),plt.yticks(np.arange(0,1,.1)),plt.ylim(0,1) , plt.show()
'''