import random
import os
import sys
import math
import numpy as np
 

inpath = "D://python2.7.6//MachineLearning//FFM_CTR//train0.csv"
v_k=5 
epoch=3
step=0.1
lbd=0.01
######################

def loadData():
    global xmatrix,yvec
    dataList=[];labelList=[]
    content=open(inpath,'r')
    line=content.readline().strip('\n')
    featList=line.split(',');#print featList;
    featList=featList[2:]
    line=content.readline().strip('\n')
    while line:
        obs=line.split(',');
        labelList.append(int(obs[1]))
        obs=obs[2:]
        dataList.append(obs)
        line=content.readline().strip('\n')

    print 'data',len(dataList),len(labelList)#len(dataList[0]),len(featList),dataList[0]
    ##############
    feat_all=[] #[[],[],[v1 v2 ..],[],[]]
    for i in range(len(featList)): 
        featSet=[]
        for obs in dataList:
            if obs[i] not in featSet:
                featSet.append(obs[i])

        ####
        feat_all.append(featSet)

    #######
    nx=len(dataList)
    featSpace=0.0
    for featset in feat_all:
        if len(featset)<nx*0.5:
            print feat_all.index(featset),'<100',len(featset)
            featSpace+=len(featset)
        if len(featset)>=nx*0.5:
            print feat_all.index(featset),'>100',len(featset)
         
        
    print 'space of feat',featSpace
    #feat_all0=feat_all[:10]+feat_all[11:]
    #### string to hash feat dict
    i=0
    hash_feat={}
    for feat in feat_all:
        if len(feat)<nx*0.5:
            for val in feat:
                hash_feat[(feat_all.index(feat),val)]=i#
                i+=1
    print 'dim',len(hash_feat)
    ######
    dim=len(hash_feat)
    n=len(dataList)
    xmatrix=np.zeros((n,dim))
    yvec=np.array([float(y) for y in labelList])
    print yvec.shape,xmatrix.shape
    for xi in range(len(dataList)):
        for fi in range(len(featList)):
            if fi!=10: #feat10 too much value not useful
                str_feat_val=dataList[xi][fi]
                f_ind=hash_feat[(fi,str_feat_val)]
                xmatrix[xi,f_ind]=1

     
     
    ##########

def initial_para():
    global w0,w1,v_mat
    global xmatrix,yvec
    n,d=xmatrix.shape
    w0=0.0
    w1=np.random.uniform(low=-0.1,high=0.1,size=(1,d))
    v_mat=np.random.uniform(low=-0.1,high=0.1,size=(d,v_k))
    
    



def initial_grad():
    global w0_g,w1_g,v_g
    global xmatrix,yvec
    n,d=xmatrix.shape
    w0_g=0.0
    w1_g=np.zeros((1,d))
    v=np.zeros((d,v_k))
    
def train():
    global w0_g,w1_g,v_g
    global w0,w1,v_mat
    global xmatrix,yvec
    loss=calc_loss();print 'loss',loss
    n,d=xmatrix.shape 
    for ep in range(epoch):
        obs_list=range(n)
        random.shuffle(obs_list)
        ####based on each one xi
        for i in obs_list[:]:#[2,1,6,3...]
            ###grad based on one obs x1
            f=calc_f(i)#sigm(wx)
            grad_loss_f=-f*(1-f)*(yvec[i]-f);#print 'grad f',f,grad_loss_f#wx358 sigm(wx)1e-130 too small grad~0
            
            ###grad w w
            w0_g=1*grad_loss_f
            w1_g=xmatrix[i,:]*grad_loss_f#+lbd*np.abs(w1)*2; #[d,]
            
            ##grad v
            g1=xmatrix[i,:]+np.zeros((1,d))#[d,]->[1,d]
            g1=np.tile(g1.T,(1,v_k))#[d,1]->[d,k]
            g2=g1*g1*v_mat#[d,k]
            gdot=np.dot(xmatrix[i,:],v_mat)#[d,][d,k]->[k,]
            g3=np.tile(gdot+np.zeros((1,v_k)),(d,1))#[k,]->[1,k]->[d,k]
            v_g=(g1*g3-g2);#print 'v', v_g[:4,:4]
            v_g=v_g*grad_loss_f#+lbd*np.abs(v_mat)*2;#print 'v',v_g[:4,:4]
            ####normalize grad
            w1_g,v_g=normalize(w1_g,v_g)
            #print 'grad',w1_g[:4],v_g[:4,:4]
            ####update para
            w0=w0-w0_g*step;#print w0 #not quite change
            w1=w1-w1_g*step;#print w1[:4]
            v_mat=v_mat-v_g*step;#print v_mat[:5,:5]
        #####
        loss=calc_loss();
        print 'epoch %d loss %f  '%(ep,loss)
            
            
            
            
def normalize(wg,vg):#[d,] [d,k] inorder to make mode=1 |vec|=1
    ss=np.dot(wg,wg.T)
    import math
    ss=math.sqrt(ss)
    wg=wg/(ss+0.00001)
    ###
    ss=vg*vg
    ss=np.sum(ss)
    ss=math.sqrt(ss)
    vg=vg/(ss+0.00001)
    return wg,vg
    
        
def calc_f(xi):#based on one xi
    global w0,w1,v_mat
    global xmatrix
    ##wx
    f1=np.dot(w1,xmatrix[xi,:])#[1,d] [d,]->[1,nobs]
     
    ####vvxx
    f21=np.dot(xmatrix[xi,:],v_mat)#[1,d] [d,k]->[1,k]
    f21=f21*f21;f21=f21.sum(); #1x1

    v1=v_mat*v_mat;xm=xmatrix[xi,:]*xmatrix[xi,:]
    f22=np.dot(xm,v_mat)#[1,k]
    f22=f22.sum()#1x1
    f2=(f21-f22)*0.5
    #####
    w1_reg=np.sum(w1*w1);
    v_reg=np.sum(v_mat*v_mat);
    f=(f1+f2+w0)#+lbd*(w1_reg+v_reg);#print 'ff',f
    f=sigm(f);#print 'ff',f
    return f

    
        
        
def calc_loss():#based on all x
    global w0,w1,v_mat
    global xmatrix,yvec
    
    
    ####[y-sigm(f)]**2
    ##wx
    f1=np.dot(w1,xmatrix.T)#[1,d] [d,2]->[1,2obs]
     
    ####vvxx
    f21=np.dot(xmatrix,v_mat)#[n,d] [d,k]->[n,k]
    f21=f21*f21;f21=f21.sum(1);#print '1',f21.shape#[n,]

    v1=v_mat*v_mat;xm=xmatrix*xmatrix
    f22=np.dot(xm,v_mat)#[n,k]
    f22=f22.sum(1)#[n,]
    f2=(f21-f22)*0.5
    #####
    f=(f1+f2+w0) #[n,] n obs
    f=sigm(f)
    loss=(yvec-f)**2; #[n,]
    
    return 0.5*loss.sum()
    
def sigm(z): #[n,]
    return np.exp(z)/(1.0+np.exp(z))
    
        
        
def test():
    global w0,w1,v_mat
    global xmatrix,yvec
    n,d=xmatrix.shape
    ntest=1000
    testList=random.sample(range(n),ntest)
    wxb=np.zeros((ntest,))+w0#[1000,]
    ###
    w1x=np.dot(xmatrix[np.array(testList),:],w1.T); #[n,d][d,1] [1000,1]
    
    vx2=np.dot(xmatrix[np.array(testList),:],v_mat)#[n,d][d,k]   [n,k]
    vx2=vx2**2
    v2x2=np.dot(xmatrix[np.array(testList),:]**2,v_mat**2)#[n,k]
    vvxx=(vx2-v2x2);vvxx=0.5*vvxx.sum(1)#[1000,]
    ###
    wxb=wxb+w1x.T+vvxx;#(1,1000) (1000,)#[1,10]+[10,1]=[10,10] or [10,1]+[10,]=[10,10]
    wxb=sigm(wxb);#print wxb.shape#[1,1000]
    pred=1.0*(wxb>0.5);#print pred.shape,yvec[np.array(testList)].shape#[1,1000][1000,]
    compare=(pred-yvec[np.array(testList)])**2
    print 'err percetage',compare.sum()/float(ntest)
    
    
    
      





#####
if __name__=='__main__':
    loadData() #(2541,) (2541, 1804)
    initial_para()
    initial_grad()
    train()
    ######
    ####test
    test()
     
        
        
    
    
    







    
    
