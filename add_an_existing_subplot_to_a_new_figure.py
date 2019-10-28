
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from numpy import exp, pi, sqrt
from scipy.stats import norm

def pdf_multivariate(x, mu, cov):
    n = len(cov)
    likelihood = exp(-0.5 * (x-mu) @ inv(cov) @ (x-mu).reshape(-1,1)) / sqrt((2*pi)**n *det(cov))
    return likelihood[0]


def make_data():
    data = np.random.normal(loc=[0,6], scale=[2,1], size=(6,2))
    x = data.T.ravel()[:-2]
    x -= x.min()-1
    y = np.array([0]*6 + [1]*4, dtype="uint8")
    return(x,y)


def visualize_points_and_their_gaussians(x,y):
    for k in sorted(set(y)):
        color = ["g","orange"]
        mu,sd = x[y==k].mean(), x[y==k].std(ddof=0)
        pdf = norm(mu,sd).pdf
        xx = np.linspace(x[y==k].min()-10, x[y==k].max()+10, 100)
        yy = pdf(xx)
        mask = yy > 0.001
        xx,yy = (a[mask] for a in (xx,yy))
        plt.plot(xx,yy, color='#999999')
        plt.fill(xx,yy, alpha=0.5, color=color[k])
        plt.plot(x[y==k], [0]*len(x[y==k]), 'o', color=color[k], mec='#999999', ms=8)
        sp = plt.gca()
    return(sp)
    
##############################################################################


x,y = make_data()
m = len(x)
pdf = norm.pdf


#initialize
mu0, mu1 = x.take(np.random.permutation(len(x))[:2])
sd0, sd1 = 1,1
prior0, prior1 = 0.5, 0.5
max_iter = 100

#loop
log = list()
subplots = list()
for epoch in range(max_iter):
    #expectation
    likelihoods0 = pdf(x, mu0, sd0) 
    likelihoods1 = pdf(x, mu1, sd1)
    
    posteriors0 = (likelihoods0*prior0) / (likelihoods0*prior0 + likelihoods1*prior1)
    posteriors1 = (likelihoods1*prior1) / (likelihoods0*prior0 + likelihoods1*prior1)  
        
    #maximization
    mu0 = sum(posteriors0*x) / sum(posteriors0)
    mu1 = sum(posteriors1*x) / sum(posteriors1)
    
    sd0 = sum((x-mu0)**2*posteriors0) / sum(posteriors0)
    sd0 = sqrt(sd0)
    sd1 = sum((x-mu1)**2*posteriors1) / sum(posteriors1)
    sd1 = sqrt(sd1)
    
    prior0 = sum(posteriors0)/m
    prior1 = sum(posteriors1)/m
    
    #predict
    ypred = (posteriors1 >= 0.5).astype("uint8")
    if len(set(ypred))<len(set(y)):
        mu0, mu1 = x.take(np.random.permutation(len(x))[:2])
        sd0, sd1 = 1,1
        prior0, prior1 = 0.5, 0.5
        continue
   
    #vizualize
    sp = visualize_points_and_their_gaussians(x,ypred)
    sp.text(0.4,0.9, f"epoch: {epoch+1}", transform=sp.transAxes, fontsize='large')
    subplots.append(sp)
    plt.close(sp.figure)
    
    #log and check convergence
    snapshot = np.array([mu0, mu1, sd0, sd1, prior0, prior1])
    log.append(snapshot)
    if len(log)>=2 and np.allclose(log[-2], log[-1], rtol=1e-2):
        from pandas import DataFrame
        df = DataFrame(log[-1].reshape(3,len(set(y))), columns=["0","1"], index=["mean","std","mix"])
        print(df)
        break

    
m = int(np.ceil(len(subplots)**0.5))
fig = plt.figure(figsize=(m,m))

for i,sp in enumerate(subplots):
    sp.remove()
    sp.figure = fig
    fig.axes.append(sp)
    fig.add_axes(sp)
    
    dummy = fig.add_subplot(m,m,i+1)
    bbox = dummy.get_position()
    sp.set_position(bbox)
    dummy.remove()

