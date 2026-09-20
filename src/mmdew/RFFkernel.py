# import sys
# from pathlib import Path

# REPO_ROOT = Path.cwd().parent              # notebooks -> mmdew-change-detector-main
# SRC_DIR   = REPO_ROOT / "src"              # -> mmdew-change-detector-main/src

# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

import numpy as np
from scipy.special import comb as nchoosek
from sklearn.metrics.pairwise import rbf_kernel
from onlinecp import algos
import onlinecp.utils.feature_functions as feat

class Gauss:
    def __init__(self, gamma):
        self.gamma = gamma

    def eval(self, X, Y):
        return rbf_kernel(X, Y, gamma=self.gamma)
    
    def rff_sampler(self, d):
        sigma = 1/np.sqrt(2*self.gamma)
        return lambda n: np.random.randn(n,d) / sigma
    
    @staticmethod
    def _est_nu(X_, max_len=500):
        n = min(len(X_), max_len)
        dists = []
        X = np.random.default_rng(1234).choice(X_,n)
        for i in range(n):
            for j in range(i+1,n):
                dists += [np.linalg.norm(X[i]-X[j],ord=2)**2]
        nu = np.median(dists)
        return np.sqrt(nu*0.5)

    @staticmethod
    def _nu2gamma(nu):
        return 1/(2*nu**2)

    @staticmethod
    def est_gamma(X_, max_len=500):
        return Gauss._nu2gamma(Gauss._est_nu(X_=X_, max_len=max_len))
    
class BiasedMMD:
    def __init__(self, kernel):
        self.kernel = kernel

    def mmd(self, X,Y):
        XX = self.kernel.eval(X,X)
        YY = self.kernel.eval(Y,Y)
        XY = self.kernel.eval(X,Y)
        return np.sqrt(np.mean(XX) + np.mean(YY) - 2*np.mean(XY))
    
    @staticmethod
    def threshold(n, m, alpha):
        # assumes $k(x,y) \le 1$ for all $x,y$
        return np.sqrt((n+m)/(n*m))*(np.sqrt(2) + np.sqrt(2*np.log(1/alpha)))
     
    
class RFFMMD(BiasedMMD):
    def __init__(self, kernel, num_omegas=100):
        self.kernel = kernel
        self.num_omegas = num_omegas        
    
    def mmd(self, X, Y, omegas=None):
        d = X.shape[1]
        if omegas is None:
            omegas = (self.kernel.rff_sampler(d))(self.num_omegas)
        tmp = X @ omegas.T
        zX = 1/np.sqrt(len(omegas))* np.concatenate((np.cos(tmp), np.sin(tmp)),axis=1)
        tmp = Y @ omegas.T
        zY = 1/np.sqrt(len(omegas))* np.concatenate((np.cos(tmp), np.sin(tmp)),axis=1)
        return np.linalg.norm(np.mean(zX,axis=0)-np.mean(zY,axis=0))

class ChangeDetector:
    def insert(self, element):
        pass

    def statistic(self):
        pass


class StreamingRFFMMD(ChangeDetector):
    class Window:
        def __init__(self, z, c):
            self.z = z
            self.c = c
    
    def __init__(self, kernel, d, num_omegas=100):
        self.windows = []
        self.omegas = (kernel.rff_sampler(d))(num_omegas)
        self.num_omegas = num_omegas

    def insert(self, element):
        element = element.reshape(1,-1)
        tmp = element @ self.omegas.T
        z = 1/np.sqrt(self.num_omegas) * np.concatenate((np.cos(tmp), np.sin(tmp)),axis=1)
        W = StreamingRFFMMD.Window(z=z,c=1)
        self.windows.append(W)
        
        while len(self.windows) >= 2:
            W1 = self.windows[-1]
            W2 = self.windows[-2]
            if W1.c == W2.c:
                W = StreamingRFFMMD.Window(z=W1.z + W2.z,c=W1.c + W2.c)
                self.windows = self.windows[:-2] + [W]
            else:
                break

    def mmd_values(self):
        """Returns a list of MMD computed for every split between windows."""
        zs = [w.z for w in self.windows]
        cs = [w.c for w in self.windows]
        lz = 0
        lc = 0
        rz = np.sum(zs, axis=0)
        rc = np.sum(cs, axis=0)
        vals=np.zeros(len(zs)-1)
        for i,pos in enumerate(range(len(zs)-1)):
            lz += zs[pos]
            rz -= zs[pos]
            lc += cs[pos]
            rc -= cs[pos]
            vals[i] = np.linalg.norm(1/lc*lz - 1/rc*rz)
        return vals

    def has_change(self,min_arl):
        if len(self.windows) < 2:
            return False
        lambda_n = np.sqrt(2*np.log(4*min_arl*np.log2(2*min_arl)))
        return (np.array(self.normalized_mmd()) >= np.sqrt(2) + lambda_n).any()

    def normalized_mmd(self):
        vals = self.mmd_values()
        cs = [w.c for w in self.windows]
        ns = np.cumsum(cs[:-1],axis=0)
        ms = list(reversed(np.cumsum(cs[::-1][:-1],axis=0)))
        return [np.sqrt(n*m/(n+m))*mmd for m,n,mmd in zip(ms,ns,vals)]
    
    def statistic(self):
        return np.max(self.normalized_mmd() + [0])
    

class ScanBStatistic(ChangeDetector):
    def __init__(self, reference_sample, B0, N, gamma):
        self.rng = np.random.default_rng()
        self.B0 = B0 # block size
        self.N = N # number of blocks
        self.gamma = gamma #Gauss.est_gamma(reference_sample)

        self.ref_blocks = self.rng.choice(reference_sample, size=N*B0, replace=False) # store the blocks that make the references
        self.ref_grams_XX = []
        for i in range(self.N):
            x = self.ref_blocks[i*self.B0:(i+1)*self.B0]
            self.ref_grams_XX += [rbf_kernel(x, gamma=self.gamma)]

        self.post_block = reference_sample[-B0:] # store the block with the most recent data
        self.var = 1/nchoosek(B0,2)*(1/N*self.expectation_h_squared(reference_sample) + (N-1)/N*self.covariance_h(reference_sample))
        self.stats = []

    def insert(self, input_value):
        self.post_block = np.concatenate((self.post_block[1:], input_value.reshape(1,-1)))
        acc = 0
        gram_YY = rbf_kernel(self.post_block, gamma=self.gamma)
        for i in range(self.N):
            acc += self.mmd_u(self.ref_blocks[i*self.B0:(i+1)*self.B0], self.ref_grams_XX[i], self.post_block, gram_YY)
        self.stats += [acc / (self.N * np.sqrt(self.var))]

    def statistic(self):
        return self.stats[-1] 


    def mmd_u(self, sample_x, gram_XX, sample_y, gram_YY):
        n_x = len(sample_x)
        n_y = len(sample_y)
        XX = gram_XX - np.eye(n_x)
        YY = gram_YY - np.eye(n_y)
        XY = rbf_kernel(sample_x, sample_y, gamma=self.gamma)

        return np.sum(XX)/(n_x*(n_x-1)) + np.sum(YY)/(n_y*(n_y-1)) - 2*np.mean(XY)

    def expectation_h_squared(self, sample): # corresponds to (7) in their paper
        # we assume a shuffled sample
        n = int(len(sample)/4)
        x   = sample[0*n:1*n]
        x_p = sample[1*n:2*n]
        y   = sample[2*n:3*n]
        y_p = sample[3*n:4*n]

        K_xx = rbf_kernel(x,x_p, gamma=self.gamma)
        K_yy = rbf_kernel(y,y_p, gamma=self.gamma)
        K_xy1 = rbf_kernel(x,y_p, gamma=self.gamma)
        K_xy2 = rbf_kernel(x_p,y, gamma=self.gamma)

        return np.mean((K_xx + K_yy - K_xy1 - K_xy2)**2)

    def covariance_h(self, sample): # corresponds to (7) in their paper
        # we assume a shuffled sample
        n     = int(len(sample)/6)
        x     = sample[0*n:1*n]
        x_p   = sample[1*n:2*n]
        x_pp  = sample[2*n:3*n]
        x_ppp = sample[3*n:4*n]
        y     = sample[4*n:5*n]
        y_p   = sample[5*n:6*n]

        K1 = rbf_kernel(x,x_p, gamma=self.gamma)
        K2 = rbf_kernel(y,y_p, gamma=self.gamma)
        K3 = rbf_kernel(x,y_p, gamma=self.gamma)
        K4 = rbf_kernel(x_p,y, gamma=self.gamma)
        K5 = rbf_kernel(x_pp,x_ppp, gamma=self.gamma)
        K6 = rbf_kernel(x_pp,y_p, gamma=self.gamma)
        K7 = rbf_kernel(x_ppp,y, gamma=self.gamma)

        h1 = K1 + K2 - K3 - K4
        h2 = K5 + K2 - K6 - K7

        return np.mean(h1*h2) - np.mean(h1)*np.mean(h2)


class OKCUSUM(ChangeDetector):
    def __init__(self, reference_sample, B_max, N, gamma, B_min=2,alpha=0.05):
        self.rng = np.random.default_rng()
        self.gamma = gamma
        self.B_max = B_max
        self.B_min = B_min
        self.N = N

        self.reference_sample = self.rng.choice(reference_sample, size=N*B_max, replace=False) # store the blocks that make the references
        self.XX = rbf_kernel(self.reference_sample, gamma=self.gamma)

        self.post_block = reference_sample[-B_max:] # store the block with the most recent data

        self.vars = []
        e_h_sq = self.expectation_h_squared(self.reference_sample)
        cov_h = self.covariance_h(self.reference_sample)
        for B0 in range(B_min,B_max+1,2):
            self.vars += [1/nchoosek(B0,2)*(1/N*e_h_sq + (N-1)/N*cov_h)] # store the variances corresponding to each B0
        self.stats = []
        self.alpha = alpha

    def expectation_h_squared(self, sample): # corresponds to (7) in their paper
        # we assume a shuffled sample
        n = int(len(sample)/4)
        x   = sample[0*n:1*n]
        x_p = sample[1*n:2*n]
        y   = sample[2*n:3*n]
        y_p = sample[3*n:4*n]

        K_xx = rbf_kernel(x,x_p, gamma=self.gamma)
        K_yy = rbf_kernel(y,y_p, gamma=self.gamma)
        K_xy1 = rbf_kernel(x,y_p, gamma=self.gamma)
        K_xy2 = rbf_kernel(x_p,y, gamma=self.gamma)

        return np.mean((K_xx + K_yy - K_xy1 - K_xy2)**2)

    def covariance_h(self, sample): # corresponds to (7) in their paper
        # we assume a shuffled sample
        n     = int(len(sample)/6)
        x     = sample[0*n:1*n]
        x_p   = sample[1*n:2*n]
        x_pp  = sample[2*n:3*n]
        x_ppp = sample[3*n:4*n]
        y     = sample[4*n:5*n]
        y_p   = sample[5*n:6*n]

        K1 = rbf_kernel(x,x_p, gamma=self.gamma)
        K2 = rbf_kernel(y,y_p, gamma=self.gamma)
        K3 = rbf_kernel(x,y_p, gamma=self.gamma)
        K4 = rbf_kernel(x_p,y, gamma=self.gamma)
        K5 = rbf_kernel(x_pp,x_ppp, gamma=self.gamma)
        K6 = rbf_kernel(x_pp,y_p, gamma=self.gamma)
        K7 = rbf_kernel(x_ppp,y, gamma=self.gamma)

        h1 = K1 + K2 - K3 - K4
        h2 = K5 + K2 - K6 - K7

        return np.mean(h1*h2) - np.mean(h1)*np.mean(h2)

    def insert(self, input_value):
        self.post_block = np.concatenate((self.post_block[1:], input_value.reshape(1,-1)))
        YY = rbf_kernel(self.post_block, gamma=self.gamma)
        XY = rbf_kernel(self.reference_sample, self.post_block, gamma=self.gamma)

        stats = -np.inf

        for i, B0 in enumerate(range(self.B_min,self.B_max+1,2)): # different window sizes
            acc = 0
            yy = YY[-B0:,-B0:]
            for n in range(self.N):
                xx = self.XX[n*self.B_max:n*self.B_max+B0,n*self.B_max:n*self.B_max+B0]
                xy = XY[n*self.B_max:n*self.B_max+B0,-B0:]
                n_x = len(xx)
                n_y = len(yy)
                acc += np.sum(xx-np.eye(n_x))/(n_x*(n_x-1)) + np.sum(yy-np.eye(n_y))/(n_y*(n_y-1)) - 2*np.mean(xy)
            tmp = acc / (self.N * np.sqrt(self.vars[i]))

            stats = max(tmp, stats)

        self.stats += [stats]

    def statistic(self):
        return self.stats[-1]


def offdiag_pairwise_median(X_, max_len=500):
    gamma = Gauss.est_gamma(X_, max_len=max_len)
    return 1 / (2 * gamma)


class NewMAAdapter(ChangeDetector):
    def __init__(self, reference_sample, d, B=50):
        self.recent_stat=0
        self.B = B
        big_Lambda, small_lambda = algos.select_optimal_parameters(B)  # forget factors chosen with heuristic in the paper
        thres_ff = small_lambda
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
        def feat_func(x):
            return feat.fourier_feat(x, W)
        
        sigmasq = offdiag_pairwise_median(reference_sample)
        W, sigmasq = feat.generate_frequencies(m, d, sigmasq=sigmasq, choice_sigma="fixed")

        self.detector = algos.NEWMA(reference_sample[0], forget_factor=big_Lambda, forget_factor2=small_lambda, feat_func=feat_func,
                       adapt_forget_factor=thres_ff)

    def insert(self,element):
        self.recent_stat = self.detector.update_stat(element)

    def statistic(self):
        return self.recent_stat
