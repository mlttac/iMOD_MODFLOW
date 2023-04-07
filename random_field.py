# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:20:56 2022
https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/
@author: cnmlt
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random



## OPTION 1: Interface between two classes to generate a discontinuity in a domain with 2 classes
# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def k_discrete2(Nx , Ny,  k1, k2, length_scale = 0.2):
       # # Generate a GP sample
       N = 512
       xmin = 0 
       xmax = 1
       
       gp_params = (1.0, length_scale)
       jitter = 1e-10
       X = np.linspace(xmin, xmax, N)[:,None]
       K = RBF(X, X, gp_params)
       L = np.linalg.cholesky(K + jitter*np.eye(N))
       
       
       gp_sample = np.dot(L, np.random.normal(size=(N,)))
       # Create a callable interpolation function  
       f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
       x = np.linspace(xmin, xmax, Nx )
       y = np.linspace(xmin, xmax, Ny )
       ffn = f_fn(y)
       ffn_norm = (32*(ffn - np.min(ffn))/np.ptp(ffn)).astype(int)
       k = np.zeros((Nx,Ny)) # initial k
       k_rand = np.random.permutation([k1, k2]) # randomly set k1 and k2 on the left and right
       for i in range(Nx):
              k[:,i] = np.where(i < ffn_norm, k_rand[0], k_rand[1])
       
       return x, ffn_norm, k


## OPTION 2: Random field which can be split into classes
def fftIndgen(n):
    a = range(0, int(n/2+1))
    b = range(1, int(n/2))
    b = list(reversed(b))
    b = [-i for i in b]
    return list(a) + b

def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100, fixed_seed = False):
    if fixed_seed:
        np.random.seed(10)

    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size = (size, size))) 
    amplitude = np.zeros((size,size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)

def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b



def split_intervals(k, minK, maxK,n_intervals):
    """Select the random min and max values and 
    Round each value up to a limit of an arbitrary grid
    https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range """

    

    intervals = np.linspace(start=minK, stop=maxK, num=n_intervals+1)

    k = np.array(intervals)[np.searchsorted(intervals, k)]
    
    # one class has only one value: I remove it 
    unique , counts = np.unique(k, return_counts=True)
    d_unique = dict(zip(counts, unique))
    if 1 in d_unique: 
           i, j = np.where(k == d_unique.get(1))
           k[i, j] = k[i-1, j] 
    
    return np.round(k, 6) 

def reshuffle(k):
    """Reshuffle the values to allow large difference between neighbouring materials """
    
    # create list with current values
    k_int = np.unique(k)
    
    # create list with the new ones(unique values but shuffled)
    k_int_shuffled = np.unique(k)
    random.shuffle(k_int_shuffled)

    # replace the old values with the new ones 
    k = k_int_shuffled[np.searchsorted(k_int,k)]

    return k

def create_Krf(alpha=-4.0, discrete = False , minK = 5 , maxK = 25, n_classes = 5, size=32, reshuffle_k = True, log_shape = False, fixed_seed = False):

       out = gaussian_random_field(Pk = lambda k: k**alpha, size=size, fixed_seed= fixed_seed)
       k = out.real
       
       if log_shape == True:
              k = np.exp(k)
       
       # minK = random.randint(5, 10)
       # maxK = random.randint(20, 25)

       k = ((k - k.min()) /( k.max() - k.min() ) ) * (maxK - minK) + minK
       
       if discrete == True:
              k = split_intervals(k, minK, maxK , n_intervals = n_classes)
              
       if reshuffle_k == True: 
               k = reshuffle(k)
       
       return k



## OPTION 3: create a K field made of N chunks all with random k


def ressample(arr, N):
    """Split array into N chunks """
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        block = np.hsplit(v, arr.shape[0] // N)
        A.extend([*block])
    return np.array(A)

def split_chunks(size = 32 , chunk_size = 4, mink= 5, maxk=25, log_intervals = False):
       """Split array into N chunks and assign random k to each chunk"""
       arr = np.zeros((size, size))  
       
       arr_reas =  ressample(arr, chunk_size) #--> chunk size 4
       
       #assign k to each chunk
       new_array = np.empty((size, size))
       count = 0
       for j in range(0, size, chunk_size):
              for i in range(0, size, chunk_size):
                     if log_intervals == False:
                        arr_reas[count] = random.randint(mink, maxk)
                     else:
                        arr_reas[count] = random.choice(np.geomspace(mink, maxk,num=100))
                        new_array[i:i+chunk_size, j:j+chunk_size] = arr_reas[count, :, :]
                        count += 1

       return new_array


if __name__ == '__main__':
       
       
       # Option:
       # - 'interface' : 2 materials split by an interface
       # - 'classes': a gaussian random field then discretized into classes
       # - 'chunks': a domain made up of 4x4 blocks of random material

       
       Option = 'classes'
       
       
       if Option=='interface': 
              
              x, interface, k = k_discrete2(Nx = 32 , Ny = 32, k1 = 0.5 , k2 = 2.5)

              fig, (ax1, ax2) = plt.subplots(1,2)
              ax1.plot(interface, x)
              ax1.set_title(r"Interface")
              ax1.set_ylabel("y")
              ax1.set_xlabel("x")
              ax2.set_title('Discontinuous k field')
              im = ax2.imshow(k,  origin='lower', aspect="auto")
              fig.colorbar(im, ax=ax2, orientation='vertical')
              fig.show()


              
       else:
              
              a = 2  # number of rows
              b = 3  # number of columns
              c = 1  # initialize plot counter
              fig = plt.figure(figsize=(14,10))
              
              k_mid = []
              n_cl_l = []
              
              for realizations in range(6):
                 
                  plt.subplot(a, b, c)
                  if realizations < 3:
                         n_cl = np.random.randint( 2, 5 ) 
                         typeclass = 'Train'
                  else: 
                         n_cl = np.random.randint( 2, 5 )
                         typeclass = 'Test'
                         
                  plt.title('$n = {}$'.format(n_cl))
                  
                  if Option == 'classes':
                         k = create_Krf(discrete = False, minK = 0.0001 , maxK = 0.005, n_classes = n_cl, size = 32, reshuffle_k = False, log_shape = True)
                         # n_cl = 5 
                         # k1 = 0.0001
                         # k2 = 0.005

                  elif Option=='chunks':
                         k = split_chunks(size = 32 , chunk_size = 4, mink= 0.0001, maxk=0.005, log_intervals = True)
                         
                  cmap = mpl.cm.viridis
                  classes = np.unique(k)
                  classes_b = classes
                  classes_b[0] -= 0.1
                  classes_b[0] += 0.1
                  # norm = mpl.colors.BoundaryNorm(list(classes), cmap.N)

                  im = plt.imshow(k , 
                           origin='lower', aspect='auto', cmap = cmap)
                  plt.colorbar(im, cmap=cmap,
                               ticks=classes, format='%1i')
                  c = c + 1
                  
                  k_mid.append(k[16])
                  n_cl_l.append(n_cl)     
              
              plt.show()
              
              fig = plt.figure()
              gs = fig.add_gridspec(2, hspace=0)
              axs = gs.subplots(sharex=True, sharey=True)
              for realizations in range(6):
                     ax_ind = 0 if realizations<3 else 1
                     axs[ax_ind].plot(k_mid[realizations], '-', label='$n = {}$'.format(n_cl_l[realizations] ))
              # Hide x labels and tick labels for all but bottom plot.
              for ax in axs:
                  ax.label_outer()
              axs[0].legend(loc="upper right")         
              axs[1].legend(loc="upper right")

              # Set common labels
              fig.text(0.5, 0.04, 'x [m]', ha='center', va='center')
              fig.text(0.06, 0.5, 'K [m/day]', ha='center', va='center', rotation='vertical')





