import numpy as np
import scipy
import pandas as pd
from numpy import linalg
from scipy.fft import fft
import matplotlib.pyplot as plt
import sklearn 
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import numba
from sklearn.decomposition import IncrementalPCA
from numba import jit
import json, gzip, warnings

##################################### FILTERING TOOLS #######################################################

def notch_filter(signal,fsamp,to_han = False):

    """ Implementation of a notch filter, where the frequencies of the line interferences are unknown. Therefore, interference is defined
    as frequency components with magnitudes greater than 5 stds away from the median frequency component magnitude in a window of the signal
    - assuming you will iterate this function over each grid 
    """

    bandwidth_as_index = round(4/(fsamp/np.shape(signal)[1]))
    # bandwidth_as_index = int(round(4*(np.shape(signal)[1]/fsamp)))
    # width of the notch filter's effect, when you intend for it to span 4Hz, but converting to indices using the frequency resolution of FT
    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])

    for chan in range(np.shape(signal)[0]):

        if to_han:
            hwindow = scipy.signal.hann(np.shape(signal[chan,:])[0])
            final_signal = signal[chan,:]* hwindow
        else:
            final_signal = signal[chan,:]

        fourier_signal = np.fft.fft(final_signal)
        fourier_interf = np.zeros(len(fourier_signal),dtype = 'complex_')
        interf2remove = np.zeros(len(fourier_signal),dtype=int)
        
        #create a copy of interf2remove, with instead of zeros NaNs 
        interf2remove2 = np.empty(len(fourier_signal)) 
        interf2remove2[:] = np.nan
        
        window = fsamp
        tracker = 0
    
        for interval in range(0,len(fourier_signal)-window + 1,window): # so the last interval will start at len(fourier_emg) - window
            median_freq = np.median(abs(fourier_signal[interval+1:interval+window+1])) 
            std_freq = np.std(abs(fourier_signal[interval+1:interval+window+1]), ddof = 1) #ddof to 1 to get Matlab results 
            
            # interference is defined as when the magnitude of a given frequency component in the fourier spectrum
            # is greater than 5 times the std, relative to the median magnitude
            label_interf = list(np.where(abs(fourier_signal[interval+1:interval+window+1]) > median_freq+5*std_freq)[0]) # np.where gives tuple, element access to the array
            # need to shift these labels to make sure they are not relative to the window only, but to the whole signal
            label_interf = [x + interval + 1 for x in label_interf] # + 1 since the interval starts with 0? -> 
    
            if label_interf: # if a list exists
                for i in range(int(-np.floor(bandwidth_as_index/2)),int(np.floor(bandwidth_as_index/2)+1)): # so as to include np.floor(bandwidth_as_index/2)
                    temp_shifted_list = [x + i for x in label_interf]
                    interf2remove[tracker: tracker + len(label_interf)] = temp_shifted_list
                    interf2remove2[tracker: tracker + len(label_interf)] = temp_shifted_list
                    tracker = tracker + len(label_interf)
        
        # we only take the first half of the signal, we need a compensatory step for the second half given we haven't wrapped the FT yet
        # use interf2remove2 instead of interf2remove, as interf2remove is initialized using 0's, this comparison cannot be made! 
        # interfremove2 is utilized here, as other wise the indices don't equal the matlab version (matlab has +1, so everything is 1 index lower here
        # results in some 0 indices in the matrix, which are already used in the placeholder!
        #TODO: make this more elegant!!
        indexf2remove = np.where(np.logical_and(interf2remove2 >= 0 , interf2remove2 < (len(fourier_signal)/2+1)))[0]

        # use interf2remove here, as interf2remove2 contains floating numbers which cannot be used for indexing 
        fourier_interf[interf2remove[indexf2remove]] = fourier_signal[interf2remove[indexf2remove]]
        corrector = int(len(fourier_signal) - np.floor(len(fourier_signal)/2)*2)  # will either be 0 or 1 (0 if signal length is even, 1 if signal length is odd)
        # wrapping FT
        fourier_interf[int(np.ceil(len(fourier_signal)/2)):] = np.flip(np.conj(fourier_interf[1: int(np.ceil(len(fourier_signal)/2)+1- corrector)])) # not indexing first because this is 0Hz, not to be repeated
        filtered_signal[chan,:] = signal[chan,:] - np.fft.ifft(fourier_interf).real
      

    return filtered_signal

def bandpass_filter(signal,fsamp, emg_type = 'surface'):

    """ Generic band-pass filter implementation and application to EMG signal  - assuming that you will iterate this function over each grid """

    """IMPORTANT!!! There is a difference in the default padding length between Python and MATLAB. For MATLAB -> 3*(max(len(a), len(b)) - 1),
    for Python scipy -> 3*max(len(a), len(b)). So I manually adjusted the Python filtfilt to pad by the same amount as in MATLAB, if you don't the results will not match across
    lanugages. NOTE OF CHIARA GIBBS """   

    if emg_type == 0:
        lowfreq = 20
        highfreq = 500
        order = 2
    elif emg_type == 1:
        lowfreq = 100
        highfreq = 4400
        order = 3

    # get the coefficients for the bandpass filter
    nyq = fsamp/2
    lowcut = lowfreq/nyq
    highcut = highfreq/nyq
    [b,a] = scipy.signal.butter(order, [lowcut,highcut],'bandpass') # the cut off frequencies should be inputted as normalised angular frequencies

    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])
    # construct and apply filter
    for chan in range(np.shape(signal)[0]):
        
        filtered_signal[chan,:] = scipy.signal.filtfilt(b,a,signal[chan,:],padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    
    return filtered_signal

def moving_mean1d(v,w):
    """ Moving average filter that replicates the method of movmean in MATLAB
    v is a 1 dimensional vector to be filtered via a moving average
    w is the window length of this filter """

    u = v.copy()
    w_temp = w
    n = len(v)-1

    if w_temp % 2 != 0:

        w = int(np.ceil(w_temp/2))
        for i in range(w):
            u[i] = np.mean(v[0:w+i])
            u[n-i] = np.mean(v[n-(w-1)-i:])

        n1 = 1 + w
        n2 = n - w

        for i in range(n1-1,n2+1):
        
            u[i] = np.mean(v[i - w + 1:i + w])

    else:

        w = int(w_temp/2)
        for i in range(w):
            u[i] = np.mean(v[0:w+i])
            u[n-i] = np.mean(v[n-(w-1)-(i+1):])

        n1 = 1 + w
        n2 = n - w

        for i in range(n1-1,n2+1):
            u[i] = np.mean(v[i - w:i + w ])

    return u
    

################################# CONVOLUTIVE SPHERING TOOLS ##########################################################

def extend_emg(extended_template, signal, ext_factor):

    """ Extension of EMG signals, for a given window, and a given grid. For extension, R-1 versions of the original data are stacked, with R-1 timeshifts.
    Structure: [channel1(k), channel2(k),..., channelm(k); channel1(k-1), channel2(k-1),...,channelm(k-1);...;channel1(k - (R-1)),channel2(k-(R-1)), channelm(k-(R-1))] """

    # signal = self.signal_dict['batched_data'][tracker][0:] (shape is channels x temporal observations)

    nchans = np.shape(signal)[0]
    nobvs = np.shape(signal)[1]
    for i in range(ext_factor):

        extended_template[nchans*i:nchans*(i+1), i:nobvs+i] = signal
    return extended_template


def whiten_emg(signal):
    
    """ Whitening the EMG signal imposes a signal covariance matrix equal to the identity matrix at time lag zero. Use to shrink large directions of variance
    and expand small directions of variance in the dataset. With this, you decorrelate the data. """

    # get the covariance matrix of the extended EMG observations
    cov_mat = np.cov(np.squeeze(signal),bias=True)
    
    # get the eigenvalues and eigenvectors of the covariance matrix
    evalues, evectors  = scipy.linalg.eigh(cov_mat) # changed scipy.lignals.eigh(cov_mat) eigh to eig 
    # in MATLAB: eig(A) returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D
    # sort the eigenvalues in descending order, and then find the regularisation factor = "average of the smallest half of the eigenvalues of the correlation matrix of the extended EMG signals" (Negro 2016)
    sorted_evalues = np.sort(evalues)[::-1]
    penalty = np.mean(sorted_evalues[len(sorted_evalues)//2-1:]) # int won't wokr for odd numbers, to nearest whole number 
    print('penalty = ', np.mean(sorted_evalues[len(sorted_evalues)//2-1:]))
    penalty = max(0, penalty)

    rank_limit = np.sum(evalues > penalty)-1
    if rank_limit < np.shape(signal)[0]:

        hard_limit = (np.real(sorted_evalues[rank_limit]) + np.real(sorted_evalues[rank_limit + 1]))/2

    # use the rank limit to segment the eigenvalues and the eigenvectors
    evectors = evectors[:,evalues > hard_limit] #1 shorter 
    evalues = evalues[evalues>hard_limit]
    diag_mat = np.diag(evalues)
    
    whitening_mat = evectors @ np.linalg.inv(np.sqrt(diag_mat)) @ np.transpose(evectors)
    dewhitening_mat = evectors @ np.sqrt(diag_mat) @ np.transpose(evectors)
    whitened_emg =  np.matmul(whitening_mat, signal).real 

    return whitened_emg, whitening_mat, dewhitening_mat

###################################### DECOMPOSITION TOOLS ##################################################################

# orthogonalisation update on 5th feb 20:24
@numba.njit(fastmath = True)
def ortho_gram_schmidt(w,B):

    """ This is the recommended method of orthogonalisation in Negro et.al 2016,
    documented in Hyvärinen et.al 2000 (fast ICA) """
    basis_projection = np.zeros(np.shape(w))
    for i in range(B.shape[1]):
        w_history = B[:, i]
        if np.all(w_history == 0):
            continue
        # Estimate the independent components one by one. When we have estimated p independent components, 
        # or p vectors w1; ...; wp; we run the one-unit fixed- point algorithm for wp11; 
        # and after every iteration step subtract from wp11 the “projections”  of the previously estimated p vectors, and then
        # renormalize 
        basis_projection = basis_projection + np.divide(np.dot(w, w_history),np.dot(w_history,w_history)) * w_history
    w = w - basis_projection
    return w

@numba.njit 
def skew(x):
    return np.square(x)

@numba.njit  
def kurt(x):
    return x**3

@numba.njit #TO DO: adjust functions 
def exp(x):
    return np.exp(-np.square(x)/2)

@numba.njit 
def logcosh(x):
    return np.log(np.cosh(x))

@numba.njit 
def dot_skew(x):
    return 2*x

@numba.njit 
def dot_kurt(x):
    return 3*(np.square(x))

@numba.njit #TO DO: adjust functions 
def dot_exp(x):
    # derivative: -e^{-(x^2)/2}
    return -1*(np.exp(-np.square(x)/2)) + np.dot((np.square(x)), np.exp(-np.square(x)/2))

@numba.njit 
def dot_logcosh(x):
    return np.tanh(x)

@numba.njit(fastmath=True) #TO DO: adjust functions 
def fixed_point_alg(w_n, B, Z,cf, dot_cf, its = 500,ortho_type='ord_deflation'):

    """ Update function for source separation vectors. The code user can select their preferred contrast function using a string input:
    1) skew --> x^2
    2) kurt --> x^3 
    3) logcosh --> log(cosh(x))
    3) exp  --> e(-x^2/2)
    
    Upon meeting a threshold difference between iterations of the algorithm, separation vectors are discovered 
    The maximum number of iterations (its) and the contrast function type (cf) are already specified, unless alternative input is provided. """
   
    assert B.ndim == 2
    assert Z.ndim == 2
    assert w_n.ndim == 1
    assert its in [500]

    counter = 0
    its_tolerance = 0.0001 # tolerance between 2 iterations 
    sep_diff = np.ones(its) # to check for the tolerance 
    B_T_B = B @ B.T
    Z_meaner = Z.shape[1] #size of the rows (amount of the columns)

    while sep_diff[counter] > its_tolerance and counter < its:

        # transfer current separation vector as the previous arising separation vector
        w_n_1 = w_n.copy()
        # use update function to get new, current separation vector
        wTZ = w_n_1.T @ Z 
        A = dot_cf(wTZ).mean()
        w_n = Z @ cf(wTZ).T / Z_meaner  - A * w_n_1 #same as taking the mean over the columns 

        # orthogonalise separation vectors
        if ortho_type == 'ord_deflation':
            w_n -= np.dot(B_T_B, w_n)
        elif ortho_type == 'gram_schmidt':
            # as recommended in Negro et.al 2016
            w_n = ortho_gram_schmidt(w_n,B)

        # normalise separation vectors
        w_n /= np.linalg.norm(w_n)
        
        counter += 1 # update amount of iterations
        sep_diff[counter] = np.abs(w_n @ w_n_1 - 1) # calculate tolerance 
    print(counter)
    return w_n

def pcaesig(signal):
    """
    Perform PCA on a row-wise signal and return the eigenvectors (E) and eigenvalues (D).

    Args:
        signal (numpy.ndarray): Input row-wise signal (channels x time points).

    Returns:
        E (numpy.ndarray): Matrix of eigenvectors (columns correspond to eigenvectors).
        D (numpy.ndarray): Diagonal matrix of eigenvalues.
    """
    # Calculate the covariance matrix of the transposed signal (time points x channels)
    covariance_matrix = np.cov(signal, bias=True)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices for sorting in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the rank tolerance (regularization factor)
    rank_tolerance = np.mean(eigenvalues[len(eigenvalues) // 2:])
    #print(f'Rank tolerance {rank_tolerance}')
    
    # If rank tolerance is negative, set it to 0
    if rank_tolerance < 0:
        rank_tolerance = 0
    
    # Determine the cutoff for significant eigenvalues
    max_last_eig = np.sum(eigenvalues > rank_tolerance)
    #print(f'Rank tolerance {max_last_eig}')
    
    if max_last_eig < signal.shape[0]:
        lower_limit_value = (eigenvalues[max_last_eig - 1] + eigenvalues[max_last_eig]) / 2
    else:
        lower_limit_value = eigenvalues[max_last_eig - 1]
    
    # Select eigenvectors and eigenvalues corresponding to significant eigenvalues
    significant_indices = eigenvalues > lower_limit_value
    
    # Select eigenvectors and eigenvalues
    E = eigenvectors[:, significant_indices]
    D = np.diag(eigenvalues[significant_indices])

    # Sort E and D to return in ascending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues[significant_indices])  # Sort by selected eigenvalues
    E = E[:, sorted_indices]  # Reorder eigenvectors
    D = D[sorted_indices][:, sorted_indices]  # Reorder diagonal matrix of eigenvalues
    
    return E, D

def whiteesig(signal, E, D):
        """
        Whitens the EMG signal.

        Parameters:
        signal: ndarray
            Row-wise signal (2D array where rows correspond to different channels).
        E: ndarray
            Full matrix whose columns are the corresponding eigenvectors.
        D: ndarray
            Diagonal matrix of eigenvalues.
        
        Returns:
        whitensignals: ndarray
            The whitened EMG signal.
        whiteningMatrix: ndarray
            The whitening matrix.
        dewhiteningMatrix: ndarray
            The dewhitening matrix.
        """
        
        whiteningMatrix = E @ np.linalg.inv(np.sqrt(D)) @ E.T
        dewhiteningMatrix = E @ np.sqrt(D) @ E.T
        whitensignals = whiteningMatrix @ signal
        
        return whitensignals, whiteningMatrix, dewhiteningMatrix

# updadted get spikes on feb 19th 11:26am
def get_spikes(w_n, Z, fsamp, std_thr = 3):

    """ Based on gradient convolutive kernel compensation. Aim to remove spurious discharges to improve the source separation
    vector estimate. Results in a reduction in ISI vairability (by seeking to minimise the covariation in MU discharges)"""

    # Step 4a: 
    source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    # Step 4b:
    # peaks = detect_peaks(np.squeeze(source_pred), mpd = np.round(fsamp*0.02)) 
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.02)+1) # peaks variable holds the indices of all peaks
    # Find 
    source_pred /=  np.mean(maxk(source_pred[peaks], 10)) #Find k largest elements of the predicted peaks and normalize the peaks  
    
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        # reshape 1d array into a column vector
        spikes_ind = np.argmax(kmeans.cluster_centers_) #find index of the spikes, using means of points in each cluster 
        spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
        # remove outliers from the spikes cluster with a std-based threshold
        spikes = spikes[source_pred[spikes] <= np.mean(source_pred[spikes]) + std_thr*np.std(source_pred[spikes], ddof=1)] #adjusted: to unbiased 
    else:
        spikes = peaks

    return source_pred, spikes


def min_cov_isi(w_n,Z,fsamp,cov_n): 
    cov_last = cov_n + 0.1
    # cov_n_1 = 2 * cov_n last version
    spikes = np.array([1])
    k = 0
    while cov_n < cov_last:

        cov_last = cov_n.copy() # save the lastCoV
        # w_n = np.expand_dims(w_n,axis=1)
        wlast = w_n.copy() # save the last MU filter 
        spikes_last = spikes # save the last discharge times
        _ , spikes = get_spikes(w_n,Z,fsamp)
        # determine the interspike interval
        ISI = np.diff(spikes/fsamp)
        # determine the coefficient of variation
        cov_n = np.std(ISI, ddof=1)/np.mean(ISI) #adjusted, to unbiased 
        # update the sepearation vector by summing all the spikes
        w_n = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
        k += 1
    # if you meet the CoV minimisation condition, but with a single-spike-long train, use the updated
    # separation vector and recompute the spikes

    if len(spikes_last) < 2:
        _ , spikes_last = get_spikes(w_n,Z,fsamp) # saves the last discharge times 

    return wlast, cov_n


################################ VALIDATION TOOLS ########################################

# updated silohuette measure on feb 5th 18:20pm

def maxk(signal, k):
    # indirect partial sort on last axis, k = k smallest elements are moved to the left 
    return np.partition(signal, -k, axis=-1)[..., -k:] 

def get_silohuette(w_n,Z,fsamp):

    # Step 4a: 
    source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.02)+1) # this is approx a value of 20, which is in time approx 10ms
    # peaks = detect_peaks(np.squeeze(source_pred), mpd = np.round(fsamp*0.02))
    
    print(np.mean(maxk(source_pred[peaks], 10)))
    source_pred /=  np.mean(maxk(source_pred[peaks], 10)) #normalization of MU pulse train
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2,init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        # indices of the spike and noise clusters (the spike cluster should have a larger value)
        spikes_ind = np.argmax(kmeans.cluster_centers_) 
        noise_ind = np.argmin(kmeans.cluster_centers_)
        # get the points that correspond to each of these clusters
        spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
        noise = peaks[np.where(kmeans.labels_ == noise_ind)]
        # calculate the centroids
        spikes_centroid = kmeans.cluster_centers_[spikes_ind]
        noise_centroid = kmeans.cluster_centers_[noise_ind]
        # difference between the within-cluster sums of point-to-centroid distances for spikes (i.e. spikes spoints to spikes cluster centre)
        intra_sums = (((source_pred[spikes]- spikes_centroid)**2).sum()) 
        # difference between the between-cluster sums of point-to-centroid distance for spikes (i.e. spikes points to noise cluster centre)
        inter_sums = (((source_pred[spikes] - noise_centroid)**2).sum())
        sil = (inter_sums - intra_sums) / max(intra_sums, inter_sums)  

    else:

        sil = 0

    return source_pred, spikes, sil


def peel_off(Z,spikes,fsamp):
    #NOTE: DID NOT CHECK THIS FUNCTION 
    windowl = round(0.05*fsamp)
    waveform = np.zeros([windowl*2+1])
    firings = np.zeros([np.shape(Z)[1]])
    firings[spikes] = 1 # make the firings binary
    EMGtemp = np.empty(Z.shape) # intialise a temporary EMG signal

    for i in range(np.shape(Z)[0]): # iterating through the (extended) channels
        temp = cutMUAP(spikes,windowl,Z[i,:])
        waveform = np.mean(temp,axis=0)
        EMGtemp[i,:] =  scipy.signal.convolve(firings, waveform, mode = 'same',method='auto')

    Z -= EMGtemp; # removing the EMG representation of the source spearation vector from the signal, avoid picking up replicate content in future iterations
    return Z


def get_binary_pulse_trains(discharge_times, ltime):
    mu_count = len(discharge_times)
    binary_pulse_trains = np.zeros([mu_count, ltime])
    for mu_batch_count in range(len(discharge_times)):   
        binary_pulse_trains[mu_batch_count, discharge_times[mu_batch_count]] = 1 
    return binary_pulse_trains
############################## POST PROCESSING #####################################################


def gausswin(M, alpha=2.5):
    #NOTE: DID NOT CHECK THIS FUNCTION
    """ Python equivalent of the in-built gausswin function MATLAB (since there is no open-source Python equivalent) """
    
    n = np.arange(-(M-1) / 2, (M-1) / 2 + 1,dtype=np.longdouble) #was dtype=np.float128
    w = np.exp((-1/2) * (alpha * n / ((M-1) / 2)) ** 2)
    return w

def cutMUAP(MUPulses, length, Y):
    #NOTE: DID NOT CHECK THIS FUNCTION  
    """ Direct converion of MATLAB code in-lab. Extracts consecutive MUAPs out of signal Y and stores
    them row-wise in the out put variable MUAPs.
    Inputs: 
    - MUPulses: Trigger positions (in samples) for rectangualr window used in extraction of MUAPs
    - length: radius of rectangular window (window length = 2*len +1)
    - Y: Single signal channel (raw vector containing a single channel of a recorded signals)
    Outputs:
    - MUAPs: row-wise matrix of extracted MUAPs (algined signal intervals of length 2*len+1)"""
 
    while len(MUPulses) > 0 and MUPulses[-1] + 2 * length > len(Y):
        MUPulses = MUPulses[:-1]

    c = len(MUPulses)
    edge_len = round(length / 2)
    tmp = gausswin(2 * edge_len) # gives the same output as the in-built gausswin function in MATLAB
    # create the filtering window 
    win = np.ones(2 * length + 1)
    win[:edge_len] = tmp[:edge_len]
    win[-edge_len:] = tmp[edge_len:]
    MUAPs = np.empty((c, 1 + 2 * length))
    for k in range(c):
        start = max(MUPulses[k] - length, 1) - (MUPulses[k] - length)
        end = MUPulses[k] + length- min(MUPulses[k] + length, len(Y))
        MUAPs[k, :] = win * np.concatenate((np.zeros(start), Y[max(MUPulses[k] - length, 1):min(MUPulses[k] + length, len(Y))+1], np.zeros(end)))

    return MUAPs

def batch_process_filters(MU_filters, wSIG, plateau_coord, exfactor,diff,ltime,fsamp):

    """ dis_time: the distribution of spiking times for every identified motor unit, but at this point we don't check to see
    whether any of these MUs are repeats"""

    # Pulse trains has shape no_mus x original signal duration
    # dewhitening matrix has shape no_windows x exten chans x exten chans
    # mu filters has size no_windows x exten chans x (maximum of) x no iterations  --> less if iterations failed to reach SIL threshold
    # ltime: size(signal.data,2)
    mu_count = 0
    print(np.shape(wSIG)[0])
    for win in range (np.shape(wSIG)[0]): #amount of windows  
        mu_count += np.size(MU_filters[win][0]) # columns of MU_filters 
    
    pulse_trains = np.zeros([mu_count, ltime]) #rows: MUs, #columns: pulse trains 
    discharge_times = [None] * mu_count
    mu_batch_count = 0

    for win_1 in range(np.shape(wSIG)[0]): # amount of windows 
        for exchan in range(np.shape(MU_filters[win_1])[1]): #amount of columns --> amount of MU filters  
            # below can be deleted (?)
            for win_2 in range(np.shape(wSIG)[0]): #amount of windows   
                #check the differrence 'plateau_coord[win_2*2]:plateau_coord[(win_2+1)*2 - 1]+exfactor-1-diff]
                pulse_trains[mu_batch_count, plateau_coord[win_2*2]:plateau_coord[win_2*2+1]+exfactor-diff] = np.matmul(MU_filters[win_1][:,exchan], wSIG[win_1][:,:])
            # r. 475 (below) not performed in matlab script 
            # pulse_trains[mu_batch_count,:] = pulse_trains[mu_batch_count,:]/ np.max(pulse_trains[mu_batch_count,:]) # normalization? 
            pulse_trains[mu_batch_count,:] = np.multiply(pulse_trains[mu_batch_count,:],abs(pulse_trains[mu_batch_count,:])) #
            # peaks = peakutils.peak.indexes(np.squeeze(pulse_trains[mu_batch_count,:]), min_dist = np.round(fsamp*0.005))                                 
            # scipy.signal.find_peaks_cwt(np.squeeze(pulse_trains[mu_batch_count,:])) 
            peaks = detect_peaks(np.squeeze(pulse_trains[mu_batch_count,:]), mpd = np.round(fsamp*0.005))
            
            pulse_trains[mu_batch_count,:] /=  np.mean(maxk(pulse_trains[mu_batch_count,:], 10))
            
            kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[mu_batch_count,peaks].reshape(-1,1)) # 2D array with 1 column
            spikes_ind = np.argmax(kmeans.cluster_centers_) # Determine highest centroid
            # fix below 
            discharge_times[mu_batch_count] = peaks[np.where(kmeans.labels_ == spikes_ind)]  
            print(f"Batch processing MU#{mu_batch_count} out of {mu_count} MUs")
            mu_batch_count += 1
    pulse_trains = pulse_trains[:, 0:ltime] #hesitate about -1 or not
    return pulse_trains, discharge_times

def xcorr(x,y): # https://stackoverflow.com/questions/25830840/python-cross-correlation
    norm_x = x/np.linalg.norm(x)
    norm_y = y/np.linalg.norm(y)
    corr = scipy.signal.correlate(norm_x, norm_y, mode="full") #coefficient of correlation
    lags = scipy.signal.correlation_lags(len(x), len(y), mode="full") #lags of correlation 
    return lags, corr

def remove_duplicates(MU_filters, pulse_trains, discharge_times, discharge_times_aligned, maxlag, jitter, fsamp, dup_thr): #old remove_duplicates!
    '''
    Input 
    PulseT = Pulse train of each MU
    distime = discharge times of the motor units 
    distime2 = discharge times of the motor units realigned with the MUAP
    maxlag = maximal lag between motor unit spike trains
    jitter = tolerance in sec for the estimation of discharge times
    tol = percentage of shared discharge times to define a duplicate
    fsamp = sampling frequency

    Output:
    Pulsenew = Pulse train of non-duplicated MU
    distimenew = discharge times of non-duplicated MU
    
    '''
    jit = round(jitter * fsamp) #from time to samples 
    spike_trains = np.zeros((np.shape(pulse_trains)[0],np.shape(pulse_trains)[1])) #array same as metlab r. 27 
    
    # Making binary spike trains for each established MU
    #distimmp = [np.empty(shape=[np.shape(pulse_trains)[0], 0])] # initializing a list, len=3
    distimmp = [None]*(np.shape(pulse_trains)[0])
    
    for i in range(np.shape(pulse_trains)[0]): #for each MU 
        spike_trains[i,discharge_times_aligned[i]] = 1 #spike train at discharge times == 1 
        distimmp_i = []
        for j in range(1, jit + 1):   
           distimmp_i = np.concatenate((discharge_times_aligned[i]-j, discharge_times_aligned[i]+j))
           # distimmp_i = [discharge_times_aligned[i]-j, discharge_times_aligned[i]+j] 
        distimmp_i = np.concatenate((distimmp_i, discharge_times_aligned[i]))
        #distimmp[i] = np.append(distimmp[i],[discharge_times_aligned[i]])    
        distimmp[i]=distimmp_i

 
    # for waitbar!
    ''' 
    MUn = len(discharge_times_aligned) #amount of MUs?  
    x = 1/MUn
    ''' 
    # With these binary trains, you can readily identify duplicate MUs
    i = 0 
    
    
    MU_filters_new = np.zeros((np.shape(MU_filters[0])[0],np.shape(MU_filters[0])[1]))
    discharge_times_new = [None] * len(distimmp) #rows with 0's if distimmp is empty!! -> delete 
    pulse_trains_new = np.zeros((len(distimmp), np.shape(pulse_trains)[1])) #rows with 0's! if distimmp is empty!! -> delete 
     # Remove duplicates 
    
    while distimmp:
        temp_discharge_times = [None] * len(distimmp)
        for mu in range(len(distimmp)):
            # Remove lag that may exist between MU 
            lags, corr = xcorr(spike_trains[0,:], spike_trains[mu,:])
            lim_lags = np.where(np.logical_and(lags>=-2*maxlag, lags<=2*maxlag))[0]
            lags = lags[lim_lags]
            corr = corr[lim_lags]
            corr_max = np.max(corr) #corr_max larger than 1? --> use normalized values!
            ind_max = np.argmax(corr)
            if corr_max > 0.2:
                temp_discharge_times[mu] = distimmp[mu] + lags[ind_max]
            else:
                temp_discharge_times[mu] = distimmp[mu]

        # Find common discharge times
        common_discharges = np.zeros(len(pulse_trains)) #check whether np.zeros(len(pulse_trains))
        for mu in range(1,len(pulse_trains)):
            #com = distimmp[1].intersection(temp_discharge_times[mu]) #check if intersect 1d 
            com = np.intersect1d(distimmp[0], temp_discharge_times[mu])
            # com([false,diff(com) == 1]) = [];
            idx = np.append(False, np.diff(com) == 1)
            if com.size != 0: 
                com = com[~idx]
            #concatenate false at beginning of logical array, removes array where diff=1  
            common_discharges[mu] = len(com)/max(len(discharge_times[0]), len(discharge_times[mu])) #check sizes 
        
        # Flag duplicates and keep the MU with the lostest CoV of ISI 
        duplicates = np.where(common_discharges >= dup_thr) #np array +1 --> otherwise it does not work  
         
        duplicates = np.append(0, duplicates) #as com with the 1st MU! 
        
        CoV = np.zeros(len(duplicates)) # initialize covariance  
        for j in range(len(duplicates)):
            ISI = np.diff(discharge_times[duplicates[j]])
            CoV[j] = np.std(ISI, ddof=1) / np.mean(ISI) #adjusted to unbiased!
        survivor = np.argmin(CoV) # find index of minimum value of array CoV 
        
        # Delete duplicates and save the surviving MU 
        MU_filters_new[:, i] = MU_filters[0][:, duplicates[survivor]].copy()
        discharge_times_new[i] = discharge_times[duplicates[survivor]].copy() #check if this works if not 
        pulse_trains_new[i] = pulse_trains[duplicates[survivor]].copy() #other pulse trains are 0! 
        
        # Update firings and discharge times -> check this!!
        for j in range(len(duplicates)):
            discharge_times[duplicates[j]] = [] #type error? 
            discharge_times_aligned[duplicates[j]] = [] #type error? 
            distimmp[duplicates[j]] = [] 
        
        discharge_times = list(filter(lambda x: len(x) > 0, discharge_times)) #None, discharge_times) #[arr.tolist() for arr in original_list if arr.sizee for ele in discharge_times if ele !=[]]
        discharge_times_aligned = list(filter(lambda x: len(x) > 0, discharge_times_aligned))
        distimmp = list(filter(lambda x: len(x) > 0, distimmp))
        """
        discharge_times = discharge_times[discharge_times != []]
        discharge_times_aligned = discharge_times_aligned[discharge_times_aligned != []]
        distimmp = distimmp[distimmp != []]      
        """
        
        spike_trains = np.delete(spike_trains, duplicates, axis=0)
        pulse_trains = np.delete(pulse_trains, duplicates, axis=0) 
        #pulse_trains[duplicates,:] = []
        print(f"{len(discharge_times)} Remaining MUs to check")
        
        i += 1
        
    discharge_times_new = list(filter(lambda x: x is not None, discharge_times_new))   
    pulse_trains_new = pulse_trains_new[~np.all(pulse_trains_new == 0, axis=1)] #check if this does not work!! 
    MU_filters_new = MU_filters_new[:, ~np.all(MU_filters_new == 0, axis=0)]
    return pulse_trains_new, discharge_times_new, MU_filters_new              
            
def remove_outliers(pulse_trains, discharge_times, fsamp, max_its = 30):
    #NOTE: DID NOT CHECK THIS FUNCTION
    
    for mu in range(np.shape(discharge_times)[0]):
        its = 0 
        # isn't this quite a coarse way of calculating firing rate? i.e. without any smoothing?
        discharge_rates = 1/(np.diff(discharge_times[mu]) / fsamp)
        while (np.std(discharge_rates)/np.mean(discharge_rates)) > 0.4 and its < max_its: #optional: add ddof 

            artifact_limit = np.mean(discharge_rates) + 3*np.std(discharge_rates)
            # identify the indices for which this limit is exceeded
            artifact_inds = np.squeeze(np.argwhere(discharge_rates > artifact_limit))
            if len(artifact_inds) > 0:

                # vectorising the comparisons between the numerator terms used to calculate the rate, for indices at rate artifacts
                diff_artifact_comp = pulse_trains[mu,discharge_times[mu][artifact_inds]] < pulse_trains[mu, discharge_times[mu][artifact_inds + 1]]
                # 0 means discharge_times[mu][artifact_inds]] was less, 1 means discharge_times[mu][artifact_inds]] was more
                less_or_more = np.argmax([diff_artifact_comp, ~diff_artifact_comp], axis=0)
                discharge_times[mu] = np.delete(discharge_times[mu], artifact_inds + less_or_more)
                its += 1
        discharge_rates = 1/(np.diff(discharge_times[mu]) / fsamp)
    
    return discharge_times


def refine_mus(signal,signal_mask, pulse_trains_n_1, discharge_times_n_1):
    # TODO: not checked
    """ signal is no_chans x time, where no_chans is the total for one grid
        signal_mask is the channels to discard
    
    signal.data(i*64-63:i*64,:), signal.EMGmask{i}, PulseT, distimenew);"""

    print("Refining MU pulse trains...")
    signal = [x for i, x in enumerate(signal) if signal_mask[i] != 1]
    nbextchan = 1500
    extension_factor = round(nbextchan/np.shape(signal)[0])
    extend_obvs = np.zeros([np.shape(signal)[0]*(extension_factor), np.shape(signal)[0] + extension_factor -1 ])
    extend_obvs = extend_emg(extend_obvs,signal,extension_factor)
    re_obvs = np.matmul(extend_obvs, extend_obvs.T)/np.shape(extend_obvs)[1]
    invre_obvs = np.linalg.pinv(re_obvs)
    pulse_trains_n = np.zeros([np.shape(pulse_trains_n_1)[0], np.shape(pulse_trains_n_1)[1]])
    discharge_times_n = [None] * len(pulse_trains_n_1)

    # recalculating the mu filters

    for mu in range(len(pulse_trains_n_1)):

        mu_filters = np.sum(extend_obvs[:,discharge_times_n_1[mu]],axis=1)
        IPTtmp = np.dot(np.dot(mu_filters.T,invre_obvs),extend_obvs)
        pulse_trains_n[mu,:] = IPTtmp[:np.shape(signal)[1]]

        pulse_trains_n[mu,:] = pulse_trains_n[mu,:]/ np.max(pulse_trains_n[mu,:])
        pulse_trains_n[mu,:] = np.multiply( pulse_trains_n[mu,:],abs(pulse_trains_n[mu,:])) 
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains_n[mu,:]))  # why no distance threshold anymore?
        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains_n[mu,peaks].reshape(-1,1)) 
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        discharge_times_n[mu] = peaks[np.where(kmeans.labels_ == spikes_ind)] 

   
    print(f"Refined {len(pulse_trains_n_1)} MUs")

    return discharge_times_n

def sort_raw_emg(rawemg, grid_format, fsamp, emgtype):
    # from openHDEMG https://github.com/GiacomoValliPhD/openhdemg/blob/main/openhdemg/library/electrodes.py
    # necessary to look at the xcorr of the STA of the MAUPs of the MUs 
    if grid_format == '4-8-L':
        sorting_order = [
            7, 6, 5, 4, 3, 2, 1, 0,
            13, 14, 15, 12, 11, 10, 9, 8,
            18, 17, 16, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        ]
        # n_rows = 8
        # n_cols = 4 

    if grid_format == '8-8-L':
        sorting_order = [
            16, 15, 14, 13, 12, 8, 4, 0,
            21, 20, 19, 18, 17, 9, 5, 1,
            26, 25, 24, 23, 22, 10, 6, 2,
            31, 30, 29, 28, 27, 11, 7, 3,
            32, 33, 34, 35, 36, 52, 56, 60,
            37, 38, 39, 40, 41, 53, 57, 61,
            42, 43, 44, 45, 46, 54, 58, 62,
            47, 48, 49, 50, 51, 55, 59, 63,
        ]
    # rawemg = notch_filter(rawemg, fsamp)
    
    # rawemg = bandpass_filter(rawemg,fsamp, emg_type = emgtype)  

    sorted_rawemg = rawemg[sorting_order, :]
    """
    rawemg = pd.DataFrame(rawemg).T
    sorted_rawemg = rawemg.reindex(columns=sorting_order) #reordering columns of rawemg file based on sorting order 
    sorted_rawemg.columns = range(sorted_rawemg.columns.size)
    """
    """
    empty_dict = {f"col{n}": None for n in range(n_cols)}
    for pos, col in enumerate(empty_dict.keys()):
        empty_dict[col] = sorted_rawemg.iloc[:, n_rows*pos:n_rows*(pos+1)]
    sorted_rawemg = empty_dict
    """
    return sorted_rawemg

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None, title=True):
    # from https://github.com/demotu/BMC/blob/master/functions/detect_peaks.py
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    title : bool or string, optional (default = True)
        if True, show standard title. If False or empty string, doesn't show
        any title. If string, shows string as title.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        # _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

    return ind

def delete_begin_peaks(peaks):
    """ 
    Function that deletes indices of peaks at the beginning of the pulse trains. 
    This is the result of applying the filters over the begin of the extended data where zeros are located.
    """
    peaks = peaks[np.where(peaks > 50)]
    return peaks

def post_process_begin(pulse_trains, discharge_times, binary_dis_times):
    """ 
    Function that deletes peaks at the beginning of the pulse trains. 
    This is the result of applying the filters over the begin of the extended data where zeros are located.
    """
    binary_dis_times[:,:50] = np.zeros((np.shape(binary_dis_times)[0], 50))

    for i in range(len(discharge_times)):
            if len(discharge_times[i]) > 10:
                for j in range(len(discharge_times[i])-2):
                    if discharge_times[i][j] < 50: 
                        del discharge_times[i][j]    

    pulse_trains[:,:50] = np.zeros((np.shape(pulse_trains)[0], 50))
    return pulse_trains, discharge_times, binary_dis_times
###############################################################################################################
################################ ADDITIONAL REAL-TIME DECOMPOSITION TOOLS #####################################
###############################################################################################################

#TODO: did not check these tools!

def get_pulse_trains(data, rejected_channels, mu_filters, chans_per_electrode, fsamp,g=0):
    """
    Function that retrieves pulse trains and discharge times based on K-means clustering
    First, all the bad channels are deleted. Then the data is extended.
    Motor unit filters are applied to the data and peaks are detected. 
    Peaks are classified in spikes or noise by K-means clustering.
    Pulse trains and discharge times are saved.
    """ 
    # channel rejection again, but on the PRE-FILTERED data
    # OR: if filtering was not used in the pre processing, the batched data could be used (?)
    data_slice = data[chans_per_electrode[g]*(g):(g+1)* chans_per_electrode[g],:] # will need to be generalised
    rejected_channels_slice = rejected_channels == 1
    cleaned_data = np.delete(data_slice, rejected_channels_slice, 0)

    # get the first estimate of pulse trains using the previously derived mu filters, applied to the emg data
    ext_factor = int(np.round(1000/np.shape(cleaned_data)[0]))
    extended_data = np.zeros([np.shape(cleaned_data)[0]*(ext_factor), np.shape(cleaned_data)[1] + ext_factor -1]) # no differential mode used here (?)
    extended_data =  extend_emg(extended_data,cleaned_data,ext_factor)
    # get the real and inverted versions
    sq_extended_data = np.dot(extended_data, extended_data.T)/np.shape(extended_data)[1]
    inv_extended_data = np.linalg.pinv(sq_extended_data)
    
    # initialisations for extracting pulse trains in clustering
    mu_count =  np.shape(mu_filters)[1]
    pulse_trains = np.zeros([mu_count, np.shape(data)[1]]) 
    discharge_times = [None] * mu_count # do not know size yet, so can only predefine as a list
    for mu in range(mu_count):
        mu_filter = mu_filters[:,mu].reshape(np.shape(mu_filters)[0],)
        # Calculate the mu-th pulse train
        pulse_temp =  (mu_filter.T @ inv_extended_data) @ extended_data 
        # step 4a 
        pulse_trains[mu,:] = pulse_temp[:np.shape(data)[1]]
        # source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
        pulse_trains[mu,:] = np.multiply(pulse_trains[mu,:],abs(pulse_trains[mu,:])) # keep the negatives 
        # Step 4b:
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[mu,:]), distance = np.round(fsamp*0.02)+1) # peaks variable holds the indices of all peaks

        #delete any artefacts at the beginning of the file.
        peaks = delete_begin_peaks(peaks)
        pulse_trains[mu,:] /=  np.mean(maxk(pulse_trains[mu,peaks], 10))
        
        if len(peaks) > 1:
            # If peaks are detected, k-means clustering is used to identify spikes and noise.
            kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[mu,peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
            spikes_ind = np.argmax(kmeans.cluster_centers_)
            spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
            # remove outliers from the spikes cluster with a std-based threshold
            discharge_times[mu] = spikes[pulse_trains[mu,spikes] <= np.mean(pulse_trains[mu,spikes]) + 3*np.std(pulse_trains[mu,spikes])]
        else:
            discharge_times[mu] = peaks
    return pulse_trains, discharge_times, ext_factor


def get_mu_filters(data,rejected_channels, discharge_times, chans_per_electrode, g=0):
    """
    Function that retrieves mu filters based on data and discharge times.
    First, all the bad channels are deleted. Then the data is extended.
    Motor unit filters are applied to the data and peaks are detected. 
    Mu filters are calculated as the sum of the values at discharge times.
    Based on the CKC method.
    """ 
    # channel rejection again, but on the PRE-FILTERED data
    # OR: if filtering was not used in the pre processing, the batched data could be used (?)
    data_slice = data[chans_per_electrode[g]*(g):(g+1)* chans_per_electrode[g],:] # will need to be generalised
    rejected_channels_slice = rejected_channels == 1
    cleaned_data = np.delete(data_slice, rejected_channels_slice, 0)

    # get the first estimate of pulse trains using the previously derived mu filters, applied to the emg data
    ext_factor = int(np.round(1000/np.shape(cleaned_data)[0]))
    extended_data = np.zeros([np.shape(cleaned_data)[0]*(ext_factor), np.shape(cleaned_data)[1] + ext_factor -1]) # no differential mode used here (?)
    extended_data =  extend_emg(extended_data,cleaned_data,ext_factor)
    # recalculate MU filters

    mu_filters = np.zeros((np.shape(extended_data)[0],len(discharge_times))) # need to check that np.shape(discharge_times)[1] is equiv to no. motor units
    for mu in range(len(discharge_times)):
        # mu filters are the sum of the values at discharge times (CKC method)
        mu_filters[:,mu] = np.sum(extended_data[:,discharge_times[mu]],axis=1)
        
    return mu_filters

def get_online_parameters(data, rejected_channels, mu_filters, chans_per_electrode, fsamp,g=0):
    """
    Function that retrieves mu filters based on data and discharge times.
    First, all the bad channels are deleted. Then the data is extended.
    Motor unit filters are applied to the data and peaks are detected. 
    Peaks are classified in spikes or noise by K-means clustering.
    Pulse trains and discharge times are saved.
    Norm, i.e. mean value of the 10 highest spikes, is used for normalisation of the pulse trains.
    Centroids are calculated for normalized data. 
    """
    # channel rejection again, but on the PRE-FILTERED data
    # OR: if filtering was not used in the pre processing, the batched data could be used (?)
    data_slice = data[chans_per_electrode[g]*(g):(g+1)* chans_per_electrode[g],:] # will need to be generalised
    rejected_channels_slice = rejected_channels == 1
    cleaned_data = np.delete(data_slice, rejected_channels_slice, 0)

    # get the first estimate of pulse trains using the previously derived mu filters, applied to the emg data
    ext_factor = int(np.round(1000/np.shape(cleaned_data)[0]))
    extended_data = np.zeros([np.shape(cleaned_data)[0]*(ext_factor), np.shape(cleaned_data)[1] + ext_factor -1]) # no differential mode used here (?)
    extended_data =  extend_emg(extended_data,cleaned_data,ext_factor)

    sq_extended_data = np.dot(extended_data, extended_data.T)/np.shape(extended_data)[1]
    inv_extended_data = np.linalg.pinv(sq_extended_data)
    
    # initialisations for extracting pulse trains and centroids in clustering
    mu_count =  np.shape(mu_filters)[1]
    pulse_trains = np.zeros((mu_count, np.shape(data)[1])) 
    discharge_times = [None] * mu_count # do not know size yet, so can only predefine as a list
    norm = np.zeros(mu_count)
    centroids = np.zeros((mu_count,2)) # first column is the spike centroids, second column is the noise centroids
    for mu in range(mu_count):
        mu_filter = mu_filters[:,mu].reshape(np.shape(mu_filters)[0],)
        # Calculate the mu-th pulse train
        pulse_temp = (mu_filter.T @ inv_extended_data) @ extended_data #CKC method
        # step 4a 
        pulse_trains[mu,:] = pulse_temp[:np.shape(data)[1]]
        # source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
        pulse_trains[mu,:] = np.multiply(pulse_trains[mu,:],abs(pulse_trains[mu,:])) # keep the negatives 
        # Step 4b:
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[mu,:]), distance = np.round(fsamp*0.02)+1) # peaks variable holds the indices of all peaks
        #delete any artefacts at the beginning of the file.
        peaks = delete_begin_peaks(peaks)
        if len(peaks) > 10:
            kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[mu,peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
            # need both spikes and noise to determine cluster centres for the online decomposition
            # spikes
            spikes_ind = np.argmax(kmeans.cluster_centers_)
            spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
            spikes = spikes[pulse_trains[mu,spikes] <= np.mean(pulse_trains[mu,spikes]) + 3*np.std(pulse_trains[mu,spikes])]
            # noise
            noise_ind = np.argmin(kmeans.cluster_centers_)
            noise = peaks[np.where(kmeans.labels_ == noise_ind)]
            # normalize the pulse trains
            norm[mu] = np.mean(maxk(pulse_trains[mu,spikes], 10))
            pulse_trains[mu,:] /= norm[mu]
            # calculate centroids
            centroids[mu,0] = KMeans(n_clusters = 1, init = 'k-means++',n_init = 1).fit(pulse_trains[mu,spikes].reshape(-1,1)).cluster_centers_
            centroids[mu,1] = KMeans(n_clusters = 1, init = 'k-means++',n_init = 1).fit(pulse_trains[mu,noise].reshape(-1,1)).cluster_centers_
    return ext_factor, inv_extended_data, norm, centroids

def getspikesonline(EMGtmp, extensionfactor, extend2, MUfilters, norm, centroids, fsamp):
    """
    Bad channels have already been deleted in the data.
    Data is extended. Extend2 contains the last columns of the previous block and is added to the data, zo delete any zeros.
    Mu filters are applied to the data and pulse trains are retrieved.
    Pulse trains are normalized based on saved norm of training measurement.
    Peaks are detected and classified based on shortest Euclidian distance to the saved centroids of training measurement.
    Pulse trains, discharge times and binary discharge trains are given back.
    """
    # get the first estimate of pulse trains using the previously derived mu filters, applied to the emg data
    ext_factor = int(np.round(1000/np.shape(EMGtmp)[0]))
    extended_data = np.zeros([np.shape(EMGtmp)[0]*(extensionfactor), np.shape(EMGtmp)[1] + extensionfactor -1]) # no differential mode used here (?)
    extended_data =  extend_emg(extended_data,EMGtmp,ext_factor)
    # Use extend2 from the previous block to fill zeros at the beginning of the extended data
    # select the same amount of columns as the data block
    extend1 = extended_data[:, :np.shape(EMGtmp)[1]]
    # fill zeros at the begin of the extended data with data from previous block (extend2)
    extend1[:,:extensionfactor-1] = extend1[:,:extensionfactor-1] + extend2
    # retrieve the extend2 data for next block
    extend2 = extended_data[:, np.shape(EMGtmp)[1]:]

    mu_count = np.shape(MUfilters)[1]
    pulse_trains = np.zeros((mu_count, np.shape(EMGtmp)[1]))
    distimes_binary = np.zeros((mu_count, np.shape(EMGtmp)[1]))
    distimes = [None] * mu_count
    for mu in range(mu_count):
        mu_filter = MUfilters[:,mu].reshape(np.shape(MUfilters)[0],)
        pulse_temp = mu_filter.T @ extend1 
        # step 4a 
        pulse_trains[mu,:] = pulse_temp
        # source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
        pulse_trains[mu,:] = np.multiply(pulse_trains[mu,:], abs(pulse_trains[mu,:]))  # keep the negatives 
        # Step 4b:
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[mu,:]), distance = np.round(fsamp*0.02)+1) # peaks variable holds the indices of all peaks
        #Normalize data 
        pulse_trains[mu,:] /= norm[mu]
        boolean_array = np.full(np.shape(pulse_trains)[1], False)  # Initialize with False
        boolean_array[peaks] = True  # Set peaks indices to True
        #classify peaks based on saved centroids
        distimes_binary[mu,:] = (abs(pulse_trains[mu,:] * boolean_array - centroids[mu,1]) > abs(pulse_trains[mu,:] * boolean_array -  centroids[mu,0]))
        distimes[mu] = np.where(distimes_binary[mu,:] == 1)[0].tolist()
    return pulse_trains, distimes_binary, distimes, extend2

def extend_and_clip_emg_online(exandclip_template, packet2extend, ext_factor, buffer4fill):

    """ Extension of EMG signals, for a given window, and a given grid. For extension, R-1 versions of the original data are stacked, with R-1 timeshifts.
    Structure: [channel1(k), channel2(k),..., channelm(k); channel1(k-1), channel2(k-1),...,channelm(k-1);...;channel1(k - (R-1)),channel2(k-(R-1)), channelm(k-(R-1))] """
    
    nchans, nobvs = np.shape(packet2extend) 
    for i in range(1,ext_factor):

        exandclip_template[nchans*i:nchans*(i+1),i:] = packet2extend[:-i] # clip on RHS
        exandclip_template[nchans*i:nchans*(i+1),:i] = buffer4fill[:,-nobvs-i:-nobvs]
   
    return exandclip_template


def get_trains_online(Z,sep_matrix):

    return (sep_matrix.T @ Z).real

def euc_distance(points,single_point):

    dist = (points - single_point)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    return dist

def knn_online(trains, fsamp, cluster_centers):

    spike_ind = np.argmax(cluster_centers)
    spike_cluster = cluster_centers[spike_ind]
    noise_cluster = cluster_centers[1-spike_ind]
    # get the Euclidean distance between the trains and the cluster centers
    # data2spikes = euc_distance(peaks,spike_cluster)
    # data2noise = euc_distance(peaks, noise_cluster)
    # cluster centers is assumed to be an array : spike_cluster_center, noise_cluster_center
    
    discharge_times = 1


    return discharge_times

def emg_from_json(filepath):
    """
    Load the emgfile or emg_refsig stored in json format.

    Parameters
    ----------
    filepath : str or Path
        The directory and the name of the file to load (including file
        extension .json).
        This can be a simple string, the use of Path is not necessary.

    Returns
    -------
    emgfile : dict
        The dictionary containing the emgfile.

    See also
    --------
    - save_json_emgfile : Save the emgfile or emg_refsig as a JSON file.
    - askopenfile : Select and open files with a GUI.

    Notes
    -----
    The returned file is called ``emgfile`` for convention
    (or ``emg_refsig`` if SOURCE in ["OTB_REFSIG", "CUSTOMCSV_REFSIG", "DELSYS_REFSIG"]).

    Examples
    --------
    For an extended explanation of the imported emgfile use:

    >>> import openhdemg.library as emg
    >>> emgfile = emg.emg_from_json(filepath="path/filename.json")
    >>> info = emg.info()
    >>> info.data(emgfile)
    """

    # Read and decompress json file
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        jsonemgfile = json.load(f)

    """
    print(type(jsonemgfile))
    <class 'dict'>
    """

    # Access the dictionaries and extract the data.
    source = json.loads(jsonemgfile["SOURCE"])
    filename = json.loads(jsonemgfile["FILENAME"])

    if source in ["DEMUSE", "OTB", "CUSTOMCSV", "DELSYS"]:
        # RAW_SIGNAL
        # df are stored in json as a dictionary, it can be directly extracted
        # and converted into a pd.DataFrame.
        # index and columns are imported as str, we need to convert it to int.
        raw_signal = pd.read_json(jsonemgfile["RAW_SIGNAL"], orient='split')
        # Check dtypes for safety, little computational cost
        raw_signal.columns = raw_signal.columns.astype(int)
        raw_signal.index = raw_signal.index.astype(int)
        raw_signal.sort_index(inplace=True)
        # REF_SIGNAL
        ref_signal = pd.read_json(jsonemgfile["REF_SIGNAL"], orient='split')
        ref_signal.columns = ref_signal.columns.astype(int)
        ref_signal.index = ref_signal.index.astype(int)
        ref_signal.sort_index(inplace=True)
        # ACCURACY
        accuracy = pd.read_json(jsonemgfile["ACCURACY"], orient='split')
        try:
            accuracy.columns = accuracy.columns.astype(int)
        except Exception:
            accuracy.columns = [*range(len(accuracy.columns))]
            warnings.warn(
                "Error while loading accuracy, check or recalculate accuracy"
            )
            # TODO error occurring when accuracy was recalculated on empty MUs.
            # Check if the error is present also for other params.
        accuracy.index = accuracy.index.astype(int)
        accuracy.sort_index(inplace=True)
        # IPTS
        ipts = pd.read_json(jsonemgfile["IPTS"], orient='split')
        ipts.columns = ipts.columns.astype(int)
        ipts.index = ipts.index.astype(int)
        ipts.sort_index(inplace=True)
        # MUPULSES
        # It is s list of lists but has to be converted in a list of ndarrays.
        mupulses = []
    
        mupulses = json.loads(jsonemgfile["MUPULSES"])
        for num, element in enumerate(mupulses):
            mupulses[num] = np.array(element)

        if "MU_filters" in jsonemgfile:
            MU_filters = np.array(json.loads(jsonemgfile["MU_filters"]))
        else:
            MU_filters = np.array([])

   
        # FSAMP
        # Make sure to convert it to float
        fsamp = float(json.loads(jsonemgfile["FSAMP"]))
        # IED
        ied = float(json.loads(jsonemgfile["IED"]))
        # EMG_LENGTH
        # Make sure to convert it to int
        emg_length = int(json.loads(jsonemgfile["EMG_LENGTH"]))
        # NUMBER_OF_MUS
        number_of_mus = int(json.loads(jsonemgfile["NUMBER_OF_MUS"]))
        # BINARY_MUS_FIRING
        binary_mus_firing = pd.read_json(
            jsonemgfile["BINARY_MUS_FIRING"],
            orient='split',
        )
        binary_mus_firing.columns = binary_mus_firing.columns.astype(int)
        binary_mus_firing.index = binary_mus_firing.index.astype(int)
        binary_mus_firing.sort_index(inplace=True)
        # EXTRAS
        # Don't alter index and columns as these could contain anything.
        extras = pd.read_json(jsonemgfile["EXTRAS"], orient='split')

        emgfile = {
            "SOURCE": source,
            "FILENAME": filename,
            "RAW_SIGNAL": raw_signal,
            "REF_SIGNAL": ref_signal,
            "ACCURACY": accuracy,
            "IPTS": ipts,
            "MUPULSES": mupulses,
            "FSAMP": fsamp,
            "IED": ied,
            "EMG_LENGTH": emg_length,
            "NUMBER_OF_MUS": number_of_mus,
            "BINARY_MUS_FIRING": binary_mus_firing,
            "EXTRAS": extras,
            "MU_filters": MU_filters,
        }

    elif source in ["OTB_REFSIG", "CUSTOMCSV_REFSIG", "DELSYS_REFSIG"]:
        # FSAMP
        fsamp = float(json.loads(jsonemgfile["FSAMP"]))
        # REF_SIGNAL
        ref_signal = pd.read_json(jsonemgfile["REF_SIGNAL"], orient='split')
        ref_signal.columns = ref_signal.columns.astype(int)
        ref_signal.index = ref_signal.index.astype(int)
        ref_signal.sort_index(inplace=True)
        # EXTRAS
        extras = pd.read_json(jsonemgfile["EXTRAS"], orient='split')

        emgfile = {
            "SOURCE": source,
            "FILENAME": filename,
            "FSAMP": fsamp,
            "REF_SIGNAL": ref_signal,
            "EXTRAS": extras,
        }

    else:
        raise Exception("\nFile source not recognised\n")

    return emgfile

def convert_channel_names_to_indices(channel_names, ElChannelMap):
        """
        Convert list of channel names (e.g., ['R1C8', 'R2C1']) to integers based on ElChannelMap.

        Parameters
        ----------
        channel_names : list of str
            List of channel names formatted as 'R1C8', 'R2C1', etc.
        ElChannelMap : list of list of int
            2D list that maps row-column to integer values.

        Returns
        -------
        list of int
            List of integers corresponding to the channel names.
        """
        channel_indices = []

        for ch_name in channel_names:
            # Extract the row and column numbers from the channel name
            row = int(ch_name[1]) - 1  # Convert R1 -> row index 0
            col = int(ch_name[3])   # Convert C1 -> column index 0
            
            # Get the corresponding index from the ElChannelMap
            channel_index = ElChannelMap[-col][row]
            channel_indices.append(channel_index)

        return channel_indices