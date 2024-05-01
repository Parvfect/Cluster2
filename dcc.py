
import utils
import os
import galois
import coding_matrices as r
import math
import numpy as np
from itertools import combinations
import datetime
import sys
from tqdm import tqdm
from protograph.protograph_interface import get_Harr_sc_ldpc, get_dv_dc
from tanner import VariableTannerGraph
import random
import matplotlib.pyplot as plt
from pstats import Stats
import re
from cProfile import Profile
from cc import get_parameters, get_parameters_sc_ldpc
from tanner_qspa import TannerQSPA, get_max_symbol
from more_itertools import distinct_permutations
import uuid

def hw_likelihoods(k_motifs, codeword_noise, eps, threshold=1e10):

    n_motifs = len(codeword_noise)
    R = sum(codeword_noise)

    prob_base = eps / n_motifs
    prob_high = (1 - eps) / k_motifs

    alphabet = distinct_permutations(
        [0] * (n_motifs - k_motifs) + [1] * k_motifs,
        r=n_motifs
    )
    log_likelihoods = []
    for symbol in alphabet:
        reads_high = sum(np.array(symbol) * codeword_noise)
        reads_base = R - reads_high

        # Subtract maximum common likelihood
        m = min(codeword_noise)
        reads_high = reads_high - m * k_motifs
        reads_base = reads_base - m * (n_motifs - k_motifs)
        
        # Compute log-likelihood
        log_likelihoods.append(
            reads_base * math.log2(prob_base)
            + reads_high * math.log2(prob_base + prob_high)
        )

    idx_original = np.argsort(np.argsort(log_likelihoods))
    log_likelihoods = np.sort(log_likelihoods)

    i = 0
    max_idx_zero = 0
    threshold = math.log2(threshold)
    while i < len(log_likelihoods) - 1:
        if log_likelihoods[i + 1] - log_likelihoods[i] > threshold:
            max_idx_zero = i + 1
        i += 1

    log_likelihoods = log_likelihoods - log_likelihoods[max_idx_zero]
    likelihoods = 2 ** log_likelihoods
    likelihoods[0:max_idx_zero] = 0

    likelihoods = likelihoods[idx_original]
    likelihoods = np.flip(likelihoods)

    return list(likelihoods / sum(likelihoods))

def choose_symbols(n_motifs, picks):
    """Creates Symbol Array as a combination of Motifs
    
    Args: 
        n_motifs (int): Total Number of Motifs
        picks (int): Number of Motifs per Symbol
    Returns: 
        symbols (list): List of all the Symbols as motif combinations
    """

    # Reference Motif Address starts from 1 not 0
    return [list(i) for i in (combinations(np.arange(1, n_motifs+1), picks))]

def distracted_coupon_collector_channel(symbol, R, P, n_motifs):
    """Model of the Distracted Coupon Collector Channel. Flips a coin, if the probability is within interference, randomly attach a motif from the set of all motifs. Otherwise randomly select from the set of motifs for the passed symbol
    
    Args: 
        symbol (list) : List of motifs as a symbol
        R (int): Read Length
        P (float): Probability of Interference 
        n_motifs (int): Number of motifs in Total
    
    Returns: 
        reads (list) : List of Reads for the Symbol
    """

    reads = []
    for i in range(R):
        if random.random() > P:
            reads.append(random.choice(symbol))
        else:
            reads.append(random.randint(1, n_motifs))    
    return reads


def get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, zero_codeword=False, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)
    
    # Popping last three symbols
    symbols.pop()
    symbols.pop()
    symbols.pop()
        
    symbol_keys = np.arange(0, ffdim)

    #graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph = TannerQSPA(dv, dc, k, n, ffdim=ffdim)

    if Harr is None:
        Harr = r.get_H_arr(dv, dc, k, n)
    
    H = r.get_H_Matrix(dv, dc, k, n, Harr)
        #print(H)

    if zero_codeword:
        G = np.zeros([k,n], dtype=int)
    else:
        G = r.parity_to_generator(H, ffdim=ffdim)
    
    graph.establish_connections(Harr)
    
    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    input_arr = [random.choice(symbol_keys) for i in range(k)]

    # Encode the input array
    C = np.dot(input_arr, G) % ffdim

    # Check if codeword is valid
    if np.any(np.dot(C, H.T) % ffdim != 0):
        print("Codeword is not valid, aborting simulation")
        exit()

    return H, G, graph, C, symbols, motifs

def get_parameters_sc_ldpc(n_motifs, n_picks, L, M, dv, dc, k, n, ffdim, zero_codeword=False, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)
    
    # Popping last three symbols
    symbols.pop()
    symbols.pop()
    symbols.pop()
    

    symbol_keys = np.arange(0, ffdim)
    
    if Harr is None:
        Harr, dv, dc, k, n = get_Harr_sc_ldpc(L, M, dv, dc)   
    else:
        dv, dc = get_dv_dc(dv, dc, k, n, Harr)
    
    #graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph = TannerQSPA(dv, dc, k, n, ffdim=ffdim)
    graph.establish_connections(Harr)

    if H is None and G is None:
        H = r.get_H_matrix_sclpdc(dv, dc, k, n, Harr)
        if zero_codeword:
            G = np.zeros([k,n], dtype=int)
        else:
            G = r.parity_to_generator(H, ffdim=ffdim)

    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    input_arr = [random.choice(symbol_keys) for i in range(k)]
    
    C = np.dot(input_arr, G) % ffdim

    # Check if codeword is valid
    if np.any(np.dot(C, H.T) % ffdim != 0):
        print("Codeword is not valid, aborting simulation")
        exit()

    return k, n, H, G, graph, C, symbols, motifs


def get_symbol_likelihood(n_motifs, n_picks, ffdim, motif_occurences, P, pop=True):
    """Generates Likelihood Array for a Symbol after it's passed through the channel, using the number of times each motif is encountered

        Args:
            n_picks (int): Number of motifs per symbol
            motif_occurences (array) (n_motifs,): Array of Occurence of Each Motif Encountered [0,0,1,1,2,3,0] 
            P (float): Interference Probability
        Returns:
            likelihoods: array (n_motifs choose k_motifs, ) - Normalized likelihood for each symbol (in lexicographical order).
    """

    # Getting the Likelihoods from Alberto's Likelihood Generator
    likelihoods = hw_likelihoods(n_picks, motif_occurences, P)

    # Popping the last three symbols
    likelihoods.pop()
    likelihoods.pop()
    likelihoods.pop()

    if sum(likelihoods) == 0: # Prevent divide by zero
        likelihoods = list(np.ones(67)/67)
    else:
        norm_factor = 1/sum(likelihoods)
        likelihoods = [norm_factor*i for i in likelihoods]
        
    # Precision - summing up to 0.9999999999999997
    assert sum(likelihoods) >= 0.99 and sum(likelihoods) < 1.01

    return likelihoods

def simulate_reads(C, symbols, read_length, P, n_motifs, n_picks):
    """Simulates reads using the QSPA Decoder
        Args:
            C (list) (n,): Codeword
            read_length (int): Read Length
            P (Float): Interference Probability
            n_motifs (int): Number of Motifs in Total
            n_picks (int): Number of Motifs Per Symbol
        Returns: 
            reads (list) : [length of Codeword, no. of symbols] list of all the reads as likelihoods
    """

    
    likelihood_arr = []
    for i in C:
        motif_occurences = np.zeros(n_motifs)
        reads = distracted_coupon_collector_channel(symbols[i], read_length, P, n_motifs)

        # Collecting Motifs Encountered
        for i in reads:
            motif_occurences[i-1] += 1

        symbol_likelihoods = get_symbol_likelihood(n_motifs, n_picks, ffdim, motif_occurences, P)
        likelihood_arr.append(symbol_likelihoods)
    

    #motif_occurrences = np.array([[2,0,0,0], [0,1,1,0], [1,1,0,0]])
    #likelihood_arr = np.array([get_symbol_likelihood(n_picks, i, P) for i in motif_occurrences])
        
    return likelihood_arr


def unmask_reordering(symbol_likelihood_arr, mask, ffdim):
    """Unmasks the Codeword after generating likelihood arrays post channel transmission"""

    unmasked_likelihood_arr = np.zeros((len(mask), ffdim))

    for j in range(len(mask)):
        for i in range(ffdim):
            unmasked_likelihood_arr[j,i] = symbol_likelihood_arr[j,(i+mask[j]) % ffdim]
        
    return unmasked_likelihood_arr


def decoding_errors_fer(k, n, dv, dc, ffdim, P, H, G, GF, graph, C, symbols, n_motifs, n_picks, decoder=None, masked=False, decoding_failures_parameter=20, max_iterations=20, iterations=500, uncoded=False, bec_decoder=False, label=None, code_class="", read_lengths=np.arange(1,20)):

    decoding_failures_parameter = max_iterations # Change this for long compute

    frame_error_rate = []
    symbol_keys = np.arange(0, ffdim)

    uid_filepath = str(datetime.datetime.now()) + " " + str(uuid.uuid4())
    
    write_path = os.path.join(os.environ['HOME'], os.path.join("results", f"{uid_filepath}.txt")) 
    
    with open(write_path, "w") as f:
        f.write(f"\n Result Tracking file {uid_filepath}")

    decoding_failures, iterations, counter = 0, 0, 0

    # Updating result file per iterations
    writing_per_iterations = 10

    for iteration in tqdm(range(max_iterations)):   
            
        input_arr = [random.choice(symbol_keys) for i in range(k)]
        
        C = np.dot(input_arr, G) % ffdim

        # Masking
        mask = [np.random.randint(ffdim) for i in range(n)]
        C2 = [(C[i] + mask[i]) % ffdim for i in range(len(C))]
        
        # Channel Simulation
        symbol_likelihoods_arr = np.array(simulate_reads(C2, symbols, read_lengths[0], P, n_motifs, n_picks))
        
        # Unmasking
        symbol_likelihoods_arr = unmask_reordering(symbol_likelihoods_arr, mask, ffdim)

        # Decoding
        z = graph.qspa_decode(symbol_likelihoods_arr, H, GF)
        
        # Validating
        
        if not np.array_equal(C, z):
            decoding_failures+=1
            
        iterations += 1
        
        # if iteration % writing_per_iterations == 0: Write every iteration - not the overhead
        with open(write_path, "a") as f:
            f.write(f"\nIterations {iterations} Failures {decoding_failures}")
        
    final_write_path = os.path.join(os.environ['HOME'], "results_12.txt")
    with open(final_write_path, "a") as f:
        f.write(f"\nIterations {iterations} Failures {decoding_failures}")
         
    return frame_error_rate


def run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, P, code_class="", iterations=10, cc_decoder=False, masked=False, bec_decoder=False, uncoded=False, read_lengths=np.arange(1,20), max_iter=10,  zero_codeword=False, graph_decoding=False, label=None, Harr=None):
    
    if code_class == "sc_":
        k, n, H, G, graph, C, symbols, motifs = get_parameters_sc_ldpc(n_motifs, n_picks, L, M, dv, dc, k, n, ffdim, zero_codeword=zero_codeword, display=False, Harr=None, H=None, G=None)
    else:
        H, G, graph, C, symbols, motifs = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, zero_codeword=zero_codeword, display=False, Harr=None, H=None, G=None)
    
    GF = galois.GF(ffdim)
    GFH = GF(H.astype(int)) # * GF(np.random.choice(GF.elements[1:], siz
    GFK = GF(G.astype(int))

    decoding_errors_fer(k, n, dv, dc, ffdim, P, GFH, G, GF, graph, C, symbols, n_motifs, n_picks, masked=masked, label=label + " Graph QSPA", read_lengths=read_lengths)  
    
    
if __name__ == "__main__":
    n_motifs, n_picks = 8, 4
    dv, dc, ffdim, P = 3, 9, 67, 2 * 0.038860387943791645 
    k, n = 30, 45
    L, M = 50, 1002
    read_lengths = np.arange(12,13)
        
    Harr = []
    masked=False

    run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, P, code_class="sc_",  uncoded=False, zero_codeword=True, masked=masked, bec_decoder=False, graph_decoding=True, read_lengths=read_lengths, label="Zero", Harr=Harr)


    # P = 2 * 0.038860387943791645                                                                                                                                                  