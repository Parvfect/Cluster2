
import random
import numpy as np
from tanner import VariableTannerGraph
import coding_matrices as r
from scipy.linalg import null_space
import sympy as sympy
from itertools import combinations
from pstats import Stats
import re
import math
from cProfile import Profile
from tqdm import tqdm
import matplotlib.pyplot as plt
from protograph.protograph_interface import get_Harr_sc_ldpc, get_dv_dc
import sys
import os

def choose_symbols(n_motifs, picks):
    """ Returns Symbol Dictionary given the motifs and the number of picks """

    # Reference Motif Address starts from 1 not 0
    return [list(i) for i in (combinations(np.arange(1, n_motifs+1), picks))]

def coupon_collector_channel(symbol, R, visibility=1):
    reads = []
    for i in range(R):
        if random.random() < visibility:
            reads.append(random.choice(symbol))
    return reads

def get_symbol_index(symbols, symbol):

    for i in symbols:
        if set(i) == set(symbol):
            return symbols.index(i)

def get_possible_symbols(reads, symbols, motifs, n_picks):
    
    reads = [set(i) for i in reads]
    
    symbol_possibilities = []
    for i in reads:

        # Will only work for the Coupon Collector Channel
        motifs_encountered = i
        motifs_not_encountered = set(motifs) - set(motifs_encountered)
        
        read_symbol_possibilities = []

        # For the case of distraction
        if len(motifs_encountered) > n_picks:
            return symbols

        if len(motifs_encountered) == n_picks:
            read_symbol_possibilities = [get_symbol_index(symbols, motifs_encountered)]
        
        else:
            
            # The symbol possibilites are the motifs that are encountered in combination with the motifs that are not encountered

            remaining_motif_combinations = [set(i) for i in combinations(motifs_not_encountered, n_picks - len(motifs_encountered))]
            
            for i in remaining_motif_combinations:
                possibe_motifs = motifs_encountered.union(i)
                symbols = [set(i) for i in symbols]
                if possibe_motifs in symbols:
                    read_symbol_possibilities.append(get_symbol_index(symbols, motifs_encountered.union(i)))
        
        symbol_possibilities.append(read_symbol_possibilities)
    
    return symbol_possibilities
 
def simulate_reads(C, read_length, symbols):
    """ Simulates the reads from the coupon collector channel """
    
    reads = []
    # Simulate one read
    for i in C:
        read = coupon_collector_channel(symbols[i], read_length)
        reads.append(read)

    # Make reads a set
    return reads

def read_symbols(C, read_length, symbols, motifs, picks):
    reads = simulate_reads(C, read_length, symbols)
    return get_possible_symbols(reads, symbols, motifs, picks)


def display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C, ffdim):

    print("The number of motifs are {}".format(n_motifs))
    print("The number of picks are {}".format(n_picks))
    print("The dv is {}".format(dv))
    print("The dc is {}".format(dc))
    print("The k is {}".format(k))
    print("The n is {}".format(n))
    print("GF{}".format(ffdim))
    print("The Motifs are \n{}\n".format(motifs))
    print("The Symbols are \n{}\n".format(symbols))
    print("The Harr is \n{}\n".format(Harr))
    print("The Parity Matrice is \n{}\n".format(H))
    print("The Generator Matrix is \n{}\n".format(G))
    print("The Codeword is \n{}\n".format(C))
    return

def get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, zero_codeword=False, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)

    # Pop the last n symbols as per difference in finite field and unique symbols through combinations
    symbols.pop()
    symbols.pop()
    symbols.pop()
    

    symbol_keys = np.arange(0, ffdim)

    #graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph = TannerGraph(dv, dc, k, n, ffdim=ffdim)

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

    return graph, G, symbols, motifs

def get_parameters_sc_ldpc(n_motifs, n_picks, L, M, dv, dc, k, n, ffdim, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)
    
    symbols.pop()
    symbols.pop()
    symbols.pop()
    
    symbol_keys = np.arange(0, ffdim)
    
    if Harr is None:
        Harr, dv, dc, k, n = get_Harr_sc_ldpc(L, M, dv, dc)   
    else:
        dv, dc = get_dv_dc(dv, dc, k, n, Harr)
    
    graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph.establish_connections(Harr)

    if H is None and G is None:
        H = r.get_H_matrix_sclpdc(dv, dc, k, n, Harr)
        G = np.zeros([k,n], dtype=int)

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

    if display:
        display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C)

    return k, n, graph, G, symbols, motifs


def run_singular_decoding(graph, C, read_length, symbols, motifs, n_picks):
    
    reads = simulate_reads(C, read_length, symbols)

    # Convert to possible symbols
    possible_symbols = read_symbols(C, read_length, symbols, motifs, n_picks)
    #possible_symbols = get_possible_symbols(reads, symbol_arr)

    # Assigning values to Variable Nodes
    graph.assign_values(possible_symbols)

    decoded_values = graph.coupon_collector_decoding()
 
    # Check if it is a homogenous array - if not then decoding is unsuccessful
    if sum([len(i) for i in decoded_values]) == len(decoded_values):
        if np.all(np.array(decoded_values).T[0] == C):
            print("Decoding successful")
            return np.array(decoded_values).T[0]
    else:
        print("Decoding unsuccessful")
        return None

def decoding_errors_fer(k, n, dv, dc, graph, G, symbols, motifs, n_picks, decoding_failures_parameter=20, max_iterations=20, iterations=20, uncoded=False, masked = False, bec_decoder=False, label=None, code_class="", read_lengths=np.arange(1,20)):
    """ Returns the frame error rate curve - for same H, same G, same C"""

    frame_error_rate = []
    symbol_keys = np.arange(0, ffdim)

    decoding_failures = 0
    iterations = 0
    counter = 0

    for read_length in tqdm(read_lengths):
        for iteration in tqdm(range(max_iterations)):
            
            input_arr = [random.choice(symbol_keys) for i in range(k)]
            C = np.dot(input_arr, G) % ffdim

            if masked:
                mask = [np.random.randint(ffdim) for i in range(n)]
                C2 = [(C[i] + mask[i]) % ffdim for i in range(len(C))]
                symbols_read = read_symbols(C2, read_length, symbols, motifs, n_picks)
                symbols_read = [[(i - mask[p])  % ffdim for i in symbols_read[p]] for p in range(len(symbols_read))]       
                
            else:
                symbols_read = read_symbols(C, read_length, symbols, motifs, n_picks)

            if not uncoded:
                graph.assign_values(symbols_read)
                if bec_decoder:
                    decoded_values = graph.coupon_collector_erasure_decoder()
                else:
                    decoded_values = graph.coupon_collector_decoding()
            else:
                decoded_values = symbols_read
           
            if sum([len(i) for i in decoded_values]) == len(decoded_values):
                if np.all(np.array(decoded_values).T[0] == C):
                    counter += 1
            else: 
                decoding_failures+=1

            iterations += 1
    
    final_write_path = os.path.join(os.environ['HOME'], f"results_{read_lengths[0]}_cc.txt")
    with open(final_write_path, "a") as f:
        f.write(f"\nIterations {iterations} Failures {decoding_failures}")
         
    return frame_error_rate


def run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, code_class="", iterations=5, bec_decoder=False, uncoded=False, saved_code=False, singular_decoding=False, fer_errors=True, read_lengths=np.arange(1,20), zero_codeword=False, label="", Harr=None, masked=False):

    if code_class == "sc_":
        k, n, graph, G, symbols, motifs = get_parameters_sc_ldpc(n_motifs, n_picks, L, M, dv, dc, k, n, ffdim, display=False)
    else:
        graph, G, symbols, motifs = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, display=False, zero_codeword=zero_codeword, Harr=Harr)
    
    print(decoding_errors_fer(k, n, dv, dc, graph, G, symbols, motifs, n_picks, iterations=iterations, label=f'{label} CC Decoder', code_class=code_class, read_lengths=read_lengths, masked=masked))
        


if __name__ == "__main__":
    n_motifs, n_picks = 8, 4
    dv, dc, ffdim = 4, 12, 67
    k, n = 30, 45
    L, M = 50, 1002
    read_length = 6
    read_lengths = np.arange(7, 8)

    masked = True

    run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, code_class="sc_", saved_code=False,  uncoded=False, bec_decoder=False, read_lengths=read_lengths, zero_codeword=True, label="ZeroCW", masked=masked)



