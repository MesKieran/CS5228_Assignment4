import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths import *


    
##########################################################################
##
## Non-Negative Matrix Factorization
##
##########################################################################

class NMF:
    
    def __init__(self, M, k=100):
        self.M, self.k = M, k
    
        num_users, num_items = M.shape
        
        self.Z = np.argwhere(M != 0)
        self.W = np.random.rand(num_users, k)
        self.H = np.random.rand(k, num_items)

        
        
    def calc_loss(self):
        
        loss = np.sum(np.square((self.M - np.dot(self.W, self.H)))[self.M != 0])

        return loss    
    
    
    
    def fit(self, learning_rate=0.0001, lambda_reg=0.1, num_iter=100, verbose=False):
        for it in range(num_iter):

            #########################################################################################
            ### Your code starts here ############################################################### 
            # Calculate the current approximation of M
            for entry in self.Z:
                i, j = entry

                # Calculate the error
                error = self.M[i, j] - np.dot(self.W[i, :], self.H[:, j])
    
                # Update W and H using Gradient Descent
                for k in range(self.k):
                    self.W[i, k] += learning_rate * (2 * error * self.H[k, j] - 2 * lambda_reg * self.W[i, k])
                    self.H[k, j] += learning_rate * (2 * error * self.W[i, k] - 2 * lambda_reg * self.H[k, j])
            ### Your code ends here #################################################################
            #########################################################################################           

            # Print loss every 10% of the iterations
            if verbose == True:
                if(it % (num_iter/10) == 0):
                    print('Loss: {:.5f} \t {:.0f}%'.format(self.calc_loss(), (it / (num_iter/100))))

        # Print final loss        
        if verbose == True:
            print('Loss: {:.5f} \t 100%'.format(self.calc_loss()))        
        
        
    def predict(self):
        #
        return np.dot(self.W, self.H)
    
    
    
    

    
##########################################################################
##
## Closeness Centrality
##
##########################################################################


def closeness(G):
    
    closeness_scores = { node:0.0 for node in G.nodes }
    
    #########################################################################################
    ### Your code starts here ############################################################### 
    for node in G.nodes:
        total_distance = 0
        # Compute the shortest paths from the current node to all other nodes
        shortest_paths = nx.single_source_shortest_path_length(G, node)
        total_distance = sum(shortest_paths.values())
        # Calculate Closeness Centrality as the reciprocal of the total distance
        closeness_scores[node] = len(G.nodes) / total_distance
    ### Your code ends here #################################################################
    #########################################################################################         
        
    return closeness_scores
    
    
    
    
    
    
    
##########################################################################
##
## PageRank Centrality
##
##########################################################################


def create_transition_matrix(A):
   
    # Divide each value by the sum of its column
    # Matrix M is column stochastic
    M = A / (np.sum(A, axis=1).reshape(1, -1).T)
    
    # Set NaN value to 0 (default value of nan_to_num)
    # Required of the sum of a columns was 0 (if directed graph is not strongly connected)
    M = np.nan_to_num(M).T
    
    return np.asarray(M)



def pagerank(G, alpha=0.85, eps=1e-06, max_iter=1000):
   
    node_list = list(G.nodes())

    ## Convert NetworkX graph to adjacency matrix (numpy array)
    A = nx.to_numpy_array(G)
    
    ## Generate transition matrix from adjacency matrix A
    M = create_transition_matrix(A)

    
    E, c = None, None
    
    #########################################################################################
    ### Your code starts here ############################################################### 

    ## Initialize E and v
    n = len(node_list)
    E = np.ones((n, 1)) / n
    c = np.ones((n, 1)) / n

    ### Your code ends here #################################################################
    ######################################################################################### 

    # Run the power method: iterate until differences between steps converges
    num_iter = 0
    while True:
        
        num_iter += 1

        #########################################################################################
        ### Your code starts here ###############################################################  
        new_c = alpha * np.dot(M, c) + (1 - alpha) * E
        # Check for convergence
        if np.linalg.norm(new_c - c) < eps:
            break
        c = new_c
        
        ### Your code ends here #################################################################
        #########################################################################################            
            
        pass

    c = c / np.sum(c)
        
    ## Return the results as a dictiory with the nodes as keys and the PageRank score as values
    return { node_list[k]:score for k, score in enumerate(c.squeeze()) }


    