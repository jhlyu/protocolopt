
import torch
import psutil
import os
from time import sleep
import argparse

iterations = 10
data_size = 5E8

def script1(N = iterations, L = data_size):
    means = []
    x1 = torch.empty( size=(1, int(L) ), requires_grad=True, device='mps') 
    for i in range(N):
        x1.data = torch.normal(0,1,size=(1,N))
        b = x1**2
        b.sum().backward()
        means.append(x1.grad.mean())
        x1.grad = None
        return means

def script2(N = iterations, L = data_size):
    means = []
    for i in range(N):
        x1 = torch.normal(0,1,size=(1,int(L) ), requires_grad=True, device='mps')
        b = x1**2
        b.sum().backward()
        means.append(x1.grad.mean())
        x1.grad = None
        return means

def run_memory_test(test_script, n):
    for i in range(n):
        
        if test_script == '1':
            _ = script1()
        elif test_script == '2':
            _ = script2()
        else:
            raise ValueError("Invalid test script. Choose '1' or '2'.")
            return
        
        sleep(10/n)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run memory test on MPS tensors.')
    parser.add_argument('test_script_number', type=str, choices=['1', '2'])
    parser.add_argument('number_runs', type=int, default = 10)
    
    args = parser.parse_args()
    print(f"Running {args.number_runs} iterations to test script{args.test_script_number}")
    run_memory_test(args.test_script_number, args.number_runs)  
