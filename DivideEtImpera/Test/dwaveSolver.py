import sys
import os
from numba.core.decorators import jit
import torch
import pprint
from utils import *
from time import sleep, time_ns, time
from QUBOMatrix import calcQUBOMatrix, calcQUBOMatrixForDivideEtImpera
from QUBOValues import getValuesForSubproblems
from solverUtils import getPath, getExpectedSolution, printInfoResults, getPath, printAdjMat, getAdjMat

from dimod.reference.samplers import ExactSolver
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite


def getParams():
  path = getPath()
  if len(sys.argv) < 3:
    method = 'SA'
  else:
    method = sys.argv[2]
  if len(sys.argv) < 4:
    nReads = 100
  else:
    nReads = int(sys.argv[3])
  if len(sys.argv) < 5:
    annealTime = 1
  else:
    annealTime = int(sys.argv[4])
  return path, method, nReads, annealTime

def getDwaveQubo(Q, indexQUBO):
  qubo = {}
  for i in range(len(indexQUBO)):
    for j in range(i,len(indexQUBO)):
      if Q[i][j] != 0:
        qubo[(indexQUBO[i],indexQUBO[j])] = Q[i,j].item()
  return qubo

def getSampler(dwave_conf_path, method='SA'):
  sampler = None
  if method == 'SA':
    sampler = SimulatedAnnealingSampler()
  elif method == 'QA':
    sampler = EmbeddingComposite(DWaveSampler(dwave_conf_path,solver={'topology__type__eq':'pegasus'}))
  else:
    sampler = ExactSolver()
  return sampler

def getMinXt(bestSample, indexQUBO, posOfIndex):
  minXt = torch.zeros(len(indexQUBO))
  for index, value in bestSample.items():
    pos = posOfIndex[index]
    minXt[pos] = value
  return minXt

def getMinInfo(record):
  readFound = None
  occurrences = None
  minEnergy = float('inf')
  for i, (_, energy, occ, *_) in enumerate(record):
    if energy < minEnergy:
      minEnergy = energy
      occurrences = occ
      readFound = i
  return readFound, occurrences
    
def writeCSV(n, probName, alpha, method, nReads, annealTime, dsName, calcQUBOTime, annealTimeRes, readFound, occurrences, minY, expY, minXt):
  with open('./tests/testsAnneal.csv', 'a') as file:
    if '10K' in dsName:
      examples = 10000
    elif '100K' in dsName:
      examples = 100000
    elif '1M' in dsName:
      examples = 1000000
    if method != 'QA':
      annealTime = '-'
    template = '{},'*12 + ',,' + '{},'*2 + '\'{}\'' + '\n'
    testResult = template.format(n,probName,alpha,examples,method,nReads,annealTime,dsName,calcQUBOTime/10**6,annealTimeRes/10**6,readFound,occurrences,minY,expY,minXt.int().tolist())
    file.write(testResult)


def dwaveSolve(dwave_conf_path, Q, indexQUBO, posOfIndex, label, method='SA', nReads=100, annealTime=1):
  qubo = getDwaveQubo(Q,indexQUBO)
  sampler = getSampler(dwave_conf_path, method=method)
  startAnneal = time_ns()
  if method == 'QA':
    sampleset = sampler.sample_qubo(qubo,num_reads=nReads,label=label,annealing_time=annealTime)
  else:
    sampleset = sampler.sample_qubo(qubo,num_reads=nReads,label=label)
  endAnneal = time_ns()
  #TODO uncomment this part
  #if 'timing' in sampleset.info.keys():
  #  print(sampleset.info['timing'])
  #  annealTime = sampleset.info['timing']['qpu_access_time']
  #else:
  #  annealTime = (endAnneal - startAnneal)//10**3
  minXt = getMinXt(sampleset.first.sample,indexQUBO,posOfIndex)
  minX = minXt.view(-1,1)
  minY = torch.matmul(torch.matmul(minXt,Q),minX).item()
  readFound, occurrences = getMinInfo(sampleset.record)
  return minXt, minY, readFound, occurrences, annealTime

def main():
  pass
 
  
if __name__ == '__main__':
  main()
