"""
python3 -m trace --count <cut.py>
"""
import time
import itertools # consider using
import cProfile # cProfile.run(.)

E = 2
N = 100000000

def orig(): 
  cube_numbers = []
  for n in range(0,N):
    if n % 2 == 1:
      cube_numbers.append(n**3)
    if(0):
      pass

def new():
  cube_numbers = [n**3 for n in range(1,N) if n%2 == 1]

def test():
  s = 0
  t0 = time.time()
  orig();
  return time.time() - t0

def main():
  s = 0
  for i in range(E):
    s += test()
  print(s/E)
main()
