class Node:
  def __init__(self, leader, size):
    self.leader = leader
    self.size = size
  
def make_set(a, sets):
  sets[a] = Node(a, 1)
 
def get_set(a, sets):
  if sets[a].leader == a:
    return a
  sets[a].leader = get_set(sets[a].leader, sets)
  return sets[a].leader
 
def union_sets(a, b, sets):
  a = get_set(a, sets)
  b = get_set(b, sets)
  if a != b:
    if sets[a].size < sets[b].size:
      a, b = b, a
    sets[b].leader = a
    sets[a].size += sets[b].size
 
n, m, k = map(int, input().split())
sets = [Node(i, 1) for i in range(n + 1)]
edges = []
for _ in range(m):
  edges.append(list(map(int, input().split())))
 
commands = []
for _ in range(k):
  commands.append(input().split())

answers = []
for command, a, b in commands[::-1]:
  a, b = int(a), int(b)
  if command == 'ask':
    answers.append('YES' if get_set(a, sets) == get_set(b, sets) else 'NO')
  else:
    union_sets(a, b, sets)
  
for answer in answers[::-1]:
  print(answer)