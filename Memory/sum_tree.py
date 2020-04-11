import numpy

""" Original Code by @jaara: https://github.com/jaara/AI-blog/blob/master/SumTree.py
"""

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        #create the sumtree with double capacity of buffer
        self.tree = numpy.zeros( 2*capacity - 1 )
        #actual data are sotred in self.data
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        #retrieving by this method leads to high likelihood of sampling high priorities
        #only randomness can pick smaller errors
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        #works for a particular replayed transition, updates the cumulative sums
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])