import matplotlib
import numpy
import pandas
from torch.utils.data import IterableDataset

print("Numpy version:")
print(numpy.version.version)
print(numpy.__version__)

print("Pandas version:")
print(pandas._version.version_json)
print(pandas.__version__)

print("Numpy version:")
print(matplotlib._version.version_json)
print(matplotlib.__version__)

if __name__ == '__main__':
    narr = numpy.array([1, 2, 3])
    print(narr)

    norm = numpy.linalg.norm(narr)
    print(norm)


class OneShotDataset(IterableDataset):

    def __init__(self, trials):
        self._trials = trials
        self._currentIdx = (0, 0)
        self._way = len(self._trials[0][0])

    def __iter__(self):
        return self

    def __next__(self):
        if self._currentIdx[0] == len(self._trials) - 1:
            self._currentIdx = (0, 0)
            raise StopIteration
        if self._currentIdx[1] == self._way - 1:
            self._currentIdx = (self._currentIdx[0] + 1, 0)
        else:
            self._currentIdx = (self._currentIdx[0], self._currentIdx[1] + 1)

        return [[self._trials[self._currentIdx[0]][0][self._currentIdx[1]],
                 self._trials[self._currentIdx[0]][1]], self._currentIdx[1]]  # [[img,way x imgs], label]

        # for t in self._trials:
        #     for i, first in enumerate(t[0]):
        #         return [[first, t[1]], i]

    def __len__(self):
        return len(self._trials) * len(self._trials[0][0])
