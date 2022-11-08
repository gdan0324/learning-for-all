import nashpy
import numpy

a = numpy.array([[6, 4],
                 [10, 0]])
b = numpy.array([[0, 4],
                 [-2, 0]])
rps = nashpy.Game(a, b)
print(rps)

equilibrium = rps.support_enumeration()
for eq in equilibrium:
    print(eq)

sigma_row = numpy.array([1, 0])
sigma_column = numpy.array([0, 1])

print(rps[sigma_row, sigma_column])
