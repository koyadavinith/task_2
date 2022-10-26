# 1st answer

import numpy as np

# 2nd answer
from numpy.lib import stride_tricks

print(np.__version__, np.show_config())

# 3rd answer

print(np.zeros(10))  

# 4th answer
x = np.array([10, 20, 35, 45]) 
print('The memory size of the array in bytes is: ', (x.size * x.itemsize))

# 5th answer
print(np.info(np.add))  


# 6th answer
y = np.zeros(10)  
y[4] = 1
print(y)

# 7th answer
z = np.arange(10, 50 )  
print(z)

# 8th answer
print('reverse of the vector: ', z[::-1])  

# 9th answer
a = np.arange(9)  
a = np.reshape(a, (3, 3))
print(a)

# 10th answer
a9 = np.array([1, 2, 0, 0, 4, 0])  
k9 = np.nonzero(a9)
print("The non-zero indices are: ", k9)

# 11th answer
I = np.eye(3, dtype=np.int32)  
print(I) 

#12th answer
print('The matrix with random values of size 3x3x3 is: ')  
print( np.random.rand(3,3,3))

#13th answer
a13 = np.random.rand(10, 10)  # 13th answer
print('A 10x10 matrix with random values is: ')
print(a13)
print('Maximum and Minimum values of the matrix are: ', np.max(a13), 'and', np.min(a13))

# 14th answer
a14 = np.random.rand(30)
print('The random vector of size 30 is: ')
print(a14)
print('The mean of the vector is: ', np.mean(a14))

# 15th answer
a15 = np.full((5, 5), 1)
a15[1:-1, 1:-1] = 0
print('The 2d array with 1 on the border and 0 on the insides is: ', a15)

# 16th answer
a16 = np.array([1, 2, 3, 4, 5, 9, 8, 6])
print('for zeroes to surround an array: ', np.pad(a16, (2,2), 'constant', constant_values=0))

# 17th answer
print('Result for the expression 0 * np.nan is:', 0 * np.nan)
print('Result for the expression np.nan == np.nan is:', np.nan == np.nan)
print('Result for the expression np.inf > np.nan is:', np.inf > np.nan)
print('Result for the expression np.nan - np.nan is:', np.nan - np.nan)
print('Result for the expression 0.3 == 3 * 0.1 is:', 0.3 == 3 * 0.1)

# 18th answer
print('A 5x5 matrix with values 1,2,3,4 just below the diagonal is:')
print(np.diagflat([1, 2, 3, 4], k=-1))

# 19th answer
print('A 8x8 matrix and fill it with a checkerboard pattern is: ')
a19 = np.full((8, 8), 0)
for i in range(0,8,2):
    for j in range(1,8,2):
        a19[i][::2] = 1
        a19[j][1::2] = 1
print(a19)

#20th answer
print('The index (x,y,z) of the 100th element in a (6,7,8) shape array is:')
print(np.unravel_index(100,(6,7,8)))

# 21st answer
print('A checkerboard 8x8 matrix using the tile function: ')
a21 = np.array([[1, 0], [0, 1]])
print(np.tile(np.tile(a21, (2, 2)), (2, 2)))

# 22nd answer
print('Normalizing an array a22')
a22 = np.random.rand(5, 5)
print(a22)
norm_array = (a22 - np.min(a22)) / np.max(a22) - np.min(a22)
print(norm_array)

# 23rd answer
print('')
RGBA = np.dtype([('Red', np.uint8), ('Green', np.uint8), ('Blue', np.uint8), ('Alpha', np.uint8)])
colour = np.array([1, 2, 3, 4], dtype=RGBA)
print(type(RGBA))

# 24th answer
a1 = np.random.rand(5, 3)
a2 = np.random.rand(3, 2)
print(np.dot(a1, a2))

#25th answer
a25 = np.arange(20)
for i in a25:
    if 3 <= i <= 8:
        a25[i] = -1
print(a25)

# 26th answer
print('The output of the given script is: ')
print(sum(range(5), -1))
from numpy import *
print(sum(range(5), -1))
print('The output of the above script is: 9 and 10')

# 27th answer
z = np.arange(10)
print(z ** z)
print('z**z is a legal expression')
print(2 << z >> 2)
print('2<<z>>2 is an legal expression')
print(z < - z)
print('z <- z is an legal expression')
print(1j * z)
print('1j*z is an legal expression')
print(z / 1 / 1)
print('z/1/1 is an legal expression')
try:
    print(z < z > z)
except:
    print('z<z>z is not a legal expression')

#28th answer
print('The result of np.array(0) / np.array(0): ',np.array(0) / np.array(0))
print('The result of np.array(0) // np.array(0): ', np.array(0) // np.array(0))
print('The result of np.array([np.nan]).astype(int).astype(float): ',np.array([np.nan]).astype(int).astype(float))


#29th answer
a29 = np.random.rand(3,3)
print(np.round(a29))

#30th answer
a30_1 = np.array([1,2,3,3,5,5,95])
a30_2 = np.array([2,3,4,5,6,3,100,98])
print(np.intersect1d(a30_1, a30_2))

# 31st answer
np.seterr(all="ignore")
print(np.ones(5)/0)

#32nd answer
print(np.sqrt(-1) == np.emath.sqrt(-1))
print("The given expression is not True")

# 33rd answer
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday, today, tomorrow)

# 34th answer
print(np.arange('2016-07', '2016-08', dtype='datetime64[D]'))

# 35th answer
A = np.ones(3) * 5
B = np.ones(3) * 10
C = np.ones(3) * 9
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
print(' ((A+B)*(-A/2)) is : ', np.dot(A, B))

# 36th answer
a36 = np.random.uniform(0, 20, 10)
print(a36)
print(a36 - a36 % 1)
print(np.floor(a36))
print(np.ceil(a36)-1)
print(a36.astype(int))
print(np.trunc(a36))

#37th answer
a37 = np.arange(0,5)
print(np.tile(a37,(5,1)))


# 38th answer
def generator():
    iter1 = (i for i in range(10))
    return iter1


a38 = np.fromiter(generator(), dtype=float)
print(a38)

#39th answer
a39 = np.linspace(0,1,11, endpoint = False)[1:]
print(a39)

#40th answer
a40 = np.random.rand(10)
np.sort(a40)
print(a40)

#41st answer
a41 = np.arange(10)
print(np.add.reduce(a41))

#42nd answer
a42_1 = np.random.randint(0,5,10)
a42_2 = np.random.randint(0,5,10)
compare = np.allclose(a42_1, a42_2)
print(compare)
compare = np.array_equal(a42_1,a42_2)
print(compare)
print("Therefore, Both arrays are not equal")

#43rd answer
a43 = np.arange(10)
print("To make the array immutable we use: ")
a43.flags.writeable = False

# 44th answer
a44 = np.random.randint(0, 10, (10, 2))
a44_1, a44_2 = a44[:, 0], a44[:, 1]
r = np.sqrt(a44_1 ** 2 + a44_2 ** 2)
t = np.arctan2(a44_2, a44_1)
polar_coordinates = np.c_[r, t]
print(polar_coordinates)

# 45th answer
a45 = np.random.random(15)
print(a45)
a45[a45.argmax()] = 0
print(a45)

# 46th answer
x46 = np.linspace(0, 1, 10)
y46 = np.linspace(0, 1, 8)
xx, yy = np.meshgrid(x46, y46)
print(xx)
print(yy)


#47th answer
x47 = np.random.randint(0,100,(3,3))
y47 = np.random.randint(0,100,(3,3))
cauchy = x47 - y47
cauchy_matrix = 1/cauchy
print(cauchy_matrix)

#48th answer
for dtype in [np.int8 , np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)

#49th answer
a49 = np.random.rand(10)
print(a49)

# 50th answer
a50 = np.random.uniform(0, 10, 10)
print(a50)
scalar = 6
arr50 = np.abs(a50 - 6)
index = arr50.argmin()
print(a50[index])

# 51st answer
a51 = np.zeros(10, [('position', [('x', float),
                                  ('y', float)]),
                    ('color', [('r', float),
                               ('g', float),
                               ('b', float)])])
print(a51)

# 52nd answer
a52 = np.random.random((100, 2))
X52, Y52 = np.atleast_2d(a52[:, 0], a52[:, 1])
D = np.sqrt((X52 - X52.T) ** 2 + (Y52 - Y52.T) ** 2)
print(D)

#53rd answer
a53 = np.arange(15, dtype = np.float32)
print(a53)
a53_1 = a53.astype(np.int32)
print(a53_1)

# 54th answer
from io import StringIO

s54 = StringIO("""1, 2, 3, 4, 5
                6,  ,  , 7, 8
                 ,  , 9,10,11""")
a54 = np.genfromtxt(s54, delimiter=",", dtype=np.int32)
print(a54)


#55th answer
a55 = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(a55):
    print(index, value)
for index in np.ndindex(a55.shape):
    print(index, a55[index])

# 56th answer
a56_x, a56_y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
d = np.sqrt(a56_x ** 2 + a56_y ** 2)
sigma, mu = 1.0, 0.0
a56 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
print("2D Gaussian-like array:")
print(a56)

#57th answer
p = 3
a57 = np.zeros((10,10))
np.put(a57, np.random.choice(100 , 3 , replace= False),1)
print(a57)

# 58th answer
a58 = np.random.rand(5, 10)
a58_1 = a58 - a58.mean(axis=1, keepdims=True)
print(a58_1)

#59th answer
a59 = np.random.randint(0,100,(6,6))
print(a59)
print(a59[:,a59[1,:].argsort()])

#60th answer
a60 = np.random.randint(0,5,(5,10))
a60[:, 2] = 0
print(a60)
a60 = a60.astype(bool)
print(a60)
a60 = a60.any(axis=0)
print('*' * 60)
print("The column with all zeroes is represented as False: ")
print(a60)

#61st answer
a61 = np.random.uniform(0, 10, 10)
print(a61)
scalar = 6
arr61 = np.abs(a61 - 6)
index = arr61.argmin()
print(a61[index])

# 62nd answer
a62_1 = np.random.randint(1, 10, (1, 3))
a62_2 = np.random.randint(1, 10, (3, 1))
print(a62_1)
print(a62_2)
result62 = 0
for i in np.nditer(a62_1):
    result62 += i
for i in np.nditer(a62_2):
    result62 += i
print(result62)

#63rd answer
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

a63 = NamedArray(np.arange(10), "range_10")
print (a63.name)

#64th answer
a64_1 = np.random.randint(1,100,10)
print(a64_1)
a64_2 = np.random.randint(0,len(a64_1),20)
print(a64_2)
a64_1 += np.bincount(a64_2, minlength=len(a64_1))
print(a64_1)
# 65th answer
X65 = [1, 2, 3, 4, 5, 6]
I65 = [1, 3, 9, 3, 4, 1]
F65 = np.bincount(I65, X65)
print(F65)

#66th answer

w, h = 256, 256
I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
colors = np.unique(I.reshape(-1, 3), axis=0)
a66 = len(colors)
print(a66)

#67th asnwer
a67 = np.random.randint(0,10,(3,4,3,4))
sum = a67.sum(axis=(-2,-1))
print(sum)

# 68th answer
D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# 69th answer
a69_1 = np.random.uniform(0, 20, (4, 4))
a69_2 = np.random.uniform(0, 20, (4, 4))
print(np.diag(np.dot(a69_1, a69_2)))

#70th answer
a70 = np.array([1,2,3,4,5])
k70 = 3
z70 = np.zeros(len(a70) + (len(a70)-1)*(k70))
z70[::k70+1] = a70
print(z70)

# 71st answer
a71_1 = np.ones((5, 5, 3)) * 2
a71_2 = np.ones((5, 5))
print(a71_1 * a71_2[:, :, None])

#72nd answer
a72 = np.random.randint(1,100,(3,3))
print(a72)
a72[[0,1]] = a72[[1,0]]
print(a72)

# 73rd answer
faces = np.random.randint(0, 100, (10, 3))
a73 = np.roll(faces.repeat(2, axis=1), -1, axis=1)
a73 = a73.reshape(len(a73) * 3, 2)
a73 = np.sort(a73, axis=1)
a73_1 = a73.view(dtype=[('p0', a73.dtype), ('p1', a73.dtype)])
a73_1 = np.unique(a73_1)
print(a73_1)

# 74th answer
C74 = np.bincount([1, 1, 2, 3, 4, 4, 6])
A74 = np.repeat(np.arange(len(C74)), C74)
print(A74)


# 75th asnwer
def compute_avg(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
a75 = np.arange(20)
print(compute_avg(a75, n=3))


# 76th asnwer
import numpy as np  # 1st answer

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return stride_tricks.as_strided(a, shape=shape, strides=strides)


a76 = rolling(np.arange(10), 3)
print(a76)

#77th answer
a77 = np.random.randint(0,2,100)
print(a77)
np.logical_not(a77, out=a77)
print(a77)
b77 = np.random.uniform(-1.0,1.0,100)
np.negative(b77, out=b77)
print(b77)


# 78th answer
def distance(P0, P1, p):
    T = P1 - P0
    L = (T ** 2).sum(axis=1)
    U = -((P0[:, 0] - p[..., 0]) * T[:, 0] + (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
    U = U.reshape(len(U), 1)
    D = P0 + U * T - p
    return np.sqrt((D ** 2).sum(axis=1))


P0_78 = np.random.uniform(-10, 10, (10, 2))
P1_78 = np.random.uniform(-10, 10, (10, 2))
p_78 = np.random.uniform(-10, 10, (1, 2))
print(distance(P0_78, P1_78, p_78))


# 79th answer

def distance(P0, P1, p):
    T = P1 - P0
    L = (T ** 2).sum(axis=1)
    U = -((P0[:, 0] - p[..., 0]) * T[:, 0] + (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
    U = U.reshape(len(U), 1)
    D = P0 + U * T - p
    return np.sqrt((D ** 2).sum(axis=1))


P0_79 = np.random.uniform(-10, 10, (10, 2))
P1_79 = np.random.uniform(-10, 10, (10, 2))
p_79 = np.random.uniform(-10, 10, (10, 2))
print(np.array([distance(P0_79, P1_79, p_i) for p_i in p_79]))

#80th answer
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)

#81st answer
a81 = np.arange(1,15,dtype=np.uint32)
R81 = stride_tricks.as_strided(a81,(11,4),(4,4))
print(R81)

#82nd answer
a82 = np.random.randint(1,100,(10,10))
rank = np.linalg.matrix_rank(a82)
print(rank)

#83rd answer
a83 = np.random.randint(1,10,30)
print(np.bincount(a83).argmax())

#84th asnwer
a84 = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (a84.shape[0]-3)
j = 1 + (a84.shape[1]-3)
C84 = stride_tricks.as_strided(a84, shape=(i, j, n, n), strides=a84.strides + a84.strides)
print(C84)


# 85th answer
class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i, j = index
        super(Symetric, self).__setitem__((i, j), value)
        super(Symetric, self).__setitem__((j, i), value)


def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)


S = symetric(np.random.randint(0, 10, (5, 5)))
S[2, 3] = 42
print(S)

# 86th answer
p86, n86 = 10, 20
M86 = np.ones((p86, n86, n86))
V86 = np.ones((p86, n86, 1))
S86 = np.tensordot(M86, V86, axes=[[0, 2], [0, 1]])
print(S86)

# 87th answer
a87 = np.ones((16, 16))
k = 4
S87 = np.add.reduceat(np.add.reduceat(a87, np.arange(0, a87.shape[0], k), axis=0),
                      np.arange(0, a87.shape[1], k), axis=1)
print(S87)


# 88th answer
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
         Z[1:-1, 0:-2] + Z[1:-1, 2:] +
         Z[2:, 0:-2] + Z[2:, 1:-1] + Z[2:, 2:])

    # Apply rules
    birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)
    Z[...] = 0
    Z[1:-1, 1:-1][birth | survive] = 1
    return Z


a88 = np.random.randint(0, 2, (50, 50))
for i in range(100): a88 = iterate(a88)
print(a88)

#89th answer
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
print (Z[np.argsort(Z)[-n:]])

#90th answer
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))

#91st answer
Z = np.array([("hi", 2.5, 3),
              ("vini", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)

#92nd answer

x = np.random.rand(int(5e7))

np.power(x,3)
x*x*x
np.einsum('i,i,i->i',x,x,x)

#93rd answer
A93 = np.random.randint(0,5,(8,3))
B93 = np.random.randint(0,5,(2,2))

C93 = (A93[..., np.newaxis, np.newaxis] == B93)
rows = np.where(C93.any((3,1)).all(1))[0]
print(rows)

#94th asnwer
a94 = np.random.randint(0,5,(10,3))
print(a94)
# solution for arrays of all dtypes (including string arrays and record arrays)
E94 = np.all(a94[:,1:] == a94[:,:-1], axis=1)
U94 = a94[~E94]
print(U94)
# soluiton for numerical arrays only, will work for any number of columns in Z
U94 = a94[a94.max(axis=1) != a94.min(axis=1),:]
print(U94)

#95th answer
I95 = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B95 = ((I95.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B95[:,::-1])

#96th answer
a96 = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(a96).view(np.dtype((np.void, a96.dtype.itemsize * a96.shape[1])))
_, idx = np.unique(T, return_index=True)
b96 = a96[idx]
print(b96)

b96 = np.unique(a96, axis=0)
print(b96)

#97th answer

A97 = np.random.uniform(0,1,10)
B97 = np.random.uniform(0,1,10)

np.einsum('i->', A97)
np.einsum('i,i->i', A97, B97) 
np.einsum('i,i', A97, B97)
np.einsum('i,j->ij', A97, B97)

#98th answer
phi = np.arange(0, 10*np.pi, 0.1)
a98 = 1
x98 = a98*phi*np.cos(phi)
y98 = a98*phi*np.sin(phi)

dr = (np.diff(x98)**2 + np.diff(y98)**2)**.5
r = np.zeros_like(x98)
r[1:] = np.cumsum(dr)
r_int = np.linspace(0, r.max(), 200)
x_int = np.interp(r_int, r, x98)
y_int = np.interp(r_int, r, y98)

#99th answer
X99 = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M99 = np.logical_and.reduce(np.mod(X99, 1) == 0, axis=-1)
M99 &= (X99.sum(axis=-1) == n)
print(X99[M99])

#100th answer
X100 = np.random.randn(100) # random 1D array
N100 = 1000 # number of bootstrap samples
idx = np.random.randint(0, X100.size, (N100, X100.size))
means = X100[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)