API still unstable and liable to change.

# Creating a tensor
```
const scalar0 = numeri.scalar(3);
const vector0 = numeri.vector([3, 4]);
const matrix0 = numeri.fromFlat([3, 4, 5, 6, 7, 8], [3, 2]); //a 3x2 matrix whose first row is `[3, 4]`
const matrix1 = numeri.fromNested([[3, 4], [5, 6], [7, 8]]); //the same 3x2 matrix, but is a tiny bit slower than `fromFlat`
const tens4d0 = numeri.fromFlat([5, 7], [1, 2, 1, 1]) //a 1x2x1x1 tensor
const zeroMat = numeri.zeros([4, 5]);
const fourMat = numeri.fill([4, 5], 4);
const z123456 = numeri.range(7);
const eyeMatr = numeri.identity(5); //5x5 identity matrix
```

# Miscellaneous
```
matrix0.shape //returns [3, 2]
matrix0.getLength() //total number of elements; 6
matrix0.toNested() //returns as nested arrays
```

# Slicing and accessing
```
matrix0.get(2, 1)

matrix0.slice(0) //returns the 0th row vector
matrix0.slice([0, 2]) //returns the submatrix with only the first two rows
matrix0.slice(undefined, 0) //returns the 0th column vector
matrix0.slice(undefined, [0, 4, 2]) //returns the submatrix with every even column from the 0th up to (not including) the 4th
```
Note that `.slice` returns a view.

# Reshaping
```
matrix0.transpose([1, 0]) //if matrix0 is n x m, this returns an m x n matrix
matrix0.transpose() //shorthand, only possible if the tensor is a matrix
tens4d0.transpose([0, 2, 3, 1]) //(a x b x c x d) -> (a x c x d x b)

matrix0.reshape([2, 3]) //returns a new matrix [[3, 4, 5], [6, 7, 8]]
```
Note that `.transpose` returns a view.

# Setting
```
matrix0.set([1, 2], 3) //sets the [1, 2] entry to 3
matrix0.setAll(11)
matrix0.slice(1).setAll(12)
```

# Unary operators
```
matrix0.exp() //returns a new matrix with elementwise exponentiation
matrix0.map((x) => x * x)
matrix0.mapInPlace((x) => x * 2)
```

# Binary operators
```
matrix0.plus(matrix1) //returns a new matrix by elementwise addition
matrix0.minus(matrix1)
matrix0.times(matrix1)
matrix0.div(matrix1)
matrix0.elemwiseBinaryOp(matrix1, (a, b) => Math.pow(a, b))
matrix0.elemwiseBinaryOpInPlace(matrix1, (a, b) => Math.pow(a, b))

matrix0.matMul(numeri.fromFlat([1, 2, 3, 4, 5, 6], [2, 3])) //matrix multiplication; only works on 2-dimensional tensors
vector0.dot(vector0) //vector dot product
```

# Broadcasting
```
matrix0.plus(1) //returns a matrix with 1 added to each element
matrix0.plus(numeri.scalar(1).broadcastOn(0, 1)) //equivalent; explicitly broadcast scalar to both dimensions

matrix0.plus(numeri.vector([1, 2]).broadcastOn(0)) //returns a matrix with [1, 2] added to each row
matrix0.plus(numeri.vector([1, 2, 3]).broadcastOn(1)) //returns a matrix with [1, 2, 3] added to each column
```

# Reducing
```
matrix0.sum()
matrix0.norm() //Eulidean norm
matrix0.lpNorm(3) //L3 norm
```

# Eigen / Eigenvalue / Eigenvector / Hessenberg Operations
```
const symMat = numeri.fromFlat([1, 2, 3, 2, 4, 5, 3, 5, 6], [3, 3]);
const tridiagonalMat = numeri.fromFlat([1, 2, 0, 2, 4, 5, 0, 5, 6], [3, 3]);

const { vals } = numeri.symEig(symMat); //returns vals as tensor
const { vals, vecs } = numeri.symEig(symMat, {includeVecs: true}); //and vecs as columns
const { vals, vecs } = numeri.tridiagonalEig(tridiagonalMat, {includeVecs: true});
const { hessenberg, q } = numeri.symHessenberg(symMat, {includeQ: true}); //hessenberg of symmetric matrix is tridiagonal
```

# A word about views

The `.slice` and `.transpose` operators return *views* into the original tensors.
This is also standard behavior in other major numerical libraries, like numpy and Tensorflow.
A view shares the same data as the original tensor, but may have different shape and access the data at different indices.

This makes slicing and transposition `O(1)` time with respect to the data size.
It also saves memory by not copying the data into each tensor.
However, sometimes having data laid out in order makes operations faster.
For instance, if we define `a = parentTensor.slice(...sliceArgs).transpose(permutation)` and then do many evaluations of `a.plus(tensor0)`, `a.plus(tensor1)`, ..., it could save time do a copy with `a = parentTensor.slice(...sliceArgs).transpose(permutation).copy()`.
This will make each of the following `.plus` operations faster.
