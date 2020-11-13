import numpy as np

# array
arr1 = np.array([1, 2, 3])
print("--- arr1 ---")
print(arr1)

arr2 = np.asarray([[3, 5, 7], [1, 6, 9]], dtype=np.int32)
print("--- arr2 ---")
print(arr2)
print(arr2.shape)

zero = np.zeros((2, 3))
print("--- zero ---")
print(zero)

one = np.ones((3, 4))
print("--- one ---")
print(one)

rand = np.random.rand(3, 4)
print("--- random ---")
print(rand)
print()


# calc
print("--- plus ---")
print(arr2 + 4)

print("--- multi ---")
print(rand * 6)

print("--- mean ---")
print(np.mean(rand))

print("--- std ---")
print(np.std(rand))
print()


# search
print("--- index ---")
print(np.where(arr2==9))
print()

# calc Euclidean distance

data = np.random.randn(100, 3)

squared = data**2
sum_data = squared.sum(axis=1)

dist = np.sqrt(sum_data)

mean_dist = np.mean(dist)
print("mean : euclidean distance")
print(mean_dist)
