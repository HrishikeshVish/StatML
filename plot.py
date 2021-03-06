import matplotlib.pyplot as plt

depth = ["0 (Random val)", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
train = [0.43, 0.93, 0.94,0.99,0.99, 0.995, 0.9956, 0.9956, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
val = [0.42, 0.96, 0.98,0.95,0.95, 0.8939, 0.8939, 0.8939, 0.8939, 0.8939, 0.8939, 0.8939, 0.8939, 0.8939, 0.8939, 0.8939]
test = [0.49, 0.52, 0.56, 0.60, 0.619, 0.61, 0.619, 0.6194, 0.6194, 0.6194, 0.6194, 0.6194, 0.6194, 0.6194, 0.6194, 0.6194]

plt.plot(depth, train, 'blue',label='train')
plt.plot(depth, test, 'red', label='test')
plt.xlabel('depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()


