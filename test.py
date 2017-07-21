from six.moves import xrange
epoch = 60
num_iterations = 1000

start_step = 0

for i in xrange(epoch):
    for j in xrange(num_iterations):
        step = start_step + i * num_iterations + j
        print(step)