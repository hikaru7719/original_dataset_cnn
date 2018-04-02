input = [batch_size,64,64,3]

↓conv1

[batch_size,64,64,32]

↓pool1

[batch_size,32,32,32]

↓conv2

[batch_size,32,32,64]

↓pool2

[batch_size,16,16,64]

↓conv3

[batch_size,16,16,128]

↓pool3

[batch_size,8,8,128]

↓flat

[batch_size,8*8*128]

↓fc1

[batch_size,1024]

↓fc2

[batch_size,3]

↓softmax

[batch_size,3]
