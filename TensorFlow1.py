from collections import Counter
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
import pandas as pd

# 导入sklearn集合中的数据集,有监督学习，即里面的数据已经分好了类别
# 具体参见http://scikit-learn.org/stable/datasets/twenty_newsgroups.html
categories = ["comp.graphics","sci.space","rec.sport.baseball"]
train_set = fetch_20newsgroups(subset='train',categories=categories)
test_set = fetch_20newsgroups(subset='test', categories=categories)
# print('total texts in train:',len(train_set.data))
# print('total texts in test:',len(test_set.data))
# 建立数据集单词字典，最终形式是text_index['the'] = 数量
vocab = Counter()
for data in train_set.data:
    for word in data.split(' '):
        vocab[word.lower()] += 1

for test_data in test_set.data:
    for word in test_data.split(' '):
        vocab[word] += 1
print(len(vocab))
total_words = len(vocab)

def get_index(vocab):
    # 先声明word是字典，否则word[element]报错
    word={}
    for i, element in enumerate(vocab):
        word[element.lower()] = i
    return word

text_index = get_index(vocab)

print("the is %s" % text_index['the'])

# 每层神经元数，包括输入神经元，隐藏神经元，输出神经元"comp.graphics","sci.space","rec.sport.baseball"
n_hidden1 = 100
n_hiddent2 = 100
n_input_number = total_words
n_class = 3

# 在 神经网络的术语里，一次 epoch = 一个向前传递（得到输出的值）和一个所有训练示例的向后传递（更新权重）。
learning_rate = 0.01
batch_size = 150
training_epochs = 10
display_step = 1



# shape的None 元素对应于大小可变的维度在测试模型时，我们将用更大的批处理来提供字典，
# 这就是为什么需要定义一个可变的批处理维度。
input_tensor = tf.placeholder(tf.float32, [None, n_input_number], name='input')
output_tensor = tf.placeholder(tf.float32, [None, n_class], name='output')

# 神经元计算
def out_prediction(input_tensor, weights, biases):
    # 定义乘法运算矩阵乘法
    # relu是激活函数
    layer_1_multiplication = tf.matmul(input_tensor,weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_addition)

    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2_activation = tf.nn.relu(layer_2_addition)

    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition
# shape参数含义；[]表示一个数，[3]表示长为3的向量，
# [2,3]表示矩阵或者张量(tensor)同一个线性变换在不同的基下的表示
# https://www.zhihu.com/question/20695804
# 利用正态分布启动权值和偏差值
weights = {
    'h1':tf.Variable(tf.random_normal([n_input_number, n_hidden1])),
    'h2':tf.Variable(tf.random_normal([n_hidden1, n_hiddent2])),
    'out':tf.Variable(tf.random_normal([n_hiddent2, n_class]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden1])),
    'b2':tf.Variable(tf.random_normal([n_hiddent2])),
    'out':tf.Variable(tf.random_normal([n_class]))
}


prediction = out_prediction(input_tensor, weights, biases)

# 由于分类问题，所以使用交叉熵误差进行优化，不断更新权值和output_tensor
cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
loss = tf.reduce_mean(cross_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 数据初始化
init = tf.global_variables_initializer()

# 批处理数据函数
def get_batch(df, i, batch_size):
    batches = []
    results = []
    texts = df.data[i * batch_size:i * batch_size + batch_size]
    categories = df.target[i * batch_size:i * batch_size + batch_size]
# 构建矩阵索引
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[text_index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        y = np.zeros((3), dtype=float)
        if category == 0:
            y[0] = 1
        elif category == 1:
            y[1] = 1
        else:
            y[2] = 1
        results.append(y)

    return np.array(batches), np.array(results)

# Session定义了Operation操作对象执行环境，在这里进行模型训练
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_set.data)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x,batch_y = get_batch(train_set,i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            # op（source op），当运行该函数，启动默认图，即运行out_prediction，并不断更新权值和分类结果
            # tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
            # feed_dict 参数是我们为每步运行所输入的数据。为了传递这个数据，我们需要定义tf.placeholders（提供给 feed_dict）
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")


    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(train_set.target)
    batch_x_test, batch_y_test = get_batch(test_set,0,total_test_data)
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))












