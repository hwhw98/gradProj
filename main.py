

import tensorflow as tf

import sys
import time
import numpy as np

argument = sys.argv
file_path = argument[1]
checkpoint_path = argument[2]

train_split=0.8
val_split=0.1
test_split=0.1

BUFFER_SIZE = 873406
BATCH_SIZE = 128
EPOCHS = 100
d_model = 256
num_layers = 4
num_heads = 4
dff = 256
rate = 0.1
epsilon = 1e-6

THRESHOLD = 0.5
MAX_POSITIONAL_ENCODING = 401
MZ_EMBEDDING_SIZE = 200000 # normal : 200000, z-scored normalized : 5849, min-max normalized : 1001
INTENSITY_EMBEDDING_SIZE = 10001 # normal : 10001, z-scored normalized : 1001, min-max normalized : 1001

dataset = tf.data.experimental.load(file_path, element_spec=(tf.RaggedTensorSpec(shape=([None,]), dtype=(tf.int32)),
                                                             tf.RaggedTensorSpec(shape=([None,]), dtype=(tf.int32)),
                                                             tf.RaggedTensorSpec(shape=([None,]), dtype=(tf.int32))))

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

# GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

ds_size = 873406

loss_object = tf.keras.losses.BinaryCrossentropy()

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b = True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += ( mask * -1e9 )
        
    attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis = -1)
    output = tf.matmul(attention_weights, v)

    return output

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(d_model)
    ])

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model and num_heads are not multiples"

        self.depth = d_model // self.num_heads # self.depth = int

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0] # .numpy function 안쓰기? 여기선 안써도 정상 작동

        q = self.wq(q)  # q, k, v : (batch, peak, d_model)
        k = self.wk(k)  
        v = self.wv(v)  

        q = self.split_heads(q, batch_size)  # q, k, v : (batch, num_heads, peak, depth)
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  

        scaled_attention = scaled_dot_product_attention(q, k, v, mask) # mask : (batch, 1, 1, peak)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # scaled_attention : (batch, peak, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, self.d_model)) # concat_attention : (batch, peak, d_model)

        output = self.dense(concat_attention) # attention 결과에 dense layer => normalize 효과를 기대하고 쓰는 것

        return output

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=epsilon)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate): # d_model 만 넘기고 나머지 3개는 Transformer 에서 임베딩해서 넣는게 어떤가?
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        # Embedding : in transformer
        self.pos_encoding = positional_encoding(MAX_POSITIONAL_ENCODING, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        peak_num = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # normalize
        x += self.pos_encoding[:, peak_num, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask) # 생각해야 할 부분 : 인코더 레이어 하나에서는 Attention 이 한번만 일어남 => 2~3개로 추가해도 되지 않을까?

        return x

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super().__init__()

        self.d_model = d_model

        self.encoder = Encoder(num_layers, self.d_model, num_heads, dff, rate)
        
        #memory leak => put embedding to cpu

        self.embedding_mz = tf.keras.layers.Embedding(MZ_EMBEDDING_SIZE, self.d_model)
        self.embedding_intensity = tf.keras.layers.Embedding(INTENSITY_EMBEDDING_SIZE, self.d_model)

        self.final_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid) # tf.squeeze 를 사용할 수 있는지는 차후에 판단합시다
    
    def call(self, inputs, training):  

        # inp : (batch_size, num_peak)
        mz, intensity = inputs

        # mask size : (batch, 1, 1, num_peak)
        padding_mask = self.create_padding_mask(mz) # padding mask 함수 수정
        
        # put embedding to cpu?

        mz_input = self.embedding_mz(mz)
        intensity_input = self.embedding_intensity(intensity)

        enc_input = mz_input + intensity_input
        enc_output = self.encoder(enc_input, training, padding_mask)
        # Encoder 에 Attention weight 나오는 부분 추가

        final_output = self.final_layer(enc_output)

        return tf.squeeze(final_output, [2]) # final output : logit tensor, 확률 / attention weights : 쓸데가 있는지는 잘 모르겠음 / padding_mask : loss/accuracy 에서 끌어다 쓰기 위함

    def create_padding_mask(self, mz):
        mask = tf.math.equal(0, mz)
        mask = tf.cast(mask, dtype=tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :] # 0 : data, 1 : mask

def get_ds_splits(ds, ds_size, train_split, val_split, test_split, shuffle):
    assert (train_split + test_split + val_split ) == 1
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE, seed=8) # seed is for comparison of hyerparameters
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def padding(mz, intensity, truth):
    return mz.to_tensor(default_value=0), intensity.to_tensor(default_value=0), truth.to_tensor(default_value=0)

def make_batches(ds): # already shuffled in get_ds_splits, so no shuffle here
    return (
        ds
        .batch(BATCH_SIZE)
        .map(padding)
        .prefetch(tf.data.AUTOTUNE))
    
def process_ds(dataset): # dataset is tfds
    
    train_ds, valid_ds, test_ds = get_ds_splits(dataset, ds_size, train_split, val_split, test_split, shuffle=True)
    
    train_batches = make_batches(train_ds).with_options(options)
    valid_batches = make_batches(valid_ds).with_options(options)
    test_batches = make_batches(test_ds).with_options(options)

    return train_batches, valid_batches, test_batches

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule): # Learning rate is same to the paper
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

train_batches, valid_batches, test_batches = process_ds(dataset) # use distributed dataset

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9)
model = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, rate=rate) # create Transformer

ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

acc_100 = 0
accuracy_object = tf.keras.metrics.Accuracy()
precision_object = tf.keras.metrics.Precision()
recall_object = tf.keras.metrics.Recall()

def loss_function(real, pred, mask):
    real = tf.cast(real, dtype=tf.float32)
    pred = tf.cast(pred, dtype=tf.float32)

    real *= mask
    pred *= mask

    loss_ = loss_object(real, pred)

    return tf.reduce_sum(loss_)

def accuracy_function(real, pred, mask, valid):
    global acc_100
    
    real = tf.cast(real, dtype=tf.float32)

    accuracy = tf.math.equal(real, pred)
    accuracy = tf.cast(accuracy, dtype=tf.float32)
    accuracy *= mask

    if valid:
        global accuracy_object
        global precision_object
        global recall_object
        
        for i in range(tf.shape(mask)[0]):
            elements = int(tf.reduce_sum(mask[i]).numpy())
            accuracy_object.reset_states()
            if accuracy_object(real[i][:elements], pred[i][:elements]).numpy() == 1.:
                acc_100 += 1 # accuracy
            
            precision_object(real[i][:elements], pred[i][:elements])
            recall_object(real[i][:elements], pred[i][:elements])
    
    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask) * 100

def process_pred(pred):
    return tf.cast(tf.math.greater(pred, tf.constant([THRESHOLD])), dtype=tf.float32)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.Mean(name='valid_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

def train_step(mz, intensity, truth):
    
    mask = tf.math.not_equal(mz, 0)
    mask = tf.cast(mask, dtype=tf.float32)

    with tf.GradientTape() as tape:
        pred = model([mz, intensity], training=True)
        loss = loss_function(truth, pred, mask)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(truth, process_pred(pred), mask, valid=False))
    
def valid_step(mz, intensity, truth):
    
    mask = tf.math.not_equal(mz, 0)
    mask = tf.cast(mask, dtype=tf.float32)

    with tf.GradientTape() as tape:
        pred = model([mz, intensity], training=True)
        loss = loss_function(truth, pred, mask)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    valid_loss(loss)
    valid_accuracy(accuracy_function(truth, process_pred(pred), mask, valid=True))
    
def test_step(mz, intensity, truth):
    
    mask = tf.math.not_equal(mz, 0)
    mask = tf.cast(mask, dtype=tf.float32)

    pred = model([mz, intensity], training=False)
    loss = loss_function(truth, pred, mask)

    test_loss(loss)
    test_accuracy(accuracy_function(truth, process_pred(pred), mask, valid=True))

for epoch in range(EPOCHS):
    acc_100 = 0

    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()

    for (batch, (mz, intensity, truth)) in enumerate(train_batches):

        train_step(mz, intensity, truth)

        if batch % 3000 == 0:
            print(f'Epoch {epoch+1} train Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}%')
            print(f'Time taken until {batch} batches : {time.time() - start:.2f} secs')

    print(f'acc 100% : {acc_100} peaks\n')
    print('Validation start')
    acc_100 = 0
    precision_object.reset_states()
    recall_object.reset_states()

    for (batch, (mz, intensity, truth)) in enumerate(valid_batches):

        valid_step(mz, intensity, truth)

    print(f'Epoch {epoch+1} valid Loss : {valid_loss.result():.4f}')
    print(f'Epoch {epoch+1} valid Accuracy : {valid_accuracy.result():.4f}%')
    print(f'Epoch {epoch+1} precision : {precision_object.result():.4f}')
    print(f'Epoch {epoch+1} recall : {recall_object.result():.4f}')
    print(f'acc 100% : {acc_100} peaks\n')
    
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}\n')

acc_100 = 0
test_loss.reset_states()
test_accuracy.reset_states()
precision_object.reset_states()
recall_object.reset_states()

for (batch, (mz, intensity, truth)) in enumerate(test_batches):

    test_step(mz, intensity, truth)
    
print(f'Test loss : {test_loss.result():.4f}')
print(f'Test accuracy : {test_accuracy.result():.4f}')
print(f'Test precision : {precision_object.result():.4f}')
print(f'Test recall : {recall_object.result():.4f}')
print(f'acc 100% : {acc_100} peaks\n')
print('Success!')
