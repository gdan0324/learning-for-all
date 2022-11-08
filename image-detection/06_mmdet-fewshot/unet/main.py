# import os
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from network import unet
#
# data_dir = './dataset'
# train_normal_dir = data_dir + '/datasets_normal/'
# train_outlier_dir = data_dir + '/datasets_outlier/'
# train_outlier_label_dir = data_dir + '/Annotations/'
# AUTOTUNE = tf.data.experimental.AUTOTUNE
#
#
# def decode_and_resize(filename):
#     image_string = tf.io.read_file(filename)
#     image_decoded = tf.image.decode_png(image_string)
#     image_resized = tf.image.resize(image_decoded, [600, 600]) / 255.0
#     return image_resized
#
#
# def test():
#     train_normal_file = tf.constant([train_normal_dir + filename for filename in os.listdir(train_normal_dir)])
#     train_outlier_file = tf.constant([train_outlier_dir + filename for filename in os.listdir(train_outlier_dir)])
#     train_outlier_label_file = tf.constant(
#         [train_outlier_label_dir + filename for filename in os.listdir(train_outlier_label_dir)])
#
#     train_dataset = tf.data.Dataset.from_tensor_slices(train_normal_file)
#     train_dataset = train_dataset.map(decode_and_resize)
#     # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
#     train_dataset = train_dataset.shuffle(buffer_size=23000)
#     train_dataset = train_dataset.batch(10)
#     print(train_dataset.take())
#
#     net = unet(input_shape=(600, 600, 3), out_channel=3)
#     num_epochs = 10
#     # loss_object = tf.keras.losses.MeanSquaredError()
#     # optimizer = tf.keras.optimizers.Adam()
#
#     net.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#         loss=tf.keras.losses.mean_squared_error,
#         metrics=[tf.keras.metrics.mean_squared_error]
#     )
#     net.fit(train_dataset)
#
#     # for epoch in range(epochs):
#     #     with tf.GradientTape() as tape:
#     #         y_tilde = net(train_dataset)
#     #         loss = loss_object(train_dataset, y_tilde)
#     #         loss = tf.reduce_mean(loss)
#     #         print(f"loss:{loss.numpy()}")
#     #     grads = tape.gradient(loss, net.variables)
#     #     optimizer.apply_gradients(zip(grads, net.variables))
#
#     print("hello world!")


if __name__ == "__main__":
    train_normal_file = tf.constant([train_normal_dir + filename for filename in os.listdir(train_normal_dir)])
    path_ds = tf.data.Dataset.from_tensor_slices(train_normal_file)
    image_ds = path_ds.map(decode_and_resize)
    train_ds = tf.data.Dataset.zip((image_ds, image_ds))
    ds = train_ds.shuffle(buffer_size=55)
    ds = ds.batch(1)

    net = unet(input_shape=[600, 600, 3], out_channel=3)
    net.load_weights("./result/tf_model.h5")
    num_epochs = 10

    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.mean_absolute_error,
    )
    net.fit(ds)
    net.save('./result/tf_model.h5')
