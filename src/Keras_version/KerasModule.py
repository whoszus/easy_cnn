import numpy

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint


class NEEModules(KerasModel):

    def __init__(self, batch_x, batch_y):
        super().__init__()
        self.epochs = 3
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        # self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model(batch_x, batch_y)
        # self.fit(X_train, y_train, X_val, y_val)

    # def preprocessing(self, X):
    #     X_list = split_features(X)
    #     return X_list

    def __build_keras_model(self, batch_x, batch_y):

        # 数据[dev_name：724, dev_type：33, city：22, time：int, alarm_level：12]
        # 进行设备名Embedding
        # 将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        # 该层只能用作模型中的第一层。
        input_dev_name = Input(shape=(batch_x,))
        output_dev_name = Embedding(724, 50, name='devName_embedding')(input_dev_name)
        output_dev_name = Reshape(target_shape=(batch_x, 50))(output_dev_name)
        print(output_dev_name)
        # dev_type
        input_dev_type =Input(shape=(batch_x,))
        output_dev_type = Embedding(33, 17, name='devType_embedding')(input_dev_type)
        output_dev_type = Reshape(target_shape=(batch_x,17))(output_dev_type)
        # city
        input_city = Input(shape=(batch_x,))
        output_city = Embedding(22, 6, name='city_embedding')(input_city)
        output_city = Reshape(target_shape=(batch_x,6,))(output_city)
        # 全连接层。
        input_time = Input(shape=(batch_x,))
        output_time = Dense(1)(input_time)

        # alarm_level
        input_level = Input(shape=(batch_x,))
        output_level = Dense(1)(input_level)

        input_model_i = [input_dev_name, input_dev_type, input_city, input_time, input_level]
        input_model = [numpy.array(input_model_i)]

        output_embeddings_i = [output_dev_name, output_dev_type, output_city, output_time, output_level]
        output_embeddings = [numpy.array(output_embeddings_i)]
        # output_embeddings = tf.convert_to_tensor(output_embeddings)
        output_model = Concatenate()(output_embeddings_i)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1)(output_model)
        output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=input_model_i, outputs=output_model)
        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    # def _val_for_fit(self, val):
    #     val = numpy.log(val) / self.max_log_y
    #     return val
    #
    # def _val_for_pred(self, val):
    #     return numpy.exp(val * self.max_log_y)

    # def fit(self, X_train, y_train, X_val, y_val):
    #     # self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
    #     #                validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
    #     #                epochs=self.epochs, batch_size=128,
    #     #                # callbacks=[self.checkpointer],
    #     #                )
    #     # self.model.load_weights('best_model_weights.hdf5')
    #
    #
    #
    #     self.model.fit(X_train, y_train, epochs=1, batch_size=64)
    #     print("Result on validation data: ", self.evaluate(X_val, y_val))
    #
    # def guess(self, features):
    #     features = self.preprocessing(features)
    #     result = self.model.predict(features).flatten()
    #     return self._val_for_pred(result)
if __name__ == '__main__':
    cnn = NEEModules()

