import numpy

from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint


class NEEModules(KerasModel):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.epochs = 3
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        # self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def __build_keras_model(self):


        # 进行设备名Embedding
        # 将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        # 该层只能用作模型中的第一层。
        input_dev_name = Input(shape=(1,))
        output_dev_name = Embedding(724, 8, name='devName_embedding')(input_dev_name)
        output_dev_name = Reshape(target_shape=(10,))(output_dev_name)
        # city
        input_city = Input(shape=(1,))
        output_city = Embedding(22, 6, name='city_embedding')(input_city)
        output_city = Reshape(target_shape=(6,))(output_city)

        # 全连接层。
        input_promo = Input(shape=(1,))
        output_promo = Dense(1)(input_promo)

        input_year = Input(shape=(1,))
        output_year = Embedding(3, 2, name='year_embedding')(input_year)
        output_year = Reshape(target_shape=(2,))(output_year)

        input_month = Input(shape=(1,))
        output_month = Embedding(12, 6, name='month_embedding')(input_month)
        output_month = Reshape(target_shape=(6,))(output_month)

        input_day = Input(shape=(1,))
        output_day = Embedding(31, 10, name='day_embedding')(input_day)
        output_day = Reshape(target_shape=(10,))(output_day)

        input_germanstate = Input(shape=(1,))
        output_germanstate = Embedding(12, 6, name='state_embedding')(input_germanstate)
        output_germanstate = Reshape(target_shape=(6,))(output_germanstate)

        input_model = [input_dev_name, input_city, input_promo,
                       input_year, input_month, input_day, input_germanstate]

        output_embeddings = [output_dev_name, output_city, output_promo,
                             output_year, output_month, output_day, output_germanstate]

        output_model = Concatenate()(output_embeddings)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1)(output_model)
        output_model = Activation('sigmoid')(output_model)
        self.model = KerasModel(inputs=input_model, outputs=output_model)
        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       epochs=self.epochs, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)



def split_features(X):
    X_list = []

    store_index = X[..., [1]]
    X_list.append(store_index)

    day_of_week = X[..., [2]]
    X_list.append(day_of_week)

    promo = X[..., [3]]
    X_list.append(promo)

    year = X[..., [4]]
    X_list.append(year)

    month = X[..., [5]]
    X_list.append(month)

    day = X[..., [6]]
    X_list.append(day)

    State = X[..., [7]]
    X_list.append(State)

    return X_list


