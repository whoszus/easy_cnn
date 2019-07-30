import pickle
import  numpy
from AY.NeeModule import NEEModules

batch_x = 20
batch_y = 5
# 第一次进入为false
embeddings_as_input = False
save_embeddings = True
saved_embeddings_fname = "embeddings.pickle"  # set save_embeddings to True to create this file
train_ratio = 0.8
f = open('feature_train_data.pickle', 'rb')
(X_INIT_DATA, Y_INIT_DATA) = pickle.load(f)
num_records = len(X_INIT_DATA)
train_size = int(train_ratio * num_records)


x_train_all = []
x_data_tmp = []
xy_data_tmp = []
data_index = 1

y_train_all = []
y_data_tmp = []


# for i in range(5):
#     models.append()
#

nee_moudle = NEEModules()

for x in range(X_INIT_DATA) :
    x_data_tmp.append(x)
    if data_index % batch_x == 0:
        x_train_all.append(x_data_tmp)
        x_data_tmp = []
    data_index +=1

data_index = 1
for y in range(Y_INIT_DATA):
    y_data_tmp.append(y)
    if data_index % batch_x == 0:
        y_train_all.append(y_data_tmp)
        y_data_tmp = []
    data_index +=1


X_train = x_train_all[:train_size]
X_val = x_train_all[train_size:]
y_train = y_train_all[:train_size]
y_val = y_train_all[train_size:]

nee_moudle.fit(X_train, y_train, epochs=1, batch_size=64)
print("Result on validation data: ", nee_moudle.evaluate(X_val, y_val))


#
# if save_embeddings:
#     model = models[0].model
#     devName_embedding = model.get_layer('devName_embedding').get_weights()[0]
#     devType_embedding = model.get_layer('devType_embedding').get_weights()[0]
#     city_embedding = model.get_layer('city_embedding').get_weights()[0]
#     with open(saved_embeddings_fname, 'wb') as f:
#         pickle.dump([devName_embedding, devType_embedding, city_embedding], f, -1)
#
#
#
#
#
# def embed_features(X):
#     # f_embeddings = open("embeddings_shuffled.pickle", "rb")
#     f_embeddings = open(saved_embeddings_fname, "rb")
#     embeddings = pickle.load(f_embeddings)
#     index_embedding_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}
#     X_embedded = []
#     (num_records, num_features) = X.shape
#     for record in X:
#         embedded_features = []
#         for i, feat in enumerate(record):
#             feat = int(feat)
#             if i not in index_embedding_mapping.keys():
#                 embedded_features += [feat]
#             else:
#                 embedding_index = index_embedding_mapping[i]
#                 embedded_features += embeddings[embedding_index][feat].tolist()
#         X_embedded.append(embedded_features)
#     return numpy.array(X_embedded)

# def sample(X, y, n):
#     num_row = X.shape[0]
#     indices = numpy.random.randint(num_row, size=n)
#     return X[indices, :], y[indices]


