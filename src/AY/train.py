import pickle
import numpy



# 第一次进入为false
embeddings_as_input = False
save_embeddings = True
saved_embeddings_fname = "embeddings.pickle"  # set save_embeddings to True to create this file
train_ratio = 0.8
f = open('feature_train_data.pickle', 'rb')
(X, y) = pickle.load(f)
num_records = len(X)
train_size = int(train_ratio * num_records)

X_train = X[:train_size]
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]
models = []


for i in range(5):
    models.append(NN_with_EntityEmbedding(X_train, y_train, X_val, y_val))


if save_embeddings:
    model = models[0].model
    store_embedding = model.get_layer('store_embedding').get_weights()[0]
    dow_embedding = model.get_layer('dow_embedding').get_weights()[0]
    year_embedding = model.get_layer('year_embedding').get_weights()[0]
    month_embedding = model.get_layer('month_embedding').get_weights()[0]
    day_embedding = model.get_layer('day_embedding').get_weights()[0]
    german_states_embedding = model.get_layer('state_embedding').get_weights()[0]
    with open(saved_embeddings_fname, 'wb') as f:
        pickle.dump([store_embedding, dow_embedding, year_embedding,
                     month_embedding, day_embedding, german_states_embedding], f, -1)





def embed_features(X):
    # f_embeddings = open("embeddings_shuffled.pickle", "rb")
    f_embeddings = open(saved_embeddings_fname, "rb")
    embeddings = pickle.load(f_embeddings)
    index_embedding_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    X_embedded = []
    (num_records, num_features) = X.shape
    for record in X:
        embedded_features = []
        for i, feat in enumerate(record):
            feat = int(feat)
            if i not in index_embedding_mapping.keys():
                embedded_features += [feat]
            else:
                embedding_index = index_embedding_mapping[i]
                embedded_features += embeddings[embedding_index][feat].tolist()
        X_embedded.append(embedded_features)
    return numpy.array(X_embedded)

# def sample(X, y, n):
#     num_row = X.shape[0]
#     indices = numpy.random.randint(num_row, size=n)
#     return X[indices, :], y[indices]


