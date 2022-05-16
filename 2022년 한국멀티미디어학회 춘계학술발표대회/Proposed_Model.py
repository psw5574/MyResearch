import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os

def pre_training():
    os.makedirs("./model/pre_trained_model", exist_ok=True)
    os.makedirs("./loss/pre_trained_model", exist_ok=True)

    file_path_list = glob.glob("dataset\\*.csv")
    location_list = []
    for file_path in file_path_list:
        location_list.append(file_path[len("dataset\\"):-4])

    for file_path in file_path_list:
        current_file_name = file_path[len("dataset\\"):-4]
        source_location_list = location_list.copy()
        source_location_list.remove(current_file_name)

        train_dataset = pd.DataFrame()
        validation_dataset = pd.DataFrame()
        for source_location in source_location_list:
            current_dataset = pd.read_csv("./dataset/" + source_location + ".csv", engine="python")

            current_train_dataset = current_dataset.iloc[:-65083,:].dropna().reset_index(drop=True)
            current_validation_dataset = current_train_dataset.sample(frac=0.1)
            train_dataset_index = list(set(current_train_dataset.index.values.tolist()) - set(current_validation_dataset.index.values.tolist()))
            current_train_dataset = current_train_dataset.iloc[train_dataset_index,:].reset_index(drop=True)

            train_dataset = pd.concat([train_dataset, current_train_dataset], axis=0)
            validation_dataset = pd.concat([validation_dataset, current_validation_dataset], axis=0)

        X = tf.placeholder(tf.float32, [None, INPUT_LAYER_NODE])
        Y = tf.placeholder(tf.float32, [None, 1])

        Layers = {}
        for i in range(1, NUM_HIDDEN_LAYER + 2):
            if i == 1:
                shape_1 = INPUT_LAYER_NODE
                shape_2 = HIDDEN_LAYER_NODE
            elif i == NUM_HIDDEN_LAYER + 1:
                shape_1 = HIDDEN_LAYER_NODE
                shape_2 = 1
            else:
                shape_1 = HIDDEN_LAYER_NODE
                shape_2 = HIDDEN_LAYER_NODE
            Layer = {"weights": tf.get_variable("W" + str(i), shape=[shape_1, shape_2],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                     "biases": tf.Variable(tf.random_normal([shape_2]))}
            Layers[i - 1] = Layer
        Out = {}
        Out[0] = {"layers": X}
        for i in range(0, NUM_HIDDEN_LAYER + 1):
            weight = Layers[i]["weights"]
            bias = Layers[i]["biases"]
            if i == NUM_HIDDEN_LAYER:
                layer = tf.matmul(Out[i]["layers"], weight) + bias
            else:
                layer = tf.nn.relu(tf.matmul(Out[i]["layers"], weight) + bias)
            Out[i + 1] = {"layers": layer}

        cost = tf.losses.mean_squared_error(labels=Y, predictions=Out[NUM_HIDDEN_LAYER + 1]["layers"])
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        loss_log_file = pd.DataFrame(columns=["Train_Loss", "Validation_Loss"])
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        train_batch = int(len(train_dataset) / BATCH_SIZE)

        validation_cost_list = []
        minimum_validation_loss = np.inf

        saver = tf.train.Saver()
        for epoch in range(EPOCH):
            total_cost = 0
            for i in range(train_batch):
                batch_data = train_dataset.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :-1]
                label_data = train_dataset.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, -1]
                batch_data = batch_data.values
                label_data = np.reshape(label_data.values, (len(label_data.values), 1))
                feed_dict = {X: batch_data, Y: label_data}
                train_cost, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                total_cost += train_cost
            batch_data = train_dataset.iloc[(i + 1) * BATCH_SIZE:, :-1]
            label_data = train_dataset.iloc[(i + 1) * BATCH_SIZE:, -1]
            batch_data = batch_data.values
            label_data = np.reshape(label_data.values, (len(label_data.values), 1))
            feed_dict = {X: batch_data, Y: label_data}
            train_cost, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            total_cost += train_cost

            validation_cost = sess.run(cost, feed_dict={X: validation_dataset.iloc[:, :-1],
                                                  Y: np.reshape(validation_dataset.iloc[:, -1].values, (len(validation_dataset.iloc[:, -1].values), 1))})

            loss_log_file.append({"Train_Loss": (total_cost / (train_batch + 1)), "Validation_Loss": validation_cost}, ignore_index=True)

            print('Epoch:', '%04d' % (epoch + 1), 'train loss =', '{:.9f}'.format(total_cost / (train_batch + 1)), 'test loss =', '{:.9f}'.format(validation_cost))

            if (epoch == 0) or (epoch % 10 == 0):
                if validation_cost <= minimum_validation_loss:
                    minimum_validation_loss = validation_cost
                    saver.save(sess, "./model/pre_trained_model/"+current_file_name+".ckpt")

            validation_cost_list.append(validation_cost)
            if epoch >= 1000:
                if validation_cost - np.mean(validation_cost_list[int(round(len(validation_cost_list) * 9 / 10)):]) >= 0.0001:
                    print("Early Stop!")
                    break
        print('Learning Finished!')
        sess.close()

        loss_log_file.to_csv("./loss/pre_trained_model/" + current_file_name + "_loss.csv")

        tf.reset_default_graph()

def fine_tuning():
    os.makedirs("./model/fine_tuned_model", exist_ok=True)
    os.makedirs("./loss/fine_tuned_model", exist_ok=True)
    os.makedirs("./result", exist_ok=True)

    file_path_list = glob.glob("dataset\\*.csv")
    for file_path in file_path_list:
        current_file_name = file_path[len("dataset\\"):-4]

        current_dataset = pd.read_csv(file_path, engine="python")

        train_dataset = current_dataset.iloc[-128426:-65083, :].dropna().reset_index(drop=True)

        validation_dataset = train_dataset.sample(frac=0.1)
        train_dataset_index = list(set(train_dataset.index.values.tolist()) - set(validation_dataset.index.values.tolist()))
        train_dataset = train_dataset.iloc[train_dataset_index, :].reset_index(drop=True)

        test_dataset = current_dataset.iloc[-65083:, :].dropna().reset_index(drop=True)

        X = tf.placeholder(tf.float32, [None, INPUT_LAYER_NODE])
        Y = tf.placeholder(tf.float32, [None, 1])

        Layers = {}
        for i in range(1, NUM_HIDDEN_LAYER + 2):
            if i == 1:
                shape_1 = INPUT_LAYER_NODE
                shape_2 = HIDDEN_LAYER_NODE
            elif i == NUM_HIDDEN_LAYER + 1:
                shape_1 = HIDDEN_LAYER_NODE
                shape_2 = 1
            else:
                shape_1 = HIDDEN_LAYER_NODE
                shape_2 = HIDDEN_LAYER_NODE
            Layer = {"weights": tf.get_variable("W" + str(i), shape=[shape_1, shape_2], initializer=tf.contrib.layers.xavier_initializer()),
                     "biases": tf.Variable(tf.random_normal([shape_2]))}
            Layers[i - 1] = Layer
        Out = {}
        Out[0] = {"layers": X}
        for i in range(0, NUM_HIDDEN_LAYER + 1):
            weight = Layers[i]["weights"]
            bias = Layers[i]["biases"]
            if i == NUM_HIDDEN_LAYER:
                layer = tf.matmul(Out[i]["layers"], weight) + bias
            else:
                layer = tf.nn.relu(tf.matmul(Out[i]["layers"], weight) + bias)
            Out[i + 1] = {"layers": layer}

        cost = tf.losses.mean_squared_error(labels=Y, predictions=Out[NUM_HIDDEN_LAYER + 1]["layers"])
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        loss_log_file = pd.DataFrame(columns=["Train_Loss", "Validation_Loss"])
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, "./model/fine_tuned_model/"+current_file_name+".ckpt")

        train_batch = int(len(train_dataset) / BATCH_SIZE)

        validation_cost_list = []
        minimum_validation_loss = np.inf

        saver = tf.train.Saver()
        for epoch in range(EPOCH):
            total_cost = 0
            for i in range(train_batch):
                batch_data = train_dataset.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :-1]
                label_data = train_dataset.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, -1]
                batch_data = batch_data.values
                label_data = np.reshape(label_data.values, (len(label_data.values), 1))
                feed_dict = {X: batch_data, Y: label_data}
                train_cost, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                total_cost += train_cost
            batch_data = train_dataset.iloc[(i + 1) * BATCH_SIZE:, :-1]
            label_data = train_dataset.iloc[(i + 1) * BATCH_SIZE:, -1]
            batch_data = batch_data.values
            label_data = np.reshape(label_data.values, (len(label_data.values), 1))
            feed_dict = {X: batch_data, Y: label_data}
            train_cost, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            total_cost += train_cost

            validation_cost = sess.run(cost, feed_dict={X: validation_dataset.iloc[:, :-1],
                                                  Y: np.reshape(validation_dataset.iloc[:, -1].values, (len(validation_dataset.iloc[:, -1].values), 1))})

            loss_log_file.append({"Train_Loss": (total_cost / (train_batch + 1)), "Validation_Loss": validation_cost}, ignore_index=True)

            print('Epoch:', '%04d' % (epoch + 1), 'train loss =', '{:.9f}'.format(total_cost / (train_batch + 1)), 'test loss =', '{:.9f}'.format(validation_cost))

            if (epoch == 0) or (epoch % 10 == 0):
                if validation_cost <= minimum_validation_loss:
                    minimum_validation_loss = validation_cost
                    saver.save(sess, "./model/fine_tuned_model/" + current_file_name + ".ckpt")

            validation_cost_list.append(validation_cost)
            if epoch >= 1000:
                if validation_cost - np.mean(validation_cost_list[int(round(len(validation_cost_list) * 9 / 10)):]) >= 0.00001:
                    print("Early Stop!")
                    break
        print('Learning Finished!')
        sess.close()

        loss_log_file.to_csv("./loss/fine_tuned_model/" + current_file_name + "_loss.csv")

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, "./model/fine_tuned_model/" + current_file_name + ".ckpt")

        Y_Pred = sess.run(Out[NUM_HIDDEN_LAYER + 1]["layers"], feed_dict={X: test_dataset.iloc[:, :-1],
                                                                          Y: np.reshape(test_dataset.iloc[:, -1].values,(len(test_dataset.iloc[:,-1].values), 1))})
        Y_Pred = pd.DataFrame(Y_Pred, columns=["Real"])

        result = pd.concat([test_dataset.iloc[:, -1].reset_index(drop=True), Y_Pred], axis=1)
        result.columns = ["Real", "Prediction"]

        result.to_csv("./result/" + current_file_name + "_Proposed_Model.csv", header=True, index=False)

        tf.reset_default_graph()


def start():
    global LEARNING_RATE, BATCH_SIZE, EPOCH, INPUT_LAYER_NODE, HIDDEN_LAYER_NODE, NUM_HIDDEN_LAYER

    LEARNING_RATE = 0.0001
    BATCH_SIZE = 1000
    EPOCH = 10000
    INPUT_LAYER_NODE = 16
    HIDDEN_LAYER_NODE = 11
    NUM_HIDDEN_LAYER = 5

    pre_training()
    fine_tuning()

start()