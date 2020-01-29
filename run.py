from helperFunction import *
from info import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from model_with_dwt1 import *

EPOCH_NUM = 100
minibatch_size = 50
learning_rate = 1e-2

# load images
csImgs, csLabels = loadImages(imgs_path_withfaces, label=1)
ctImgs, ctLabels = loadImages(imgs_path_nofaces, label=0)
print(sizeof_fmt(csImgs))
print(sizeof_fmt(ctImgs))

# split train and test images
Xtrain_cs, Xtest_cs, ytrain_cs, ytest_cs = train_test_split(csImgs, csLabels, test_size=0.25)
Xtrain_ct, Xtest_ct, ytrain_ct, ytest_ct = train_test_split(ctImgs, ctLabels, test_size=0.25)
print("sum of y cs:{}, sum of y ct:{}".format(ytrain_cs.sum()+ytest_cs.sum(), ytrain_ct.sum()+ytest_ct.sum()))
print("Xtrain cs:{}, ytrain cs:{}, Xtest cs:{}, ytest cs:{}".format(Xtrain_cs.shape, ytrain_cs.shape,
                                                                  Xtest_cs.shape, ytest_cs.shape))
print("Xtrain ct:{}, ytrain ct:{}, Xtest ct:{}, ytest ct:{}".format(Xtrain_ct.shape, ytrain_ct.shape,
                                                                  Xtest_ct.shape, ytest_ct.shape))
del csImgs, ctImgs

# integrate cs and ct images
Xtrain = np.vstack((Xtrain_cs, Xtrain_ct))
ytrain = np.hstack((ytrain_cs, ytrain_ct))
Xtrain, ytrain = shuffle(Xtrain, ytrain)
Xtrain = np.expand_dims(Xtrain, axis=1)
del Xtrain_cs, Xtrain_ct

Xtest = np.vstack((Xtest_cs, Xtest_ct))
ytest = np.hstack((ytest_cs, ytest_ct))
Xtest, ytest = shuffle(Xtest, ytest)
Xtest = np.expand_dims(Xtest, axis=1)
del Xtest_cs, Xtest_ct
print("Xtrain shape:{}, ytrain shape:{}\nXtest shape:{}, ytest shape:{}".format(Xtrain.shape, ytrain.shape,
                                                                                Xtest.shape, ytest.shape))
# prepare model
model = Net()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
print("start training...")
# num_of_batches = np.int(np.ceil(Xtrain.shape[0] / minibatch_size))
for epoch in range(EPOCH_NUM):
    Xtrain, ytrain = shuffle(Xtrain, ytrain)    # for each epoch, shuffle training data
    for idx in range(0, Xtrain.shape[0], minibatch_size):
        X = Xtrain[idx:idx+minibatch_size, ...]     # (N, 1, xdim, ydim, zdim)
        y = ytrain[idx:idx+minibatch_size]          # (N,)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print("\t\tidx: {}, loss: {:.5f}".format(idx, loss.item()))

    if (epoch + 1) % 10 == 0:
        print("=======Epoch: {} / {}, Loss: {:.5f}=======".format(epoch + 1, EPOCH_NUM, loss.item()))

# testing
print("start testing...")
y_prob = []
with torch.no_grad():
    for idx in range(0, Xtest.shape[0], minibatch_size):
        X = Xtest[idx:idx+minibatch_size, ...]
        X = torch.from_numpy(X)
        outputs = model(X)
        # print("outputs.numpy shape: ", outputs.numpy().shape)
        prob = list(outputs.detach().numpy())
        y_prob += prob
# print("y_prob len: ", len(y_prob))
# print(y_prob)
y_prob_np = np.array(y_prob).reshape((-1,))
print(y_prob_np)
y_pred = (y_prob_np >= 0.5) * 1
print("y_pred shape: ", y_pred.shape)
print(ytest)
print("y_pred shape:{}, ytest shape: {}".format(y_pred.shape, ytest.shape))
num_correct = np.sum(y_pred == ytest)
print("Accuracy: ", num_correct / len(ytest))


