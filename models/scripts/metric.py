import numpy as np
import matplotlib.pyplot as plt

def rmse(y, y_pred):
  return np.sqrt(((y - y_pred)**2).mean())

def r2(y, y_pred):
  sse = ((y - y_pred)**2).sum()
  sst = ((y - y.mean())**2).sum()
  r2 = 1.0 - sse / sst
  return r2

def evaluate(y, y_pred, title=None, fig_name=None):
  print("RMSE: {}".format(rmse(y, y_pred)))
  print("R2: {}".format(r2(y, y_pred)))
  if title is not None:
    plt.scatter(y, y_pred)
    plt.title(title, size=15)
    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    if (fig_name) is not None:
      plt.savefig(fig_name)
    plt.show()
