import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Read file
rdata = pd.read_csv('cc_clients.csv')

# Check The data
rdata.head()