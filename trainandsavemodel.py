from pipeline import pipeline
import argparse
from utils import load_model
import joblib
import numpy as np
def main():
    model, scaler = load_model("models_trained")


    perf = np.array([3.17e9,3.81e-04,22]).reshape(1,3)
    perf = np.hstack([np.array([0,0,0]).reshape(1,3),perf])

    perf = scaler.transform(perf)[:,3:]

    print('perf after', perf)
    #bandwidth, power, gain
#     [2.175997e+09 3.816326e-04 2.094111e+01]
# [8.501272e+09 5.610970e-04 2.823009e+01]

    
    params = model.predict(perf)

    data = np.hstack((params, perf))
    
    # print(data)

    inverse_trans = scaler.inverse_transform(data)
    #params, perf
    print(inverse_trans[:,0:3],inverse_trans[:,3:6])

if __name__ == "__main__":
    main()