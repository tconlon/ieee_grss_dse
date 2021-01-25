import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def calculate_band_pdfs(train_ds):

    perc_list = []

    for n, (input_batch, target_batch) in train_ds.enumerate():

        # input_batch = np.reshape(input_batch, (input_batch.shape[0]*input_batch.shape[1]*input_batch.shape[
        #     2]*input_batch.shape[3], input_batch.shape[4]))
        # print(input_batch.shape)

        perc_value = np.percentile(input_batch, q=95, axis=(0,1,2,3))
        perc_list.append(perc_value)

    perc_array = np.array(perc_list)

    fig, ax = plt.subplots(nrows=1, ncols=2)


    for ix in range(12):
        if ix <6:
            ax[0].plot(range(len(perc_array)), sorted(perc_array[:, ix]), label=ix)
        else:
            ax[1].plot(range(len(perc_array)), sorted(perc_array[:, ix]), label=ix)


    ax[0].legend()
    ax[1].legend()

    ax[0].grid()
    ax[1].grid()

    ax[0].axvline(x=880)
    ax[1].axvline(x=880)

    plt.show()