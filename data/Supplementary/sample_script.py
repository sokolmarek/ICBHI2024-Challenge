import numpy as np
import os


def load_npz_data(fpath):
    all_data = np.load(fpath)
    return all_data['data']


if __name__ == '__main__':

    # Example

    fpath_fmri_p1 = os.path.join(
        "..",
        "Train",
        "P01",
        "fMRI_data.npz"
    )

    fpath_ppg_p1 = os.path.join(
        "..",
        "Train",
        "P01",
        "PPG_data.npz"
    )

    fpath_resp_p1 = os.path.join(
        "..",
        "Train",
        "P01",
        "resp_data.npz"
    )

    fpath_labels_p1 = os.path.join(
        "..",
        "Train",
        "P01",
        "labels.npz"
    )

    fmri_p1_data = load_npz_data(fpath_fmri_p1)
    ppg_p1_data = load_npz_data(fpath_ppg_p1)
    resp_p1_data = load_npz_data(fpath_resp_p1)
    labels_p1_data = load_npz_data(fpath_labels_p1)

    print(f"fMRI data: {fmri_p1_data.shape}\nPPG data: {ppg_p1_data.shape}\nRespiration data: {resp_p1_data.shape}\nLabels: {labels_p1_data.shape}")
