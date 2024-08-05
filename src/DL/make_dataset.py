import numpy as np
import h5py
import sys
sys.path.append('../../../src')
from PlumeDataset import plume_dataset

class EqualRangeNormalizer:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, array):
        self.min_vals = np.min(array, axis=0)
        self.max_vals = np.max(array, axis=0)
        print("Original labels stats:")
        print("Min:", np.min(array, axis=0))
        print("Max:", np.max(array, axis=0))
        print("Mean:", np.mean(array, axis=0))
        print("Std:", np.std(array, axis=0))

    def transform(self, array):
        normalized_labels = (array - self.min_vals) / (self.max_vals - self.min_vals)
        print("\nNormalized labels stats:")
        print("Min:", np.min(normalized_labels, axis=0))
        print("Max:", np.max(normalized_labels, axis=0))
        print("Mean:", np.mean(normalized_labels, axis=0))
        print("Std:", np.std(normalized_labels, axis=0))
        return normalized_labels

    def inverse_transform(self, array):
        reconstructed_labels = array * (self.max_vals - self.min_vals) + self.min_vals
        print("\nReconstructed labels stats:")
        print("Min:", np.min(reconstructed_labels, axis=0))
        print("Max:", np.max(reconstructed_labels, axis=0))
        print("Mean:", np.mean(reconstructed_labels, axis=0))
        print("Std:", np.std(reconstructed_labels, axis=0))
        return reconstructed_labels

def make_dataset(target_file, input_files, df_condition, selected_frame, growth_name_dict, normalize_labels=False):

    selected_frame = (2, 36)
    growth_names = list(growth_name_dict.keys())

    length = 0
    for file in input_files:
        plume_ds = plume_dataset(file_path=file, group_name='PLD_Plumes')
        keys = plume_ds.dataset_names()
        plumes = plume_ds.load_plumes('1-SrRuO3')
        length += len(plumes)
    print(plumes.shape, plumes.dtype, np.min(plumes), np.max(plumes), length)

    with h5py.File(target_file, 'w') as f:
        f.create_dataset('plumes', shape=(length, 34, 250, 400), dtype=np.uint8)
        f.create_dataset('growth_rate(angstrom_per_pulse)', shape=(length, 1), dtype=np.float32)
        f.create_dataset('growth_rate(nm_per_min)', shape=(length, 1), dtype=np.float32)
        f.create_dataset('Pressure (mTorr)', shape=(length, 1), dtype=np.float32)
        f.create_dataset('Fluence (J/cm2)', shape=(length, 1), dtype=np.float32)
        f.create_dataset('labels', shape=(length, 3), dtype=np.float32)
        f.create_dataset('growth_name', shape=(length, 1), dtype=np.uint8)

        index = 0
        for growth, file in zip(growth_names, input_files):
            print(file)
            plume_ds = plume_dataset(file_path=file, group_name='PLD_Plumes')
            plumes = plume_ds.load_plumes('1-SrRuO3')[:, selected_frame[0]:selected_frame[1]]
            f['plumes'][index:index+len(plumes)] = plumes
            print(len(plumes))

            f['growth_rate(angstrom_per_pulse)'][index:index+len(plumes)] = df_condition[df_condition['Growth'] == growth]['Growth rate (Å/pulse)'].values[0]
            f['growth_rate(nm_per_min)'][index:index+len(plumes)] = df_condition[df_condition['Growth'] == growth]['Growth rate (nm/min)'].values[0]
            f['Pressure (mTorr)'][index:index+len(plumes)] = df_condition[df_condition['Growth'] == growth]['Pressure (mTorr)'].values[0]
            f['Fluence (J/cm2)'][index:index+len(plumes)] = df_condition[df_condition['Growth'] == growth]['Fluence (J/cm2)'].values[0]
            
            labels = np.array([df_condition[df_condition['Growth'] == growth]['Pressure (mTorr)'].values[0],
                            df_condition[df_condition['Growth'] == growth]['Fluence (J/cm2)'].values[0],
                            df_condition[df_condition['Growth'] == growth]['Growth rate (Å/pulse)'].values[0]])
            f['labels'][index:index+len(plumes)] = labels
            f['growth_name'][index:index+len(plumes)] = growth_name_dict[growth]

            index += len(plumes)


    if normalize_labels:
        # normalize the labels and create another dataset for it
        with h5py.File(target_file, 'r') as f:
            labels = np.array(f['labels'])
            
        # Create and fit the normalizer
        normalizer = EqualRangeNormalizer()
        normalizer.fit(labels)

        # Normalize the labels
        normalized_labels = normalizer.transform(labels)# normalize the labels and create another dataset for it
        with h5py.File(target_file, 'a') as f:
            f.create_dataset('normalized_labels', data=normalized_labels)