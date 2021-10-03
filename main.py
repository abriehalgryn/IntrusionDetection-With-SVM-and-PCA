from attack_categories import attack_categories
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import csv
import time
import pickle
from os.path import exists

from utils import *

TRAIN_FILE = "NSL-KDD/KDDTrain+.txt"
MODEL_DIR = "Models"
TEST_FILE = "NSL-KDD/KDDTest+.txt"

# megabytes
CACHE_SIZE = 800

MAX_ITER = 100_000


class PCA_SVM:
    def __init__(self, n_eigenvectors=20, normalize_method="column", kernel="rbf", whiten_eigenvectors=False, gamma="scale", C=1.0, verbose=False, category_classifications=False):
        self.length = 0
        self.test_data = []
        self.train_data = []
        self.reset(n_eigenvectors, normalize_method, kernel, whiten_eigenvectors, gamma, C, category_classifications)
        self.verbose = verbose

    def create_classification_lookup(self):
        self.classification_lookup = {}
        self.category_idx_table = {}

        for cat_idx, category in enumerate(attack_categories):
            for attack in attack_categories[category]:
                if attack in self.classification_lookup:
                    raise Exception(attack + " exists")
                if self.category_classifications:
                    self.classification_lookup[attack] = cat_idx
                    self.category_idx_table[cat_idx] = cat_idx
                else:
                    idx = len(self.classification_lookup)
                    self.classification_lookup[attack] = idx
                    self.category_idx_table[idx] = cat_idx


    def reset(self, n_eigenvectors, normalize_method, kernel, whiten_eigenvectors, gamma, C, category_classifications):
        self.n_eigenvectors = n_eigenvectors
        if normalize_method not in ("row", "column", None):
            raise Exception("Invalid normalization method")
        self.normalize_method = normalize_method
        self.kernel = kernel
        self.whiten_eigenvectors = whiten_eigenvectors
        self.classifier = None
        self.trained = False
        self.gamma = gamma
        self.C = float(C)
        self.category_classifications = category_classifications

    def exists(self):
        return exists(MODEL_DIR + "/" + self.file_name + ".pkl")

    @property
    def file_name(self):
        return "__".join([
            str(self.n_eigenvectors),
            str(self.normalize_method),
            str(self.kernel),
            str(int(self.category_classifications)),
            str(int(self.whiten_eigenvectors)),
            str(self.gamma),
            str(self.C),
        ]).replace(".", "_")

    def print(self, message, newline=False):
        if self.verbose:
            print("\r" + self.length * " ", end="")
            print("\r" + message, end="\n" if newline else "", flush=True)
            self.length = len(message)

    def start(self):
        if self.trained:
            return False

        t = time.time()

        self.print("loading data...")
        self.load_data()

        self.print("creating classifications...")
        self.create_classification_lookup()

        self.print("extracting classifications...")
        self.extract_classifications()

        self.print("tokenizing data...")
        self.tokenize_data()

        self.print("preprocessing data...")
        self.preprocess_data()

        self.print("applying pca...")
        self.apply_pca()

        self.print("training svm...")
        self.train()

        self.duration = time.time() - t
        self.print(f"finished in {round(self.duration, 2)} seconds.", newline=True)

    def load_data(self):
        with open(TRAIN_FILE, newline='') as f:
            self.train_data = list(csv.reader(f))

        with open(TEST_FILE, newline='') as f:
            self.test_data = list(csv.reader(f))

        # deletes the difficulty column, since it's not part
        # of the raw data, it was added on and shouldn't be used
        # for training
        delete_column(self.train_data, -1)
        delete_column(self.test_data, -1)

    def extract_classifications(self):
        # since we don't care about the specific type of attack,
        # we convert the attack type string to indicated whether an
        # attack occurred at all
        #   1 = attack occurred
        #   0 = no attack occurred
        self.train_classifications = [
            self.classification_lookup[record.pop(-1)] for record in self.train_data
        ]
        self.test_classifications = [
            self.classification_lookup[record.pop(-1)] for record in self.test_data
        ]

    def tokenize_data(self):
        for col, val in enumerate(self.train_data[0]):
            if not isnumeric(val):
                # symbolic features (strings) should be
                # converted to numerical values
                # the following list will contain a dictionary
                # for every column with string values
                symbolic_map = {}
                for data in [self.train_data, self.test_data]:
                    for row in data:
                        if val not in symbolic_map:
                            symbolic_map[val] = len(symbolic_map)
                        row[col] = symbolic_map[val]

    def preprocess_data(self):
        if not self.normalize_method:
            return

        # the minimum and maximum value for each column/row
        # these will be used for normalization
        min_max_columns = [
            [None, None] for i in range(len(self.train_data[0]))
        ]
        min_max_rows = [
            [[None, None] for i in range(len(self.train_data))],
            [[None, None] for i in range(len(self.test_data))]
        ]

        for i, data in enumerate([self.train_data, self.test_data]):
            for row, columns in enumerate(data):
                for col, val in enumerate(columns):
                    columns[col] = val = float(val)

                    low, high = min_max_columns[col]
                    if low is None or val < low:
                        min_max_columns[col][0] = val
                    if high is None or val > high:
                        min_max_columns[col][1] = val

                    low, high = min_max_rows[i][row]
                    if low is None or val < low:
                        min_max_rows[i][row][0] = val
                    if high is None or val > high:
                        min_max_rows[i][row][1] = val

        if self.normalize_method == "row":
            normalize_rows(self.train_data, min_max_rows[0])
            normalize_rows(self.test_data, min_max_rows[1])

        elif self.normalize_method == "column":
            normalize_columns(self.train_data, min_max_columns)
            normalize_columns(self.test_data, min_max_columns)

    def apply_pca(self):
        pca = PCA(n_components=self.n_eigenvectors, whiten=self.whiten_eigenvectors)
        pca.fit(self.train_data)
        self.train_descriptors = pca.transform(self.train_data)
        self.test_descriptors = pca.transform(self.test_data)

    def train(self):
        self.classifier = SVC(kernel=self.kernel, gamma=self.gamma, C=self.C, cache_size=CACHE_SIZE, max_iter=MAX_ITER)
        self.classifier.fit(self.train_descriptors, self.train_classifications)
        self.trained = True

    def test(self):
        if not self.trained:
            return False

        predictions = self.classifier.predict(self.test_descriptors)
        total = len(predictions)

        exact_matches = 0
        category_matches = 0
        attacks_identified = 0

        for i, predicted_attack in enumerate(predictions):
            attack = self.test_classifications[i]
            category = self.category_idx_table[attack]
            predicted_category = self.category_idx_table[predicted_attack]

            if predicted_attack == attack:
                exact_matches += 1

            if predicted_category == category:
                category_matches += 1

            if bool(predicted_category) == bool(category):
                attacks_identified += 1

        return exact_matches / total, category_matches / total, attacks_identified / total

    def load_from_file(self):
        try:
            with open(MODEL_DIR + "/" + self.file_name + ".pkl", "rb") as f:
                self.classifier = pickle.load(f)
            self.trained = True
            return True
        except FileNotFoundError:
            return False

    def save_to_file(self):
        if self.trained:
            with open(MODEL_DIR + "/" + self.file_name + ".pkl", "wb") as f:
                pickle.dump(self.classifier, f)
            return True
        else:
            return False


def huge_test():
    pca_svm = PCA_SVM(verbose=True)

    for category_classification in [False]:
        for kernel in ["rbf", "linear", "poly"]:
            for whiten_eigenvectors in [False, True]:
                for n_eigenvectors in [7, 13, 23, 41]:
                    for normalize_method in ["row", "column"]:
                        for C in [1, 1024, 2048]:
                            for gamma in ["auto", "scale"]:
                                pca_svm.reset(
                                    whiten_eigenvectors=whiten_eigenvectors,
                                    kernel=kernel,
                                    n_eigenvectors=n_eigenvectors,
                                    normalize_method=normalize_method,
                                    gamma=gamma,
                                    C=C,
                                    category_classifications=category_classification
                                )

                                if pca_svm.exists():
                                    print(pca_svm.file_name, "already exists (skipped)")
                                    continue

                                pca_svm.start()
                                exact, category, attack = pca_svm.test()
                                print(f"Exact matches:      {round(exact * 100, 2)}%")
                                print(f"Category matches:   {round(category * 100, 2)}%")
                                print(f"Attacks identified: {round(attack * 100, 2)}%")

                                with open("scores.txt", "a") as f:
                                    f.write(f"\n{pca_svm.file_name} {exact} {category} {attack} {duration}")

                                pca_svm.save_to_file()
