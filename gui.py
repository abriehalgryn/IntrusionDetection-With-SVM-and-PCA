import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import time
import pickle

from main import PCA_SVM
import multiprocessing


RESULTS_TEMPLATE = """
The model was {}% accurate at matching the attack categories.
It identified whether or not attacks occured {}% of the time.
Training took {} seconds to complete.
"""

def train_worker(
        return_dict, whiten_eigenvectors, n_eigenvectors,
        kernel, normalize_method, C, gamma):

    model = PCA_SVM(
        n_eigenvectors=n_eigenvectors,
        normalize_method=normalize_method,
        kernel=kernel,
        verbose=True,
        category_classifications=True,
        C=C,
        gamma=gamma
    )
    model.start()
    return_dict["model"] = model

def test_worker(return_dict, model):
    return_dict["results"] = model.test()

PADDING = 3

# frame config
F_RELIEF = "groove"
F_BORDERWIDTH = 2

# button config
B_HEIGHT = 3
B_RELIEF = "groove"
B_BORDERWIDTH = 2

INPUT_PADDING=5

MAX_EIGENVECTORS = 41
MIN_EIGENVECTORS = 1

class Window(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.after_id = None
        self.process_manager = multiprocessing.Manager()

        self.title("SVM-PCA Intrusion Detection Manager")
        self.minsize(500, 300)
        self.trained = False
        self.tested = False
        self.results = []

        frame = tk.Frame(self)
        left_frame = tk.Frame(frame)
        right_frame = tk.Frame(frame)

        self.create_button_frame(left_frame)
        self.create_config_frame(right_frame)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        left_frame.pack(
            side="left", fill="both",
            expand=True, pady=PADDING, padx=PADDING
        )
        right_frame.pack(side="right", fill="both", expand=True)

        frame.pack(
            side="top", fill="both", expand=True,
            padx=PADDING, pady=PADDING
        )

        self.create_hooks()

    def create_config_frame(self, frame):
        self.n_eigenvectors = tk.StringVar()
        self.n_eigenvectors.set("15")
        self.n_eigenvectors_entry = tk.Entry(frame, textvariable=self.n_eigenvectors)

        self.whiten_eigenvectors = tk.IntVar()
        self.whiten_checkbox = tk.Checkbutton(frame, variable=self.whiten_eigenvectors)

        self.normalize_method = tk.StringVar(self)
        self.normalize_method.set("Column")
        self.normalize_method_options = tk.OptionMenu(frame, self.normalize_method, "Column", "Row", "None")

        self.kernel = tk.StringVar(self)
        self.kernel.set("RBF")
        self.kernel_options = tk.OptionMenu(frame, self.kernel, "RBF", "Linear", "Poly", "Sigmoid")

        self.C = tk.StringVar(self)
        self.C.set("1.0")
        self.C_entry = tk.Entry(frame, textvariable=self.C)

        self.gamma = tk.StringVar(self)
        self.gamma.set("Scale")
        self.gamma_options = tk.OptionMenu(frame, self.gamma, "Scale", "Auto")

        config_names = [
            "Eigenvectors", "Whiten Eigenvectors",
            "Normalize Method", "Kernel", "Regularization Parameter",
            "Gamma"
        ]
        config_widgets = [
            self.n_eigenvectors_entry, self.whiten_checkbox,
            self.normalize_method_options, self.kernel_options,
            self.C_entry, self.gamma_options,
        ]

        for i, name in enumerate(config_names):
            label = tk.Label(
                frame, text=name + " ",
                anchor="e", justify="right"
            )
            label.grid(column=0, row=i, sticky="nsew")
            if not isinstance(config_widgets[i], tk.Checkbutton):
                config_widgets[i].grid(column=1, row=i, sticky="nsew")
            else:
                config_widgets[i].grid(column=1, row=i, sticky="nsw")
            frame.rowconfigure(i, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)



    def create_button_frame(self, frame):
        self.save_button = tk.Button(frame, text="Save")
        self.save_button.pack(side="top", fill="both", expand=True, ipady=B_HEIGHT, padx=PADDING, pady=PADDING)
        self.load_button = tk.Button(frame, text="Load")
        self.load_button.pack(side="top", fill="both", expand=True, ipady=B_HEIGHT, padx=PADDING, pady=PADDING)
        self.train_button = tk.Button(frame, text="Train")
        self.train_button.pack(side="top", fill="both", expand=True, ipady=B_HEIGHT, padx=PADDING, pady=PADDING)
        self.test_button = tk.Button(frame, text="Test")
        self.test_button.pack(side="top", fill="both", expand=True, ipady=B_HEIGHT, padx=PADDING, pady=PADDING)
        self.reset_button = tk.Button(frame, text="Reset")
        self.reset_button.pack(side="top", fill="both", expand=True, ipady=B_HEIGHT, padx=PADDING, pady=PADDING)

    def create_hooks(self):
        self.save_button.configure(command=self.save)
        self.load_button.configure(command=self.load)
        self.train_button.configure(command=self.train)
        self.test_button.configure(command=self.test)
        self.reset_button.configure(command=self.reset)

    def load(self):
        file_name = filedialog.askopenfilename(filetypes=[("Serialized Data", "*.pkl")], defaultextension="pkl")
        if file_name == "":
            return

        try:
            with open(file_name, "rb") as f:
                pickle_rick = pickle.load(f)

            whiten_eigenvectors, n_eigenvectors, kernel, normalize_method, \
            C, gamma, model, tested, results = pickle_rick

            self.enable_buttons()
            self.disable_configs()

            self.whiten_eigenvectors.set(whiten_eigenvectors)
            self.n_eigenvectors.set(n_eigenvectors)
            self.kernel.set(kernel)
            self.normalize_method.set(normalize_method)
            self.gamma.set(gamma)
            self.C.set(C)
            self.model = model
            self.tested = tested
            self.results = results

            self.trained = True
            self.train_button.configure(
                text="Trained" if self.trained else "Train"
            )
            self.test_button.configure(
                text="Tested" if self.tested else "Test"
            )

        except IndexError:
            messagebox.showerror(
                title="Error loading file",
                message="There was an error loading the data file."
            )
            return

    def save(self):
        if not self.trained:
            messagebox.showerror(
                title="Error",
                message="Model must be trained before it can be saved."
            )
            return

        whiten_eigenvectors = self.whiten_eigenvectors.get()
        n_eigenvectors = self.n_eigenvectors.get()
        kernel = self.kernel.get()
        normalize_method = self.normalize_method.get()
        gamma = self.gamma.get()
        C = self.C.get()

        pickle_rick = [
            whiten_eigenvectors,
            n_eigenvectors,
            kernel,
            normalize_method,
            C,
            gamma,
            self.model,
            self.tested,
            self.results
        ]

        file = filedialog.asksaveasfile(filetypes=[("Serialized Data", "*.pkl")], defaultextension="pkl")
        with open(file.name, "wb") as f:
            pickle.dump(pickle_rick, f)

    def disable_buttons(self):
        self.save_button.configure(state="disabled")
        self.load_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        self.test_button.configure(state="disabled")

    def enable_buttons(self):
        self.save_button.configure(state="normal")
        self.load_button.configure(state="normal")
        self.train_button.configure(state="normal")
        self.test_button.configure(state="normal")

    def disable_configs(self):
        self.whiten_checkbox.configure(state="disabled")
        self.n_eigenvectors_entry.configure(state="disabled")
        self.kernel_options.configure(state="disabled")
        self.normalize_method_options.configure(state="disabled")
        self.C_entry.configure(state="disabled")
        self.gamma_options.configure(state="disabled")

    def enable_configs(self):
        self.whiten_checkbox.configure(state="normal")
        self.n_eigenvectors_entry.configure(state="normal")
        self.kernel_options.configure(state="normal")
        self.normalize_method_options.configure(state="normal")
        self.C_entry.configure(state="normal")
        self.gamma_options.configure(state="normal")

    def validate_C(self):
        try:
            C = float(self.C.get())
        except:
            messagebox.showerror(
                title="Error",
                message="Regularization Parameter must be numerical"
            )
            return False
        if C <= 0:
            messagebox.showerror(
                title="Error",
                message="Regularization Parameter must be positive"
            )
            return False
        if C < 0.1:
            messagebox.showerror(
                title="Error",
                message="Regularization Parameter cannot be below 0.1"
            )
            return False
        if C > 2500:
            messagebox.showerror(
                title="Error",
                message="Regularization Parameter cannot be above 2500"
            )
            return False
        return True


    def validate_eigenvectors(self):
        try:
            n_eigenvectors = float(self.n_eigenvectors.get())
        except:
            messagebox.showerror(
                title="Error",
                message="Number of eigenvectors must be numerical"
            )
            return False

        if "." in self.n_eigenvectors.get():
            messagebox.showerror(
                title="Error",
                message="Number of eigenvectors must be an integer"
            )
            return False

        if not MIN_EIGENVECTORS <= n_eigenvectors <= MAX_EIGENVECTORS:
            messagebox.showerror(
                title="Error",
                message="Number of eigenvectors must be within range "
                    f"{MIN_EIGENVECTORS}-{MAX_EIGENVECTORS} inclusive.\n"
                    "(The number of eigenvectors may not exceed the "
                    "data dimensionality)"
            )
            return False
        return True

    def train(self):
        if self.trained:
            messagebox.showerror(
                title="Already trained",
                message="The model has already been trained for these parameters"
            )
            return

        if not self.validate_eigenvectors():
            return

        if not self.validate_C():
            return

        whiten_eigenvectors = self.whiten_eigenvectors.get()
        n_eigenvectors = int(self.n_eigenvectors.get())
        kernel = self.kernel.get().lower()
        normalize_method = self.normalize_method.get().lower()
        C = float(self.C.get())
        gamma = self.gamma.get().lower()

        self.disable_buttons()
        self.disable_configs()
        self.train_button.configure(text="Training...")
        self.reset_button.configure(text="Cancel")

        self.return_dict = self.process_manager.dict()
        self.worker = multiprocessing.Process(
            target=train_worker,
            args=(
                self.return_dict, whiten_eigenvectors, n_eigenvectors,
                kernel, normalize_method, C, gamma
            )
        )
        self.worker.start()

        messagebox.showinfo(
            title="Training started",
            message="Please be patient. This may take some time (up to 15 minutes).\n\nGrab a coffee :-)"
        )

        self.after_id = self.after(5000, self.check_if_trained)

    def check_if_trained(self):
        if not self.worker.is_alive():
            self.model = self.return_dict["model"]
            self.training_finished()
        else:
            self.after_id = self.after(5000, self.check_if_trained)

    def training_finished(self):
        messagebox.showinfo(
            title="Training finished",
            message=f"Took {round(self.model.duration, 2)} seconds"
        )
        self.enable_buttons()
        self.reset_button.configure(text="Reset")
        self.train_button.configure(text="Trained")
        self.trained = True

    def test(self):
        if self.tested:
            self.show_test_results()
            return

        if not self.trained:
            messagebox.showerror(
                title="Not trained",
                message="You have to train the model before you can test it"
            )
            return

        self.disable_buttons()
        self.test_button.configure(text="Testing...")
        self.reset_button.configure(text="Cancel")

        self.worker = multiprocessing.Process(
            target=test_worker,
            args=(self.return_dict, self.model)
        )
        self.worker.start()

        messagebox.showinfo(
            title="Testing started",
            message="Testing started, this should only take a couple seconds"
        )

        self.after_id = self.after(5000, self.check_if_tested)

    def check_if_tested(self):
        if not self.worker.is_alive():
            self.results = self.return_dict["results"]
            self.test_button.configure(text="Tested")
            self.reset_button.configure(text="Reset")
            self.tested = True
            self.enable_buttons()
            self.show_test_results()
        else:
            self.after_id = self.after(2000, self.check_if_tested)

    def show_test_results(self):
        exact, category, attack_yes_no = self.results

        messagebox.showinfo(
            title="Testing Results",
            message=RESULTS_TEMPLATE.format(
                round(category * 100, 2),
                round(attack_yes_no * 100, 2),
                round(self.model.duration, 2)
            )
        )

    def on_close(self):
        if self.after_id is not None and self.worker.is_alive():
            self.worker.terminate()
        self.destroy()

    def kill_worker(self):
        print("\r", " "*20, end="", flush=True)
        self.after_cancel(self.after_id)
        self.worker.terminate()
        if not self.trained:
            self.enable_configs()
        self.enable_buttons()
        self.train_button.configure(
            text="Trained" if self.trained else "Train"
        )
        self.test_button.configure(
            text="Tested" if self.tested else "Test"
        )
        self.reset_button.configure(text="Reset")

    def reset(self):
        if self.after_id is not None and self.worker.is_alive():
            self.kill_worker()
            return
        self.trained = False
        self.tested = False
        self.train_button.configure(text="Train")
        self.test_button.configure(text="Test")
        self.enable_configs()


if __name__ == "__main__":
    window = Window()
    window.mainloop()
