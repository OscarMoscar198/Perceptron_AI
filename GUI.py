import tkinter as tk
from tkinter import filedialog, ttk
import perceptron_logic as pl
import numpy as np

def select_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    if filename:
        file_label.config(text=filename.split('/')[-1])  # Show only the file name
    return filename

def start_training():
    learning_rate = float(learning_rate_entry.get())
    epochs = int(epochs_entry.get())
    file_path = file_label.cget("text")
    pl.train_perceptron(learning_rate, epochs, file_path)

    # Show report after training
    generate_report()

def show_graphics():
    pl.show_results()

def generate_report():
    np.set_printoptions(precision=4, suppress=True)
    initial_weights, final_weights, epochs, error = pl.get_weights()
    report = (f"Number of Epochs: {epochs}\n"
              f"Permissible Error: {error}\n\n"
              f"Initial Weights Configuration:\n{initial_weights}\n\n"
              f"Final Weights Configuration:\n{final_weights}")
    report_text.config(state=tk.NORMAL)
    report_text.delete('1.0', tk.END)
    report_text.insert(tk.END, report)
    report_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Perceptron Training")
root.geometry('900x400')

style = ttk.Style()
style.theme_use('clam')

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Frame for input parameters
parameters_frame = ttk.LabelFrame(main_frame, text="Input Parameters")
parameters_frame.pack(fill=tk.BOTH, side=tk.LEFT, padx=10, pady=10)

ttk.Label(parameters_frame, text="Learning Rate (eta):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
learning_rate_entry = ttk.Entry(parameters_frame)
learning_rate_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(parameters_frame, text="Number of Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
epochs_entry = ttk.Entry(parameters_frame)
epochs_entry.grid(row=1, column=1, padx=5, pady=5)

# Frame for file selection
file_frame = ttk.LabelFrame(main_frame, text="Select CSV File")
file_frame.pack(fill=tk.BOTH, side=tk.LEFT, padx=10, pady=10)

ttk.Button(file_frame, text="Select File", command=select_file).pack(pady=5)
file_label = ttk.Label(file_frame, text="")
file_label.pack(pady=5)

# Frame for action buttons
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill=tk.BOTH, side=tk.LEFT, padx=10, pady=10)

ttk.Button(button_frame, text="Start Training", command=start_training).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Show Graphics", command=show_graphics).pack(side=tk.LEFT, padx=5)

# Frame for report
report_frame = ttk.LabelFrame(main_frame, text="Report", width=400)  # Set width to occupy the whole right side
report_frame.pack(fill=tk.BOTH, side=tk.RIGHT, padx=10, pady=10, expand=True)

report_text = tk.Text(report_frame, wrap=tk.WORD, height=20)
report_text.pack(fill=tk.BOTH, expand=True)
report_text.config(state=tk.DISABLED)

root.mainloop()
