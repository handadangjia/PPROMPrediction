import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# ==========================
# Fixed model path
# ==========================
MODEL_PATH = r"E:\MyCode\AIForPPROM\model\xgboost_model.json"


class PPROMPredictionApp:
    """
    GUI application for PPROM prediction with SHAP interpretability.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("PPROM Risk Prediction Tool")
        self.root.geometry("760x520")

        self.data_path = tk.StringVar()

        # Load model
        self.model = self.load_model()

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Cache last input data
        self.X = None

        self.create_widgets()

    @staticmethod
    def load_model():
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        return model

    def create_widgets(self):
        # =====================
        # Data selection
        # =====================
        tk.Label(
            self.root,
            text="Feature Data File (CSV / Excel)"
        ).pack(anchor="w", padx=10, pady=8)

        tk.Entry(
            self.root,
            textvariable=self.data_path,
            width=90
        ).pack(padx=10)

        tk.Button(
            self.root,
            text="Browse Data",
            command=self.select_data
        ).pack(pady=6)

        # =====================
        # Buttons
        # =====================
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=12)

        tk.Button(
            button_frame,
            text="Run Prediction",
            command=self.run_prediction,
            bg="#1976D2",
            fg="white",
            width=18,
            height=2
        ).pack(side="left", padx=10)

        tk.Button(
            button_frame,
            text="Show SHAP Explanation",
            command=self.show_shap,
            bg="#388E3C",
            fg="white",
            width=22,
            height=2
        ).pack(side="left", padx=10)

        # =====================
        # Result table
        # =====================
        self.result_table = ttk.Treeview(
            self.root,
            columns=("Index", "Probability", "Prediction"),
            show="headings",
            height=12
        )

        self.result_table.heading("Index", text="Sample Index")
        self.result_table.heading("Probability", text="Predicted Probability")
        self.result_table.heading("Prediction", text="Prediction Result")

        self.result_table.column("Index", width=120, anchor="center")
        self.result_table.column("Probability", width=220, anchor="center")
        self.result_table.column("Prediction", width=220, anchor="center")

        self.result_table.pack(
            fill="both", expand=True, padx=10, pady=10
        )

    def select_data(self):
        path = filedialog.askopenfilename(
            title="Select Feature Data",
            filetypes=[("Data Files", "*.csv *.xlsx *.xls")]
        )
        if path:
            self.data_path.set(path)

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format.")

    def run_prediction(self):
        try:
            if not self.data_path.get():
                raise ValueError("Please select a feature data file.")

            X = self.load_data(self.data_path.get())

            if X.shape[1] != self.model.num_features():
                raise ValueError(
                    f"Feature mismatch: model expects "
                    f"{self.model.num_features()}, got {X.shape[1]}"
                )

            X = X.astype(float)
            self.X = X  # cache for SHAP

            dmatrix = xgb.DMatrix(X, missing=np.nan)
            prob = self.model.predict(dmatrix)
            label = (prob >= 0.5).astype(int)

            # Clear table
            for row in self.result_table.get_children():
                self.result_table.delete(row)

            for i, (p, l) in enumerate(zip(prob, label)):
                interpretation = "PPROM" if l == 1 else "non-PPROM"
                self.result_table.insert(
                    "",
                    "end",
                    values=(i, f"{p:.4f}", interpretation)
                )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_shap(self):
        try:
            if self.X is None:
                raise ValueError("Please run prediction first.")

            selected = self.result_table.selection()
            if not selected:
                raise ValueError("Please select one sample in the table.")

            item = self.result_table.item(selected[0])
            sample_index = int(item["values"][0])

            x_sample = self.X.iloc[sample_index:sample_index + 1]

            shap_values = self.explainer.shap_values(x_sample)
            shap.force_plot(
                base_value=self.explainer.expected_value,
                shap_values=shap_values[0],
                features=x_sample.iloc[0],
                matplotlib=True,
                show=False
            )

            plt.title(f"SHAP Explanation for Sample {sample_index}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = PPROMPredictionApp(root)
    root.mainloop()
