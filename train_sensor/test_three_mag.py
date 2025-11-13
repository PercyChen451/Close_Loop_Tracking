import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class ForceTorqueEvaluator:
    def __init__(self, model_path, n_steps, input_size, hidden_dim=512, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = n_steps

        # Build model
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 3 forces + 3 torques
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model weights loaded from {model_path}")

        # Load scalers
        self.scaler_X = StandardScaler()
        self.scaler_X.mean_ = np.load("model_params/X_mean.npy")
        self.scaler_X.scale_ = np.load("model_params/X_std.npy")

        self.scaler_y = StandardScaler()
        self.scaler_y.mean_ = np.load("model_params/y_mean.npy")
        self.scaler_y.scale_ = np.load("model_params/y_std.npy")

        # 8-25_2
        # self.baseline_mag = np.array([-728.0, -478.0, 6609.0, 3.0, -36.0, 7034.0, -801.0, 1128.0, 8969.0,], dtype=np.float32)
        # 8-25_1
        # self.baseline_mag = np.array([-730.0, -478.0, 6615.0, 10.0, -43.0, 7036.0, -803.0, 1142.0, 8962.0], dtype=np.float32)
        # cali 11-11
        self.baseline_mag = np.array([-865.0,-273.0,6310.0,-69.0,-9.0,6453.0,-803.0,1154.0,8518.0], dtype=np.float32)
        # percy 86
        # self.baseline_mag = np.array([-811.0, -429.0, 6634.0, 43.0, -127.0, 7178.0, -973.0, 1209.0, 8699.0], dtype=np.float32)

    def _destandardize(self, y):
        return y * self.scaler_y.scale_ + self.scaler_y.mean_

    def evaluate_csv(self, csv_file, plot=False, plot_F = False):
        df = pd.read_csv(csv_file)
        mag_data = df.iloc[:, 1:10].values.astype(np.float32)
        labels = df.iloc[:, 10:16].values.astype(np.float32)

        mag_data = mag_data - self.baseline_mag

        # print(df.head())
        # print(df.columns)
        # print(mag_data[:5])
        # print(labels[:5])

        # Standardize input first (per original 9 features)
        mag_data_std = (mag_data - self.scaler_X.mean_) / self.scaler_X.scale_

        # Build sequences
        X_seq, y_seq = [], []
        for i in range(len(mag_data_std) - self.n_steps):
            X_seq.append(mag_data_std[i:i+self.n_steps].flatten())
            y_seq.append(labels[i+self.n_steps])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        # Predict
        with torch.no_grad():
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            y_pred_std = self.model(X_tensor).cpu().numpy()

        # Destandardize predictions
        y_pred = self._destandardize(y_pred_std)
        y_true = y_seq  # already in original units

        print(np.shape(y_true), np.shape(y_pred))
        # Compute metrics
        mse = np.mean((y_true[0:3] - y_pred[0:3]) ** 2, axis=0)
        print(np.shape(mse))
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred)**2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)

        print(f"MSE: {mse}, RMSE: {rmse}")
        print(f"R² [Fx, Fy, Fz, Tx, Ty, Tz]: {np.round(r2, 4)}")

        i1, i2 = 30200, 31000
        f_true = np.array([y_true[i1:i2, 0], y_true[i1:i2, 1], y_true[i1:i2, 2]])
        f_norm_true = np.linalg.norm(f_true, axis=0)

        f_pred = np.array([y_pred[i1:i2, 0], y_pred[i1:i2, 1], y_pred[i1:i2, 2]])
        f_norm_pred = np.linalg.norm(f_pred, axis=0)

        
        # Plot actual vs predicted
        # plot_F = False
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(y_true[i1:i2, 0], label="True Force")
            plt.plot(y_pred[i1:i2, 0], label="Predicted Force", linestyle="--")

            # plt.plot(f_norm_true, label="True Force")
            # plt.plot(f_norm_pred, label="Predicted Force", linestyle="--")

            # plt.plot(y_true[i1:i2, 1], label="Fy Actual")
            # plt.plot(y_pred[i1:i2, 1], label="Fy Predicted", linestyle="--")

            # plt.plot(y_true[:, 1], label="Fz Actual")
            # plt.plot(y_pred[:, 1], label="Fz Predicted", linestyle="--")


            plt.title("Magnetometer vs ATI Forces")
            plt.xlabel("Sample")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.show()
        elif plot_F:
            labels_names = ["Fx", "Fy", "Fz"]
            plt.figure(figsize=(15, 8))
            i1, i2 = 29700, 31500
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.plot(y_true[i1:i2, i], label="Actual")
                plt.plot(y_pred[i1:i2, i], label="Predicted", linestyle="--")
                plt.title(labels_names[i])
                plt.xlabel("Sample")
                plt.ylabel("Value")
                plt.legend()
                plt.ylim([-1.1, 1.1])
            plt.tight_layout()
            plt.show()
        
        else:
            labels_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
            plt.figure(figsize=(15, 8))
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.plot(y_true[0:, i], label="Actual")
                plt.plot(y_pred[0:, i], label="Predicted", linestyle="--")
                # plt.plot(y_true[:, i], label="Actual")
                # plt.plot(y_pred[:, i], label="Predicted", linestyle="--")
                plt.title(labels_names[i])
                plt.xlabel("Sample")
                plt.ylabel("Value")
                plt.legend()
            plt.tight_layout()
            plt.show()

        return y_true, y_pred


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    evaluator = ForceTorqueEvaluator(
        model_path="model_params/force_torque_model.pth",
        n_steps=5,
        input_size=5*9,  # n_steps * 9 magnetometer values
        hidden_dim=512
    )

    y_true, y_pred = evaluator.evaluate_csv("raw_data/cali_1_11-11.csv")
