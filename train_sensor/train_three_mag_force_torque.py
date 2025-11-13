import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ForceTorquePredictor:
    def __init__(self, csv_file, n_steps=5, hidden_dim=512, lr=1e-3, device=None):
        self.csv_file = csv_file
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


        # 8-25_2
        # self.baseline_mag = np.array([-728.0, -478.0, 6609.0, 3.0, -36.0, 7034.0, -801.0, 1128.0, 8969.0,], dtype=np.float32)
        # 8-25_1
        # self.baseline_mag = np.array([-730.0, -478.0, 6615.0, 10.0, -43.0, 7036.0, -803.0, 1142.0, 8962.0], dtype=np.float32)
        # percy 86
        # self.baseline_mag = np.array([-811.0, -429.0, 6634.0, 43.0, -127.0, 7178.0, -973.0, 1209.0, 8699.0], dtype=np.float32)
        # cali 11-11
        self.baseline_mag = np.array([-865.0,-273.0,6310.0,-69.0,-9.0,6453.0,-803.0,1154.0,8518.0], dtype=np.float32)


        # Load and preprocess data
        self._load_data()
        self._build_model()

    def _load_data(self):
        df = pd.read_csv(self.csv_file)

        # Separate inputs and outputs
        mag_data = df.iloc[:, 1:10].values   # 9 magnetometer values
        labels = df.iloc[:, 10:].values    # 3 forces + 3 torques

        mag_data = mag_data - self.baseline_mag

        # print(df.head())
        # print(df.columns)
        # print(mag_data[:5])
        # print(labels[:5])

        # Normalize
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        mag_data = self.scaler_X.fit_transform(mag_data)
        labels = self.scaler_y.fit_transform(labels)

        # plt.plot(mag_data[5:10000+5,2])
        # plt.plot(labels[:10000, 2])
        # plt.legend(['Bx1', 'Bx2', 'Bx3', 'Fx','Fy','Fz','Tx','Ty','Tz'])
        # plt.show()

        # print(df.head())
        # print(df.columns)
        # print(mag_data[:5])
        # print(labels[:5])

        # print("mag_data size", np.shape(mag_data))
        # print("force_data size", np.shape(labels))

        # Create sequences of n_steps
        X, y = [], []
        step_shift = 0
        for i in range(step_shift, len(mag_data) - self.n_steps):
            X.append(mag_data[i:i+self.n_steps].flatten())  # (n_steps * 9)
            y.append(labels[i+self.n_steps-step_shift])                # predict at next timestep

        X, y = np.array(X), np.array(y)

        # print("X size", np.shape(X))
        # print("y size", np.shape(y))
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False
        )

        # Convert to tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        self.input_dim = self.X_train.shape[1]  # n_steps * 9
        self.output_dim = self.y_train.shape[1] # 6 (forces+torques)

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        ).to(self.device)

        # self.criterion = nn.MSELoss()
        self.criterion = nn.MSELoss(reduction="none") 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Define output weights: reduce contribution of z-axis (index 2 for Fz, index 5 for Tz if needed)
        self.output_weights = torch.tensor([1.0, 1.0, .6, .03, .03, .1],  
                                           dtype=torch.float32,
                                           device=self.device)
        
    def train(self, epochs=50, batch_size=128):
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)

                # compute loss with weights
                raw_loss = self.criterion(preds, y_batch)
                loss = (raw_loss * self.output_weights).mean() 

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.6f}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            raw_loss = self.criterion(preds, self.y_test)
            loss = (raw_loss * self.output_weights).mean()
        print(f"Test Loss (MSE): {loss:.6f}")
        return loss

    def save(self, model_path="model_params/force_torque_model.pth"):
        """Save model weights and scalers separately."""
        # Save model weights only
        torch.save(self.model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

        # Save scalers as numpy arrays
        np.save("model_params/X_mean.npy", self.scaler_X.mean_)
        np.save("model_params/X_std.npy", self.scaler_X.scale_)
        np.save("model_params/y_mean.npy", self.scaler_y.mean_)
        np.save("model_params/y_std.npy", self.scaler_y.scale_)
        print("Scalers saved as X_mean.npy, X_std.npy, y_mean.npy, y_std.npy")

    def load(self, model_path="model_params/force_torque_model.pth"):
        """Load model weights and scalers from separate files."""
        # Load model weights
        self._build_model()  # ensures model architecture exists
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model weights loaded from {model_path}")

        # Load scalers
        self.scaler_X.mean_ = np.load("model_params/X_mean.npy")
        self.scaler_X.scale_ = np.load("model_params/X_std.npy")
        self.scaler_y.mean_ = np.load("model_params/y_mean.npy")
        self.scaler_y.scale_ = np.load("model_params/y_std.npy")
        print("Scalers loaded from X_mean.npy, X_std.npy, y_mean.npy, y_std.npy")

    def preprocess_input(self, mag_sequence):
        """
        Subtract baseline and scale using the training scaler.
        mag_sequence: np.array of shape (n_steps, 9)
        """
        mag_sequence = mag_sequence - self.baseline_mag
        return self.scaler_X.transform(mag_sequence)

    def predict(self, mag_sequence):
        """
        mag_sequence: np.array of shape (n_steps, 9)
        """
        self.model.eval()
        X_scaled = self.preprocess_input(mag_sequence)
        X_tensor = torch.tensor(X_scaled.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
        return self.scaler_y.inverse_transform(y_pred_scaled).flatten()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Training
    predictor = ForceTorquePredictor("raw_data/cali_1_11-11.csv", n_steps=5, hidden_dim=512)
    predictor.train(epochs=200, batch_size=32)
    predictor.evaluate()
    predictor.save("model_params/force_torque_model.pth")

    if False:
        # Testing
        predictor.load("model_params/force_torque_model.pth")
        # test_seq =  np.zeros((5,9))
        # test_seq =  np.array( [[ 0.3512, -0.3505, -0.3253,  0.4313,  0.2917, -0.1898,  0.2931, -0.2956, -0.2746],
        #             [ 0.3774, -0.3897, -0.3486,  0.3157,  0.3238, -0.1505,  0.2931, -0.2956, -0.2658],
        #             [ 0.3599, -0.2916, -0.3183,  0.4458,  0.2356, -0.1445,  0.3067, -0.3885, -0.2552],
        #             [ 0.2987, -0.3505, -0.3183,  0.5758,  0.3479, -0.1838,  0.3135, -0.3537, -0.264 ],
        #             [ 0.2987, -0.429,  -0.3183,  0.4313,  0.3158, -0.1868,  0.2931, -0.3537, -0.2764]] )
        
        test_seq =  np.array(  [[-812.0, -424.0, 6630.0, 43.0, -126.0, 7176.0, -974.0, 1215.0, 8695.0],
                    [-809.0, -428.0, 6620.0, 35.0, -122.0, 7189.0, -974.0, 1215.0, 8700.0],
                    [-811.0, -418.0, 6633.0, 44.0, -133.0, 7191.0, -972.0, 1207.0, 8706.0],
                    [-818.0, -424.0, 6633.0, 53.0, -119.0, 7178.0, -971.0, 1210.0, 8701.0],
                    [-818.0, -432.0, 6633.0, 43.0, -123.0, 7177.0, -974.0, 1210.0, 8694.0]] )

        pred = predictor.predict(test_seq[:5,:])
        print("Predicted [Fx, Fy, Fz, Tx, Ty, Tz]:", pred)
