# import flwr as fl
# from flwr.server import ServerConfig
# from flwr.server.strategy import FedAvgM
# from flwr.common import parameters_to_ndarrays  # ✅ Fix is here
# import torch
# import torch.nn as nn

# # Define the same model used in client.py
# def get_model():
#     class SimpleNN(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.net = nn.Sequential(
#                 # nn.Linear(9, 128), nn.ReLU(),
#                 # nn.Dropout(0.3),
#                 # nn.Linear(128, 64), nn.ReLU(),
#                 # nn.Dropout(0.2),
#                 # nn.Linear(64, 32), nn.ReLU(),
#                 # nn.Linear(32, 2)
#                 nn.Linear(9, 256), nn.ReLU(),
#                 nn.Linear(256, 128), nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(128, 64), nn.ReLU(),
#                 nn.Linear(64, 2)

#             )
#         def forward(self, x):
#             return self.net(x)
#     return SimpleNN()

# class SaveModelStrategy(FedAvgM):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.model = get_model()

#     def aggregate_fit(self, server_round, results, failures):
#         aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
#         if aggregated_parameters is not None:
#             weights = parameters_to_ndarrays(aggregated_parameters)  #  works now
#             params_dict = zip(self.model.state_dict().keys(), weights)
#             state_dict = {k: torch.tensor(v) for k, v in params_dict}
#             self.model.load_state_dict(state_dict, strict=True)
#             if server_round == self.config.num_rounds:
#                 torch.save(self.model.state_dict(), "global_model.pt")
#                 print("[✓] Global model saved as 'global_model.pt'")
#         return aggregated_parameters, {}

# if __name__ == "__main__":
#     strategy = SaveModelStrategy()
#     strategy.config = ServerConfig(num_rounds=10)

#     print("[🚀] Starting Flower server with model saving...")
#     fl.server.start_server(
#         server_address="127.0.0.1:9090",
#         config=strategy.config,
#         strategy=strategy
#     )


import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvgM
from flwr.common import parameters_to_ndarrays
import torch
import torch.nn as nn
import csv
from nn_models import SimpleNN, ConvNN, get_model

# Define the model


# Custom Strategy with model saving after last round
class SaveModelStrategy(FedAvgM):
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_model()
        self.metrics_log = []
        self.num_rounds = num_rounds

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            weights = parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(self.model.state_dict().keys(), weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
            
            # Save model and metrics if it's the final round
            if server_round == self.num_rounds:
                torch.save(self.model.state_dict(), "global_model.pt")
                print(f"[✓] Global model saved as 'global_model.pt'")
                with open("round_metrics.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["round", "loss"])
                    writer.writerows(self.metrics_log)
                print(f"[✓] Round metrics saved to 'round_metrics.csv'")
        return aggregated_parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_loss is not None:
            self.metrics_log.append((server_round, aggregated_loss))
            print(f"[📉] Round {server_round} aggregated loss: {aggregated_loss:.6f}")
        return aggregated_loss, {}

if __name__ == "__main__":
    num_rounds = 5
    strategy = SaveModelStrategy(num_rounds=num_rounds)

    print("[🚀] Starting Flower server with logging and model saving...")
    fl.server.start_server(
        server_address="127.0.0.1:9090",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
