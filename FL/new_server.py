import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvgM
from flwr.common import parameters_to_ndarrays
import torch
import torch.nn as nn
import csv
from nn_models import SimpleNN, ConvNN, get_model

# Custom Strategy with model saving and final global accuracy
class SaveModelStrategy(FedAvgM):
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_model()
        self.metrics_log = []
        self.num_rounds = num_rounds
        self.final_accuracy = None  # to store final global accuracy

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
                print(f" Round metrics saved to 'round_metrics.csv'")
                
                # If global accuracy was stored in the last round, show it
                if self.final_accuracy is not None:
                    print(f"Final Global Accuracy: {self.final_accuracy:.2%}")

        return aggregated_parameters, {}

    # def aggregate_evaluate(self, server_round, results, failures):
    #     aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

    #     # Calculate global accuracy manually from client reports
    #     total_correct = 0
    #     total_examples = 0
    #     for _, evaluate_res in results:
    #         loss, num_examples, metrics = evaluate_res
    #         acc = 1.0 - loss  # since clients return (1 - acc) as loss
    #         total_correct += acc * num_examples
    #         total_examples += num_examples

    #     if total_examples > 0:
    #         global_accuracy = total_correct / total_examples
    #         print(f" Round {server_round} Global Accuracy: {global_accuracy:.2%}")
    #         if server_round == self.num_rounds:
    #             self.final_accuracy = global_accuracy

    #     if aggregated_loss is not None:
    #         self.metrics_log.append((server_round, aggregated_loss))
    #         print(f" Round {server_round} aggregated loss: {aggregated_loss:.6f}")
    #     return aggregated_loss, {}

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

    # Compute global accuracy manually from (client, EvaluateRes) tuples
        total_correct = 0
        total_examples = 0
        for _, res in results:  # Unpack (client, EvaluateRes)
            loss = res.loss
            num_examples = res.num_examples
            acc = 1.0 - loss  # Clients send (1 - acc) as loss
            total_correct += acc * num_examples
            total_examples += num_examples

        if total_examples > 0:
            global_accuracy = total_correct / total_examples
            print(f"Round {server_round} Global Accuracy: {global_accuracy:.2%}")
            if server_round == self.num_rounds:
                self.final_accuracy = global_accuracy

        if aggregated_loss is not None:
            self.metrics_log.append((server_round, aggregated_loss))
            print(f"Round {server_round} aggregated loss: {aggregated_loss:.6f}")

        return aggregated_loss, {}

if __name__ == "__main__":
    num_rounds = 5
    strategy = SaveModelStrategy(num_rounds=num_rounds)

    print("Starting Flower server with logging and model saving...")
    fl.server.start_server(
        server_address="127.0.0.1:9090",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
