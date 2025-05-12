"""
Random-Forest version of the previous CNN training script.
Compatible with Python < 3.10 (uses typing.Optional instead of | None).
"""

import pandas as pd
from typing import Optional             # ← NEW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# ------------------ Trainer --------------------------------------------------
class RFTrainer:
    def __init__(self, csv_path: str,
                 n_estimators: int = 300,
                 max_depth: Optional[int] = None,     # ← CHANGED
                 test_size: float = .15,
                 val_size: float = .15,
                 seed: int = 42):

        (self.X_train,
         self.X_val,
         self.X_test,
         self.y_train,
         self.y_val,
         self.y_test) = self._prepare_data(csv_path, test_size, val_size, seed)

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )

    # ---------------- data pipeline ----------------
    @staticmethod
    def _prepare_data(path, test_frac, val_frac, seed):
        df = pd.read_csv(path)

        X = df[["Trans_Rate", "PDR", "Delay",
                "Energy", "Packet_Loss", "Malicious_Prob"]].values
        y = df["Label"].values

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_frac, stratify=y, random_state=seed
        )

        relative_val = val_frac / (1 - test_frac)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=relative_val,
            stratify=y_train_val,
            random_state=seed
        )

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ---------------- training / evaluation ----------------
    def fit(self):
        self.model.fit(self.X_train, self.y_train)

        train_acc = accuracy_score(self.y_train, self.model.predict(self.X_train))
        val_acc   = accuracy_score(self.y_val,   self.model.predict(self.X_val))

        print(f"Training accuracy   : {train_acc:6.3%}")
        print(f"Validation accuracy : {val_acc:6.3%}")
        return self

    def test(self):
        y_pred = self.model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred)

        print(f"\nTest accuracy       : {test_acc:6.3%}")
        print("\nClassification report (test):")
        print(classification_report(self.y_test, y_pred, digits=3))
        return test_acc


# ------------------ Run ------------------------------------------------------
if __name__ == "__main__":
    trainer = RFTrainer(
        csv_path="trust_aware_iiot_dataset_balanced.csv",
        n_estimators=400,
        max_depth=None,
        test_size=0.15,
        val_size=0.15,
        seed=42,
    )
    trainer.fit()
    trainer.test()
