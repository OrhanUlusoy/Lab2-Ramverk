"""Entrypoint för att köra hela flödet.

Krav i labben: hela flödet ska kunna startas via `main.py`.
Här läser vi hyperparametrar från `params.json`, kör träning och skriver ut
slutlig accuracy som "Final Accuracy".
"""

import json
from train import train_model


def main():
    # Läs in experiment-/träningsparametrar.
    with open("params.json", "r") as f:
        params = json.load(f)

        # Träna och få tillbaka accuracy från sista epoken.
        accuracy = train_model(
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["learning_rate"],
        )

    # Loggas som del av experimenten i README.
    print(f"\nFinal Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
