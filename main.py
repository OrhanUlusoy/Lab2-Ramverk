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

    # Backwards compatibility: om någon bara har en config utan lista.
    experiments = params.get("experiments")
    if not experiments:
        experiments = [
            {
                "name": "single",
                "epochs": params["epochs"],
                "batch_size": params["batch_size"],
                "learning_rate": params["learning_rate"],
            }
        ]

    data_root = params.get("data_root", "data/cifar10")
    download = bool(params.get("download", False))
    seed = params.get("seed", 42)

    results = []
    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"Experiment: {exp.get('name', 'unnamed')}")
        print(
            f"epochs={exp['epochs']} | batch_size={exp['batch_size']} | learning_rate={exp['learning_rate']} | seed={seed}"
        )

        test_accuracy = train_model(
            epochs=exp["epochs"],
            batch_size=exp["batch_size"],
            lr=exp["learning_rate"],
            seed=seed,
            data_root=data_root,
            download=download,
        )
        results.append({"name": exp.get("name", "unnamed"), "test_accuracy": test_accuracy})

    print("\n" + "-" * 80)
    print("Sammanfattning (test accuracy, sista epoken):")
    for r in results:
        print(f"- {r['name']}: {r['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
