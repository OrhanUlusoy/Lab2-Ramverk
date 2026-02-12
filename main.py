import json
from train import train_model


def main():

    with open("params.json", "r") as f:
        params = json.load(f)

        accuracy = train_model(
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["learning_rate"],
        )


    print(f"\nFinal Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
