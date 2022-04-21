import wandb


def main():
    # Config
    wandb.init(mode="offline", config={"optimizer": "adam", "lr": 0.001})

    # History
    wandb.log({"metric": 120.0})
    wandb.log({"metric": 50.0})
    wandb.log({"metric": 10.0})
    wandb.log({"metric": 5.0})
    wandb.log({"metric": 1.0})
    wandb.log({"metric": 15.0})


if __name__ == "__main__":
    main()
