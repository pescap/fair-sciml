import argparse
import glob

from ml.deeponet_trainer import DeepONetTrainer, FieldLoader


def main():
    parser = argparse.ArgumentParser(
        description="Train a DeepONet on the sphere-array Helmholtz dataset"
    )
    parser.add_argument("--data", type=str, default="simulations/spheres/*/*.h5")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()

    trainer = DeepONetTrainer(
        branch_hidden_layers=[256, 256, 256],
        trunk_hidden_layers=[256, 256, 256],
        data_loader=FieldLoader(sorted(glob.glob(args.data))),
    )
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    (branch_test, trunk_test), output_test = trainer.prepare_data()[2:]
    prediction = trainer.model.predict((branch_test, trunk_test))
    print(trainer.evaluate(output_test, prediction))


if __name__ == "__main__":
    main()
