import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run JR4CE.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="glit2021",
        help="Choose a dataset from {glit2021, glit2022}.",
    )
    parser.add_argument(
        "--data_path", nargs="?", default="datasets", help="Input data path."
    )
    parser.add_argument("--dim", type=int, default=32, help="Number of dimension.")
    parser.add_argument("--epoch", type=int, default=50, help="Number of epoch.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument(
        "--patience", type=int, default=10, help="Number of epoch for early stopping."
    )
    parser.add_argument(
        "--Ks",
        nargs="?",
        default="[5,10,20]",
        help="Calculate metric@K when evaluating.",
    )
    parser.add_argument(
        "--val_interval", type=int, default=1, help="Validation interval."
    )
    parser.add_argument(
        "--save_path",
        nargs="?",
        default="trained_model",
        help="Model path for saving.",
    )
    parser.add_argument(
        "--model_path", nargs="?", default="", help="Model path for evaluation."
    )
    parser.add_argument(
        "--neg_size", type=int, default=2, help="Negative sampling size."
    )
    parser.add_argument(
        "--div_lambda", type=float, default=0.1, help="Lambda for diversity loss."
    )
    parser.add_argument(
        "--threshold_user",
        type=float,
        default=0.5,
        help="Similarity threshold for user.",
    )
    parser.add_argument(
        "--threshold_item",
        type=float,
        default=0.5,
        help="Disimilarity threshold for item.",
    )
    parser.add_argument(
        "--num_gcn_layer", type=int, default=2, help="Number of GCN layers."
    )
    parser.add_argument(
        "--kgl_module", type=int, default=1, help="Knowledge graph learning module."
    )
    parser.add_argument(
        "--cf_module", type=int, default=1, help="Collaborative filtering module."
    )

    return parser.parse_args()
