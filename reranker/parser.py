import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMR/DPP reranking.")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="glit2021",
        help="Choose a dataset from {glit2021, glit2022}.",
    )
    parser.add_argument(
        "--data_path", nargs="?", default="datasets", help="Input data path."
    )
    parser.add_argument(
        "--Ks",
        nargs="?",
        default="[5,10,20]",
        help="Calculate metric@K when evaluating.",
    )
    parser.add_argument(
        "--seed", type=int, default=2022, help="Seed for ranking model."
    )
    parser.add_argument("--model", type=str, default="mmr", help="Model name.")
    parser.add_argument(
        "--lambda_factor", type=float, default=0.7, help="Lambda factor for MMR."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for DPP."
    )
    parser.add_argument(
        "--user_embeddings_path",
        type=str,
        default=None,
        help="Path to user embeddings file.",
    )
    parser.add_argument(
        "--item_embeddings_path",
        type=str,
        default=None,
        help="Path to item embeddings file.",
    )
    parser.add_argument(
        "--use_parallel",
        action="store_true",
        help="Use parallel processing for reranking.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing.",
    )

    args = parser.parse_args()

    return args
