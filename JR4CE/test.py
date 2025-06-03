import torch

from .dataset import TrainDataset, eval_dataset
from .metrics import evaluate
from .parser import parse_args
from .utils import seed_everything

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    seed_everything(args.seed)

    data_path = f"{args.data_path}/{args.dataset}"

    train_dataset = TrainDataset(
        data_path=data_path,
        neg_sample_size=args.neg_size,
        threshold_user=args.threshold_user,
        threshold_item=args.threshold_item,
        neg_uniform=args.neg_uniform == 1,
    )
    graph_data = train_dataset.data(device)
    test_data = eval_dataset(data_path, "test.txt")

    model = torch.load(args.model_path, weights_only=False).to(device)
    Ks = eval(args.Ks)
    evaluate(test_data, model, Ks, device, graph_data, log=print, save_ranking=True)
