import matplotlib
import pytorch_lightning as pl
import torch
from clearml import Task
from itmo_pirsii_2023_diploma.src.relation.dataset import RawDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Disable tk warnings
matplotlib.use('Agg')


def get_gemini_model(params):
    from itmo_pirsii_2023_diploma.src.relation.gemini_model import GeminiClassifierModel
    return GeminiClassifierModel(**params), "gemini"


def plot_confusion_matrix(task: Task, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # fig, ax = plt.subplots(figsize=(10, 8))
    # fig = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax).get_figure()
    # ax.set_xlabel("Predicted labels")
    # ax.set_ylabel("True labels")
    # plt.close(fig)

    # Log confusion matrix to TensorBoard
    task.get_logger().report_confusion_matrix(f"test_confusion_matrix", series="result", iteration=0,
                                              matrix=cm, xaxis="Predicted", yaxis="True labels", yaxis_reversed=True)


if __name__ == "__main__":
    ## Set GEMINI_API_KEY env variable to use Gemini model

    from loader import get_loaders
    from dotenv import load_dotenv
    import logging

    load_dotenv()

    logger = logging.getLogger(__name__)
    logging.basicConfig()

    use_clearml = True  # True = use clearML, False = dont use
    continue_last_model = False  # True = use last model, need to set ckpt_path in section below, False = train model from zero

    if use_clearml:
        from clearml import Task

        project_name = "Diploma"
        task_tag = "relation"
        task = Task.init(project_name=project_name, tags=[task_tag],  # , "fix:1", "bert-encoding"
                         continue_last_task=continue_last_model,
                         )  # , continue_last_task=True, reuse_last_task_id='283151c414814d9793e80284e2b8d1d7'

    # Set params
    max_length = 110
    dataset_version = "v4.1"
    # Set seed
    seed = 0

    params = {
        "seed": seed,
        "loss_type": "binary_cross_entropy",
        "max_epochs": 1,
        "batch_size": 5,
        "output_dim": 1,
        "max_length": max_length,
        "dataset_version": dataset_version,
        "debug_size": 10,
        "prompt_dir": r"src\relation\prompts",
        "prompt_id": "5",
        "model_name": "gemini-2.0-flash-lite-001",
    }

    params["vocab_size"] = 0

    if use_clearml: task.connect(params)

    pl.seed_everything(seed)

    # Load data
    train_loader, val_loader, test_loader, debug_loader = get_loaders(
        test_path=f"dataset/relation/test.tsv",
        batch_size=params["batch_size"],
        max_length=max_length,
        debug_size=params["debug_size"],
        output_dim=params["output_dim"],
        raw_dataset=True)

    # Get model
    model, tag = get_gemini_model(params)

    if use_clearml:
        task.set_model_config(config_text=str(model))
        task.set_name(model.name)
        task.add_tags(tag)
        task.add_tags(f"ds:{params['dataset_version']}")
    #
    # # Configure trainer
    # trainer = pl.Trainer(max_epochs=params["max_epochs"],
    #                      logger=TensorBoardLogger("D:/Projects/diplom/tb_logs", name=model.name),
    #                      )
    test_dataset = RawDataset.from_csv(f"dataset/relation/test.tsv", None,
                                       max_length, sep="\t", output_dim=params["output_dim"])
    results = []
    targets = []
    for idx in tqdm(range(0, len(test_dataset), params["batch_size"])):
        sents, y = test_dataset[idx:idx + params["batch_size"]]
        results.extend(model(*sents))
        targets.extend(y)
    t = torch.concat(targets)
    r = torch.tensor(results)
    mask = (r != -1)

    t = t[mask]
    r = r[mask]
    skipped = mask.shape[0] - mask.sum().item()

    acc = accuracy_score(t, r)
    f1 = f1_score(t, r)

    task.get_logger().report_scalar("test", "f1_score", f1, 0)
    task.get_logger().report_scalar("test", "acc", acc, 0)
    task.get_logger().report_scalar("skipped", "number", skipped, 0)

    plot_confusion_matrix(task, t, r)

    if use_clearml:
        task.close()
