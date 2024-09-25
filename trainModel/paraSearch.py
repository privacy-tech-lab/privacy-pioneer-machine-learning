# model was tuned and tested using Python 3.11.9
from datasets import DatasetDict, Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer
)
from ray import tune
import wandb

def main():
    """
    Main function to run the hyperparameter search using Ray Tune.
    """
    # Initialize Weights & Biases
    wandb.login()
    
    # Load datasets
    val_dataset_path = 'allVal.jsonl'
    train_dataset_path = 'allTrain.jsonl'
    dataset = load_datasets(train_dataset_path, val_dataset_path)
    
    # Perform hyperparameter search
    perform_hyperparameter_search(dataset)

def load_datasets(train_path, val_path):
    """
    Load datasets from the given paths.

    Args:
        train_path (str): Path to the training dataset.
        val_path (str): Path to the validation dataset.

    Returns:
        DatasetDict: A dictionary containing the training and validation datasets.
    """
    train_dataset = Dataset.from_json(train_path)
    val_dataset = Dataset.from_json(val_path)
    dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})
    return dataset

def perform_hyperparameter_search(dataset):
    """
    Perform hyperparameter search using Ray Tune.

    Args:
        dataset (DatasetDict): The dataset containing training and validation sets.
    """
    # Set the task and model checkpoint
    task = "cola"
    model_checkpoint = "huawei-noah/TinyBERT_General_4L_312D"

    # Prepare the dataset
    dataset = prepare_dataset(dataset, model_checkpoint)

    # Define model creation function
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=2
        )

    # Define hyperparameter search space for Ray Tune
    def ray_hp_space(trial):
        return {
            "learning_rate": tune.loguniform(1e-8, 1e-2),
            "per_device_train_batch_size": tune.choice([4, 8, 16]),
            "per_device_eval_batch_size": tune.choice([4, 8, 16]),
            "warmup_steps": tune.choice([100, 500, 1000, 1500, 2000]),
            "num_train_epochs": tune.choice([2, 4, 6, 8, 10, 12]),
            "weight_decay": tune.loguniform(1e-4, 1e-1),
        }

    # Initialize Trainer
    trainer = Trainer(
        model_init=model_init,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=AutoTokenizer.from_pretrained(model_checkpoint),
        data_collator=DataCollatorWithPadding(
            tokenizer=AutoTokenizer.from_pretrained(model_checkpoint), return_tensors="pt"
        ),
    )

    # Perform hyperparameter search
    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="ray",
        n_trials=1,
        hp_space=ray_hp_space,
        resources_per_trial={"cpu": 16, "gpu": 1},
        compute_objective=lambda metrics: metrics['eval_loss'],
    )

    # Output the best hyperparameters found
    print("Best hyperparameters found:")
    print(best_trial.hyperparameters)

def prepare_dataset(dataset, model_checkpoint):
    """
    Prepare the dataset by tokenizing and setting up labels.

    Args:
        dataset (DatasetDict): The original dataset.
        model_checkpoint (str): The model checkpoint for the tokenizer.

    Returns:
        DatasetDict: The prepared dataset.
    """
    # Create new label column with ClassLabel
    for split in ['train', 'validation']:
        new_features = dataset[split].features.copy()
        new_features["label"] = ClassLabel(num_classes=2, names=['True', 'False'])
        dataset[split] = dataset[split].cast(new_features)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Preprocess datasets
    def preprocess_function(examples):
        return tokenizer(examples["data"], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    return encoded_dataset

if __name__ == '__main__':
    main()
