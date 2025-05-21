# model was trained and tested using Python 3.11.9
import os
import numpy as np
from datasets import DatasetDict, Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import wandb

def main():
    """
    Main function to run the training and evaluation.
    """
    # Initialize Weights & Biases
    wandb.login()

    # Load datasets. You can found the training and validation datasets from our Hugging Face
    # https://huggingface.co/datasets/privacy-tech-lab/ppAllTrain
    # https://huggingface.co/datasets/privacy-tech-lab/ppAllVal
    val_dataset_path = 'allVal.jsonl'
    train_dataset_path = 'allTrain.jsonl'

    dataset = load_datasets(train_dataset_path, val_dataset_path)

    # Define hyperparameters
    # This is the best hyperparameter found by ray tune.
    setup_hyperparameters = [
        {
            'batch_size': 8,
            'learning_rate': 8.041e-05,
            'early_stopping_patience': 10,
            'warmup_steps': 2000,
            'weight_decay': 0.01465,
            'logging_steps': 5000,
            'save_steps': 5000,
            'num_epochs': 10
        }
    ]

    # Initialize run number
    run_num = 1

    for params in setup_hyperparameters:
        train_model(dataset, params, run_num)
        run_num += 1

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

def train_model(dataset, params, run_num):
    """
    Train the model with the given dataset and parameters.

    Args:
        dataset (DatasetDict): The dataset containing training and validation sets.
        params (dict): A dictionary of hyperparameters.
        run_num (int): The run number for logging purposes.
    """
    # Unpack parameters
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    early_stopping_patience = params['early_stopping_patience']
    warmup_steps = params['warmup_steps']
    weight_decay = params['weight_decay']
    logging_steps = params['logging_steps']
    save_steps = params['save_steps']
    num_epochs = params['num_epochs']

    print(f"Current training parameters: {params}")

    # Set the task and model checkpoint
    task = "cola"
    model_checkpoint = "huawei-noah/TinyBERT_General_4L_312D"

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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_dataset = encoded_dataset['train']
    val_dataset = encoded_dataset['validation']

    # Define the model
    def create_model():
        num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels
        )
        return model

    model = create_model()

    # Define compute_metrics
    def compute_metrics(prediction):
        logits = prediction.predictions
        labels = prediction.label_ids
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        }

    # Define model name and output directory
    model_name = model_checkpoint.split("/")[-1]
    project_folder = "compoundDistilled-pt-tuning"
    output_dir = os.path.join(project_folder, f"{model_name}-finetuned-{task}-pp-{project_folder}-{run_num}")

    # Initialize Weights & Biases run
    wandb.init(project=project_folder, name=f'run_{run_num}', config=params)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, 'logs'),
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_steps=logging_steps,
        save_steps=save_steps,
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        report_to="wandb",
        save_safetensors=False 
    )

    # Define early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback],
        model_init=create_model
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)

    # Evaluate the model
    evaluate_model(encoded_dataset['validation'], output_dir, project_folder, run_num, num_epochs, batch_size, learning_rate, early_stopping_patience)

    # Finish Weights & Biases run
    wandb.finish()

def evaluate_model(validation_dataset, model_dir, project_folder, run_num, num_epochs, batch_size, learning_rate, early_stopping_patience):
    """
    Evaluate the trained model on the validation dataset.

    Args:
        validation_dataset (Dataset): The validation dataset.
        model_dir (str): The directory where the trained model is saved.
        project_folder (str): The project folder name.
        run_num (int): The run number.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size used during training.
        learning_rate (float): Learning rate used during training.
        early_stopping_patience (int): Early stopping patience used during training.
    """
    # Load the trained model
    classifier = pipeline('text-classification', model=model_dir, tokenizer=model_dir)

    # Prepare true labels and texts
    texts = validation_dataset['data']
    true_labels = validation_dataset['label']

    # Make predictions in batches
    preds = []
    batch_size_eval = 32  
    for i in range(0, len(texts), batch_size_eval):
        batch_texts = texts[i:i+batch_size_eval]
        results = classifier(batch_texts)
        for res in results:
            label = res['label']
            preds.append(0 if label == 'LABEL_0' else 1)

    # Generate classification report
    report = classification_report(true_labels, preds, target_names=['False', 'True'])
    print(f"Model: {model_dir}\n\nNumber of Epochs: {num_epochs}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nEarly Stopping Patience: {early_stopping_patience}\n\nClassification Report:\n{report}")

    # Additional evaluation per type
    for ty in ['city', 'region', 'lat', 'lng', 'zip']:
        type_indices = [i for i, text in enumerate(texts) if text.startswith(ty)]
        if type_indices:
            type_texts = [texts[i] for i in type_indices]
            type_true_labels = [true_labels[i] for i in type_indices]
            # Make predictions
            type_preds = []
            for i in range(0, len(type_texts), batch_size_eval):
                batch_texts = type_texts[i:i+batch_size_eval]
                results = classifier(batch_texts)
                for res in results:
                    label = res['label']
                    type_preds.append(0 if label == 'LABEL_0' else 1)
            type_report = classification_report(type_true_labels, type_preds, target_names=['False', 'True'])
            print(f"Type: {ty}\n\nClassification Report:\n{type_report}")

if __name__ == '__main__':
    main()
