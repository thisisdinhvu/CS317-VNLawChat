from metaflow import FlowSpec, step, Parameter
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback, get_scheduler
from torch.optim import AdamW
import pandas as pd

from prepare_data import download_and_prepare


def create_contrastive_dataset(filtered_data):
    dataset_dict = {
        "anchor": [],
        "positive": [],
    }
    for _, row in filtered_data.iterrows():
        dataset_dict["anchor"].append(row['query'])
        dataset_dict["positive"].append(row['answer'])

    return Dataset.from_dict(dataset_dict)


class TrainSentenceTransformerFlow(FlowSpec):

    epochs = Parameter("epochs", default=5)
    batch_size = Parameter("batch_size", default=32)
    lr = Parameter("lr", default=2e-5)
    model_name = Parameter("model_name", default="bkai-foundation-models/vietnamese-bi-encoder")

    @step
    def start(self):
        print("Downloading and preparing data...")
        self.data = download_and_prepare()
        self.next(self.split_data)

    @step
    def split_data(self):
        train_data, eval_data = train_test_split(self.data, test_size=0.1, random_state=42)
        print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")
        self.train_dataset = create_contrastive_dataset(train_data)
        self.eval_dataset = create_contrastive_dataset(eval_data)
        self.next(self.load_model)

    @step
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True).to('cuda')
        self.loss = losses.CachedMultipleNegativesRankingLoss(self.model, mini_batch_size=1024)
        self.next(self.train)

    @step
    def train(self):
        args = SentenceTransformerTrainingArguments(
            output_dir="models/BKAI",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_ratio=0.1,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            learning_rate=self.lr,
            save_strategy="steps",
            save_steps=300,
            save_total_limit=2,
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=300,
            metric_for_best_model="eval_loss",
            warmup_steps=50,
            fp16=True,
            report_to="wandb",
            weight_decay=0.01,
            load_best_model_at_end=True
        )

        optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        num_training_steps = len(self.train_dataset) // args.per_device_train_batch_size * args.num_train_epochs
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(len(self.train_dataset) * 0.1),
            num_training_steps=num_training_steps
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=self.loss,
            optimizers=(optimizer, scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.005)]
        )

        trainer.train()
        self.next(self.end)

    @step
    def end(self):
        print("Training completed successfully!")


if __name__ == "__main__":
    TrainSentenceTransformerFlow()
